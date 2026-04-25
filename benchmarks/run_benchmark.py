#!/usr/bin/env python3
"""
Comprehensive benchmarking script for MSM GPU implementation.

Runs ./msm_bn254_mgpu with different input sizes and modes (even vs greedy).
Captures timing, planner parameters, and prediction accuracy.

Outputs:
  - benchmark_N*.csv (per input size)
  - benchmark_final_all.csv (combined results)
"""

import subprocess
import sys
import os
import re
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Configuration
N_VALUES = [1000000, 5000000, 10000000, 20000000, 35000000, 50000000]
WBITS = 8
MODES = {
    'even': 0,           # Static round-robin
    'greedy': 1          # Greedy planner
}
NUM_GPUS = 2
BINARY = '../build/msm_bn254_mgpu'
BENCHMARK_DIR = '.'

class MSMBenchmarker:
    def __init__(self, binary_path: str, benchmark_dir: str):
        self.benchmark_dir = Path(benchmark_dir).resolve()
        binary = Path(binary_path)
        if not binary.is_absolute():
            binary = (self.benchmark_dir / binary).resolve()
        self.binary = str(binary)
        # Keep datasets in one place: the binary's folder (build/).
        self.run_dir = str(binary.parent)
        self.results = []
        
        if not os.path.exists(self.binary):
            raise FileNotFoundError(f"Binary not found: {self.binary}")
    
    def run_msm(self, N: int, wbits: int, mode: int) -> Tuple[str, str]:
        """
        Run MSM binary with given parameters.
        
        Args:
            N: Number of scalars/points
            wbits: Window bits
            mode: 0=even (static RR), 1=greedy (planner)
        
        Returns:
            (stdout, stderr)
        """
        cmd = [
            self.binary,
            str(N),              # argv[1]: N
            str(wbits),          # argv[2]: wbits
            str(mode),           # argv[3]: use_greedy (0=even, 1=greedy)
            '0',                 # argv[4]: check (0=off)
            '0',                 # argv[5]: objective (0=throughput)
            '1',                 # argv[6]: audit_stage_timing (1=on for logging)
            '0',                 # argv[7]: force_split_test
            '1',                 # argv[8]: use_reusable_dataset (use cached!)
        ]
        
        print(f"Running: {' '.join(cmd)}", file=sys.stderr)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, cwd=self.run_dir)
        
        if result.returncode != 0:
            print(f"Error running binary: {result.stderr}", file=sys.stderr)
            raise RuntimeError(f"Binary exited with code {result.returncode}")
        
        return result.stdout, result.stderr
    
    def parse_output(self, stdout: str, stderr: str, N: int, wbits: int, mode: str) -> Dict[str, Any]:
        """
        Parse stdout/stderr to extract timing and prediction data.
        
        Returns:
            Dict with keys: N, wbits, mode, num_gpus, setup_ms, warmup_ms, window_ms, 
                           finalize_ms, total_time_ms, throughput, predicted_total_ms,
                           actual_total_ms, prediction_error_pct, plus planner params
        """
        result = {
            'N': N,
            'wbits': wbits,
            'mode': mode,
            'num_gpus': NUM_GPUS,
        }
        
        # Extract from stdout
        # Format: "N=1000000, wbits=8, GPUs=2, mode=Greedy"
        # Or: "setup_ms=..., warmup_ms=..., window_ms=..., finalize_ms=..., total_ms=..."
        
        # Try to find timing line in stdout
        for line in stdout.split('\n'):
            if 'setup' in line and 'ms' in line:
                # Extract timing breakdown
                setup_match = re.search(r'setup[_\s]*(?:time[_\s]*)?(?:in[_\s]*)?.*?([0-9.e+-]+)\s*ms', line, re.IGNORECASE)
                if setup_match:
                    result['setup_ms'] = float(setup_match.group(1))
            
            if 'throughput' in line.lower():
                tp_match = re.search(r'throughput[_\s]*(?:is[_\s]*)?([0-9.e+-]+)', line, re.IGNORECASE)
                if tp_match:
                    result['throughput'] = float(tp_match.group(1))
        
        # Extract from stderr (detailed audit output)
        result['k_compute_small'] = self._extract_param(stderr, r'k_compute_small[_\s]*=\s*([0-9.e+-]+)')
        result['k_compute_mid'] = self._extract_param(stderr, r'k_compute_mid[_\s]*=\s*([0-9.e+-]+)')
        result['k_compute_large'] = self._extract_param(stderr, r'k_compute_large[_\s]*=\s*([0-9.e+-]+)')
        result['k_digit'] = self._extract_param(stderr, r'k_digit[_\s]*=\s*([0-9.e+-]+)')
        result['k_count'] = self._extract_param(stderr, r'k_count[_\s]*=\s*([0-9.e+-]+)')
        result['k_merge'] = self._extract_param(stderr, r'k_merge[_\s]*=\s*([0-9.e+-]+)')
        result['k_suffix'] = self._extract_param(stderr, r'k_suffix[_\s]*=\s*([0-9.e+-]+)')
        result['alpha_pack'] = self._extract_param(stderr, r'alpha_pack[_\s]*=\s*([0-9.e+-]+)')
        result['num_sms'] = self._extract_param(stderr, r'num_sms[_\s]*=\s*([0-9]+)')
        result['M_g_bytes'] = self._extract_param(stderr, r'M_g[_\s]*=\s*([0-9.e+-]+)')
        
        # Extract timing from stderr (multiple window_total_time and predicted_latency lines)
        window_times = []
        predicted_times = []
        
        for line in stderr.split('\n'):
            if 'window_total_time=' in line:
                match = re.search(r'window_total_time=([\d.eE+-]+)', line)
                if match:
                    window_times.append(float(match.group(1)))
            
            if 'predicted_latency_model=' in line:
                match = re.search(r'predicted_latency_model=([\d.eE+-]+)', line)
                if match:
                    predicted_times.append(float(match.group(1)))  # Already in milliseconds from binary
        
        # Aggregate window times
        if window_times:
            result['window_count'] = len(window_times)
            result['avg_window_ms'] = sum(window_times) / len(window_times)
            result['min_window_ms'] = min(window_times)
            result['max_window_ms'] = max(window_times)
            result['total_window_time_ms'] = sum(window_times)
        
        # Aggregate predicted times
        if predicted_times:
            result['avg_predicted_ms'] = sum(predicted_times) / len(predicted_times)
            result['total_predicted_ms'] = sum(predicted_times)
        
        # Calculate prediction error
        if 'total_predicted_ms' in result and 'total_window_time_ms' in result:
            pred = result['total_predicted_ms']
            actual = result['total_window_time_ms']
            error_pct = ((pred - actual) / actual * 100) if actual > 0 else 0
            result['prediction_error_pct'] = error_pct
        
        return result
    
    def _extract_param(self, text: str, pattern: str) -> Any:
        """Extract numeric parameter from text using regex."""
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                val_str = match.group(1)
                # Try to parse as float/scientific notation
                if 'e' in val_str.lower():
                    return float(val_str)
                elif '.' in val_str:
                    return float(val_str)
                else:
                    return int(val_str)
            except (ValueError, AttributeError):
                return None
        return None
    
    def run_benchmark_suite(self):
        """Run full benchmark suite for all N values and modes."""
        print("Starting MSM GPU Benchmarking Suite...", file=sys.stderr)
        print(f"Input sizes: {N_VALUES}", file=sys.stderr)
        print(f"Modes: {list(MODES.keys())}", file=sys.stderr)
        print(f"Results directory: {self.benchmark_dir}", file=sys.stderr)
        print("", file=sys.stderr)
        
        for N in N_VALUES:
            print(f"\n{'='*70}", file=sys.stderr)
            print(f"Benchmarking N={N:,}", file=sys.stderr)
            print(f"{'='*70}\n", file=sys.stderr)
            
            results_for_n = []
            
            for mode_name, mode_val in MODES.items():
                print(f"  Mode: {mode_name} (use_greedy={mode_val})", file=sys.stderr)
                
                try:
                    stdout, stderr = self.run_msm(N, WBITS, mode_val)
                    result = self.parse_output(stdout, stderr, N, WBITS, mode_name)
                    results_for_n.append(result)
                    
                    # Print summary
                    self._print_summary(result)
                    
                except Exception as e:
                    print(f"  ERROR: {e}", file=sys.stderr)
                    continue
            
            # Save per-N CSV
            self._save_csv(results_for_n, N)
            self.results.extend(results_for_n)
        
        # Save combined CSV
        self._save_combined_csv()
        
        print(f"\n{'='*70}", file=sys.stderr)
        print(f"Benchmarking complete!", file=sys.stderr)
        print(f"Results saved in: {self.benchmark_dir}", file=sys.stderr)
        print(f"{'='*70}\n", file=sys.stderr)
    
    def _print_summary(self, result: Dict[str, Any]):
        """Print brief summary of a run."""
        mode = result.get('mode', 'unknown')
        N = result.get('N', 0)
        window_ms = result.get('avg_window_ms', 0)
        pred_ms = result.get('avg_predicted_ms', 0)
        error_pct = result.get('prediction_error_pct', 0)
        throughput = result.get('throughput', 0)
        
        print(f"    ✓ Avg window: {window_ms:.1f}ms, Predicted: {pred_ms:.1f}ms, "
              f"Error: {error_pct:+.1f}%, Throughput: {throughput:.0f}", file=sys.stderr)
    
    def _save_csv(self, results: List[Dict[str, Any]], N: int):
        """Save results for a single N to CSV."""
        filename = self.benchmark_dir / f'benchmark_N{N}.csv'
        
        if not results:
            print(f"  No results to save for N={N}", file=sys.stderr)
            return
        
        # Get all unique keys from all results
        all_keys = set()
        for r in results:
            all_keys.update(r.keys())
        
        # Sort keys for consistent output
        fieldnames = sorted(list(all_keys))
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, restval='')
            writer.writeheader()
            writer.writerows(results)
        
        print(f"  Saved: {filename}", file=sys.stderr)
    
    def _save_combined_csv(self):
        """Save all results to combined CSV."""
        filename = self.benchmark_dir / 'benchmark_final_all.csv'
        
        if not self.results:
            print("No results to save", file=sys.stderr)
            return
        
        # Get all unique keys
        all_keys = set()
        for r in self.results:
            all_keys.update(r.keys())
        
        fieldnames = sorted(list(all_keys))
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, restval='')
            writer.writeheader()
            writer.writerows(self.results)
        
        print(f"Saved combined results: {filename}", file=sys.stderr)


def main():
    try:
        benchmarker = MSMBenchmarker(BINARY, BENCHMARK_DIR)
        benchmarker.run_benchmark_suite()
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
