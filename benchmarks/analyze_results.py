#!/usr/bin/env python3
"""
Analysis utilities for MSM GPU benchmarking results.

Usage:
  python3 analyze_results.py [csv_file]
  
Examples:
  python3 analyze_results.py benchmark_final_all.csv
  python3 analyze_results.py benchmark_N1000000.csv
"""

import sys
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

def load_csv(filename: str) -> List[Dict[str, str]]:
    """Load CSV file into list of dicts."""
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)

def to_number(val: str) -> Any:
    """Convert string to number if possible."""
    if not val or val.lower() == 'nan':
        return None
    try:
        if '.' in val or 'e' in val.lower():
            return float(val)
        return int(val)
    except ValueError:
        return val

def print_comparison_table(data: List[Dict[str, str]]):
    """Print mode comparison table (even vs greedy)."""
    print("\n" + "="*100)
    print("MODE COMPARISON (Even vs Greedy)")
    print("="*100)
    print(f"{'N':>10} | {'Mode':>8} | {'Avg Window':>12} | {'Predicted':>12} | {'Error %':>10} | {'Throughput':>12}")
    print("-"*100)
    
    by_n = defaultdict(dict)
    for row in data:
        try:
            n = int(row.get('N', 0))
            mode = row.get('mode', '')
            window = to_number(row.get('avg_window_ms', ''))
            pred = to_number(row.get('avg_predicted_ms', ''))
            error = to_number(row.get('prediction_error_pct', ''))
            tp = to_number(row.get('throughput', ''))
            
            # Calculate throughput if not available (scalars per ms per window)
            if tp is None and window and window > 0:
                tp = n / window / 1000.0  # N points per window total time (ms) / 1000 for throughput
            
            by_n[n][mode] = {
                'window': window,
                'pred': pred,
                'error': error,
                'tp': tp if tp else 0.0
            }
        except (ValueError, KeyError):
            continue
    
    for n in sorted(by_n.keys()):
        for mode in ['even', 'greedy']:
            if mode in by_n[n]:
                m = by_n[n][mode]
                print(f"{n:>10} | {mode:>8} | {m['window']:>12.2f} | {m['pred']:>12.2f} | "
                      f"{m['error']:>10.2f} | {m['tp']:>12.1f}")
    
    # Speedup analysis
    print("\n" + "-"*100)
    print(f"{'N':>10} | {'Speedup (even/greedy)':>30} | {'Improvement':>15}")
    print("-"*100)
    for n in sorted(by_n.keys()):
        if 'even' in by_n[n] and 'greedy' in by_n[n]:
            even_w = by_n[n]['even']['window']
            greedy_w = by_n[n]['greedy']['window']
            if even_w and greedy_w:
                speedup = even_w / greedy_w
                improve_pct = (even_w - greedy_w) / even_w * 100
                print(f"{n:>10} | {speedup:>30.4f}x | {improve_pct:>14.2f}%")

def print_parameter_analysis(data: List[Dict[str, str]]):
    """Print planner parameters used."""
    print("\n" + "="*100)
    print("PLANNER PARAMETERS")
    print("="*100)
    
    # Get first row and extract params (same for all runs)
    if data:
        row = data[0]
        print(f"\nk_compute_mid:  {row.get('k_compute_mid', 'N/A')}")
        print(f"k_compute_small: {row.get('k_compute_small', 'N/A')}")
        print(f"k_compute_large: {row.get('k_compute_large', 'N/A')}")
        print(f"alpha_pack:      {row.get('alpha_pack', 'N/A')}")
        print(f"k_digit:         {row.get('k_digit', 'N/A')}")
        print(f"k_count:         {row.get('k_count', 'N/A')}")
        print(f"k_merge:         {row.get('k_merge', 'N/A')}")
        print(f"k_suffix:        {row.get('k_suffix', 'N/A')}")
        print(f"num_sms:         {row.get('num_sms', 'N/A')}")
        print(f"M_g_bytes:       {row.get('M_g_bytes', 'N/A')}")

def print_prediction_accuracy(data: List[Dict[str, str]]):
    """Analyze prediction accuracy across all runs."""
    print("\n" + "="*100)
    print("PREDICTION ACCURACY ANALYSIS")
    print("="*100)
    print(f"{'N':>10} | {'Mode':>8} | {'Error %':>10} | {'Status':>15}")
    print("-"*100)
    
    errors = []
    for row in data:
        try:
            n = row.get('N', '')
            mode = row.get('mode', '')
            error = to_number(row.get('prediction_error_pct', ''))
            if error is not None:
                errors.append(abs(error))
                status = '✓ Excellent' if abs(error) < 5 else ('✓ Good' if abs(error) < 20 else '✗ High')
                print(f"{n:>10} | {mode:>8} | {error:>10.2f} | {status:>15}")
        except (ValueError, KeyError):
            continue
    
    if errors:
        print("\n" + "-"*100)
        avg_error = sum(errors) / len(errors)
        max_error = max(errors)
        print(f"Average prediction error: {avg_error:.2f}%")
        print(f"Maximum prediction error: {max_error:.2f}%")
        print(f"Prediction quality: {'EXCELLENT' if avg_error < 5 else ('GOOD' if avg_error < 15 else 'FAIR')}")

def print_scaling_analysis(data: List[Dict[str, str]]):
    """Analyze scaling behavior."""
    print("\n" + "="*100)
    print("SCALING ANALYSIS (Greedy mode)")
    print("="*100)
    print(f"{'N':>10} | {'Avg Window':>12} | {'Throughput':>12} | {'Efficiency':>12}")
    print("-"*100)
    
    greedy_runs = [row for row in data if row.get('mode') == 'greedy']
    by_n = {}
    
    for row in greedy_runs:
        try:
            n = int(row.get('N', 0))
            window = to_number(row.get('avg_window_ms', ''))
            tp = to_number(row.get('throughput', ''))
            
            # Calculate throughput if not available
            if tp is None and window and window > 0:
                tp = n / window / 1000.0
            
            by_n[n] = {'window': window, 'tp': tp if tp else 0.0}
        except (ValueError, KeyError):
            continue
    
    prev_window = None
    for n in sorted(by_n.keys()):
        m = by_n[n]
        efficiency = 'N/A'
        if prev_window and m['window']:
            # Efficiency: how well does window time scale with N?
            scale_factor = n / min(by_n.keys())
            expected_window = by_n[min(by_n.keys())]['window'] * scale_factor
            efficiency = f"{m['window']/expected_window*100:.1f}%"
        
        print(f"{n:>10} | {m['window']:>12.2f} | {m['tp']:>12.1f} | {efficiency:>12}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_results.py <csv_file>", file=sys.stderr)
        print("Example: python3 analyze_results.py benchmark_final_all.csv", file=sys.stderr)
        return 1
    
    csv_file = sys.argv[1]
    
    if not Path(csv_file).exists():
        print(f"Error: File not found: {csv_file}", file=sys.stderr)
        return 1
    
    print(f"Analyzing: {csv_file}", file=sys.stderr)
    data = load_csv(csv_file)
    
    if not data:
        print(f"Error: No data found in {csv_file}", file=sys.stderr)
        return 1
    
    print_comparison_table(data)
    print_parameter_analysis(data)
    print_prediction_accuracy(data)
    print_scaling_analysis(data)
    
    print("\n" + "="*100)
    print(f"Total runs analyzed: {len(data)}")
    print("="*100 + "\n")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
