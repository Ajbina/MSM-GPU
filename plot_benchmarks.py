#!/usr/bin/env python3
"""
Benchmarking analysis and visualization script for MSM GPU pipeline.

Reads benchmark_results.csv and generates 6 analysis plots:
1. Runtime vs N (1 GPU vs 2 GPU)
2. Speedup vs N (ratio of 1 GPU to 2 GPU)
3. Throughput vs N (points/second)
4. Runtime vs wbits
5. Prediction error (predicted vs actual)
6. Stage decomposition (stacked bar chart)

CSV format (post-calibration):
  N, wbits, num_gpus, use_greedy, total_time_ms, setup_ms, warmup_ms,
  window_ms, finalize_ms, throughput, speedup, predicted_total_time_ms,
  actual_total_time_ms, prediction_error_pct, k_compute_active
  [+ suggested_k_small, suggested_k_mid, suggested_k_large if audit enabled]

Validation checks performed:
  - Speedup < 1.1 for 2 GPU (flag potential underutilization)
  - Runtime vs wbits nearly flat (flag insensitivity)
  - Prediction error magnitude > 10% (flag model calibration issues)

Usage:
    python3 plot_benchmarks.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

def load_data(filename="benchmark_results.csv"):
    """Load benchmark results from CSV file."""
    try:
        df = pd.read_csv(filename)
        print(f"Loaded {len(df)} records from {filename}")
        print(f"Columns: {list(df.columns)}")
        return df
    except FileNotFoundError:
        print(f"Error: {filename} not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        sys.exit(1)

def plot_runtime_vs_n(df, output_file="runtime_vs_N.png"):
    """Plot total runtime vs N for 1 GPU and 2 GPU."""
    print("Generating: runtime_vs_N.png")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter data: use_greedy=1 (standard runs), single wbits value
    df_1gpu = df[(df['num_gpus'] == 1) & (df['use_greedy'] == 1) & (df['wbits'] == 8)]
    df_2gpu = df[(df['num_gpus'] == 2) & (df['use_greedy'] == 1) & (df['wbits'] == 8)]
    
    if len(df_1gpu) > 0:
        df_1gpu_sorted = df_1gpu.sort_values('N')
        ax.plot(df_1gpu_sorted['N'], df_1gpu_sorted['total_time_ms'], 
                'o-', label='1 GPU', linewidth=2, markersize=6)
    
    if len(df_2gpu) > 0:
        df_2gpu_sorted = df_2gpu.sort_values('N')
        ax.plot(df_2gpu_sorted['N'], df_2gpu_sorted['total_time_ms'], 
                's-', label='2 GPU', linewidth=2, markersize=6)
    
    ax.set_xlabel('N (number of points)', fontsize=12)
    ax.set_ylabel('Total time (ms)', fontsize=12)
    ax.set_title('MSM Runtime vs N (wbits=8)', fontsize=13)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"  → Saved to {output_file}")
    plt.close()

def plot_speedup_vs_n(df, output_file="speedup_vs_N.png"):
    """Plot speedup (1 GPU time / 2 GPU time) vs N."""
    print("Generating: speedup_vs_N.png")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Merge 1 GPU and 2 GPU data on N and wbits
    df_1gpu = df[(df['num_gpus'] == 1) & (df['use_greedy'] == 1) & (df['wbits'] == 8)].copy()
    df_2gpu = df[(df['num_gpus'] == 2) & (df['use_greedy'] == 1) & (df['wbits'] == 8)].copy()
    
    if len(df_1gpu) > 0 and len(df_2gpu) > 0:
        df_1gpu.set_index('N', inplace=True)
        df_2gpu.set_index('N', inplace=True)
        
        # Compute speedup for matching N values
        common_n = df_1gpu.index.intersection(df_2gpu.index)
        if len(common_n) > 0:
            speedup = df_1gpu.loc[common_n, 'total_time_ms'] / df_2gpu.loc[common_n, 'total_time_ms']
            sorted_n = common_n.sort_values()
            ax.plot(sorted_n, speedup.loc[sorted_n], 'o-', linewidth=2, markersize=8, color='green')
            ax.axhline(y=2.0, color='red', linestyle='--', label='Ideal (2x)', alpha=0.7)
            
            ax.set_xlabel('N (number of points)', fontsize=12)
            ax.set_ylabel('Speedup (1 GPU time / 2 GPU time)', fontsize=12)
            ax.set_title('Speedup: 1 GPU vs 2 GPU (wbits=8)', fontsize=13)
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"  → Saved to {output_file}")
    plt.close()

def plot_throughput_vs_n(df, output_file="throughput_vs_N.png"):
    """Plot throughput (points/sec) vs N."""
    print("Generating: throughput_vs_N.png")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df_1gpu = df[(df['num_gpus'] == 1) & (df['use_greedy'] == 1) & (df['wbits'] == 8)]
    df_2gpu = df[(df['num_gpus'] == 2) & (df['use_greedy'] == 1) & (df['wbits'] == 8)]
    
    if len(df_1gpu) > 0:
        df_1gpu_sorted = df_1gpu.sort_values('N')
        ax.plot(df_1gpu_sorted['N'], df_1gpu_sorted['throughput'], 
                'o-', label='1 GPU', linewidth=2, markersize=6)
    
    if len(df_2gpu) > 0:
        df_2gpu_sorted = df_2gpu.sort_values('N')
        ax.plot(df_2gpu_sorted['N'], df_2gpu_sorted['throughput'], 
                's-', label='2 GPU', linewidth=2, markersize=6)
    
    ax.set_xlabel('N (number of points)', fontsize=12)
    ax.set_ylabel('Throughput (points/sec)', fontsize=12)
    ax.set_title('MSM Throughput vs N (wbits=8)', fontsize=13)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"  → Saved to {output_file}")
    plt.close()

def plot_runtime_vs_wbits(df, output_file="runtime_vs_wbits.png"):
    """Plot runtime vs wbits for different N values."""
    print("Generating: runtime_vs_wbits.png")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use 2 GPU runs, select a few N values
    df_2gpu = df[(df['num_gpus'] == 2) & (df['use_greedy'] == 1)]
    
    n_values = sorted(df_2gpu['N'].unique())
    
    for n in n_values[-3:]:  # Plot top 3 N values
        df_subset = df_2gpu[df_2gpu['N'] == n].sort_values('wbits')
        if len(df_subset) > 0:
            ax.plot(df_subset['wbits'], df_subset['total_time_ms'], 
                    'o-', label=f'N={n}', linewidth=2, markersize=6)
    
    ax.set_xlabel('wbits (window bits)', fontsize=12)
    ax.set_ylabel('Total time (ms)', fontsize=12)
    ax.set_title('MSM Runtime vs wbits (2 GPU)', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"  → Saved to {output_file}")
    plt.close()

def plot_prediction_error(df, output_file="prediction_error.png"):
    """Plot prediction error (percentage) vs N."""
    print("Generating: prediction_error.png")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Check if new column exists
    if 'prediction_error_pct' not in df.columns:
        print("  [Using fallback: computing error from predicted_total_time_ms / actual_total_time_ms]")
        df = df.copy()
        df['prediction_error_pct'] = (
            (df['predicted_total_time_ms'] - df['actual_total_time_ms']) 
            / df['actual_total_time_ms'] * 100.0
        )
    
    df_2gpu = df[(df['num_gpus'] == 2) & (df['use_greedy'] == 1) & (df['wbits'] == 8)].copy()
    
    # Filter out rows with zero actual time
    df_2gpu = df_2gpu[df_2gpu['actual_total_time_ms'] > 0]
    
    if len(df_2gpu) > 0:
        df_2gpu.sort_values('N', inplace=True)
        
        ax.plot(df_2gpu['N'], df_2gpu['prediction_error_pct'], 
                'o-', linewidth=2, markersize=8, color='purple')
        ax.axhline(y=0.0, color='green', linestyle='-', label='Perfect (0%)', linewidth=2)
        ax.axhline(y=5.0, color='orange', linestyle='--', label='±5%', alpha=0.7)
        ax.axhline(y=-5.0, color='orange', linestyle='--', alpha=0.7)
        
        ax.set_xlabel('N (number of points)', fontsize=12)
        ax.set_ylabel('Prediction Error (%)', fontsize=12)
        ax.set_title('Model Prediction Error vs N (2 GPU, wbits=8)', fontsize=13)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        # Add text annotation showing mean error
        mean_error = df_2gpu['prediction_error_pct'].mean()
        mean_abs_error = df_2gpu['prediction_error_pct'].abs().mean()
        ax.text(0.05, 0.95, f'Mean error: {mean_error:.1f}%\nMean |error|: {mean_abs_error:.1f}%', 
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"  → Saved to {output_file}")
    plt.close()

def plot_stage_decomposition(df, output_file="stage_decomposition.png"):
    """Plot stacked bar chart of stage timings (setup, warmup, window, finalize)."""
    print("Generating: stage_decomposition.png")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Use 2 GPU runs with wbits=8, sorted by N
    df_2gpu = df[(df['num_gpus'] == 2) & (df['use_greedy'] == 1) & (df['wbits'] == 8)].copy()
    df_2gpu.sort_values('N', inplace=True)
    
    if len(df_2gpu) > 0:
        x = np.arange(len(df_2gpu))
        width = 0.6
        
        setup = df_2gpu['setup_ms'].values
        warmup = df_2gpu['warmup_ms'].values
        window = df_2gpu['window_ms'].values
        finalize = df_2gpu['finalize_ms'].values
        
        ax.bar(x, setup, width, label='Setup', color='#1f77b4')
        ax.bar(x, warmup, width, bottom=setup, label='Warmup', color='#ff7f0e')
        ax.bar(x, window, width, bottom=setup+warmup, label='Window', color='#2ca02c')
        ax.bar(x, finalize, width, bottom=setup+warmup+window, label='Finalize', color='#d62728')
        
        ax.set_xlabel('N (number of points)', fontsize=12)
        ax.set_ylabel('Time (ms)', fontsize=12)
        ax.set_title('MSM Stage Decomposition (2 GPU, wbits=8)', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{int(n)}' for n in df_2gpu['N']], rotation=45)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.2, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"  → Saved to {output_file}")
    plt.close()

def validate_benchmarking_results(df):
    """Perform sanity checks on benchmark results and flag anomalies."""
    print("\n=== BENCHMARK VALIDATION ===\n")
    
    warnings = []
    
    # Check 1: Speedup for 2 GPU
    print("Check 1: Multi-GPU speedup (2 GPU vs 1 GPU)")
    df_1gpu = df[(df['num_gpus'] == 1) & (df['use_greedy'] == 1) & (df['wbits'] == 8)]
    df_2gpu = df[(df['num_gpus'] == 2) & (df['use_greedy'] == 1) & (df['wbits'] == 8)]
    
    if len(df_1gpu) > 0 and len(df_2gpu) > 0:
        df_1gpu_idx = df_1gpu.set_index('N')
        df_2gpu_idx = df_2gpu.set_index('N')
        common_n = df_1gpu_idx.index.intersection(df_2gpu_idx.index)
        
        if len(common_n) > 0:
            speedups = df_1gpu_idx.loc[common_n, 'total_time_ms'] / df_2gpu_idx.loc[common_n, 'total_time_ms']
            low_speedup = speedups[speedups < 1.1]
            
            if len(low_speedup) > 0:
                warning_msg = f"  ⚠ Low speedup detected for {len(low_speedup)} cases (speedup < 1.1):"
                print(warning_msg)
                for n_val, speedup_val in low_speedup.items():
                    print(f"    N={n_val}: speedup={speedup_val:.2f}x")
                warnings.append(warning_msg)
            else:
                print(f"  ✓ All speedups >= 1.1 (range: {speedups.min():.2f}x - {speedups.max():.2f}x)")
    else:
        print("  - Insufficient data (need both 1 GPU and 2 GPU runs)")
    
    # Check 2: Sensitivity to wbits
    print("\nCheck 2: Runtime sensitivity to wbits")
    df_2gpu_wbits = df[(df['num_gpus'] == 2) & (df['use_greedy'] == 1)]
    
    for n_val in sorted(df_2gpu_wbits['N'].unique())[-3:]:  # Check latest 3 N values
        df_subset = df_2gpu_wbits[df_2gpu_wbits['N'] == n_val]
        if len(df_subset) >= 2:
            wbits_values = df_subset.sort_values('wbits')['wbits'].values
            times = df_subset.sort_values('wbits')['total_time_ms'].values
            
            # Check relative variance
            runtime_variance = (times.max() - times.min()) / times.mean()
            
            if runtime_variance < 0.05:  # Less than 5% variation
                warning_msg = f"  ⚠ N={n_val}: Runtime nearly flat across wbits (variance {runtime_variance:.1%})"
                print(warning_msg)
                warnings.append(warning_msg)
            else:
                print(f"  ✓ N={n_val}: Runtime varies {runtime_variance:.1%} across wbits")
    
    # Check 3: Prediction accuracy
    print("\nCheck 3: Prediction model accuracy")
    if 'prediction_error_pct' not in df.columns:
        df = df.copy()
        df['prediction_error_pct'] = (
            (df['predicted_total_time_ms'] - df['actual_total_time_ms']) 
            / df['actual_total_time_ms'] * 100.0
        )
    
    df_2gpu_pred = df[(df['num_gpus'] == 2) & (df['use_greedy'] == 1) & (df['wbits'] == 8)]
    if len(df_2gpu_pred) > 0:
        mean_abs_error = df_2gpu_pred['prediction_error_pct'].abs().mean()
        large_errors = df_2gpu_pred[df_2gpu_pred['prediction_error_pct'].abs() > 10.0]
        
        print(f"  Mean absolute error: {mean_abs_error:.1f}%")
        
        if len(large_errors) > 0:
            warning_msg = f"  ⚠ {len(large_errors)} cases with error > 10%:"
            print(warning_msg)
            for idx, row in large_errors.iterrows():
                print(f"    N={int(row['N'])}: error={row['prediction_error_pct']:.1f}%")
            warnings.append(warning_msg)
        else:
            print(f"  ✓ All prediction errors within ±10%")
    
    print()
    return warnings

def main():
    """Main plotting workflow."""
    print("=== MSM GPU Pipeline Benchmarking Analysis ===\n")
    
    df = load_data("benchmark_results.csv")
    print(f"Data shape: {df.shape}")
    print(f"N range: {df['N'].min()} - {df['N'].max()}")
    print(f"wbits range: {df['wbits'].min()} - {df['wbits'].max()}")
    print(f"GPUs: {sorted(df['num_gpus'].unique())}\n")
    
    plot_runtime_vs_n(df)
    plot_speedup_vs_n(df)
    plot_throughput_vs_n(df)
    plot_runtime_vs_wbits(df)
    plot_prediction_error(df)
    plot_stage_decomposition(df)
    
    # Run validation checks
    warnings = validate_benchmarking_results(df)
    
    if len(warnings) == 0:
        print("✓ All validation checks passed")
    else:
        print(f"⚠ {len(warnings)} warning(s) detected")
    
    print("\n✓ Analysis complete")
    print("  Generated files:")
    print("    - runtime_vs_N.png")
    print("    - speedup_vs_N.png")
    print("    - throughput_vs_N.png")
    print("    - runtime_vs_wbits.png")
    print("    - prediction_error.png")
    print("    - stage_decomposition.png")

if __name__ == "__main__":
    main()

