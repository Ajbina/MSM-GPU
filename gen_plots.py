#!/usr/bin/env python3
"""
Generate benchmark analysis plots from MSM GPU benchmark data.
Handles variable column counts in CSV (audit runs have extra columns).
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def load_benchmark_data(filename="benchmark_results.csv"):
    """Load and clean benchmark CSV, handling variable columns."""
    try:
        # Read CSV manually to handle variable columns
        import csv
        data = []
        core_cols = ['N', 'wbits', 'num_gpus', 'use_greedy', 'total_time_ms', 
                     'setup_ms', 'warmup_ms', 'window_ms', 'finalize_ms', 
                     'throughput', 'speedup', 'predicted_total_time_ms', 
                     'actual_total_time_ms', 'prediction_error_pct', 'k_compute_active']
        
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Only keep the core columns
                clean_row = {col: row.get(col, 0) for col in core_cols}
                data.append(clean_row)
        
        df = pd.DataFrame(data)
        
        # Convert numeric columns
        for col in core_cols[:-1]:  # All except last
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass
        
        print(f"Loaded {len(df)} benchmark records from {filename}")
        print(f"N range: {df['N'].min():.0f} to {df['N'].max():.0f}")
        print(f"wbits values: {sorted(df['wbits'].unique())}")
        return df
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_prediction_error_vs_n(df):
    """Plot prediction error percentage vs input size N."""
    if df is None or 'N' not in df.columns:
        return
    
    print("Generating: prediction_error_vs_N.png")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get unique wbits values
    for wbits in sorted(df['wbits'].unique()):
        df_wbits = df[(df['wbits'] == wbits) & (df['num_gpus'] == 2)].copy()
        if len(df_wbits) > 0:
            df_wbits = df_wbits.sort_values('N')
            ax.plot(df_wbits['N'], df_wbits['prediction_error_pct'], 
                   'o-', label=f'wbits={wbits}', linewidth=2, markersize=7)
    
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Input size N (points)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Prediction Error (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Prediction Error vs Input Size', fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=11, loc='best')
    plt.tight_layout()
    plt.savefig('prediction_error_vs_N.png', dpi=150)
    plt.close()
    print("  ✓ Saved to prediction_error_vs_N.png")

def plot_predicted_vs_actual_time(df):
    """Plot predicted vs actual execution time."""
    if df is None or 'predicted_total_time_ms' not in df.columns:
        return
    
    print("Generating: predicted_vs_actual_time.png")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot for different wbits
    colors = {'6': 'blue', '8': 'green', '10': 'red'}
    for wbits in sorted(df['wbits'].unique()):
        df_wbits = df[(df['wbits'] == wbits) & (df['num_gpus'] == 2)].copy()
        if len(df_wbits) > 0:
            ax.scatter(df_wbits['predicted_total_time_ms'], 
                      df_wbits['actual_total_time_ms'],
                      s=100, alpha=0.6, label=f'wbits={wbits}',
                      color=colors.get(str(wbits), 'purple'))
    
    # Plot ideal line (predicted == actual)
    all_times = pd.concat([df['predicted_total_time_ms'], df['actual_total_time_ms']])
    min_time = all_times.min()
    max_time = all_times.max()
    ax.plot([min_time, max_time], [min_time, max_time], 'k--', alpha=0.5, linewidth=2, label='Perfect fit (pred=actual)')
    
    ax.set_xlabel('Predicted Time (ms)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Model Prediction Accuracy: Predicted vs Actual Time', fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=11, loc='upper left')
    plt.tight_layout()
    plt.savefig('predicted_vs_actual_time.png', dpi=150)
    plt.close()
    print("  ✓ Saved to predicted_vs_actual_time.png")

def plot_time_distribution(df):
    """Plot stacked time decomposition (setup, warmup, window, finalize)."""
    if df is None:
        return
    
    print("Generating: time_distribution.png")
    
    # Filter to 2-GPU runs for cleaner plotting
    df_2gpu = df[(df['num_gpus'] == 2) & (df['use_greedy'] == 1)].copy()
    
    # Get unique N values sorted
    n_values = sorted(df_2gpu['N'].unique())[:8]  # Limit to 8 points for readability
    
    df_subset = df_2gpu[df_2gpu['N'].isin(n_values)].copy()
    df_subset = df_subset.sort_values('N')
    
    if len(df_subset) == 0:
        print("  No data available for time distribution plot")
        return
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Prepare data for stacked bar chart
    x_pos = np.arange(len(df_subset))
    width = 0.6
    
    setup = df_subset['setup_ms'].values
    warmup = df_subset['warmup_ms'].values
    window = df_subset['window_ms'].values
    finalize = df_subset['finalize_ms'].values
    
    # Stack the bars
    p1 = ax.bar(x_pos, setup, width, label='Setup', color='#FF9999')
    p2 = ax.bar(x_pos, warmup, width, bottom=setup, label='Warmup', color='#FFB366')
    p3 = ax.bar(x_pos, window, width, bottom=setup+warmup, label='Window Processing', color='#99CCFF')
    p4 = ax.bar(x_pos, finalize, width, bottom=setup+warmup+window, label='Finalize', color='#99FF99')
    
    ax.set_xlabel('Input Size N', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Time Distribution Across Pipeline Stages (2 GPUs, wbits=8)', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{int(n/1000)}k' if n >= 1000 else f'{int(n)}' for n in df_subset['N'].values], rotation=45)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('time_distribution.png', dpi=150)
    plt.close()
    print("  ✓ Saved to time_distribution.png")

def plot_cost_difference(df):
    """Plot absolute cost difference (actual - predicted) vs N."""
    if df is None:
        return
    
    print("Generating: cost_difference_vs_N.png")
    
    df_plot = df[(df['num_gpus'] == 2) & (df['use_greedy'] == 1)].copy()
    df_plot['cost_diff'] = df_plot['actual_total_time_ms'] - df_plot['predicted_total_time_ms']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Absolute difference
    for wbits in sorted(df_plot['wbits'].unique()):
        df_w = df_plot[df_plot['wbits'] == wbits].sort_values('N')
        if len(df_w) > 0:
            ax1.plot(df_w['N'], df_w['cost_diff'], 'o-', label=f'wbits={wbits}', linewidth=2, markersize=7)
    
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Input size N', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Cost Difference (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Absolute Cost Error (Actual - Predicted)', fontsize=12, fontweight='bold')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(fontsize=10)
    
    # Percentage difference
    for wbits in sorted(df_plot['wbits'].unique()):
        df_w = df_plot[df_plot['wbits'] == wbits].sort_values('N')
        if len(df_w) > 0:
            pct_diff = (df_w['cost_diff'] / df_w['predicted_total_time_ms'] * 100)
            ax2.plot(df_w['N'], pct_diff, 's-', label=f'wbits={wbits}', linewidth=2, markersize=7)
    
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Input size N', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cost Difference (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Relative Cost Error', fontsize=12, fontweight='bold')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('cost_difference_vs_N.png', dpi=150)
    plt.close()
    print("  ✓ Saved to cost_difference_vs_N.png")

def print_summary(df):
    """Print summary statistics."""
    if df is None:
        return
    
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY STATISTICS")
    print("="*70)
    
    # Prediction accuracy by wbits
    print("\nPrediction Error by Window Bits:")
    for wbits in sorted(df['wbits'].unique()):
        errors = df[df['wbits'] == wbits]['prediction_error_pct']
        if len(errors) > 0:
            print(f"  wbits={wbits:2d}: mean={errors.mean():7.2f}% std={errors.std():6.2f}% "
                  f"[{errors.min():7.2f}%, {errors.max():7.2f}%]")
    
    # Performance scaling
    print("\nPerformance Scaling (2 GPU, wbits=8):")
    df_scale = df[(df['num_gpus'] == 2) & (df['wbits'] == 8)].sort_values('N')
    if len(df_scale) > 1:
        for _, row in df_scale.iterrows():
            n_millions = row['N'] / 1e6
            tp = row['throughput']
            print(f"  N={row['N']:7d}: throughput={tp:8.0f} Mpts/s")
    
    print("\n" + "="*70)

if __name__ == '__main__':
    print("\n=== MSM GPU Benchmark Visualization ===\n")
    
    df = load_benchmark_data()
    
    if df is not None:
        plot_prediction_error_vs_n(df)
        plot_predicted_vs_actual_time(df)
        plot_cost_difference(df)
        plot_time_distribution(df)
        print_summary(df)
        print("\n✓ All plots generated successfully!")
    else:
        print("ERROR: Could not load benchmark data")
