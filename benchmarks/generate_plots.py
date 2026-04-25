#!/usr/bin/env python3
"""
Generate meaningful plots from MSM GPU benchmarking results.
Creates visualizations of performance metrics across different input sizes and modes.
"""

import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
import os

# Configure plots
plt.style.use('seaborn-v0_8-darkgrid')
colors = {'even': '#0173B2', 'greedy': '#DE8F05'}  # Blue and Orange
fig_size = (12, 6)

def load_results(csv_file):
    """Load benchmark results from CSV."""
    results = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'N': int(row['N']),
                'mode': row['mode'],
                'window_time': float(row['avg_window_ms']),
                'predicted_time': float(row['avg_predicted_ms']),
                'error_pct': float(row['prediction_error_pct']),
                'total_time': float(row['total_window_time_ms']),
                'window_count': int(row['window_count'])
            })
    return results

def plot_mode_comparison(results):
    """Plot: Even vs Greedy execution time comparison."""
    by_n = defaultdict(dict)
    
    for r in results:
        by_n[r['N']][r['mode']] = r['window_time']
    
    ns = sorted(by_n.keys())
    even_times = [by_n[n].get('even', 0) for n in ns]
    greedy_times = [by_n[n].get('greedy', 0) for n in ns]
    
    x = range(len(ns))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=fig_size)
    ax.bar([i - width/2 for i in x], even_times, width, label='Even (static)', color=colors['even'])
    ax.bar([i + width/2 for i in x], greedy_times, width, label='Greedy (planner)', color=colors['greedy'])
    
    ax.set_xlabel('Input Size (N points)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Avg Window Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Mode Comparison: Even vs Greedy Execution Time', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{n//1e6:.0f}M' for n in ns])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plot_mode_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: plot_mode_comparison.png")
    plt.close()

def plot_throughput_scaling(results):
    """Plot: Throughput scaling across input sizes (greedy mode)."""
    greedy_results = [r for r in results if r['mode'] == 'greedy']
    greedy_results.sort(key=lambda x: x['N'])
    
    ns = [r['N'] / 1e6 for r in greedy_results]  # Convert to millions
    # Throughput: N points per window time (ms) per 1000 = scalars/ms
    throughputs = [r['N'] / r['window_time'] / 1000 for r in greedy_results]
    
    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(ns, throughputs, marker='o', linewidth=2.5, markersize=8, color=colors['greedy'])
    ax.fill_between(ns, throughputs, alpha=0.2, color=colors['greedy'])
    
    ax.set_xlabel('Input Size N (Millions)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Throughput (K scalars/ms)', fontsize=12, fontweight='bold')
    ax.set_title('Throughput Scaling: Greedy Mode Performance', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add annotations
    for n, tp in zip(ns, throughputs):
        ax.annotate(f'{tp:.1f}K', (n, tp), textcoords="offset points", 
                   xytext=(0,10), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('plot_throughput_scaling.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: plot_throughput_scaling.png")
    plt.close()

def plot_window_time_scaling(results):
    """Plot: Window execution time vs input size (both modes)."""
    by_mode = defaultdict(lambda: {'ns': [], 'times': []})
    
    for r in results:
        by_mode[r['mode']]['ns'].append(r['N'] / 1e6)
        by_mode[r['mode']]['times'].append(r['window_time'])
    
    fig, ax = plt.subplots(figsize=fig_size)
    
    for mode in ['even', 'greedy']:
        data = by_mode[mode]
        ns = [n for n, t in sorted(zip(data['ns'], data['times']))]
        times = [t for n, t in sorted(zip(data['ns'], data['times']))]
        ax.plot(ns, times, marker='s', linewidth=2.5, markersize=7, 
               label=f'{mode.capitalize()} mode', color=colors[mode])
    
    ax.set_xlabel('Input Size N (Millions)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Avg Window Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Execution Time Scaling: Window Performance vs N', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plot_window_scaling.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: plot_window_scaling.png")
    plt.close()

def plot_prediction_accuracy_for_mode(results, mode, filename, title):
    """Plot: Prediction error vs input size for a single execution mode."""
    mode_results = [r for r in results if r['mode'] == mode]
    mode_results.sort(key=lambda x: x['N'])

    ns = [r['N'] / 1e6 for r in mode_results]
    errors = [abs(r['error_pct']) for r in mode_results]  # Absolute error

    fig, ax = plt.subplots(figsize=fig_size)

    x = range(len(ns))
    bars = ax.bar(x, errors, color=colors[mode], alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, error in zip(bars, errors):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{error:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Input Size N (Millions)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Prediction Error (%)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(n)}M' for n in ns])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()


def plot_prediction_accuracy(results):
    """Plot: Prediction error vs input size for greedy mode."""
    plot_prediction_accuracy_for_mode(
        results,
        'greedy',
        'plot_prediction_error.png',
        'Planner Prediction Accuracy vs Input Size (Greedy Mode)',
    )


def plot_prediction_accuracy_even(results):
    """Plot: Prediction error vs input size for even mode."""
    plot_prediction_accuracy_for_mode(
        results,
        'even',
        'plot_prediction_error_even.png',
        'Planner Prediction Accuracy vs Input Size (Even Mode)',
    )

def plot_speedup_comparison(results):
    """Plot: Greedy vs Even speedup (always 1.0 or near it)."""
    by_n = defaultdict(dict)
    
    for r in results:
        by_n[r['N']][r['mode']] = r['window_time']
    
    ns = sorted(by_n.keys())
    speedups = []
    
    for n in ns:
        even_time = by_n[n].get('even', 1.0)
        greedy_time = by_n[n].get('greedy', 1.0)
        if greedy_time > 0:
            speedup = even_time / greedy_time
            speedups.append(speedup)
        else:
            speedups.append(1.0)
    
    ns_m = [n / 1e6 for n in ns]
    
    fig, ax = plt.subplots(figsize=fig_size)
    
    x = range(len(ns_m))
    colors_speedup = ['#DE8F05' if s > 1.0 else '#0173B2' for s in speedups]
    bars = ax.bar(x, speedups, color=colors_speedup, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax.axhline(y=1.0, color='black', linestyle='-', linewidth=2, label='Parity (1.0x)')
    
    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{speedup:.3f}x', ha='center', va='bottom' if speedup > 1 else 'top', 
               fontsize=10, fontweight='bold')
    
    # Add legend for colors
    # When speedup > 1.0: Even/Greedy > 1, so Greedy is faster (use Greedy's color: Orange)
    # When speedup ≤ 1.0: Even/Greedy ≤ 1, so Even is faster (use Even's color: Blue)
    even_patch = mpatches.Patch(color='#0173B2', alpha=0.7, edgecolor='black', label='Even faster (≤1.0x)')
    greedy_patch = mpatches.Patch(color='#DE8F05', alpha=0.7, edgecolor='black', label='Greedy faster (>1.0x)')
    ax.legend(handles=[even_patch, greedy_patch], fontsize=11, loc='upper right')
    
    ax.set_xlabel('Input Size N (Millions)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup (Even / Greedy)', fontsize=12, fontweight='bold')
    ax.set_title('Mode Performance: Ratio of Even time to Greedy time\n(< 1.0 = Greedy faster, > 1.0 = Even faster)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(n)}M' for n in ns_m])
    ax.set_ylim([0.95, 1.05])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plot_speedup.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: plot_speedup.png")
    plt.close()

def main():
    csv_file = 'benchmark_final_all.csv'
    
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found")
        return 1
    
    print(f"Loading results from {csv_file}...")
    results = load_results(csv_file)
    
    if not results:
        print("No results found")
        return 1
    
    print(f"✓ Loaded {len(results)} benchmark runs\n")
    
    print("Generating plots...")
    plot_mode_comparison(results)
    plot_throughput_scaling(results)
    plot_window_time_scaling(results)
    plot_prediction_accuracy(results)
    plot_prediction_accuracy_even(results)
    plot_speedup_comparison(results)
    
    print("\n" + "="*50)
    print("All plots generated successfully!")
    print("="*50)
    print("\nGenerated files:")
    print("  • plot_mode_comparison.png")
    print("  • plot_throughput_scaling.png")
    print("  • plot_window_scaling.png")
    print("  • plot_prediction_error.png")
    print("  • plot_prediction_error_even.png")
    print("  • plot_speedup.png")
    
    return 0

if __name__ == '__main__':
    exit(main())
