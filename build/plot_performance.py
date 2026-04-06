#!/usr/bin/env python3
"""
Visualize performance.csv metrics from MSM GPU benchmark
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read CSV
df = pd.read_csv('performance.csv')

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('MSM GPU Performance Analysis', fontsize=16, fontweight='bold')

# 1. Predicted vs Actual latency
ax = axes[0, 0]
ax.plot(df['win'], df['pred_lat'] * 1e3, 'b-o', label='Predicted', linewidth=2, markersize=4)
ax.plot(df['win'], df['act'] * 1e3, 'r-s', label='Actual', linewidth=2, markersize=4)
ax.set_xlabel('Window Index')
ax.set_ylabel('Latency (ms)')
ax.set_title('Predicted vs Actual Window Latency')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Predicted vs Actual throughput
ax = axes[0, 1]
ax.plot(df['win'], df['pred_tp'], 'b-o', label='Predicted', linewidth=2, markersize=4)
# Actual throughput = points / actual time
df['act_tp'] = df['points'] / df['act']
ax.plot(df['win'], df['act_tp'], 'r-s', label='Actual', linewidth=2, markersize=4)
ax.set_xlabel('Window Index')
ax.set_ylabel('Throughput (points/sec)')
ax.set_title('Predicted vs Actual Throughput')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Bucket skew: max_cost vs avg_cost
ax = axes[1, 0]
skew = df['max_cost'] / (df['avg_cost'] + 1e-10)
ax.bar(df['win'], skew, color='purple', alpha=0.7, edgecolor='black')
ax.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='No skew')
ax.set_xlabel('Window Index')
ax.set_ylabel('Skew Ratio (max / avg)')
ax.set_title('Bucket Cost Skew per Window')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 4. Number of buckets vs points
ax = axes[1, 1]
ax.plot(df['win'], df['buckets'], 'g-o', label='Num Buckets', linewidth=2, markersize=4)
ax.set_ylabel('Number of Buckets', color='g')
ax.tick_params(axis='y', labelcolor='g')
ax2 = ax.twinx()
ax2.plot(df['win'], df['points'], 'c-s', label='Num Points', linewidth=2, markersize=4)
ax2.set_ylabel('Number of Points', color='c')
ax2.tick_params(axis='y', labelcolor='c')
ax.set_xlabel('Window Index')
ax.set_title('Buckets and Points per Window')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('performance_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved: performance_analysis.png")

# Print summary statistics
print("\n" + "="*60)
print("PERFORMANCE SUMMARY")
print("="*60)
print(f"Total windows: {len(df)}")
print(f"\nPredicted Latency (ms):")
print(f"  Mean: {df['pred_lat'].mean()*1e3:.4f}")
print(f"  Min:  {df['pred_lat'].min()*1e3:.4f}")
print(f"  Max:  {df['pred_lat'].max()*1e3:.4f}")
print(f"\nActual Latency (ms):")
print(f"  Mean: {df['act'].mean()*1e3:.4f}")
print(f"  Min:  {df['act'].min()*1e3:.4f}")
print(f"  Max:  {df['act'].max()*1e3:.4f}")
print(f"\nPredicted Throughput (pts/s):")
print(f"  Mean: {df['pred_tp'].mean():.1f}")
print(f"\nActual Throughput (pts/s):")
print(f"  Mean: {df['act_tp'].mean():.1f}")
print(f"\nBucket Skew Ratio (max / avg):")
print(f"  Mean: {skew.mean():.2f}")
print(f"  Min:  {skew.min():.2f}")
print(f"  Max:  {skew.max():.2f}")
avg_prediction_error = abs(df['pred_lat'] - df['act']).mean() / df['act'].mean() * 100
print(f"\nAverage Prediction Error: {avg_prediction_error:.1f}%")
print("="*60)
