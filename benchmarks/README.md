# MSM GPU Benchmarking Suite

Comprehensive benchmarking script to compare **even split (static round-robin)** vs **greedy planner-driven** bucket distribution across multiple input sizes.

## Quick Start

```bash
cd benchmarks
python3 run_benchmark.py
```

## What It Does

- **Runs** ./msm_bn254_mgpu for N ∈ {1M, 5M, 10M, 20M, 35M, 50M, 60M}
- **Tests** two modes per N:
  - `even`: Static round-robin distribution (use_greedy=0)
  - `greedy`: Greedy planner-driven distribution (use_greedy=1)
- **Captures**:
  - Timing breakdown (setup, windows, finalize)
  - Planner parameters (k_compute, k_digit, k_count, k_merge, k_suffix)
  - Predicted vs actual execution time
  - Prediction error percentage
  - Throughput (points/sec)
- **Saves** results to:
  - `benchmark_N1000000.csv` through `benchmark_N60000000.csv` (per-size)
  - `benchmark_final_all.csv` (combined overview)

## Output Files

### Per-Size CSVs: `benchmark_N*.csv`

Each file contains results for a specific N value, showing both modes side-by-side:

```
N,wbits,mode,num_gpus,setup_ms,avg_window_ms,total_window_time_ms,...,prediction_error_pct
1000000,8,even,2,...,0.0604,1.9331,...,-1.40
1000000,8,greedy,2,...,0.0601,1.9236,...,-0.78
```

### Combined CSV: `benchmark_final_all.csv`

All results in one file for easy comparison:

```
N,wbits,mode,num_gpus,k_compute_mid,alpha_pack,k_digit,k_count,...
1000000,8,even,2,11.0854,2.8e-08,1.5e-09,...
1000000,8,greedy,2,11.0854,2.8e-08,1.5e-09,...
...
60000000,8,greedy,2,11.0854,2.8e-08,1.5e-09,...
```

## Fields Captured

### Core Metrics
- `N`: Number of scalars/points
- `wbits`: Window bits (8)
- `mode`: "even" or "greedy"
- `num_gpus`: Number of GPUs (2)

### Timing (milliseconds)
- `setup_ms`: Initial setup
- `avg_window_ms`: Average per-window time
- `min_window_ms`: Fastest window
- `max_window_ms`: Slowest window
- `total_window_time_ms`: Sum of all windows
- `window_count`: Number of windows

### Planner Predictions
- `avg_predicted_ms`: Average predicted per-window
- `total_predicted_ms`: Sum of predicted times
- `prediction_error_pct`: Actual vs predicted error

### Planner Parameters
- `k_compute_small`: Coefficient for wbits ≤ 6
- `k_compute_mid`: Coefficient for 7 ≤ wbits ≤ 8 (primary)
- `k_compute_large`: Coefficient for wbits ≥ 9
- `k_digit`: GPU digit extraction time per scalar
- `k_count`: GPU bucket counting time per scalar
- `k_merge`: GPU merge time per task
- `k_suffix`: GPU suffix time per bucket
- `alpha_pack`: CPU packing time per point
- `num_sms`: GPU streaming multiprocessors
- `M_g_bytes`: GPU memory budget

### Results
- `throughput`: Points/sec

## Key Insights to Look For

### 1. Mode Comparison (even vs greedy)
Compare same N with different modes:
```
N=1M,  even:   avg_window=0.0604ms
N=1M,  greedy: avg_window=0.0601ms  ← nearly equivalent
```

### 2. Prediction Accuracy
- `prediction_error_pct` in single digits is typical after model convergence
- Current full sweep: average absolute error is about 8.0%, max about 10.45%
- N=1M currently lands near zero (~-1% to -0.8%)

### 3. Scaling Behavior
- Overhead should decrease as N increases (more buckets to balance)
- Greedy should consistently outperform or match even

### 4. Throughput Comparison
- Higher throughput = better scheduling
- Use throughput as a secondary signal; avg_window_ms and prediction_error_pct are the primary health metrics

## How to Analyze Results

### Compare modes per N:
```bash
grep "^1000000" benchmark_final_all.csv
# Shows both even and greedy results for N=1M
```

### Compare across scales:
```bash
cut -d, -f1,3,7 benchmark_final_all.csv | sort -u
# Shows N, mode, and throughput progression
```

### Calculate speedup:
```python
import pandas as pd
df = pd.read_csv('benchmark_final_all.csv')
for N in df['N'].unique():
    even = df[(df['N'] == N) & (df['mode'] == 'even')]['avg_window_ms'].values[0]
    greedy = df[(df['N'] == N) & (df['mode'] == 'greedy')]['avg_window_ms'].values[0]
    speedup = even / greedy
    print(f"N={N}: speedup={speedup:.3f}x")
```

## Technical Details

### Dataset Caching
All runs use `use_reusable_dataset=1`, so datasets are **cached on disk**:
- First run for N: ~20 seconds (generates + saves)
- Subsequent runs: <1 second (loads cached)

### Planner Invocation
The planner is called **once per window** (32 times total for wbits=8):
```cpp
BucketPlan plan = make_plan(bucket_sizes, G, objective, params, ...);
```

- `even` mode: Static round-robin (no load balancing per window)
- `greedy` mode: Greedy bin-packing assignment (adapts per window)

### EWMA Learning
Both modes benefit from overhead learning:
- Initial overhead estimate: 24ms (conservative)
- After 10 windows: Converges to actual (typically 11-12ms)
- Prediction accuracy improves as MSM runs

## Troubleshooting

### Script hangs/times out
- Increase subprocess timeout in `run_benchmark.py` (line ~95)
- Check if N=35M, N=50M, or N=60M datasets exist
- Monitor GPU temperature (`nvidia-smi`)

### Missing fields in output
- Re-run with `audit_stage_timing=1` enabled (script does this)
- Check stderr for parsing errors

### Inaccurate predictions
- Planner parameters (k_compute_mid, etc.) are calibrated for RTX 4000 Ada
- If using different GPU, first 10 windows will have larger errors
- EWMA learning will adapt after convergence

## Performance Tips

- Run during low-system-load times
- Background tasks (yux220) should be minimal
- Use `nvidia-smi -l 1` to monitor GPU utilization

## Future Enhancements

- [ ] Add single-GPU benchmarks
- [ ] Export to JSON for visualization
- [ ] Compare with CPU-only baseline
- [ ] Profile individual kernels
- [ ] Test various wbits values
