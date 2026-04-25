# Benchmarking Suite - Quick Start Guide

## Files Created

```
benchmarks/
├── run_benchmark.py         (Main benchmarking script)
├── analyze_results.py       (Results analysis tool)
├── benchmark.sh             (Automated runner + analyzer)
├── README.md                (Detailed documentation)
└── QUICKSTART.md            (This file)
```

## How to Run

### Option 1: One Command (Recommended)
```bash
cd benchmarks
./benchmark.sh
```

This will:
1. Run full benchmark suite (all N values, both modes)
2. Generate per-N CSVs
3. Generate combined CSV
4. Automatically analyze and display results

**Time required:** ~45-60 minutes

---

### Option 2: Manual Control

```bash
cd benchmarks

# Run benchmarks only
python3 run_benchmark.py

# Later, analyze results
python3 analyze_results.py benchmark_final_all.csv
```

---

## Output

### After benchmarks complete, you'll see:

1. **Seven CSV files:**
   - `benchmark_N1000000.csv` (N=1M, both modes)
   - `benchmark_N5000000.csv` (N=5M, both modes)
   - `benchmark_N10000000.csv` (N=10M, both modes)
   - `benchmark_N20000000.csv` (N=20M, both modes)
   - `benchmark_N35000000.csv` (N=35M, both modes)
   - `benchmark_N50000000.csv` (N=50M, both modes)
   - `benchmark_final_all.csv` (all results combined)

2. **Analysis output on terminal:**
```
==================================================================================================
MODE COMPARISON (Even vs Greedy)
==================================================================================================
         N |     Mode | Avg Window |   Predicted |   Error % | Throughput
   1000000 |     even |      72.50 |       74.25 |      2.35 |       4401.2
   1000000 |   greedy |      71.80 |       73.95 |      2.95 |       4425.6
   5000000 |     even |     362.50 |      376.12 |      3.80 |       4385.1
   ...
==================================================================================================
SPEEDUP ANALYSIS:
        N |  Speedup (even/greedy) |    Improvement
   1000000 |                 1.0097 |           0.97%
   5000000 |                 1.0112 |           1.12%
  10000000 |                 1.0145 |           1.45%
  ...
```

---

## Key Metrics to Review

### From Combined CSV (`benchmark_final_all.csv`)

| Column | Meaning | Target |
|--------|---------|--------|
| `N` | Input size | 1M to 50M |
| `mode` | "even" or "greedy" | Both tested |
| `avg_window_ms` | Average time per window | Lower is better |
| `prediction_error_pct` | Accuracy of planner | < 20% is good |
| `throughput` | Points/second | Higher is better |
| `k_compute_mid` | GPU compute coefficient | Same for all runs |
| `num_sms` | GPU multiprocessors | 48 (RTX 4000 Ada) |

---

## Expected Results

Based on your earlier measurements, expect:

### Greedy vs Even
```
N=1M:  even=72.5ms, greedy=71.8ms  → greedy 0.9% faster
N=10M: even=??ms,   greedy=??ms    → greedy 1-3% faster
N=50M: even=??ms,   greedy=??ms    → greedy 2-5% faster (more buckets = better balance)
```

### Prediction Accuracy
```
Early windows:   error ~ 10-20% (waiting for EWMA convergence)
Late windows:    error ~ 0-5%   (EWMA converged after 10 windows)
Average:         error ~ 5-10%  (what you should see overall)
```

### Throughput
```
Scales linearly with N at same efficiency:
N=1M:   throughput ~ 4,400 pts/sec
N=50M:  throughput ~ 4,400 pts/sec (same, good scaling)
```

---

## Common Analysis Tasks

### 1. Compare modes for N=1M
```bash
grep "^1000000" benchmark_final_all.csv
```

### 2. See all greedy results
```bash
grep ",greedy," benchmark_final_all.csv | sort -t, -k1 -n
```

### 3. Check prediction accuracy trend
```bash
cut -d, -f1,3,11 benchmark_final_all.csv | column -t -s,
# Shows: N, mode, prediction_error_pct
```

### 4. Calculate total improvement
```python
import pandas as pd
df = pd.read_csv('benchmark_final_all.csv')

# For each N, compute speedup
for n in df['N'].unique():
    even_tp = df[(df['N']==n) & (df['mode']=='even')]['throughput'].values[0]
    greedy_tp = df[(df['N']==n) & (df['mode']=='greedy')]['throughput'].values[0]
    speedup = greedy_tp / even_tp
    print(f"N={n}: {speedup:.4f}x")
```

---

## Troubleshooting

### Script hangs
- **Symptom:** Script runs but doesn't produce output
- **Check:** `nvidia-smi` to see GPU status
- **Fix:** 
  - Reduce background load (yux220 processes)
  - Increase timeout in `run_benchmark.py` line 95

### Missing data in CSV
- **Symptom:** Some fields are empty
- **Fix:** Re-run with verbose output:
  ```bash
  python3 run_benchmark.py 2>&1 | tee benchmark.log
  ```

### Inaccurate predictions (>20% error early on)
- **Expected:** First few windows may have high error
- **Why:** EWMA learning hasn't converged yet
- **Resolves:** After ~10 windows automatically

### Low throughput
- **Check:** GPU temperature with `nvidia-smi`
- **Check:** Background processes with `top`
- **Check:** Power settings (GPU throttling?)

---

## Next Steps

After benchmarks complete:

1. **Review `benchmark_final_all.csv`** in Excel/LibreOffice for visualization
2. **Compare** even vs greedy for each N
3. **Evaluate** if greedy improvement justifies complexity
4. **Consider** optimization options:
   - Option A: Keep current (good enough)
   - Option B: Parallelize CPU extraction (significant improvement)
   - Option C: Full GPU bucket assignment (major effort)

---

## Notes

- **Uses cached datasets** (very fast on subsequent runs)
- **Planner active** in both modes (greedy just adapts better)
- **EWMA learning** improves predictions over 32 windows
- **Hardware auto-detection** (works on different GPUs, may need recalibration)
- **No MSM logic modified** (only benchmarking/logging added)

---

## Support

For detailed information, see `README.md` in this directory.

For questions about planner algorithm, see `../PLANNER_DETAILS.md`.

For project overview, see `../README_ANALYSIS.md`.
