# MSM GPU Benchmark Workflow

Quick start guide for running benchmarks and generating plots.

## Prerequisites

```bash
# Build the project
cd build
cmake .. && make -j4
cd ..

# Install Python requirements
pip install matplotlib pandas numpy
```

## Quick Start

### 1. Run Benchmarks
```bash
bash run_benchmarks.sh
```

**Duration:** ~15-30 minutes (27 benchmark runs across 4 test matrices)

**Output:**
- `benchmark_results.csv` - Timing results from all runs
- `audit_results.csv` - Detailed performance model data (audit runs only)

### 2. Generate Plots
```bash
python3 plot_benchmarks.py
```

**Output:** 8 PNG plots analyzing performance and model accuracy

## Benchmark Matrices

| Matrix | Purpose | Cases | Params |
|--------|---------|-------|--------|
| A | Scaling (N & GPUs) | 12 | N=[20k,100k,200k,500k,1M,2M], GPU=[1,2], wbits=8 |
| B | Window bits | 6 | N=[200k,1M], wbits=[6,8,10], GPU=2 |
| C | Model validation | 4 | audit=1 cases with detailed timing |
| D | Correctness | 5 | check=1, N=2000, wbits=[1,6,8,10,16] |

## Output Files

### benchmark_results.csv
- `timestamp`: Run time
- `gpus`, `N`, `wbits`, `mode`, `check`, `audit`: Parameters
- `total_us`: Total execution time (microseconds)

### audit_results.csv (when audit=1)
- All from benchmark_results.csv, plus:
- `predicted_core_sum_ms`: Model predicted core time
- `measured_staged_window_sum_ms`: Actual measured time
- `core_relative_error_pct`: (measured - predicted) / predicted × 100
- Stage ratios: pack, compute, merge, suffix (prediction accuracy)
- Time decomposition: setup, warmup, window_sum, finalize

## Plots Generated

| Plot | Source | Purpose |
|------|--------|---------|
| `runtime_vs_N.png` | benchmark | Runtime scaling (1 GPU vs 2 GPU, wbits=8, log-log) |
| `throughput_vs_N.png` | benchmark | Mpoints/s scaling (1 GPU vs 2 GPU, wbits=8) |
| `speedup_vs_N.png` | benchmark | Multi-GPU efficiency (speedup @ 2 GPUs vs 1 GPU) |
| `runtime_vs_wbits.png` | benchmark | Window bits sensitivity (2 GPU, N=200k & 1M) |
| `core_model_error.png` | audit | Core model prediction error by test case |
| `full_model_error.png` | audit | Full latency model prediction error |
| `runtime_decomposition.png` | audit | Stacked time: setup, warmup, windows, finalize |
| `stage_ratios.png` | audit | Stage prediction accuracy (pred/actual ratios) |

## Interpreting Results

**Speedup (speedup_vs_N.png):**
- ~2.0x = excellent scaling
- <1.5x = communication/load balance overhead
- >2.0x = unlikely (measurement noise)

**Model Error (core/full_model_error.png):**
- ±10% = well-calibrated
- ±20-30% = acceptable, minor tuning needed
- >±30% = recalibration required

**Stage Ratios (stage_ratios.png):**
- 1.0 = perfect prediction
- <1.0 = model overestimates stage cost
- >1.0 = model underestimates stage cost

## Advanced

### Custom Benchmark Run
```bash
CUDA_VISIBLE_DEVICES=0 ./build/msm_bn254_mgpu 100000 8 1 0 0 1 0
# Args: N wbits mode check objective audit force_split
# mode: 1=greedy, 0=even
# check: 1=verify correctness
# objective: 0=throughput, 1=latency
# audit: 1=detailed timing
# force_split: 1=debug memory splitting
```

### Regenerate Plots (without re-running benchmarks)
```bash
python3 plot_benchmarks.py
```

### Reset All Results
```bash
rm benchmark_results.csv audit_results.csv *.png
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No CUDA GPU found" | Check GPU drivers: `nvidia-smi` |
| "Binary not found" | Run `cd build && make -j4` first |
| No audit data | Ensure main.cu has AUDIT_* output lines + rebuild |
| Plot generation fails | `pip install --upgrade matplotlib pandas numpy` |

## Files

- `run_benchmarks.sh` – Main benchmark runner (bash)
- `plot_benchmarks.py` – Plotting utility (Python 3)
- `BENCHMARK_GUIDE.py` – Detailed documentation

For detailed information, run:
```bash
python3 BENCHMARK_GUIDE.py
```
