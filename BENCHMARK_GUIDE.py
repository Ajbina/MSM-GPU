#!/usr/bin/env python3
"""
BENCHMARK AND PLOTTING WORKFLOW GUIDE
======================================

This workflow provides a complete benchmarking pipeline for the MSM GPU project.

FILES CREATED:
    - run_benchmarks.sh         : Main benchmark runner script
    - plot_benchmarks.py        : Plotting and analysis script
    - benchmark_results.csv     : Benchmark timing results (auto-created)
    - audit_results.csv         : Detailed audit/model validation data (auto-created)

PREREQUISITES:
    1. Build the project:
       $ cd build
       $ cmake .. && make -j4
       $ cd ..

    2. Python 3 with matplotlib:
       $ pip install matplotlib pandas numpy

STEP 1: RUN BENCHMARKS
======================

Command:
    $ bash run_benchmarks.sh

This script runs 4 benchmark matrices:

    Matrix A: Scaling runs (wbits=8)
    - N: 20000, 100000, 200000, 500000, 1000000, 2000000
    - GPUs: 1, 2
    - Measures runtime scaling with problem size
    - Total cases: 12

    Matrix B: Window sweep (2 GPUs)
    - N: 200000, 1000000
    - wbits: 6, 8, 10
    - Measures sensitivity to window bits
    - Total cases: 6

    Matrix C: Audit/model validation (2 GPUs, audit=1)
    - (N, wbits): (20000,8), (100000,8), (200000,6), (200000,10)
    - Collects detailed performance model metrics
    - Total cases: 4

    Matrix D: Correctness runs (2 GPUs, check=1, audit=1)
    - N: 2000
    - wbits: 1, 6, 8, 10, 16
    - Validates correctness across wbits
    - Total cases: 5

Total: 27 benchmark runs (~15-30 minutes depending on GPU count and CUDA setup)

Output:
    - benchmark_results.csv : One row per benchmark run
    - audit_results.csv     : One row per audit run (Matrix C and D only)

If a benchmark fails, the script continues with the next one.

STEP 2: GENERATE PLOTS
======================

Command:
    $ python3 plot_benchmarks.py

This generates 8 PNG plots:

    FROM benchmark_results.csv (always generated):
    
    1. runtime_vs_N.png
       - X-axis: N (log scale)
       - Y-axis: Runtime (log scale)
       - Compare 1 GPU vs 2 GPU at wbits=8
       - Shows scalability with problem size

    2. throughput_vs_N.png
       - X-axis: N (log scale)
       - Y-axis: Throughput (Mpoints/s)
       - Compare 1 GPU vs 2 GPU at wbits=8
       - Shows computational efficiency

    3. speedup_vs_N.png
       - X-axis: N (log scale)
       - Y-axis: Speedup (1GPU time / 2GPU time)
       - Includes theoretical ideal (2.0x)
       - Shows multi-GPU scaling efficiency

    4. runtime_vs_wbits.png
       - X-axis: wbits (6, 8, 10)
       - Y-axis: Runtime (ms)
       - Multiple lines for N=200000 and N=1000000
       - Shows window bits sensitivity

    FROM audit_results.csv (if available and non-empty):

    5. core_model_error.png
       - Y-axis: Core prediction error (%)
       - X-axis: Test cases (N, wbits pairs)
       - Negative: model underestimates, Positive: model overestimates
       - Validates core computation model

    6. full_model_error.png
       - Y-axis: Full latency prediction error (%)
       - X-axis: Test cases (N, wbits pairs)
       - Validates complete performance model including overhead

    7. runtime_decomposition.png
       - Stacked bars showing time allocation per stage:
         * Setup    : Initial data generation and GPU allocation
         * Warmup   : First window (template pass, not measured in results)
         * Windows  : Actual measured windows
         * Finalize : Cleanup and final output
       - One bar per audit case
       - Shows where time is spent

    8. stage_ratios.png
       - Grouped bars for prediction accuracy:
         * Pack    : Host-side data packing
         * Compute : GPU kernel execution
         * Merge   : Host-side bucket merging
         * Suffix  : Host-side window suffix computation
       - Ratio = Predicted / Actual
       - 1.0 = perfect prediction
       - Validates per-stage model components

INTERPRETING RESULTS
====================

Speedup Analysis (speedup_vs_N.png):
    - If speedup ≈ 2.0x: Excellent multi-GPU scaling
    - If speedup < 2.0x: Communication or load balance overhead
    - If speedup > 2.0x: Unlikely; may indicate measurement noise or cache effects

Model Error Analysis (core_model_error.png, full_model_error.png):
    - ±10% error: Model is well-calibrated
    - ±20% error: Model needs minor tuning
    - >±30% error: Model needs recalibration or system parameters review

Stage Ratios (stage_ratios.png):
    - pack ≈ 1.0  : Packing cost model is accurate
    - compute ≈ 1.0 : Compute cost model is accurate
    - merge ≈ 1.0  : Merge cost model is accurate
    - suffix ≈ 1.0 : Suffix cost model is accurate
    - If ratio < 1.0: Model overestimates stage cost
    - If ratio > 1.0: Model underestimates stage cost

ADVANCED USAGE
==============

1. Running custom benchmarks:
   $ CUDA_VISIBLE_DEVICES=0 ./build/msm_bn254_mgpu N=100000 wbits=8 mode=1 check=0 objective=0 audit=1 force_split=0

   Args: N wbits mode check objective audit force_split
   - mode: 1=greedy, 0=even
   - check: 1=verify correctness, 0=skip
   - objective: 0=throughput, 1=latency
   - audit: 1=detailed timing, 0=minimal output
   - force_split: 1=debug memory splitting, 0=normal

2. Extending run_benchmarks.sh:
   - Add new benchmark matrices to the script
   - Each matrix is a simple loop over parameter combinations
   - Example: add matrix E for different objectives (latency vs throughput)

3. Custom plots:
   - plot_benchmarks.py can be extended with additional analyses
   - Data is read from CSV files which can also be opened in Excel/LibreOffice
   - Regenerate without re-running benchmarks by editing plot_benchmarks.py

TROUBLESHOOTING
===============

"Total time: ... us" not parsed:
    - Check that Binary path in run_benchmarks.sh is correct
    - Verify build succeeded: ls -la ./build/msm_bn254_mgpu

audit_results.csv values are empty:
    - Ensure audit flag (arg 6) is set to 1
    - Check that main.cu includes AUDIT_* output lines
    - Verify binary was rebuilt after adding audit output

Plots not generated:
    - Ensure matplotlib is installed: pip install matplotlib
    - Check that benchmark_results.csv has data with proper headers
    - Run manually: python3 plot_benchmarks.py

EXAMPLE SESSION
===============

$ cd msm-gpu
$ bash run_benchmarks.sh
Starting benchmark run: 2026-03-31_14-30-05
=== Matrix A: Scaling runs (wbits=8) ===
Running: ./build/msm_bn254_mgpu 20000 8 1 0 0 0 0
...
Benchmark run complete!
Results saved to:
  - benchmark_results.csv
  - audit_results.csv

$ python3 plot_benchmarks.py
Loading benchmark data...
Loading audit data...

Generating plots...
Saved: runtime_vs_N.png
Saved: throughput_vs_N.png
Saved: speedup_vs_N.png
Saved: runtime_vs_wbits.png
Saved: core_model_error.png
Saved: full_model_error.png
Saved: runtime_decomposition.png
Saved: stage_ratios.png

All plots generated successfully!

$ ls -la *.png
-rw-r--r-- 1 user group  45K runtime_vs_N.png
-rw-r--r-- 1 user group  42K throughput_vs_N.png
-rw-r--r-- 1 user group  38K speedup_vs_N.png
-rw-r--r-- 1 user group  35K runtime_vs_wbits.png
-rw-r--r-- 1 user group  32K core_model_error.png
-rw-r--r-- 1 user group  32K full_model_error.png
-rw-r--r-- 1 user group  50K runtime_decomposition.png
-rw-r--r-- 1 user group  48K stage_ratios.png

NOTES
=====

- No .log files are created (all output is captured in memory)
- CSV files are appended to, not overwritten
- Timestamps are added to each row for traceability
- Script is idempotent: can re-run and results accumulate
- To reset, delete benchmark_results.csv and audit_results.csv before re-running
- Performance varies based on system state, GPU utilization, thermal conditions
- For stable results, run benchmarks on a quiet system and average multiple runs
"""

if __name__ == '__main__':
    import subprocess
    # Print this file as usage guide
    print(__doc__)
