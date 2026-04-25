# MSM GPU - Multi-Scalar Multiplication on GPU

A high-performance GPU implementation of Multi-Scalar Multiplication (MSM) for BN254 elliptic curve cryptography with adaptive task scheduling via the Planner system.

## Overview

This project implements efficient MSM computation on NVIDIA GPUs with support for:
- **Dual GPU execution** - Automatic load balancing across 2 GPUs
- **Adaptive task scheduling** - Planner learns optimal window decomposition
- **Two execution modes**:
  - **Even mode**: Static round-robin bucket distribution (baseline)
  - **Greedy mode**: Adaptive planner-based optimization (recommended)
- **Comprehensive benchmarking** - Scaling analysis across N=1M to 60M points

## Architecture

### Core Components

```
include/
  ├── bn254_params.cuh       # BN254 curve parameters
  ├── field_bn254.cuh        # Fp (field) operations
  ├── scalar_bn254.cuh       # Fr (scalar) operations  
  ├── ec_bn254.cuh           # Elliptic curve point operations
  ├── cuda_utils.cuh         # GPU memory/stream utilities
  ├── kernels.cuh            # GPU kernel declarations
  └── msm_plan.hpp           # Planner algorithm (greedy task assignment)

src/
  ├── kernels.cu             # GPU kernel implementations
  └── main.cu                # MSM coordinator (32-window windowing)
```

### Planner System

The **Planner** (in `include/msm_plan.hpp`) optimizes window execution timing by:

1. **Per-window planning**: Called once per bit window (32 total for 254-bit scalar)
2. **Greedy bucket assignment**: Minimizes max GPU time via bin-packing
3. **EWMA overhead learning**: Adapts to hardware-specific costs
   - Formula: `total_time = phase1_digit + phase1_count + host_pack + h2d + compute + d2h + merge + suffix + overhead_ewma`
   - Overhead converges from 24ms initial to ~11ms actual over 32 windows

4. **Wave model**: Simulates parallel GPU execution to predict latencies

**Key insight**: Greedy assignment adapts per window with changing bucket distributions, and the current latency model now tracks actual runtime with single-digit average error.

## Building

### Requirements
- CUDA 12.0+
- CMake 3.20+
- C++17 compiler
- 2x NVIDIA GPU (Ada architecture recommended, tested on RTX 4000 Ada)

### Build Steps

```bash
cd /home/syn324/projects/msm-gpu
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

**Output**: `build/msm_bn254_mgpu` (6.2MB binary)

## Usage

### Quick Test

```bash
./build/msm_bn254_mgpu N wbits use_greedy [check] [objective] [audit] [force_split] [use_reusable_dataset]
```

**Parameters:**
- `N`: Number of scalars/points (e.g., 1000000 for 1M)
- `wbits`: Window bits (typically 8)
- `use_greedy`: 0=even (static), 1=greedy (planner)
- `check`: 0=off, 1=verify correctness
- `objective`: 0=throughput (default), 1=latency
- `audit`: 0=off, 1=enable timing logs
- `force_split`: Advanced; usually 0
- `use_reusable_dataset`: 1=cache, 0=regenerate

**Examples:**

```bash
# N=1M, greedy mode, audit timing
./build/msm_bn254_mgpu 1000000 8 1 0 0 1 0 1

# N=10M, even mode, verify correctness
./build/msm_bn254_mgpu 10000000 8 0 1 0 0 0 1

# N=50M, greedy, fast path (no audit)
./build/msm_bn254_mgpu 50000000 8 1 0 0 0 0 1
```

### Full Benchmarking Suite

```bash
cd benchmarks
python3 run_benchmark.py
```

Runs all N ∈ [1M, 5M, 10M, 20M, 35M, 50M, 60M] × 2 modes, generates:
- Individual CSVs: `benchmark_N*.csv`
- Combined: `benchmark_final_all.csv`

**Analysis:**
```bash
python3 analyze_results.py benchmark_final_all.csv
```

Produces:
- Mode comparison table
- Prediction accuracy analysis
- Scaling metrics
- Planner parameter summary

## Benchmark Results (Apr 24, 2026)

### Performance Summary

| N | Even Avg Window | Greedy Avg Window | Speedup (Even/Greedy) | Prediction Error (Even / Greedy) |
|---|------------------|-------------------|------------------------|-----------------------------------|
| 1M | 0.060ms | 0.060ms | 1.0049x | -1.40% / -0.78% |
| 5M | 0.235ms | 0.237ms | 0.9926x | +8.84% / +8.02% |
| 10M | 0.453ms | 0.461ms | 0.9823x | +10.45% / +8.51% |
| 20M | 0.902ms | 0.904ms | 0.9980x | +9.67% / +9.45% |
| 35M | 1.584ms | 1.567ms | 1.0106x | +8.73% / +9.86% |
| 50M | 2.271ms | 2.227ms | 1.0197x | +8.05% / +10.20% |
| 60M | 2.706ms | 2.683ms | 1.0087x | +8.79% / +9.67% |

**Speedup interpretation**: Ratio = Even / Greedy
- **< 1.0**: Greedy is faster (better) ✓
- **≈ 1.0**: Modes equivalent
- **> 1.0**: Even is faster (better)

*Example at N=60M: Greedy is about 0.87% faster (2.683ms < 2.706ms)*

### Key Findings

1. **Mode Equivalence**: Even and Greedy remain close (roughly +/-2%), indicating:
   - Overhead learning is working (EWMA converging correctly)
   - Hardware is well-balanced between 2 GPUs
   - Both modes achieve stable performance

2. **Prediction Accuracy**:
   - Average absolute error: **8.03%**
   - Maximum absolute error: **10.45%**
   - N=1M now sits near zero error (about -1% to -0.8%)

3. **Scaling**:
   - Linear trend from 1M to 60M
   - Window time grows from ~0.06ms to ~2.68-2.71ms
   - Prediction error stays in a tighter post-fix band (~8-10% for 5M-60M)

4. **Hardware Utilization**:
   - Both GPUs: 48 SMs each, 20.5GB VRAM
   - Auto-detected at runtime via CUDA APIs
   - Per-GPU budget: 90% of VRAM (~18.84GB)

## Project Structure

```
msm-gpu/
├── include/
│   ├── bn254_params.cuh        # BN254 curve parameters (p, r, G1)
│   ├── field_bn254.cuh         # Fp (field) arithmetic operations
│   ├── scalar_bn254.cuh        # Fr (scalar) arithmetic operations
│   ├── ec_bn254.cuh            # EC point operations + Jacobian coordinates
│   ├── cuda_utils.cuh          # GPU memory & stream utilities
│   ├── kernels.cuh             # GPU kernel declarations
│   └── msm_plan.hpp            # Planner: adaptive task assignment algorithm
│
├── src/
│   ├── kernels.cu              # GPU kernel implementations
│   └── main.cu                 # MSM driver & window coordination
│
├── build/
│   ├── msm_bn254_mgpu          # Compiled binary (6.2MB)
│   └── dataset_N*.bin          # Benchmark datasets (13.6GB, cached)
│
├── benchmarks/
│   ├── run_benchmark.py        # Benchmark suite orchestrator
│   ├── analyze_results.py      # Results analysis & reporting
│   ├── benchmark.sh            # Bash wrapper script
│   └── benchmark_final_all.csv # Latest complete benchmark results
│
├── CMakeLists.txt              # Build configuration
└── README.md                   # This file
```

## Datasets

Cached datasets in `build/` directory (use `use_reusable_dataset=1`):

| Size | File | Storage |
|------|------|---------|
| 1M | dataset_N1000000.bin | 123MB |
| 5M | dataset_N5000000.bin | 611MB |
| 10M | dataset_N10000000.bin | 1.2GB |
| 20M | dataset_N20000000.bin | 2.4GB |
| 35M | dataset_N35000000.bin | 2.7GB |
| 50M | dataset_N50000000.bin | 6.0GB |
| 60M | dataset_N60000000.bin | 7.2GB |

**Total: 20.8GB**

## Implementation Details

### MSM Algorithm (Windowed)

1. **Initialization**: Load random scalars and points, initialize GPU contexts
2. **32 Windows** (wbits=8, 254-bit scalars):
   - Extract 8-bit digit from each scalar
   - Distribute buckets across GPUs via Planner
   - Execute parallel: GPU digit extraction + bucket counting + aggregation
3. **Finalization**: Final point operations and result transfer

### Planner Algorithm

```
For each window {
  - Input: bucket distribution, GPU state
  - Greedy assignment: min-max optimization
  - Output: bucket-to-GPU mapping, time estimate
  - Learn: Update EWMA overhead with measured time
}
```

**Greedy pseudocode**:
```
For each bucket (sorted by size desc):
  Assign to GPU with min current load
  Update GPU load with bucket processing time
Return max GPU time as window prediction
```

### Performance Tuning

**Kernel choices per GPU type**:
- Auto-detected via `cudaGetDeviceProperties()`
- `k_compute_[small|mid|large]`: Time per point for bucket calculation
- `alpha_pack`: Host-side packing overhead coefficient

Current parameters (RTX 4000 Ada):
- `k_compute_small`: 5.84
- `k_compute_mid`: 11.0854
- `k_compute_large`: 8.91
- `alpha_pack`: 2.8e-08

## Troubleshooting

### Build Errors
- **CUDA not found**: Set `export CUDA_HOME=/usr/local/cuda`
- **CMake error**: Ensure CMake 3.20+, CUDA 12.0+

### Runtime Issues
- **Out of memory**: Reduce N or check background processes
- **Incorrect results**: Enable `check=1` mode
- **Slow execution**: Verify both GPUs active via `nvidia-smi`

### Debugging
Enable audit logging for detailed timing:
```bash
./build/msm_bn254_mgpu 10000000 8 1 0 0 1 0 1 2>&1 | grep "window\|predic\|overhead"
```

## Future Optimizations

1. **Prediction refinement**: Account for H2D/D2H overlap, kernel launch overhead
2. **Dynamic greedy**: Adjust `alpha_pack` per GPU on first run
3. **Multi-window parallelism**: Pipeline windows across GPU streams
4. **Larger curves**: Support BLS12-381, other pairing-friendly curves

## References

- **Windowed MSM**: Pippenger's algorithm with bucketing
- **BN254**: Barreto-Naehrig curve (254-bit prime order)
- **CUDA**: NVIDIA GPU programming model

## License & Attribution

Implementation for research purposes. Based on classical MSM algorithms with novel planner-based optimization.

## Contact

For questions or issues, refer to the benchmarking results in `benchmarks/benchmark_final_all.csv` and validation output in logs.
