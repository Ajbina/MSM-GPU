#!/bin/bash
# Quick benchmark runner and analyzer
# Usage: ./benchmark.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================="
echo "MSM GPU Benchmarking Suite"
echo "=================================================="
echo ""

# Check if binary exists
if [ ! -f "../build/msm_bn254_mgpu" ]; then
    echo "ERROR: Binary not found at ../build/msm_bn254_mgpu"
    echo "Please build the project first:"
    echo "  cd .."
    echo "  mkdir -p build && cd build"
    echo "  cmake .. && make -j"
    exit 1
fi

echo "✓ Binary found: ../build/msm_bn254_mgpu"
echo ""

# Run benchmarks
echo "Starting benchmark runs..."
echo "(This may take 30-60 minutes depending on system load)"
echo ""

python3 run_benchmark.py

if [ $? -ne 0 ]; then
    echo "ERROR: Benchmark failed"
    exit 1
fi

echo ""
echo "=================================================="
echo "Benchmarks Complete!"
echo "=================================================="
echo ""

# Check results
if [ -f "benchmark_final_all.csv" ]; then
    echo "Results saved:"
    ls -lh benchmark_N*.csv benchmark_final_all.csv 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
    echo ""
    
    # Run analysis
    echo "Running analysis..."
    echo ""
    python3 analyze_results.py benchmark_final_all.csv
else
    echo "WARNING: No results found"
    exit 1
fi

echo "Done!"
