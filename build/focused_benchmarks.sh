#!/bin/bash
# Focused benchmarks: small to very large data with prediction analysis

BINARY="./msm_bn254_mgpu"

echo "=== FOCUSED BENCHMARKS: Small to Very Large Data ==="
echo ""

# Test parameters: N values from 10k to 1.5M, wbits=8, 2 GPUs with audit
for N in 10000 50000 100000 200000 500000 1000000; do
    echo "Running N=$N (wbits=8, 2 GPUs, audit=1)"
    $BINARY $N 8 1 0 0 1 0
    echo ""
done

# Test different wbits with moderate N for time distribution analysis
for wbits in 6 8 10; do
    echo "Running wbits=$wbits (N=100000, 2 GPUs, audit=1)"
    $BINARY 100000 $wbits 1 0 0 1 0
    echo ""
done

echo "=== BENCHMARKS COMPLETE ==="
