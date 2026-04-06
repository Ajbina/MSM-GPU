#!/bin/bash
# run_benchmarks.sh
# Benchmark runner for MSM GPU project
# Runs predefined benchmark matrices and parses output into CSV files

set -e

BINARY="./build/msm_bn254_mgpu"
BENCHMARK_CSV="benchmark_results.csv"
AUDIT_CSV="audit_results.csv"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)

# Initialize CSV headers if they don't exist
if [ ! -f "$BENCHMARK_CSV" ]; then
    echo "timestamp,gpus,N,wbits,mode,check,audit,total_us" > "$BENCHMARK_CSV"
fi

if [ ! -f "$AUDIT_CSV" ]; then
    echo "timestamp,gpus,N,wbits,mode,check,audit,predicted_core_sum_ms,measured_staged_window_sum_ms,predicted_latency_ms,actual_gpu_time_ms,core_relative_error_pct,full_relative_error_pct,stage_ratio_pack,stage_ratio_compute,stage_ratio_merge,stage_ratio_suffix,setup_time_ms,warmup_window_time_ms,window_total_time_sum_ms,finalize_time_ms,reconstruction_gap_ms" > "$AUDIT_CSV"
fi

# Helper function to run a benchmark and parse output
run_benchmark() {
    local gpus=$1
    local N=$2
    local wbits=$3
    local mode=$4
    local check=$5
    local objective=$6
    local audit=$7
    local force_split=$8

    local cmd="$BINARY $N $wbits $mode $check $objective $audit $force_split"
    
    if [ "$gpus" -eq 1 ]; then
        cmd="CUDA_VISIBLE_DEVICES=0 $cmd"
    fi

    echo "Running: $cmd"
    
    # Capture output
    local output
    output=$(eval "$cmd" 2>&1) || return 1
    
    # Parse total time
    local total_us
    total_us=$(echo "$output" | grep -oP 'Total time: \K[0-9]+' | head -1)
    
    if [ -z "$total_us" ]; then
        echo "WARNING: Could not parse total time from output"
        return 1
    fi
    
    # Append to benchmark_results.csv
    echo "$TIMESTAMP,$gpus,$N,$wbits,$mode,$check,$audit,$total_us" >> "$BENCHMARK_CSV"
    
    if [ "$audit" -eq 1 ]; then
        # Parse audit values
        local pred_core=$(echo "$output" | grep "AUDIT_COST_CORE" | grep -oP 'predicted_core_sum_ms=\K[0-9.]+')
        local meas_win=$(echo "$output" | grep "AUDIT_COST_CORE" | grep -oP 'measured_staged_window_sum_ms=\K[0-9.]+')
        local pred_lat=$(echo "$output" | grep "AUDIT_COST_FULL" | grep -oP 'predicted_latency_ms=\K[0-9.]+')
        local actual_gpu=$(echo "$output" | grep "AUDIT_COST_FULL" | grep -oP 'actual_gpu_time_ms=\K[0-9.]+')
        local ratio_pack=$(echo "$output" | grep "AUDIT_STAGE_RATIOS" | grep -oP 'pack=\K[0-9.]+')
        local ratio_compute=$(echo "$output" | grep "AUDIT_STAGE_RATIOS" | grep -oP 'compute=\K[0-9.]+')
        local ratio_merge=$(echo "$output" | grep "AUDIT_STAGE_RATIOS" | grep -oP 'merge=\K[0-9.]+')
        local ratio_suffix=$(echo "$output" | grep "AUDIT_STAGE_RATIOS" | grep -oP 'suffix=\K[0-9.]+')
        local setup_ms=$(echo "$output" | grep "AUDIT_DECOMP" | grep -oP 'setup_ms=\K[0-9.]+')
        local warmup_ms=$(echo "$output" | grep "AUDIT_DECOMP" | grep -oP 'warmup_ms=\K[0-9.]+')
        local window_ms=$(echo "$output" | grep "AUDIT_DECOMP" | grep -oP 'window_sum_ms=\K[0-9.]+')
        local finalize_ms=$(echo "$output" | grep "AUDIT_DECOMP" | grep -oP 'finalize_ms=\K[0-9.]+')
        local recon_gap=$(echo "$output" | grep "AUDIT_DECOMP" | grep -oP 'reconstruction_gap_ms=\K[0-9.]+')
        
        # Compute error percentages
        local core_error=""
        local full_error=""
        
        if [ -n "$pred_core" ] && [ -n "$meas_win" ]; then
            core_error=$(awk "BEGIN {if ($pred_core > 0) print ($meas_win - $pred_core) / $pred_core * 100; else print \"\"}")
        fi
        
        if [ -n "$pred_lat" ] && [ -n "$actual_gpu" ]; then
            full_error=$(awk "BEGIN {if ($pred_lat > 0) print ($actual_gpu - $pred_lat) / $pred_lat * 100; else print \"\"}")
        fi
        
        # Append to audit_results.csv
        echo "$TIMESTAMP,$gpus,$N,$wbits,$mode,$check,$audit,$pred_core,$meas_win,$pred_lat,$actual_gpu,$core_error,$full_error,$ratio_pack,$ratio_compute,$ratio_merge,$ratio_suffix,$setup_ms,$warmup_ms,$window_ms,$finalize_ms,$recon_gap" >> "$AUDIT_CSV"
    fi
}

echo "Starting benchmark run: $TIMESTAMP"
echo ""

# Matrix A: Scaling runs at wbits=8
echo "=== Matrix A: Scaling runs (wbits=8) ==="
for N in 20000 100000 200000 500000 1000000 2000000; do
    for gpus in 1 2; do
        run_benchmark $gpus $N 8 1 0 0 0 0 || echo "FAILED: N=$N, gpus=$gpus"
    done
done

# Matrix B: Window sweep runs on 2 GPUs
echo ""
echo "=== Matrix B: Window sweep (2 GPUs) ==="
for N in 200000 1000000; do
    for wbits in 6 8 10; do
        run_benchmark 2 $N $wbits 1 0 0 0 0 || echo "FAILED: N=$N, wbits=$wbits"
    done
done

# Matrix C: Audit/model-validation runs
echo ""
echo "=== Matrix C: Audit runs (2 GPUs) ==="
for case in "20000 8" "100000 8" "200000 6" "200000 10"; do
    set -- $case
    N=$1
    wbits=$2
    run_benchmark 2 $N $wbits 1 0 0 1 0 || echo "FAILED: N=$N, wbits=$wbits, audit=1"
done

# Matrix D: Small correctness runs
echo ""
echo "=== Matrix D: Correctness runs (2 GPUs, check=1) ==="
for wbits in 1 6 8 10 16; do
    run_benchmark 2 2000 $wbits 1 1 0 1 0 || echo "FAILED: N=2000, wbits=$wbits, check=1"
done

echo ""
echo "Benchmark run complete!"
echo "Results saved to:"
echo "  - $BENCHMARK_CSV"
echo "  - $AUDIT_CSV"
