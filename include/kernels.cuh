// include/kernels.cuh
#pragma once

#include "ec_bn254.cuh"
#include <cuda_runtime.h>

// Initialize array to infinity points
__global__ void init_inf(G1J* out, int n);

// Test kernel - write "one" to each block's index
__global__ void test_write_kernel(G1J* out);

// Block-per-bucket reduction (shared memory), offsets length = nbuckets+1
__global__ void bucket_sum_block_per_bucket(
  const G1J* points,
  const int* offsets,
  G1J* bucket_sums,
  int nbuckets
);

// Scatter local bucket sums into full bucket array of size B
__global__ void scatter_bucket_sums(
  const G1J* local_sums,
  const int* bucket_ids,
  G1J* full_sums,
  int nlocal
);

// Merge local task bucket sums into full bucket sums (GPU-side)
// Safe for split buckets: accumulates all local sums for each bucket
__global__ void merge_local_bucket_sums(
  const int* task_bucket_ids,
  const G1J* local_bucket_sums,
  G1J* merged_bucket_sums,
  int B,
  int ntasks
);

// Window reduce using suffix trick (single-thread baseline)
__global__ void window_reduce_suffix(
  const G1J* bucket_sums,
  int B,
  G1J* out_one
);

// ============================================================================
// PHASE 1: GPU-side digit extraction and bucket counting
// ============================================================================

// Device struct for 254-bit scalar
struct ScalarR_device {
  uint64_t v[4];
};

// Compute window digits for all scalars in parallel
__global__ void kernel_compute_digits(
  const ScalarR_device* d_scalars,
  int N,
  int bit_offset,
  int wbits,
  uint32_t* d_digits
);

// Count bucket sizes from digits using atomic operations
__global__ void kernel_count_buckets(
  const uint32_t* d_digits,
  int N,
  int* d_bucket_counts
);
