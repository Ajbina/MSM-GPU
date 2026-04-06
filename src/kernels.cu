#include "ec_bn254.cuh"
#include <cuda_runtime.h>
#include <cstdio>

// Initialize an array of Jacobian points to infinity.
__global__ void init_inf(G1J* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = G1J::infinity();
}

// Debug-only write kernel.
// Uses a placeholder Jacobian value for memory-write smoke testing;
// this is not guaranteed to be an on-curve point.
__global__ void test_write_kernel(G1J* out) {
  if (threadIdx.x == 0) {
    G1J p;
    p.Z = Fp::one();
    p.X = Fp::one();
    p.Y = Fp::one();
    out[blockIdx.x] = p;
  }
}

// Block-per-bucket reduction using shared memory.
// Each block reduces points[offsets[bid] ... offsets[bid+1]-1] into bucket_sums[bid].
__global__ void bucket_sum_block_per_bucket(
    const G1J* points, const int* offsets, G1J* bucket_sums, int nbuckets) {
  int bid = blockIdx.x;
  if (bid >= nbuckets) return;

  int start = offsets[bid];
  int end   = offsets[bid + 1];

  extern __shared__ G1J sh[];

  G1J acc = G1J::infinity();
  for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
    acc = g1_add(acc, points[i]);
  }
  sh[threadIdx.x] = acc;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sh[threadIdx.x] = g1_add(sh[threadIdx.x], sh[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    bucket_sums[bid] = sh[0];
  }
}

// Merge local task bucket sums into full bucket sums (GPU-side, split-bucket safe).
// Handles multiple tasks per logical bucket (e.g., split buckets).
// Thread 0 initializes bucket sums to infinity, then accumulates all task local sums.
//
// Parameters:
//   task_bucket_ids: bucket ID for each task (length ntasks)
//   local_bucket_sums: local sum for each task (length ntasks)
//   merged_bucket_sums: output per-bucket sums (length B, pre-allocated)
//   B: total number of buckets
//   ntasks: number of local sums to merge
__global__ void merge_local_bucket_sums(
    const int* task_bucket_ids,
    const G1J* local_bucket_sums,
    G1J* merged_bucket_sums,
    int B,
    int ntasks) {
  // Single thread (0,0) performs the merge
  // This is safe and simple: serialized accumulation per bucket
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // Initialize merged sums to infinity
    for (int b = 0; b < B; ++b) {
      merged_bucket_sums[b] = G1J::infinity();
    }
    // Accumulate each task's local sum into its bucket
    for (int i = 0; i < ntasks; ++i) {
      int b = task_bucket_ids[i];
      merged_bucket_sums[b] = g1_add(merged_bucket_sums[b], local_bucket_sums[i]);
    }
  }
}

// WARNING: legacy helper, not used in the current split-aware pipeline.
// Not safe for split buckets because it overwrites instead of accumulating
// when multiple local sums map to the same logical bucket.
__global__ void scatter_bucket_sums(
    const G1J* local_sums, const int* bucket_ids, G1J* full_sums, int nlocal) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nlocal) return;
  int b = bucket_ids[i];
  full_sums[b] = local_sums[i];
}

// Legacy GPU window reduction helper.
// Safe only when launched such that a single thread performs the reduction.
__global__ void window_reduce_suffix(const G1J* bucket_sums, int B, G1J* out_one) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    G1J running = G1J::infinity();
    G1J acc = G1J::infinity();
    for (int b = B - 1; b >= 0; --b) {
      running = g1_add(running, bucket_sums[b]);
      acc = g1_add(acc, running);
    }
    *out_one = acc;
  }
}

// ============================================================================
// PHASE 1: GPU-side digit extraction and bucket counting
// ============================================================================

// Device struct for 254-bit scalar (copy of host ScalarR)
struct ScalarR_device {
  uint64_t v[4];
};

// Device-compatible version of get_window_digit
__device__ static inline uint32_t dev_get_window_digit(const ScalarR_device& s, int bit_offset, int wbits) {
  const int limb = bit_offset >> 6;      // / 64
  const int off  = bit_offset & 63;      // % 64

  uint64_t lo = (limb < 4) ? s.v[limb] : 0ULL;
  uint64_t hi = (limb + 1 < 4) ? s.v[limb + 1] : 0ULL;

  // Create 128-bit value represented asTwo 64-bit parts
  // Extract wbits starting at bit 'off' within this 128-bit region
  uint32_t result;
  if (off + wbits <= 64) {
    // All bits are in lo
    result = (uint32_t)((lo >> off) & ((1ULL << wbits) - 1ULL));
  } else if (off >= 64) {
    // All bits are in hi (shouldn't happen with limb-based extraction, but be safe)
    result = (uint32_t)((hi >> (off - 64)) & ((1ULL << wbits) - 1ULL));
  } else {
    // Bits span lo and hi
    uint32_t bits_from_lo = 64 - off;  // Number of bits from lo
    uint32_t lo_part = (uint32_t)((lo >> off) & ((1ULL << bits_from_lo) - 1ULL));
    uint32_t bits_from_hi = wbits - bits_from_lo;
    uint32_t hi_part = (uint32_t)(hi & ((1ULL << bits_from_hi) - 1ULL));
    result = lo_part | (hi_part << bits_from_lo);
  }
  return result;
}

// Kernel: compute window digit for every scalar
// Processes N scalars in parallel, outputs N digits
__global__ void kernel_compute_digits(
    const ScalarR_device* d_scalars,
    int N,
    int bit_offset,
    int wbits,
    uint32_t* d_digits) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  d_digits[i] = dev_get_window_digit(d_scalars[i], bit_offset, wbits);
}

// Kernel: count bucket sizes from digits
// One thread per point: atomically increment bucket count if digit != 0
// Assumes d_bucket_counts is pre-zeroed
__global__ void kernel_count_buckets(
    const uint32_t* d_digits,
    int N,
    int* d_bucket_counts) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  uint32_t digit = d_digits[i];
  if (digit > 0) {
    int bucket = (int)(digit - 1);
    atomicAdd(&d_bucket_counts[bucket], 1);
  }
}
