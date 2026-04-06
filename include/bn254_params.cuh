#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// Constants
static __device__ __host__ const uint64_t BN254_P_HOST[4] = {
  4332616871279656263ULL,
  10917124144477883021ULL,
  13281191951274694749ULL,
  3486998266802970665ULL
};
static __device__ __host__ const uint64_t BN254_R_HOST[4] = {
  4891460686036598785ULL,
  2896914383306846353ULL,
  13281191951274694749ULL,
  3486998266802970665ULL
};
static __device__ __host__ const uint64_t BN254_N0_HOST = 0x87d20782e4866389ULL;
static __device__ __host__ const uint64_t BN254_RMODP_HOST[4] = {
  15230403791020821917ULL,
  754611498739239741ULL,
  7381016538464732716ULL,
  1011752739694698287ULL
};
static __device__ __host__ const uint64_t BN254_R2_HOST[4] = {
  17522657719365597833ULL,
  13107472804851548667ULL,
  5164255478447964150ULL,
  493319470278259999ULL
};

// G1 generator (affine): (1,2) :contentReference[oaicite:2]{index=2}
static constexpr uint64_t BN254_GX = 1ULL;
static constexpr uint64_t BN254_GY = 2ULL;
