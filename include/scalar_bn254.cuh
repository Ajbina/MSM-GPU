#pragma once
#include <cstdint>
#include <random>
#include "bn254_params.cuh"

// 254-bit scalar modulo r (BN254 subgroup order), stored little-endian in 4x64-bit limbs.
struct ScalarR {
  uint64_t v[4]; // little-endian: v[0] is least significant

  static inline ScalarR zero() { return {{0,0,0,0}}; }
  inline bool is_zero() const { return (v[0]|v[1]|v[2]|v[3]) == 0ULL; }
};

// Compare a >= b for 256-bit little-endian limbs
static inline bool geq_u256(const uint64_t a[4], const uint64_t b[4]) {
  for (int i = 3; i >= 0; --i) {
    if (a[i] > b[i]) return true;
    if (a[i] < b[i]) return false;
  }
  return true; // equal
}

// out = a - b (assumes a >= b)
static inline void sub_u256(uint64_t out[4], const uint64_t a[4], const uint64_t b[4]) {
  unsigned __int128 borrow = 0;
  for (int i = 0; i < 4; i++) {
    unsigned __int128 ai = a[i];
    unsigned __int128 bi = (unsigned __int128)b[i] + (uint64_t)borrow;
    unsigned __int128 ri = (ai >= bi) ? (ai - bi) : (((unsigned __int128)1 << 64) + ai - bi);
    borrow = (ai < bi);
    out[i] = (uint64_t)ri;
  }
}

// Return scalar mod r in [1, r-1] using rejection sampling.
// This is "real" in the sense that the scalar is uniformly sampled modulo r (except excluding 0).
static inline ScalarR random_scalar_mod_r(std::mt19937_64& rng) {
  ScalarR s;
  while (true) {
    s.v[0] = rng();
    s.v[1] = rng();
    s.v[2] = rng();
    s.v[3] = rng();

    // Make it at most 254 bits (optional but reduces rejections).
    // r is ~254 bits, so clear the top 2 bits of limb3 to keep candidates in range more often.
    s.v[3] &= ((1ULL << 62) - 1ULL);

    if (s.is_zero()) continue;
    // use host-side constant to avoid reading uninitialized device memory
    if (geq_u256(s.v, BN254_R_HOST)) continue; // reject if s >= r
    return s;
  }
}

// Extract wbits window digit starting at bit_offset from scalar s.
// Returns digit in [0, 2^wbits - 1]. Works across limb boundaries.
// Assumes wbits <= 32 typically (Pippenger window sizes).
static inline uint32_t get_window_digit(const ScalarR& s, int bit_offset, int wbits) {
  // Which limb and which bit inside limb
  const int limb = bit_offset >> 6;      // / 64
  const int off  = bit_offset & 63;      // % 64

  // Build a 128-bit chunk from limb and limb+1 to safely extract across boundary
  uint64_t lo = (limb < 4) ? s.v[limb] : 0ULL;
  uint64_t hi = (limb + 1 < 4) ? s.v[limb + 1] : 0ULL;

  unsigned __int128 chunk = ((unsigned __int128)hi << 64) | (unsigned __int128)lo;
  unsigned __int128 shifted = (chunk >> off);

  if (wbits == 32) return (uint32_t)(shifted & 0xFFFFFFFFu);
  if (wbits == 0)  return 0;
  if (wbits >= 64) {
    // Not expected for Pippenger; clamp behavior
    return (uint32_t)(shifted & 0xFFFFFFFFu);
  }
  uint32_t mask = (wbits == 32) ? 0xFFFFFFFFu : ((1u << wbits) - 1u);
  return (uint32_t)(shifted & mask);
}

// Read a single bit at position i (0 = LSB) from ScalarR
static inline uint32_t get_bit(const ScalarR& s, int i) {
  int limb = i >> 6;
  int off  = i & 63;
  if (limb < 0 || limb >= 4) return 0;
  return (uint32_t)((s.v[limb] >> off) & 1ULL);
}
