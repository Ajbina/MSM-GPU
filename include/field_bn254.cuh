// include/field_bn254.cuh
#pragma once
#include <cstdint>
#include "bn254_params.cuh"

#if defined(__CUDA_ARCH__)
#define HD __host__ __device__ __forceinline__
#else
#define HD __host__ __device__ inline
#endif

struct Fp {
  uint64_t v[4];

  HD Fp() { v[0]=v[1]=v[2]=v[3]=0; }

  HD static Fp zero() { return Fp(); }

  // Montgomery "one" = R mod p
  HD static Fp one() {
    Fp r;
    r.v[0]=BN254_RMODP_HOST[0]; r.v[1]=BN254_RMODP_HOST[1]; r.v[2]=BN254_RMODP_HOST[2]; r.v[3]=BN254_RMODP_HOST[3];
    return r;
  }

  HD static bool geq(const uint64_t a[4], const uint64_t b[4]) {
    for (int i=3;i>=0;--i) { if (a[i]>b[i]) return true; if (a[i]<b[i]) return false; }
    return true;
  }

  HD static void sub_n(uint64_t out[4], const uint64_t a[4], const uint64_t b[4]) {
    unsigned __int128 borrow = 0;
    for (int i=0;i<4;i++) {
      unsigned __int128 ai = (unsigned __int128)a[i];
      unsigned __int128 bi = (unsigned __int128)b[i] + (uint64_t)borrow;
      unsigned __int128 base = ((unsigned __int128)1) << 64;
      unsigned __int128 ri = (ai >= bi) ? (ai - bi) : (base + ai - bi);
      borrow = (ai < bi);
      out[i] = (uint64_t)ri;
    }
  }

  HD static void add_n(uint64_t out[4], const uint64_t a[4], const uint64_t b[4]) {
    unsigned __int128 carry = 0;
    for (int i=0;i<4;i++) {
      unsigned __int128 s = (unsigned __int128)a[i] + (unsigned __int128)b[i] + (uint64_t)carry;
      out[i] = (uint64_t)s;
      carry = s >> 64;
    }
  }

  // Conditional subtract p
  HD static void reduce(uint64_t x[4]) {
    if (geq(x, BN254_P_HOST)) {
      uint64_t t[4];
      sub_n(t, x, BN254_P_HOST);
      x[0]=t[0]; x[1]=t[1]; x[2]=t[2]; x[3]=t[3];
    }
  }

  HD Fp operator+(const Fp& o) const {
    Fp r;
    add_n(r.v, v, o.v);
    reduce(r.v);
    return r;
  }

  HD Fp operator-(const Fp& o) const {
    Fp r;
    if (geq(v, o.v)) {
      sub_n(r.v, v, o.v);
    } else {
      uint64_t t[4];
      sub_n(t, BN254_P_HOST, o.v);
      add_n(r.v, v, t);
      reduce(r.v);
    }
    return r;
  }

  // Montgomery multiply: (a*b*R^-1) mod p
  HD static Fp mont_mul(const Fp& a, const Fp& b) {
    unsigned __int128 t[8] = {};

    // t = a*b
    for (int i=0;i<4;i++) {
      unsigned __int128 carry = 0;
      for (int j=0;j<4;j++) {
        unsigned __int128 cur = t[i+j] + (unsigned __int128)a.v[i]*b.v[j] + carry;
        t[i+j] = cur & (unsigned __int128)0xFFFFFFFFFFFFFFFFULL;
        carry  = cur >> 64;
      }
      t[i+4] += carry;
    }

    // Montgomery reduction
    for (int i=0;i<4;i++) {
      uint64_t m = (uint64_t)t[i] * BN254_N0_HOST;
      unsigned __int128 carry = 0;
      for (int j=0;j<4;j++) {
        unsigned __int128 cur = t[i+j] + (unsigned __int128)m * BN254_P_HOST[j] + carry;
        t[i+j] = cur & (unsigned __int128)0xFFFFFFFFFFFFFFFFULL;
        carry  = cur >> 64;
      }
      t[i+4] += carry;
    }

    Fp r;
    r.v[0] = (uint64_t)t[4];
    r.v[1] = (uint64_t)t[5];
    r.v[2] = (uint64_t)t[6];
    r.v[3] = (uint64_t)t[7];
    reduce(r.v);
    return r;
  }

  HD Fp operator*(const Fp& o) const { return mont_mul(*this, o); }

  // x -> x*R mod p (Montgomery form)
  HD static Fp from_u64(uint64_t x) {
    Fp a; a.v[0]=x; a.v[1]=a.v[2]=a.v[3]=0;
    Fp r2;
    r2.v[0]=BN254_R2_HOST[0]; r2.v[1]=BN254_R2_HOST[1]; r2.v[2]=BN254_R2_HOST[2]; r2.v[3]=BN254_R2_HOST[3];
    return mont_mul(a, r2);
  }

  HD bool operator==(const Fp& o) const {
    return v[0]==o.v[0] && v[1]==o.v[1] && v[2]==o.v[2] && v[3]==o.v[3];
  }
  HD bool is_zero() const { return v[0]==0 && v[1]==0 && v[2]==0 && v[3]==0; }
};

// host/device exponentiation helper (used for host inversion)
HD Fp fp_pow(Fp a, const uint64_t e[4]) {
  Fp r = Fp::one();
  for (int i=3;i>=0;--i) {
    for (int b=63;b>=0;--b) {
      r = r*r;
      if ((e[i] >> b) & 1ULL) r = r*a;
    }
  }
  return r;
}
