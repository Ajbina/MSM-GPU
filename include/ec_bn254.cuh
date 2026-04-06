#pragma once
#include <cstdint>
#include "field_bn254.cuh"

// Jacobian point on BN254 G1: y^2 = x^3 + 3
struct G1J {
  Fp X, Y, Z;

  HD static G1J infinity() {
    G1J P; P.X = Fp::zero(); P.Y = Fp::one(); P.Z = Fp::zero(); return P;
  }
  HD bool is_inf() const { return Z.is_zero(); }
};

struct G1Aff {
  Fp x, y;
  bool inf;
};

HD G1J g1_from_affine_u64(uint64_t x, uint64_t y) {
  G1J P;
  P.X = Fp::from_u64(x);
  P.Y = Fp::from_u64(y);
  P.Z = Fp::one();
  return P;
}

// Jacobian doubling (a=0)
HD G1J g1_dbl(const G1J& P) {
  if (P.is_inf()) return P;

  // Standard Jacobian dbl formulas (a=0)
  Fp XX = P.X * P.X;
  Fp YY = P.Y * P.Y;
  Fp YYYY = YY * YY;

  Fp S = (P.X + YY);
  S = S * S;
  S = S - XX - YYYY;
  S = S + S;

  Fp M = XX + XX + XX;

  Fp X3 = (M*M) - (S + S);

  Fp Y3 = M * (S - X3);
  Fp eightYYYY = YYYY + YYYY; eightYYYY = eightYYYY + eightYYYY; eightYYYY = eightYYYY + eightYYYY;
  Y3 = Y3 - eightYYYY;

  Fp Z3 = (P.Y * P.Z); Z3 = Z3 + Z3;

  G1J R; R.X=X3; R.Y=Y3; R.Z=Z3;
  return R;
}

// Jacobian add (complete-enough for random MSM; not a formally complete formula)
HD G1J g1_add(const G1J& P, const G1J& Q) {
  if (P.is_inf()) return Q;
  if (Q.is_inf()) return P;

  Fp Z1Z1 = P.Z * P.Z;
  Fp Z2Z2 = Q.Z * Q.Z;

  Fp U1 = P.X * Z2Z2;
  Fp U2 = Q.X * Z1Z1;

  Fp S1 = (P.Y * Q.Z) * Z2Z2;
  Fp S2 = (Q.Y * P.Z) * Z1Z1;

  if (U1 == U2) {
    if (!(S1 == S2)) return G1J::infinity();
    return g1_dbl(P);
  }

  Fp H  = U2 - U1;
  Fp HH = H * H;
  Fp HHH = HH * H;
  Fp R  = S2 - S1;
  Fp V  = U1 * HH;

  Fp X3 = (R*R) - HHH - (V + V);
  Fp Y3 = R * (V - X3) - (S1 * HHH);
  Fp Z3 = (P.Z * Q.Z) * H;

  G1J out; out.X=X3; out.Y=Y3; out.Z=Z3;
  return out;
}

// Host scalar mul (used only for random input generation and CPU reference check).
// Scalar is uint64 here for practicality; scalars for MSM are random 64-bit reduced mod r.
inline G1J g1_scalar_mul_u64(G1J P, uint64_t k) {
  G1J R = G1J::infinity();
  while (k) {
    if (k & 1ULL) R = g1_add(R, P);
    P = g1_dbl(P);
    k >>= 1ULL;
  }
  return R;
}

// Affine conversion (host: uses exp to invert Z; OK for final output/check)
inline G1Aff g1_to_affine_host(const G1J& P) {
  if (P.is_inf()) return {Fp::zero(), Fp::zero(), true};

  // inv = Z^(p-2) in Montgomery domain:
  // exponent (p-2) as 256-bit limbs:
  // use host-side constant since this runs on host
  static const uint64_t PM2[4] = {
    BN254_P_HOST[0] - 2ULL, BN254_P_HOST[1], BN254_P_HOST[2], BN254_P_HOST[3]
  };

  Fp zinv = fp_pow(P.Z, PM2);
  Fp z2 = zinv * zinv;
  Fp z3 = z2 * zinv;

  G1Aff A;
  A.x = P.X * z2;
  A.y = P.Y * z3;
  A.inf = false;
  return A;
}
