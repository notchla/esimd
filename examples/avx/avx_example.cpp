// Copyright 2026 notchla liso.lorenzo@gmail.com
// SPDX-License-Identifier: Apache-2.0
//
// esimd showcase — AVX (256-bit, 8-wide float / 4-wide double).
//
// Demonstrates processing 8 floats per instruction (vfloat8): SoA-style batch
// math, fused multiply-add, horizontal reductions, compare+select, plus the
// 4-wide double type (vdouble4).
//
// Build standalone (from this folder):
//
//   g++ -std=c++17 -I../../include \
//       -mavx -mbmi -D__AVX__ -D__SSE4_2__ -D__SSE4_1__ -D__BMI__ \
//       avx_example.cpp -o avx_example
//   ./avx_example
//
// Same #include as every other example — the -m/-D flags above unlock the 8-wide
// AVX backend.

#include <esimd/esimd.h>

#include <iostream>

using namespace esimd;

template <typename V>
static void print(const char* label, const V& v, int n) {
  std::cout << label << " = [";
  for (int i = 0; i < n; ++i) std::cout << (i ? ", " : "") << v[i];
  std::cout << "]\n";
}

int main() {
  std::cout << "== esimd AVX example (8-wide float, 4-wide double) ==\n";

  // Evaluate y = a*x + b for 8 x-values at once.
  alignas(32) float xs[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  const vfloat8 x = vfloat8::load(xs);
  const vfloat8 a(3.0f), b(1.0f);
  const vfloat8 y = madd(a, x, b); // a*x + b, one FMA across 8 lanes
  print("x", x, 8);
  print("y = 3*x + 1", y, 8);

  // Horizontal reductions collapse the 8 lanes to a scalar.
  std::cout << "reduce_add(y) = " << reduce_add(y)
            << ", reduce_min(y) = " << reduce_min(y)
            << ", reduce_max(y) = " << reduce_max(y) << "\n";

  // sqrt over 8 lanes, and a masked blend: keep sqrt(x) where x >= 4 else 0.
  const vfloat8 s = sqrt(x);
  print("sqrt(x)", s, 8);
  const vboolf8 hi = x >= vfloat8(4.0f);
  std::cout << "movemask(x >= 4) = 0x" << std::hex << movemask(hi) << std::dec << "\n";
  print("select(x>=4, sqrt(x), 0)", select(hi, s, vfloat8(zero)), 8);

  // Double precision, 4 lanes wide.
  const vdouble4 d(1.0, 2.0, 3.0, 4.0);
  const vdouble4 e(10.0, 20.0, 30.0, 40.0);
  print("d", d, 4);
  print("madd(d, 2, e) = d*2 + e", madd(d, vdouble4(2.0), e), 4);
  std::cout << "reduce_add(d) = " << reduce_add(d) << "\n";

  return 0;
}
