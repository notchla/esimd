// Copyright 2026 notchla liso.lorenzo@gmail.com
// SPDX-License-Identifier: Apache-2.0
//
// esimd showcase — AVX2 (native 256-bit integer paths).
//
// Under AVX2, the 8-wide integer types get true native codegen: vint8 gains a
// single-instruction multiply (_mm256_mullo_epi32), vuint8 gets native unsigned
// min/max, and the 4-wide 64-bit vllong4 becomes available. This example shows
// native integer arithmetic, min/max, a cross-lane permute, and 64-bit lanes.
//
// Build standalone (from this folder):
//
//   g++ -std=c++17 -I../../include \
//       -mavx2 -mfma -mf16c -mbmi -mbmi2 -mlzcnt \
//       -D__AVX2__ -D__AVX__ -D__SSE4_2__ -D__SSE4_1__ -D__LZCNT__ -D__BMI__ \
//       avx2_example.cpp -o avx2_example
//   ./avx2_example
//
// Note the include is unchanged — adding -D__AVX2__ is what swaps the two-SSE-halves
// integer backend for the native one.

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
  std::cout << "== esimd AVX2 example (native 8-wide int, 4-wide int64) ==\n";

  const vint8 a(1, 2, 3, 4, 5, 6, 7, 8);
  const vint8 b(8, 7, 6, 5, 4, 3, 2, 1);
  print("a", a, 8);
  print("b", b, 8);
  print("a * b  (native mullo_epi32)", a * b, 8);       // {8,14,18,20,20,18,14,8}
  print("min(a,b)", min(a, b), 8);
  print("max(a,b)", max(a, b), 8);
  std::cout << "reduce_add(a) = " << reduce_add(a)
            << ", reduce_max(a) = " << reduce_max(a) << "\n";

  // Cross-lane permute: reverse the 8 lanes. index[i] picks source lane index[i].
  const __m256i reverse = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
  print("permute(a, reverse)", permute(a, reverse), 8);

  // Unsigned native min/max.
  const vuint8 u(10u, 40u, 20u, 30u, 60u, 50u, 80u, 70u);
  print("u", u, 8);
  print("min(u, 45)", min(u, vuint8(45u)), 8);

  // 64-bit integer lanes (only present under AVX2 / __X86_64__).
  const vllong4 L(1000000000ll, 2000000000ll, 3ll, 4ll);
  const vllong4 M(3ll, 3ll, 3ll, 3ll);
  print("L", L, 4);
  print("L + M", L + M, 4);
  std::cout << "reduce_add(L) = " << reduce_add(L) << "\n";
  std::cout << "movemask(L < 5) = 0x" << std::hex
            << movemask(L < vllong4(5ll)) << std::dec << "\n";

  return 0;
}
