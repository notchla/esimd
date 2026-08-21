// Copyright 2026 notchla liso.lorenzo@gmail.com
// SPDX-License-Identifier: Apache-2.0
//
// esimd showcase — AVX512 (512-bit, 16-wide float / 8-wide double).
//
// Demonstrates 16 floats per instruction (vfloat16), AVX512 mask registers
// (vboolf16 is a __mmask16) driving masked select, an 8-wide 64-bit multiply
// (vllong8, native _mm512_mullo_epi64), and 8-wide double math.
//
// Build standalone (from this folder):
//
//   g++ -std=c++17 -I../../include \
//       -march=skylake-avx512 -mavx2 -mfma -mf16c -mbmi -mbmi2 -mlzcnt \
//       -D__AVX512F__ -D__AVX512VL__ -D__AVX512DQ__ -D__AVX512BW__ \
//       -D__AVX2__ -D__AVX__ -D__SSE4_2__ -D__SSE4_1__ -D__LZCNT__ -D__BMI__ \
//       avx512_example.cpp -o avx512_example
//   ./avx512_example
//
// (clang++ works identically.) Or via CMake: ./build/examples/avx512/esimd_example_avx512.
//
// The __AVX512VL__ define also promotes the smaller bool types (vboolf4/8) to
// real mask registers behind the same include.

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
  std::cout << "== esimd AVX512 example (16-wide float, 8-wide double) ==\n";

  // 16 lanes {0..15} in one register; a single FMA over all of them.
  const vfloat16 x(step);          // 0,1,2,...,15
  const vfloat16 y = madd(x, vfloat16(2.0f), vfloat16(1.0f)); // 2*x + 1
  print("x", x, 16);
  print("y = 2*x + 1", y, 16);
  std::cout << "reduce_add(x) = " << reduce_add(x)
            << ", reduce_max(x) = " << reduce_max(x) << "\n";

  // Mask registers: build a vboolf16 from a comparison, inspect it, and use it to
  // blend. Keep x where x is even, else -1  (mask via (x mod 2 == 0) emulated with
  // a compare on a precomputed vector).
  alignas(64) float evenv[16];
  for (int i = 0; i < 16; ++i) evenv[i] = (i % 2 == 0) ? 1.0f : 0.0f;
  const vboolf16 even = vfloat16::load(evenv) == vfloat16(1.0f);
  std::cout << "movemask(even lanes) = 0x" << std::hex << movemask(even) << std::dec
            << "  (popcnt = " << popcnt(even) << ")\n";
  print("select(even, x, -1)", select(even, x, vfloat16(-1.0f)), 16);

  // 8-wide double.
  const vdouble8 d(step);          // 0..7
  print("d", d, 8);
  print("d * d  (elementwise square)", d * d, 8);
  std::cout << "reduce_add(d) = " << reduce_add(d) << "\n";

  // 8-wide 64-bit integers with a native 64-bit multiply.
  const vllong8 L = vllong8(step) + vllong8(1ll); // 1..8
  print("L", L, 8);
  print("L * L  (native mullo_epi64)", L * L, 8);
  std::cout << "reduce_add(L) = " << reduce_add(L) << "\n";

  return 0;
}
