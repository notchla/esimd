// Copyright 2026 notchla liso.lorenzo@gmail.com
// SPDX-License-Identifier: Apache-2.0
//
// esimd showcase — width-agnostic double code with vdoublexd / VSIZEXD.
//
// The double-precision sibling of portable/. vdoublex is vdouble<VSIZEX>, which on
// AVX/AVX2 is 8 lanes — wider than any AVX double register, so it has no SIMD
// backend there. VSIZEXD is the native double width instead:
//
//   VSIZEXD    2 under SSE/NEON/NEON2X, 4 under AVX/AVX2, 8 under AVX512
//   vdoublexd = vdouble<VSIZEXD>   vbooldxd = vboold<VSIZEXD>
//
// Build standalone for, say, AVX2 (from this folder):
//
//   g++ -std=c++17 -I../../include \
//       -mavx2 -mfma -mf16c -mbmi -mbmi2 -mlzcnt \
//       -D__AVX2__ -D__AVX__ -D__SSE4_2__ -D__SSE4_1__ -D__LZCNT__ -D__BMI__ \
//       portable_double_example.cpp -o portable_double_example
//   ./portable_double_example
//
// Or via CMake: ./build/examples/portable_double/esimd_example_portable_double_<isa>.

#include <esimd/esimd.h>

#include <iostream>
#include <vector>

using namespace esimd;

// y[i] = a*x[i] + b[i] for an array of arbitrary length n.
static void daxpy(double a, const double* x, const double* b, double* y, int n) {
  const vdoublexd va(a);
  int i = 0;
  for (; i + VSIZEXD <= n; i += VSIZEXD) {
    const vdoublexd vx = vdoublexd::loadu(x + i);
    const vdoublexd vb = vdoublexd::loadu(b + i);
    vdoublexd::storeu(y + i, madd(va, vx, vb));
  }
  // Masked tail: lanes past the end of the array never touch memory.
  if (i < n) {
    const vbooldxd tail = vdoublexd(step) < vdoublexd(double(n - i));
    const vdoublexd vx = vdoublexd::loadu(tail, x + i);
    const vdoublexd vb = vdoublexd::loadu(tail, b + i);
    vdoublexd::storeu(tail, y + i, madd(va, vx, vb));
  }
}

int main() {
  std::cout << "== esimd portable double example (vdoublexd / VSIZEXD) ==\n";
  std::cout << "VSIZEXD (double width) = " << VSIZEXD << ", VSIZEX = " << VSIZEX << "\n";

  const vdoublexd idx(step);
  std::cout << "vdoublexd(step) = [";
  for (int i = 0; i < VSIZEXD; ++i) std::cout << (i ? ", " : "") << idx[i];
  std::cout << "], reduce_add = " << reduce_add(idx) << "\n";

  const vbooldxd low = idx < vdoublexd(2.0);
  std::cout << "vdoublexd(step) < 2 = [";
  for (int i = 0; i < VSIZEXD; ++i) std::cout << (i ? ", " : "") << bool(low[i]);
  std::cout << "]\n";

  // 11 is not a multiple of 2, 4 or 8, so the masked tail always runs.
  const int N = 11;
  std::vector<double> x(N), b(N), y(N, -1.0);
  for (int i = 0; i < N; ++i) { x[i] = double(i); b[i] = double(100 + i); }

  daxpy(2.0, x.data(), b.data(), y.data(), N);

  std::cout << "y = 2*x + b = [";
  for (int i = 0; i < N; ++i) std::cout << (i ? ", " : "") << y[i];
  std::cout << "]\n";

  bool ok = true;
  for (int i = 0; i < N; ++i)
    if (y[i] != 2.0 * x[i] + b[i]) ok = false;
  std::cout << (ok ? "OK: matches scalar reference (masked tail included)\n" : "MISMATCH\n");
  return ok ? 0 : 1;
}
