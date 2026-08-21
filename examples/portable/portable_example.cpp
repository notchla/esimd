// Copyright 2026 notchla liso.lorenzo@gmail.com
// SPDX-License-Identifier: Apache-2.0
//
// esimd showcase — width-agnostic code with vfloatx / VSIZEX.
//
// Unlike the per-ISA examples, this one names NO fixed width. It uses the default
// aliases instead:
//
//   VSIZEX   the default lane count for the build (4 under SSE, 8 under AVX/AVX2/AVX512)
//   vfloatx  = vfloat<VSIZEX>      vintx = vint<VSIZEX>      vboolx = vbool<VSIZEX>
//
//   VSIZEL   the "large" lane count (4 under SSE, 8 under AVX/AVX2, 16 under AVX512).
//            On AVX512 the default width stays 8 to avoid 512-bit downclocking, but
//            VSIZEL exposes the full 16-wide type when you want it. esimd ships the
//            VSIZEL *constant* but not matching aliases, so this example defines
//            vfloatl/vintl/vbooll = vfloat<VSIZEL> etc.
//
// The *same source* is compiled once per ISA (see this folder's CMakeLists), and
// each build processes a different number of lanes per iteration — you write the
// loop once and the width follows the flags. The canonical use is shown below: a
// SAXPY (y = a*x + b) over an array of ANY length, with a masked remainder for the
// tail that doesn't fill a full VSIZEX-wide vector.
//
// Build standalone for, say, AVX2 (from this folder):
//
//   g++ -std=c++17 -I../../include \
//       -mavx2 -mfma -mf16c -mbmi -mbmi2 -mlzcnt \
//       -D__AVX2__ -D__AVX__ -D__SSE4_2__ -D__SSE4_1__ -D__LZCNT__ -D__BMI__ \
//       portable_example.cpp -o portable_example
//   ./portable_example
//
// Swap the flag row (see examples/README.md) to build the identical file for SSE,
// AVX, or AVX512. Or via CMake: ./build/examples/portable/esimd_example_portable_<isa>.

#include <esimd/esimd.h>

#include <iostream>
#include <vector>

using namespace esimd;

// Large-width aliases, mirroring esimd's built-in vfloatx/vintx/vboolx but keyed on
// VSIZEL instead of VSIZEX. vfloat<VSIZEL> resolves to a real type at every ISA
// level: vfloat4 (SSE), vfloat8 (AVX/AVX2), vfloat16 (AVX512).
using vfloatl = vfloat<VSIZEL>;
using vintl   = vint<VSIZEL>;
using vbooll  = vbool<VSIZEL>;

// y[i] = a*x[i] + b[i] for an array of arbitrary length n, written once with the
// default-width aliases. No width literal appears anywhere in this function.
static void saxpy(float a, const float* x, const float* b, float* y, int n) {
  const vfloatx va(a);
  int i = 0;

  // Body: process VSIZEX elements per iteration with one fused multiply-add.
  for (; i + VSIZEX <= n; i += VSIZEX) {
    const vfloatx vx = vfloatx::loadu(x + i);
    const vfloatx vb = vfloatx::loadu(b + i);
    vfloatx::storeu(y + i, madd(va, vx, vb)); // va*vx + vb
  }

  // Tail: the last (n - i) elements don't fill a full vector. Build a predicate
  // mask whose first (n - i) lanes are true — vintx(step) is {0,1,...,VSIZEX-1} —
  // and use masked load/store so we never touch memory past the array.
  if (i < n) {
    const vboolx tail = vintx(step) < vintx(n - i);
    const vfloatx vx = vfloatx::loadu(tail, x + i);
    const vfloatx vb = vfloatx::loadu(tail, b + i);
    vfloatx::storeu(tail, y + i, madd(va, vx, vb));
  }
}

int main() {
  std::cout << "== esimd portable example (vfloatx / VSIZEX, vfloatl / VSIZEL) ==\n";
  std::cout << "VSIZEX (default width) = " << VSIZEX
            << ", VSIZEL (large width) = " << VSIZEL << "\n";

  // The default-width aliases in action: vintx(step) is the per-lane index vector.
  const vintx idx(step);
  std::cout << "vintx(step) = [";
  for (int i = 0; i < VSIZEX; ++i) std::cout << (i ? ", " : "") << idx[i];
  std::cout << "], reduce_add = " << reduce_add(idx) << "\n";

  // The large-width aliases: same code, VSIZEL lanes. On AVX512 this is 16-wide
  // while vintx above is only 8-wide.
  const vintl idxl(step);
  std::cout << "vintl(step) = [";
  for (int i = 0; i < VSIZEL; ++i) std::cout << (i ? ", " : "") << idxl[i];
  std::cout << "], reduce_add = " << reduce_add(idxl) << "\n";

  // Run the width-agnostic SAXPY on an array whose length (10) is deliberately not
  // a multiple of 4 or 8, so the masked tail path is always exercised.
  const int N = 10;
  std::vector<float> x(N), b(N), y(N, -1.0f);
  for (int i = 0; i < N; ++i) { x[i] = float(i); b[i] = float(100 + i); }

  saxpy(2.0f, x.data(), b.data(), y.data(), N);

  std::cout << "y = 2*x + b = [";
  for (int i = 0; i < N; ++i) std::cout << (i ? ", " : "") << y[i];
  std::cout << "]\n";

  // Verify against a plain scalar reference.
  bool ok = true;
  for (int i = 0; i < N; ++i)
    if (y[i] != 2.0f * x[i] + b[i]) ok = false;
  std::cout << (ok ? "OK: matches scalar reference (masked tail included)\n"
                   : "MISMATCH\n");
  return ok ? 0 : 1;
}
