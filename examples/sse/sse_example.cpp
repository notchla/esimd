// Copyright 2026 notchla liso.lorenzo@gmail.com
// SPDX-License-Identifier: Apache-2.0
//
// esimd showcase — SSE (128-bit, 4-wide).
//
// Demonstrates the 4-wide types (vfloat4 / vint4 / vboolf4): loads, arithmetic,
// fast reciprocal-sqrt normalization, dot & cross products, and compare+select.
//
// Build standalone (from this folder):
//
//   g++ -std=c++17 -I../../include \
//       -msse4.2 -D__SSE__ -D__SSE2__ -D__SSE4_1__ -D__SSE4_2__ \
//       sse_example.cpp -o sse_example
//   ./sse_example
//
// The include below is the *only* esimd line you need; the -D__SSE*__ flags above
// pick the SSE backend behind it.

#include <esimd/esimd.h>

#include <iostream>

using namespace esimd;

// esimd deliberately ships no iostream operators, so examples print via the
// element accessor operator[] (which every vector type provides).
template <typename V>
static void print(const char* label, const V& v, int n = 4) {
  std::cout << label << " = [";
  for (int i = 0; i < n; ++i) std::cout << (i ? ", " : "") << v[i];
  std::cout << "]\n";
}

int main() {
  std::cout << "== esimd SSE example (4-wide) ==\n";

  // A vfloat4 can hold an xyz vector (4th lane unused / 0).
  const vfloat4 a(1.0f, 2.0f, 2.0f, 0.0f); // |a| = 3
  const vfloat4 b(4.0f, 0.0f, 3.0f, 0.0f); // |b| = 5
  print("a", a);
  print("b", b);

  // Elementwise arithmetic and a fused multiply-add: a*2 + b.
  print("a + b", a + b);
  print("madd(a,2,b) = a*2+b", madd(a, vfloat4(2.0f), b));

  // Length via dot + sqrt, and a fast normalize via rsqrt (Newton-refined in the
  // library). reduce_add collapses the 4 lanes to a scalar.
  const float len_a = std::sqrt(dot(a, a));
  std::cout << "dot(a,a) = " << dot(a, a) << ", |a| = " << len_a << "\n";
  const vfloat4 a_hat = a * rsqrt(vfloat4(dot(a, a)));
  print("normalize(a)", a_hat);
  std::cout << "|normalize(a)| ~ " << std::sqrt(dot(a_hat, a_hat)) << "\n";

  // Cross product (esimd provides cross() for vfloat4).
  print("cross(a,b)", cross(a, b));

  // Comparisons yield a vboolf4 mask; select() blends lane-wise.
  const vboolf4 mask = a < b; // lanewise a < b
  std::cout << "movemask(a < b) = 0x" << std::hex << movemask(mask) << std::dec << "\n";
  print("select(a<b, a, b)  (min per lane)", select(mask, a, b));
  print("max(a, b)", max(a, b));

  // Integer lane demo: vint4 with a horizontal reduction.
  const vint4 iv(3, 1, 4, 1);
  print("iv", iv);
  std::cout << "reduce_add(iv) = " << reduce_add(iv)
            << ", reduce_max(iv) = " << reduce_max(iv) << "\n";

  return 0;
}
