// Copyright 2026 notchla liso.lorenzo@gmail.com
// SPDX-License-Identifier: Apache-2.0
//
// Correctness tests for the optional trigonometry layer (<esimd/trig.h>). Every
// function is checked lane-by-lane against a long double reference and must
// stay inside SLEEF's documented ULP bound. Compiled once per ISA, so the SLEEF
// backends and the libm fallback are held to the same bounds.

#include <esimd/trig.h>
#include "test_helpers.h"

#if defined(ESIMD_HAS_TRIG)

using namespace esimd;
using namespace esimd_test;

namespace {

// SLEEF's documented bounds. The reference is evaluated in long double, so its
// own error is negligible at float and double result precision.
constexpr double ULP_U10   = 1.0;
constexpr double ULP_U35   = 3.5;
constexpr double ULP_U3500 = 3500.0;

// Arguments spanning the small range, the reduction range, and the payne-hanek
// path that kicks in for large arguments.
const std::vector<double> kWide = {
  0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 0.7853981633974483, -0.7853981633974483,
  1.5707963267948966, -1.5707963267948966, 3.141592653589793, -3.141592653589793,
  6.283185307179586, 3.7, -3.7, 10.0, -10.0, 100.0, -100.0,
  12345.678, -12345.678, 1.0e6, -1.0e6, 1.0e9
};

// Domain of asin/acos, edges included.
const std::vector<double> kUnit = {
  0.0, -0.0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75,
  0.9, -0.9, 0.999, -0.999, 1.0, -1.0
};

// fast_sin/fast_cos are only specified for |x| < 125000.
const std::vector<double> kFast = {
  0.0, 0.5, -0.5, 1.0, -1.0, 3.7, -3.7, 10.0, -10.0,
  100.0, -100.0, 1000.0, -1000.0, 12345.678, -12345.678, 100000.0
};

template<typename V>
using lane_t = typename std::decay<decltype(std::declval<V>()[0])>::type;

// Build a V from `inputs` starting at `offset`, cycling when the table is not a
// multiple of the vector width.
template<typename V>
V pack(const std::vector<double>& inputs, size_t offset)
{
  V v;
  for (int i = 0; i < V::size; ++i)
    v[i] = lane_t<V>(inputs[(offset + i) % inputs.size()]);
  return v;
}

// Walk the whole table in V::size chunks, checking every lane in ULP. The
// reference is taken at the input as actually represented in the lane type, so
// input rounding is not counted against the function under test.
template<typename V, typename VecFn, typename RefFn>
void check_unary(const std::vector<double>& inputs, VecFn vecfn, RefFn reffn, double max_ulp)
{
  for (size_t o = 0; o < inputs.size(); o += V::size) {
    const V x = pack<V>(inputs, o);
    std::vector<double> ref(V::size);
    for (int i = 0; i < V::size; ++i)
      ref[i] = double(reffn((long double)x[i]));
    expect_ulp(vecfn(x), ref, max_ulp);
  }
}

// Same, over every (y, x) pair drawn from the table at a fixed lane offset so
// all four quadrants and both zeros are covered.
template<typename V, typename VecFn, typename RefFn>
void check_binary(const std::vector<double>& inputs, VecFn vecfn, RefFn reffn, double max_ulp)
{
  for (size_t o = 0; o < inputs.size(); o += V::size) {
    for (size_t shift = 1; shift < 4; ++shift) {
      const V y = pack<V>(inputs, o);
      const V x = pack<V>(inputs, o + shift);
      std::vector<double> ref(V::size);
      for (int i = 0; i < V::size; ++i)
        ref[i] = double(reffn((long double)y[i], (long double)x[i]));
      expect_ulp(vecfn(y, x), ref, max_ulp);
    }
  }
}

} // namespace

// Every assertion below is instantiated once per vector type the active ISA
// provides. TYPE names the suite so a failure reports which width broke.
#define ESIMD_TRIG_TESTS(SUITE, V)                                                        \
TEST(SUITE, sin_cos_tan_u10) {                                                            \
  check_unary<V>(kWide, [](const V& a) { return sin(a); },                                \
                        [](long double v) { return std::sin(v); }, ULP_U10);              \
  check_unary<V>(kWide, [](const V& a) { return cos(a); },                                \
                        [](long double v) { return std::cos(v); }, ULP_U10);              \
  check_unary<V>(kWide, [](const V& a) { return tan(a); },                                \
                        [](long double v) { return std::tan(v); }, ULP_U10);              \
}                                                                                          \
TEST(SUITE, sin_cos_tan_u35) {                                                            \
  check_unary<V>(kWide, [](const V& a) { return sin_u35(a); },                            \
                        [](long double v) { return std::sin(v); }, ULP_U35);              \
  check_unary<V>(kWide, [](const V& a) { return cos_u35(a); },                            \
                        [](long double v) { return std::cos(v); }, ULP_U35);              \
  check_unary<V>(kWide, [](const V& a) { return tan_u35(a); },                            \
                        [](long double v) { return std::tan(v); }, ULP_U35);              \
}                                                                                          \
TEST(SUITE, asin_acos_u10) {                                                              \
  check_unary<V>(kUnit, [](const V& a) { return asin(a); },                               \
                        [](long double v) { return std::asin(v); }, ULP_U10);             \
  check_unary<V>(kUnit, [](const V& a) { return acos(a); },                               \
                        [](long double v) { return std::acos(v); }, ULP_U10);             \
}                                                                                          \
TEST(SUITE, asin_acos_u35) {                                                              \
  check_unary<V>(kUnit, [](const V& a) { return asin_u35(a); },                           \
                        [](long double v) { return std::asin(v); }, ULP_U35);             \
  check_unary<V>(kUnit, [](const V& a) { return acos_u35(a); },                           \
                        [](long double v) { return std::acos(v); }, ULP_U35);             \
}                                                                                          \
TEST(SUITE, atan) {                                                                       \
  check_unary<V>(kWide, [](const V& a) { return atan(a); },                               \
                        [](long double v) { return std::atan(v); }, ULP_U10);             \
  check_unary<V>(kWide, [](const V& a) { return atan_u35(a); },                           \
                        [](long double v) { return std::atan(v); }, ULP_U35);             \
}                                                                                          \
TEST(SUITE, atan2) {                                                                      \
  check_binary<V>(kWide, [](const V& y, const V& x) { return atan2(y, x); },              \
                         [](long double y, long double x) { return std::atan2(y, x); },   \
                         ULP_U10);                                                        \
  check_binary<V>(kWide, [](const V& y, const V& x) { return atan2_u35(y, x); },          \
                         [](long double y, long double x) { return std::atan2(y, x); },   \
                         ULP_U35);                                                        \
}                                                                                          \
/* sincos is a distinct SLEEF routine (one shared range reduction feeding both
   results), so it is not bit-identical to separate sin/cos calls -- only bound
   by the same ULP contract. */                                                           \
TEST(SUITE, sincos) {                                                                     \
  for (size_t o = 0; o < kWide.size(); o += V::size) {                                    \
    const V x = pack<V>(kWide, o);                                                        \
    std::vector<double> rs(V::size), rc(V::size);                                         \
    for (int i = 0; i < V::size; ++i) {                                                   \
      rs[i] = double(std::sin((long double)x[i]));                                        \
      rc[i] = double(std::cos((long double)x[i]));                                        \
    }                                                                                     \
    V s, c;     sincos(x, s, c);                                                          \
    V s35, c35; sincos_u35(x, s35, c35);                                                  \
    expect_ulp(s,   rs, ULP_U10);                                                         \
    expect_ulp(c,   rc, ULP_U10);                                                         \
    expect_ulp(s35, rs, ULP_U35);                                                         \
    expect_ulp(c35, rc, ULP_U35);                                                         \
  }                                                                                       \
}

// fast_sin/fast_cos are single precision only -- SLEEF has no double u3500.
#define ESIMD_TRIG_FAST_TESTS(SUITE, V)                                                   \
TEST(SUITE, fast_sin_cos) {                                                               \
  check_unary<V>(kFast, [](const V& a) { return fast_sin(a); },                           \
                        [](long double v) { return std::sin(v); }, ULP_U3500);            \
  check_unary<V>(kFast, [](const V& a) { return fast_cos(a); },                           \
                        [](long double v) { return std::cos(v); }, ULP_U3500);            \
}

// Instantiate for exactly the types the active ISA defines. vfloat4 is the only
// one every backend has: SSE4.2 stops there, and ARM has no vdouble at all.
ESIMD_TRIG_TESTS(trig_vfloat4, vfloat4)
ESIMD_TRIG_FAST_TESTS(trig_vfloat4, vfloat4)

#if defined(__AVX__) // AVX, AVX2, AVX512 and NEON2X all reach the 8-wide types
ESIMD_TRIG_TESTS(trig_vfloat8, vfloat8)
ESIMD_TRIG_FAST_TESTS(trig_vfloat8, vfloat8)
#if defined(__X86_64__)
ESIMD_TRIG_TESTS(trig_vdouble4, vdouble4)
#endif
#endif

#if defined(__AVX512F__)
ESIMD_TRIG_TESTS(trig_vfloat16, vfloat16)
ESIMD_TRIG_FAST_TESTS(trig_vfloat16, vfloat16)
ESIMD_TRIG_TESTS(trig_vdouble8, vdouble8)
#endif

#endif // ESIMD_HAS_TRIG
