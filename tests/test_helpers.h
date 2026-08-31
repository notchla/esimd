// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Small helpers for comparing esimd vector types element-by-element against a
// scalar reference. All esimd vector types expose operator[], so these work for
// any width N.

#pragma once

#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <vector>

namespace esimd_test
{
  // Exact element-wise comparison against an initializer list of expected values.
  template<typename V, typename T>
  inline void expect_eq(const V& v, std::initializer_list<T> expected)
  {
    int i = 0;
    for (T e : expected) { EXPECT_EQ(v[i], e) << "at lane " << i; ++i; }
  }

  // Boolean mask comparison (operator[] yields bool for vboolf types).
  template<typename V>
  inline void expect_mask(const V& v, std::initializer_list<bool> expected)
  {
    int i = 0;
    for (bool e : expected) { EXPECT_EQ(bool(v[i]), e) << "at lane " << i; ++i; }
  }

  // Approximate element-wise comparison (for rcp/rsqrt/rounding/division).
  template<typename V>
  inline void expect_near(const V& v, std::initializer_list<float> expected, float tol)
  {
    int i = 0;
    for (float e : expected) { EXPECT_NEAR(v[i], e, tol) << "at lane " << i; ++i; }
  }

  // Error of `actual` against a higher-precision reference, in ULP of actual's
  // own type. The SLEEF-backed trig layer specifies accuracy in ULP (u10 = <=1,
  // u35 = <=3.5) over ranges where an absolute tolerance is meaningless, so
  // expect_near cannot express its bounds.
  template<typename T>
  inline double ulp_error(T actual, double expected)
  {
    const double a = double(actual);
    if (std::isnan(a) || std::isnan(expected))
      return (std::isnan(a) && std::isnan(expected)) ? 0.0 : HUGE_VAL;
    if (a == expected) return 0.0;
    if (std::isinf(a) || std::isinf(expected)) return HUGE_VAL;
    // Spacing of the binade *containing* the reference, i.e. step away from
    // zero: just above a power of two the spacing doubles, and stepping toward
    // zero there would report twice the real error.
    const T e = T(expected);
    const T inf = std::numeric_limits<T>::infinity();
    const double ulp = std::fabs(double(std::nextafter(e, e < T(0) ? -inf : inf)) - double(e));
    return std::fabs(a - expected) / ulp;
  }

  // Element-wise ULP comparison against a per-lane reference.
  template<typename V>
  inline void expect_ulp(const V& v, const std::vector<double>& expected, double max_ulp)
  {
    for (size_t i = 0; i < expected.size(); ++i)
      EXPECT_LE(ulp_error(v[i], expected[i]), max_ulp)
        << "at lane " << i << " (got " << v[i] << ", want " << expected[i] << ")";
  }
}
