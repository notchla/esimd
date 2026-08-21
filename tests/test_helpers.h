// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Small helpers for comparing esimd vector types element-by-element against a
// scalar reference. All esimd vector types expose operator[], so these work for
// any width N.

#pragma once

#include <gtest/gtest.h>
#include <cmath>

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
}
