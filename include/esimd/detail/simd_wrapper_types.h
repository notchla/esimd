// Copyright 2009-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "platform.h"

namespace esimd
{
  // Concrete wrappers around the SSE register types. They avoid attribute-bearing
  // template arguments such as __m128 in identity_wrapper<__m128>.
  struct __m128_wrapper {
    __m128 data;
    __forceinline __m128_wrapper() {}
    __forceinline __m128_wrapper(__m128 v) : data(v) {}
    __forceinline operator __m128() const { return data; }
    __forceinline operator __m128&() { return data; }
  };

  struct __m128i_wrapper {
    __m128i data;
    __forceinline __m128i_wrapper() {}
    __forceinline __m128i_wrapper(__m128i v) : data(v) {}
    __forceinline operator __m128i() const { return data; }
    __forceinline operator __m128i&() { return data; }
  };

  struct __m128d_wrapper {
    __m128d data;
    __forceinline __m128d_wrapper() {}
    __forceinline __m128d_wrapper(__m128d v) : data(v) {}
    __forceinline operator __m128d() const { return data; }
    __forceinline operator __m128d&() { return data; }
  };
}
