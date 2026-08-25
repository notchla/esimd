// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "detail/platform.h"
#include "detail/intrinsics.h"
#include "detail/alloc.h"
#include "detail/constants.h"
#include "varying.h"

namespace esimd 
{
#if defined(ESIMD_ARM64) || defined(__SSE4_1__)
  __forceinline __m128 blendv_ps(__m128 f, __m128 t, __m128 mask) {
    return _mm_blendv_ps(f,t,mask);
  }
#else
  __forceinline __m128 blendv_ps(__m128 f, __m128 t, __m128 mask) { 
    return _mm_or_ps(_mm_and_ps(mask, t), _mm_andnot_ps(mask, f)); 
  }
#endif

  /* header-only lookup tables (were extern in sse.cpp; now C++17 inline variables) */
  inline const __m128 mm_lookupmask_ps[16] = {
    _mm_castsi128_ps(_mm_set_epi32( 0, 0, 0, 0)),
    _mm_castsi128_ps(_mm_set_epi32( 0, 0, 0,-1)),
    _mm_castsi128_ps(_mm_set_epi32( 0, 0,-1, 0)),
    _mm_castsi128_ps(_mm_set_epi32( 0, 0,-1,-1)),
    _mm_castsi128_ps(_mm_set_epi32( 0,-1, 0, 0)),
    _mm_castsi128_ps(_mm_set_epi32( 0,-1, 0,-1)),
    _mm_castsi128_ps(_mm_set_epi32( 0,-1,-1, 0)),
    _mm_castsi128_ps(_mm_set_epi32( 0,-1,-1,-1)),
    _mm_castsi128_ps(_mm_set_epi32(-1, 0, 0, 0)),
    _mm_castsi128_ps(_mm_set_epi32(-1, 0, 0,-1)),
    _mm_castsi128_ps(_mm_set_epi32(-1, 0,-1, 0)),
    _mm_castsi128_ps(_mm_set_epi32(-1, 0,-1,-1)),
    _mm_castsi128_ps(_mm_set_epi32(-1,-1, 0, 0)),
    _mm_castsi128_ps(_mm_set_epi32(-1,-1, 0,-1)),
    _mm_castsi128_ps(_mm_set_epi32(-1,-1,-1, 0)),
    _mm_castsi128_ps(_mm_set_epi32(-1,-1,-1,-1))
  };

  inline const __m128d mm_lookupmask_pd[4] = {
    _mm_castsi128_pd(_mm_set_epi32( 0, 0, 0, 0)),
    _mm_castsi128_pd(_mm_set_epi32( 0, 0,-1,-1)),
    _mm_castsi128_pd(_mm_set_epi32(-1,-1, 0, 0)),
    _mm_castsi128_pd(_mm_set_epi32(-1,-1,-1,-1))
  };
}

#if defined(__AVX512VL__)
#include "vboolf4_avx512.h"
#else
#include "vboolf4_sse2.h"
#endif
#include "vint4_sse2.h"
#include "vuint4_sse2.h"
#include "vfloat4_sse2.h"
