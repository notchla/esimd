// Copyright 2026 notchla liso.lorenzo@gmail.com
// SPDX-License-Identifier: Apache-2.0
//
// What this translation unit gets from esimd, decided by its ISA flags alone.
// Standalone: include it without the rest of esimd to query the configuration.
//
//   ESIMD_ISA_<X>       1 for exactly one of SCALAR SSE AVX AVX2 AVX512 NEON NEON2X
//   ESIMD_ISA_NAME      that backend as a string, e.g. "AVX2"
//   ESIMD_HAS_<TYPE>    1 when <TYPE> has a SIMD implementation, else 0
//   ESIMD_TRIG_SLEEF    1 when <esimd/trig.h> is backed by SLEEF,
//                       0 when it is a per-lane libm loop
//
// All are always defined as 0 or 1, so test them with #if, not #ifdef.
// A 0 type still names the generic array struct from varying.h, which has no
// operators. The entry headers include each type exactly when its macro is 1.

#pragma once

#include "detail/platform.h"

////////////////////////////////////////////////////////////////////////////////
/// Backend
////////////////////////////////////////////////////////////////////////////////

// ARM first: NEON2X defines __AVX__/__AVX2__ to reach the 8-wide types.
#if defined(ESIMD_ARM64) && defined(__AVX__)
#  define ESIMD_ISA_NEON2X 1
#  define ESIMD_ISA_NAME "NEON2X"
#elif defined(ESIMD_ARM64) || defined(__ARM_NEON)
#  define ESIMD_ISA_NEON 1
#  define ESIMD_ISA_NAME "NEON"
#elif defined(__AVX512F__)
#  define ESIMD_ISA_AVX512 1
#  define ESIMD_ISA_NAME "AVX512"
#elif defined(__AVX2__)
#  define ESIMD_ISA_AVX2 1
#  define ESIMD_ISA_NAME "AVX2"
#elif defined(__AVX__)
#  define ESIMD_ISA_AVX 1
#  define ESIMD_ISA_NAME "AVX"
#elif defined(__SSE__)
#  define ESIMD_ISA_SSE 1
#  if defined(__SSE4_2__)
#    define ESIMD_ISA_NAME "SSE4.2"
#  elif defined(__SSE4_1__)
#    define ESIMD_ISA_NAME "SSE4.1"
#  else
#    define ESIMD_ISA_NAME "SSE2"
#  endif
#else
#  define ESIMD_ISA_SCALAR 1
#  define ESIMD_ISA_NAME "scalar"
#endif

#if !defined(ESIMD_ISA_SCALAR)
#  define ESIMD_ISA_SCALAR 0
#endif
#if !defined(ESIMD_ISA_SSE)
#  define ESIMD_ISA_SSE 0
#endif
#if !defined(ESIMD_ISA_AVX)
#  define ESIMD_ISA_AVX 0
#endif
#if !defined(ESIMD_ISA_AVX2)
#  define ESIMD_ISA_AVX2 0
#endif
#if !defined(ESIMD_ISA_AVX512)
#  define ESIMD_ISA_AVX512 0
#endif
#if !defined(ESIMD_ISA_NEON)
#  define ESIMD_ISA_NEON 0
#endif
#if !defined(ESIMD_ISA_NEON2X)
#  define ESIMD_ISA_NEON2X 0
#endif

////////////////////////////////////////////////////////////////////////////////
/// Types
////////////////////////////////////////////////////////////////////////////////

// 128-bit
#define ESIMD_HAS_VFLOAT4  (!ESIMD_ISA_SCALAR)
#define ESIMD_HAS_VBOOLF4  ESIMD_HAS_VFLOAT4
#define ESIMD_HAS_VINT4    ESIMD_HAS_VFLOAT4
#define ESIMD_HAS_VUINT4   ESIMD_HAS_VFLOAT4
#define ESIMD_HAS_VBOOLD2  ESIMD_HAS_VFLOAT4
#define ESIMD_HAS_VDOUBLE2 ESIMD_HAS_VFLOAT4
// 64-bit lane compares need SSE4.2, and toScalar a 64-bit target.
#if defined(__64BIT__) && (defined(__SSE4_2__) || defined(ESIMD_ARM64))
#  define ESIMD_HAS_VLLONG2 ESIMD_HAS_VFLOAT4
#else
#  define ESIMD_HAS_VLLONG2 0
#endif

// 256-bit. NEON2X emulates the 32-bit types and vboold4, not vdouble4/vllong4.
#define ESIMD_HAS_VFLOAT8  (ESIMD_ISA_AVX || ESIMD_ISA_AVX2 || ESIMD_ISA_AVX512 || ESIMD_ISA_NEON2X)
#define ESIMD_HAS_VBOOLF8  ESIMD_HAS_VFLOAT8
#define ESIMD_HAS_VINT8    ESIMD_HAS_VFLOAT8
#define ESIMD_HAS_VUINT8   ESIMD_HAS_VFLOAT8
#define ESIMD_HAS_VBOOLD4  ESIMD_HAS_VFLOAT8
#if defined(__X86_64__)
#  define ESIMD_HAS_VDOUBLE4 ESIMD_HAS_VFLOAT8
#  define ESIMD_HAS_VLLONG4  (ESIMD_ISA_AVX2 || ESIMD_ISA_AVX512)
#else
#  define ESIMD_HAS_VDOUBLE4 0
#  define ESIMD_HAS_VLLONG4  0
#endif

// 512-bit
#define ESIMD_HAS_VFLOAT16 ESIMD_ISA_AVX512
#define ESIMD_HAS_VBOOLF16 ESIMD_HAS_VFLOAT16
#define ESIMD_HAS_VINT16   ESIMD_HAS_VFLOAT16
#define ESIMD_HAS_VUINT16  ESIMD_HAS_VFLOAT16
#define ESIMD_HAS_VBOOLD8  ESIMD_HAS_VFLOAT16
#define ESIMD_HAS_VLLONG8  ESIMD_HAS_VFLOAT16
#define ESIMD_HAS_VDOUBLE8 ESIMD_HAS_VFLOAT16

////////////////////////////////////////////////////////////////////////////////
/// Optional layers
////////////////////////////////////////////////////////////////////////////////

// SLEEF ships inline headers only for FMA-capable targets. -mavx2 and
// -mavx512f do not imply -mfma.
#if ESIMD_ISA_NEON || ESIMD_ISA_NEON2X || ((ESIMD_ISA_AVX2 || ESIMD_ISA_AVX512) && defined(__FMA__))
#  define ESIMD_TRIG_SLEEF 1
#else
#  define ESIMD_TRIG_SLEEF 0
#endif
