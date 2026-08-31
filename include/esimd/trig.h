// Copyright 2026 notchla liso.lorenzo@gmail.com
// SPDX-License-Identifier: Apache-2.0
//
// Optional trigonometry layer for the esimd vector types, backed by SLEEF's
// inline headers (Boost Software License 1.0, see detail/sleef/).
//
// Opt-in: <esimd/esimd.h> never pulls this in. Define ESIMD_DISABLE_TRIG (or
// build with -DESIMD_ENABLE_TRIG=OFF) to compile it away entirely.
//
// COMPILE THIS TRANSLATION UNIT WITH -ffp-contract=off. SLEEF reconstructs the
// rounding error of an unfused a*b, which a contracted FMA discards, so the ULP
// bounds below do not hold without it. The `#pragma STDC FP_CONTRACT OFF` the
// headers carry is honoured by Clang and ignored by GCC. esimd::esimd does not
// supply the flag, as that would disable contraction in every TU linking it.
//
//   f(x)        SLEEF u10    -- <=1   ULP
//   f_u35(x)    SLEEF u35    -- <=3.5 ULP, faster
//   fast_f(x)   SLEEF u3500  -- float only, |x| < 125000
//
// Each width dispatches to the SLEEF build for that same instruction set. SLEEF
// generates inline headers only for FMA-capable targets, so SSE4.2 and plain
// AVX instead get a per-lane libm loop, flagged by ESIMD_TRIG_SCALAR_FALLBACK.

#pragma once

#include "esimd.h"

#include <cmath>
#include <cstring> // the generated SLEEF headers call memcpy without including it

// Backend selection. ARM must be tested first: the NEON2X flag set defines
// __AVX2__/__AVX__ to reach the 8-wide types even though there is no x86
// hardware, so the x86 branch would otherwise be taken on AArch64.
//
// The generated headers open with `#pragma STDC FP_CONTRACT OFF`, which GCC
// does not implement, and define a handful of helpers this layer never calls.
// SLEEF's own build silences both; do the same for the includes only.
#if defined(__GNUC__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wunknown-pragmas"
#  pragma GCC diagnostic ignored "-Wunused-function"
#endif

#if defined(ESIMD_DISABLE_TRIG)
// nothing: the layer is switched off, ESIMD_HAS_TRIG stays undefined
#elif defined(ESIMD_ARM64)
#  define ESIMD_HAS_TRIG
#  include "detail/sleef/sleefinline_advsimd.h"
#elif defined(__AVX2__)
#  define ESIMD_HAS_TRIG
#  include "detail/sleef/sleefinline_avx2128.h"
#  include "detail/sleef/sleefinline_avx2.h"
#  if defined(__AVX512F__)
#    include "detail/sleef/sleefinline_avx512f.h"
#  endif
#else // SSE4.2, plain AVX: no SLEEF inline header exists for a non-FMA target
#  define ESIMD_HAS_TRIG
#  define ESIMD_TRIG_SCALAR_FALLBACK
#endif

#if defined(__GNUC__)
#  pragma GCC diagnostic pop
#endif

#if defined(ESIMD_HAS_TRIG)

namespace esimd
{

#if defined(ESIMD_TRIG_SCALAR_FALLBACK)

  // Apply a scalar function lane by lane. Not vectorised; it keeps the API
  // uniform on the ISAs SLEEF cannot serve.
  template<typename V, typename Fn>
  __forceinline V trig_lanewise(const V& a, Fn fn)
  {
    V r;
    for (int i = 0; i < V::size; ++i) r[i] = fn(a[i]);
    return r;
  }

  template<typename V, typename Fn>
  __forceinline V trig_lanewise(const V& a, const V& b, Fn fn)
  {
    V r;
    for (int i = 0; i < V::size; ++i) r[i] = fn(a[i], b[i]);
    return r;
  }

  // The accuracy suffixes are bounds and libm already sits inside the tightest
  // of them, so the u35 names alias the plain ones.
#define ESIMD_TRIG_SCALAR_DEFS(VEC)                                                          \
  __forceinline VEC sin (const VEC& a) { return trig_lanewise(a, [](auto v) { return std::sin(v); });  } \
  __forceinline VEC cos (const VEC& a) { return trig_lanewise(a, [](auto v) { return std::cos(v); });  } \
  __forceinline VEC tan (const VEC& a) { return trig_lanewise(a, [](auto v) { return std::tan(v); });  } \
  __forceinline VEC asin(const VEC& a) { return trig_lanewise(a, [](auto v) { return std::asin(v); }); } \
  __forceinline VEC acos(const VEC& a) { return trig_lanewise(a, [](auto v) { return std::acos(v); }); } \
  __forceinline VEC atan(const VEC& a) { return trig_lanewise(a, [](auto v) { return std::atan(v); }); } \
  __forceinline VEC sin_u35 (const VEC& a) { return sin(a);  }                                \
  __forceinline VEC cos_u35 (const VEC& a) { return cos(a);  }                                \
  __forceinline VEC tan_u35 (const VEC& a) { return tan(a);  }                                \
  __forceinline VEC asin_u35(const VEC& a) { return asin(a); }                                \
  __forceinline VEC acos_u35(const VEC& a) { return acos(a); }                                \
  __forceinline VEC atan_u35(const VEC& a) { return atan(a); }                                \
  __forceinline VEC atan2(const VEC& y, const VEC& x) {                                       \
    return trig_lanewise(y, x, [](auto a, auto b) { return std::atan2(a, b); }); }             \
  __forceinline VEC atan2_u35(const VEC& y, const VEC& x) { return atan2(y, x); }              \
  __forceinline void sincos(const VEC& a, VEC& s, VEC& c) {                                   \
    for (int i = 0; i < VEC::size; ++i) { s[i] = std::sin(a[i]); c[i] = std::cos(a[i]); } }     \
  __forceinline void sincos_u35(const VEC& a, VEC& s, VEC& c) { sincos(a, s, c); }

  // Kept float-only to match the SLEEF backends, which have no double u3500.
#define ESIMD_TRIG_SCALAR_FAST_DEFS(VEC)                                                      \
  __forceinline VEC fast_sin(const VEC& a) { return sin(a); }                                 \
  __forceinline VEC fast_cos(const VEC& a) { return cos(a); }

  ESIMD_TRIG_SCALAR_DEFS(vfloat4)
  ESIMD_TRIG_SCALAR_FAST_DEFS(vfloat4)
#if defined(__AVX__) // plain AVX also has the 8-wide types
  ESIMD_TRIG_SCALAR_DEFS(vfloat8)
  ESIMD_TRIG_SCALAR_FAST_DEFS(vfloat8)
#if defined(__X86_64__) // vdouble4 is gated the same way in avx.h
  ESIMD_TRIG_SCALAR_DEFS(vdouble4)
#endif
#endif

#undef ESIMD_TRIG_SCALAR_DEFS
#undef ESIMD_TRIG_SCALAR_FAST_DEFS

#else // SLEEF backends

  // SLEEF names decompose as Sleef_<fn><TOK>_<ACC><ISA>, e.g. Sleef_sinf8_u10avx2.
#define ESIMD_TRIG_DEFS(VEC, TOK, ISA)                                                       \
  __forceinline VEC sin (const VEC& a) { return Sleef_sin##TOK##_u10##ISA(a); }               \
  __forceinline VEC cos (const VEC& a) { return Sleef_cos##TOK##_u10##ISA(a); }               \
  __forceinline VEC tan (const VEC& a) { return Sleef_tan##TOK##_u10##ISA(a); }               \
  __forceinline VEC asin(const VEC& a) { return Sleef_asin##TOK##_u10##ISA(a); }              \
  __forceinline VEC acos(const VEC& a) { return Sleef_acos##TOK##_u10##ISA(a); }              \
  __forceinline VEC atan(const VEC& a) { return Sleef_atan##TOK##_u10##ISA(a); }              \
  __forceinline VEC sin_u35 (const VEC& a) { return Sleef_sin##TOK##_u35##ISA(a); }           \
  __forceinline VEC cos_u35 (const VEC& a) { return Sleef_cos##TOK##_u35##ISA(a); }           \
  __forceinline VEC tan_u35 (const VEC& a) { return Sleef_tan##TOK##_u35##ISA(a); }           \
  __forceinline VEC asin_u35(const VEC& a) { return Sleef_asin##TOK##_u35##ISA(a); }          \
  __forceinline VEC acos_u35(const VEC& a) { return Sleef_acos##TOK##_u35##ISA(a); }          \
  __forceinline VEC atan_u35(const VEC& a) { return Sleef_atan##TOK##_u35##ISA(a); }          \
  __forceinline VEC atan2    (const VEC& y, const VEC& x) { return Sleef_atan2##TOK##_u10##ISA(y, x); } \
  __forceinline VEC atan2_u35(const VEC& y, const VEC& x) { return Sleef_atan2##TOK##_u35##ISA(y, x); } \
  __forceinline void sincos(const VEC& a, VEC& s, VEC& c) {                                   \
    const auto r = Sleef_sincos##TOK##_u10##ISA(a); s = r.x; c = r.y; }                        \
  __forceinline void sincos_u35(const VEC& a, VEC& s, VEC& c) {                               \
    const auto r = Sleef_sincos##TOK##_u35##ISA(a); s = r.x; c = r.y; }

  // SLEEF has no fasttan, and the u3500 variants are single precision only.
#define ESIMD_TRIG_FAST_DEFS(VEC, TOK, ISA)                                                  \
  __forceinline VEC fast_sin(const VEC& a) { return Sleef_fastsin##TOK##_u3500##ISA(a); }     \
  __forceinline VEC fast_cos(const VEC& a) { return Sleef_fastcos##TOK##_u3500##ISA(a); }

#if defined(ESIMD_ARM64)

  // sse2neon typedefs __m128 as float32x4_t, which is exactly what the advsimd
  // functions take, so vfloat4 converts straight through. There is no vdouble2
  // in esimd (and vdouble4 is x86-only), so ARM gets float overloads only.
  ESIMD_TRIG_DEFS(vfloat4, f4, advsimd)
  ESIMD_TRIG_FAST_DEFS(vfloat4, f4, advsimd)

#if defined(__AVX__) // NEON2X: avx2neon makes __m256 a { __m128 lo, hi; } pair.
#define ESIMD_TRIG_PAIR_1(NAME, SLEEFNAME)                                                   \
  __forceinline vfloat8 NAME(const vfloat8& a) {                                              \
    __m256 r; r.lo = SLEEFNAME(a.v.lo); r.hi = SLEEFNAME(a.v.hi); return r; }
#define ESIMD_TRIG_PAIR_2(NAME, SLEEFNAME)                                                   \
  __forceinline vfloat8 NAME(const vfloat8& y, const vfloat8& x) {                            \
    __m256 r; r.lo = SLEEFNAME(y.v.lo, x.v.lo); r.hi = SLEEFNAME(y.v.hi, x.v.hi); return r; }
#define ESIMD_TRIG_PAIR_SINCOS(NAME, SLEEFNAME)                                              \
  __forceinline void NAME(const vfloat8& a, vfloat8& s, vfloat8& c) {                         \
    const auto lo = SLEEFNAME(a.v.lo); const auto hi = SLEEFNAME(a.v.hi);                      \
    __m256 rs, rc; rs.lo = lo.x; rs.hi = hi.x; rc.lo = lo.y; rc.hi = hi.y;                     \
    s = rs; c = rc; }

  ESIMD_TRIG_PAIR_1(sin,      Sleef_sinf4_u10advsimd)
  ESIMD_TRIG_PAIR_1(cos,      Sleef_cosf4_u10advsimd)
  ESIMD_TRIG_PAIR_1(tan,      Sleef_tanf4_u10advsimd)
  ESIMD_TRIG_PAIR_1(asin,     Sleef_asinf4_u10advsimd)
  ESIMD_TRIG_PAIR_1(acos,     Sleef_acosf4_u10advsimd)
  ESIMD_TRIG_PAIR_1(atan,     Sleef_atanf4_u10advsimd)
  ESIMD_TRIG_PAIR_1(sin_u35,  Sleef_sinf4_u35advsimd)
  ESIMD_TRIG_PAIR_1(cos_u35,  Sleef_cosf4_u35advsimd)
  ESIMD_TRIG_PAIR_1(tan_u35,  Sleef_tanf4_u35advsimd)
  ESIMD_TRIG_PAIR_1(asin_u35, Sleef_asinf4_u35advsimd)
  ESIMD_TRIG_PAIR_1(acos_u35, Sleef_acosf4_u35advsimd)
  ESIMD_TRIG_PAIR_1(atan_u35, Sleef_atanf4_u35advsimd)
  ESIMD_TRIG_PAIR_1(fast_sin, Sleef_fastsinf4_u3500advsimd)
  ESIMD_TRIG_PAIR_1(fast_cos, Sleef_fastcosf4_u3500advsimd)
  ESIMD_TRIG_PAIR_2(atan2,     Sleef_atan2f4_u10advsimd)
  ESIMD_TRIG_PAIR_2(atan2_u35, Sleef_atan2f4_u35advsimd)
  ESIMD_TRIG_PAIR_SINCOS(sincos,     Sleef_sincosf4_u10advsimd)
  ESIMD_TRIG_PAIR_SINCOS(sincos_u35, Sleef_sincosf4_u35advsimd)

#undef ESIMD_TRIG_PAIR_1
#undef ESIMD_TRIG_PAIR_2
#undef ESIMD_TRIG_PAIR_SINCOS
#endif // __AVX__ (NEON2X)

#else // x86 with AVX2 or better

  ESIMD_TRIG_DEFS(vfloat4, f4, avx2128)
  ESIMD_TRIG_FAST_DEFS(vfloat4, f4, avx2128)
  ESIMD_TRIG_DEFS(vfloat8, f8, avx2)
  ESIMD_TRIG_FAST_DEFS(vfloat8, f8, avx2)
#if defined(__X86_64__) // vdouble4 is gated the same way in avx.h
  ESIMD_TRIG_DEFS(vdouble4, d4, avx2)
#endif

#if defined(__AVX512F__)
  ESIMD_TRIG_DEFS(vfloat16, f16, avx512f)
  ESIMD_TRIG_FAST_DEFS(vfloat16, f16, avx512f)
  ESIMD_TRIG_DEFS(vdouble8, d8, avx512f)
#endif

#endif // ESIMD_ARM64

#undef ESIMD_TRIG_DEFS
#undef ESIMD_TRIG_FAST_DEFS

#endif // ESIMD_TRIG_SCALAR_FALLBACK
}

#endif // ESIMD_HAS_TRIG
