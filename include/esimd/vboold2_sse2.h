// Copyright 2026 notchla liso.lorenzo@gmail.com
// SPDX-License-Identifier: Apache-2.0

#pragma once

#define vboolf vboolf_impl
#define vboold vboold_impl
#define vint vint_impl
#define vuint vuint_impl
#define vllong vllong_impl
#define vfloat vfloat_impl
#define vdouble vdouble_impl

namespace esimd
{
  /* 2-wide SSE bool type for 64bit data types */
  template<>
  struct vboold<2>
  {
    ALIGNED_STRUCT_(16);

    typedef vboold2 Bool;

    enum  { size = 2 };                        // number of SIMD elements
    union { __m128d v; long long i[2]; };      // data

    ////////////////////////////////////////////////////////////////////////////////
    /// Constructors, Assignment & Cast Operators
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline vboold() {}
    __forceinline vboold(const vboold2& a) { v = a.v; }
    __forceinline vboold2& operator =(const vboold2& a) { v = a.v; return *this; }

    __forceinline vboold(__m128d a) : v(a) {}
    __forceinline vboold(__m128i a) : v(_mm_castsi128_pd(a)) {}

    __forceinline operator const __m128d&() const { return v; }
    __forceinline const __m128i m128i() const { return _mm_castpd_si128(v); }

    __forceinline vboold(bool a) : v(mm_lookupmask_pd[(size_t(a) << 1) | size_t(a)]) {}
    __forceinline vboold(bool a, bool b) : v(mm_lookupmask_pd[(size_t(b) << 1) | size_t(a)]) {}
    __forceinline vboold(int mask) { assert(mask >= 0 && mask < 4); v = mm_lookupmask_pd[mask]; }
    __forceinline vboold(unsigned int mask) { assert(mask < 4); v = mm_lookupmask_pd[mask]; }

    ////////////////////////////////////////////////////////////////////////////////
    /// Constants
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline vboold(FalseTy) : v(_mm_setzero_pd()) {}
    __forceinline vboold(TrueTy)  : v(_mm_castsi128_pd(_mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128()))) {}

    ////////////////////////////////////////////////////////////////////////////////
    /// Array Access
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline bool       operator [](size_t index) const { assert(index < 2); return (_mm_movemask_pd(v) >> index) & 1; }
    __forceinline long long& operator [](size_t index)       { assert(index < 2); return i[index]; }
  };

  ////////////////////////////////////////////////////////////////////////////////
  /// Unary Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vboold2 operator !(const vboold2& a) { return _mm_xor_pd(a, vboold2(esimd::True)); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Binary Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vboold2 operator &(const vboold2& a, const vboold2& b) { return _mm_and_pd(a, b); }
  __forceinline vboold2 operator |(const vboold2& a, const vboold2& b) { return _mm_or_pd (a, b); }
  __forceinline vboold2 operator ^(const vboold2& a, const vboold2& b) { return _mm_xor_pd(a, b); }

  __forceinline vboold2 andn(const vboold2& a, const vboold2& b) { return _mm_andnot_pd(b, a); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Assignment Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vboold2& operator &=(vboold2& a, const vboold2& b) { return a = a & b; }
  __forceinline vboold2& operator |=(vboold2& a, const vboold2& b) { return a = a | b; }
  __forceinline vboold2& operator ^=(vboold2& a, const vboold2& b) { return a = a ^ b; }

  ////////////////////////////////////////////////////////////////////////////////
  /// Comparison Operators + Select
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vboold2 operator !=(const vboold2& a, const vboold2& b) { return _mm_xor_pd(a, b); }
  __forceinline vboold2 operator ==(const vboold2& a, const vboold2& b) { return _mm_castsi128_pd(_mm_cmpeq_epi32(a.m128i(), b.m128i())); }

  __forceinline vboold2 select(const vboold2& m, const vboold2& t, const vboold2& f) {
#if defined(ESIMD_ARM64) || defined(__SSE4_1__)
    return _mm_blendv_pd(f, t, m);
#else
    return _mm_or_pd(_mm_and_pd(m, t), _mm_andnot_pd(m, f));
#endif
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Movement/Shifting/Shuffling Functions
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vboold2 unpacklo(const vboold2& a, const vboold2& b) { return _mm_unpacklo_pd(a, b); }
  __forceinline vboold2 unpackhi(const vboold2& a, const vboold2& b) { return _mm_unpackhi_pd(a, b); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Reduction Operations
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline bool reduce_and(const vboold2& a) { return _mm_movemask_pd(a) == 0x3; }
  __forceinline bool reduce_or (const vboold2& a) { return _mm_movemask_pd(a) != 0x0; }

  __forceinline bool all (const vboold2& a) { return _mm_movemask_pd(a) == 0x3; }
  __forceinline bool any (const vboold2& a) { return _mm_movemask_pd(a) != 0x0; }
  __forceinline bool none(const vboold2& a) { return _mm_movemask_pd(a) == 0x0; }

  __forceinline bool all (const vboold2& valid, const vboold2& b) { return all((!valid) | b); }
  __forceinline bool any (const vboold2& valid, const vboold2& b) { return any(valid & b); }
  __forceinline bool none(const vboold2& valid, const vboold2& b) { return none(valid & b); }

  __forceinline size_t movemask(const vboold2& a) { return _mm_movemask_pd(a); }
  __forceinline size_t popcnt  (const vboold2& a) { return bool(a[0]) + bool(a[1]); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Get/Set Functions
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline bool get(const vboold2& a, size_t index) { return a[index]; }
  __forceinline void set  (vboold2& a, size_t index)     { a[index] = -1; }
  __forceinline void clear(vboold2& a, size_t index)     { a[index] =  0; }
}

#undef vboolf
#undef vboold
#undef vint
#undef vuint
#undef vllong
#undef vfloat
#undef vdouble
