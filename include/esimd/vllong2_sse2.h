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
  /* 2-wide SSE 64-bit long long type */
  template<>
  struct vllong<2>
  {
    ALIGNED_STRUCT_(16);

    typedef vboold2 Bool;

    enum  { size = 2 }; // number of SIMD elements
    union {             // data
      __m128i v;
      long long i[2];
    };

    ////////////////////////////////////////////////////////////////////////////////
    /// Constructors, Assignment & Cast Operators
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline vllong() {}
    __forceinline vllong(const vllong2& t) { v = t.v; }
    __forceinline vllong2& operator =(const vllong2& f) { v = f.v; return *this; }

    __forceinline vllong(const __m128i& t) { v = t; }
    __forceinline operator __m128i() const { return v; }
    __forceinline explicit operator __m128d() const { return _mm_castsi128_pd(v); }

    __forceinline vllong(long long i) {
      v = _mm_set1_epi64x(i);
    }

    __forceinline vllong(long long a, long long b) {
      v = _mm_set_epi64x(b,a);
    }

    ////////////////////////////////////////////////////////////////////////////////
    /// Constants
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline vllong(ZeroTy) : v(_mm_setzero_si128()) {}
    __forceinline vllong(OneTy)  : v(_mm_set1_epi64x(1)) {}
    __forceinline vllong(StepTy) : v(_mm_set_epi64x(1,0)) {}
    __forceinline vllong(ReverseStepTy) : v(_mm_set_epi64x(0,1)) {}

    ////////////////////////////////////////////////////////////////////////////////
    /// Loads and Stores
    ////////////////////////////////////////////////////////////////////////////////

    static __forceinline void store_nt(void* __restrict__ ptr, const vllong2& a) {
      _mm_stream_pd((double*)ptr,_mm_castsi128_pd(a));
    }

    static __forceinline vllong2 loadu(const void* addr)
    {
      return _mm_loadu_si128((__m128i*)addr);
    }

    static __forceinline vllong2 load(const vllong2* addr) {
      return _mm_load_si128((__m128i*)addr);
    }

    static __forceinline vllong2 load(const long long* addr) {
      return _mm_load_si128((__m128i*)addr);
    }

    static __forceinline void store(void* ptr, const vllong2& v) {
      _mm_store_si128((__m128i*)ptr,v);
    }

    static __forceinline void storeu(void* ptr, const vllong2& v) {
      _mm_storeu_si128((__m128i*)ptr,v);
    }

    static __forceinline void storeu(const vboold2& mask, long long* ptr, const vllong2& f) {
#if defined(__AVX512VL__)
      _mm_mask_storeu_epi64(ptr,mask,f);
#elif defined(__AVX__) && !defined(ESIMD_ARM64) && !ESIMD_MSVC_MASKSTORE_FALLBACK
      _mm_maskstore_pd((double*)ptr,mask.m128i(),_mm_castsi128_pd(f));
#else
      const size_t m = _mm_movemask_pd(mask);
      if (m & 1) ptr[0] = f[0];
      if (m & 2) ptr[1] = f[1];
#endif
    }

    static __forceinline void store(const vboold2& mask, void* ptr, const vllong2& f) {
#if defined(__AVX512VL__)
      _mm_mask_store_epi64(ptr,mask,f);
#else
      storeu(mask,(long long*)ptr,f);
#endif
    }

    ////////////////////////////////////////////////////////////////////////////////
    /// Array Access
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline       long long& operator [](size_t index)       { assert(index < 2); return i[index]; }
    __forceinline const long long& operator [](size_t index) const { assert(index < 2); return i[index]; }
  };

  ////////////////////////////////////////////////////////////////////////////////
  /// Select
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vllong2 select(const vboold2& m, const vllong2& t, const vllong2& f) {
#if defined(__AVX512VL__)
    return _mm_mask_blend_epi64(m, f, t);
#else
    return _mm_castpd_si128(_mm_blendv_pd(_mm_castsi128_pd(f), _mm_castsi128_pd(t), m));
#endif
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Unary Operators
  ////////////////////////////////////////////////////////////////////////////////

#if defined(__AVX512VL__)
  __forceinline vboold2 asBool(const vllong2& a) { return _mm_movepi64_mask(a); }
#else
  __forceinline vboold2 asBool(const vllong2& a) { return _mm_castsi128_pd(a); }
#endif

  __forceinline vllong2 operator +(const vllong2& a) { return a; }
  __forceinline vllong2 operator -(const vllong2& a) { return _mm_sub_epi64(_mm_setzero_si128(), a); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Binary Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vllong2 operator +(const vllong2& a, const vllong2& b) { return _mm_add_epi64(a, b); }
  __forceinline vllong2 operator +(const vllong2& a, long long      b) { return a + vllong2(b); }
  __forceinline vllong2 operator +(long long      a, const vllong2& b) { return vllong2(a) + b; }

  __forceinline vllong2 operator -(const vllong2& a, const vllong2& b) { return _mm_sub_epi64(a, b); }
  __forceinline vllong2 operator -(const vllong2& a, long long      b) { return a - vllong2(b); }
  __forceinline vllong2 operator -(long long      a, const vllong2& b) { return vllong2(a) - b; }

  /* only low 32bit part */
  __forceinline vllong2 operator *(const vllong2& a, const vllong2& b) { return _mm_mul_epi32(a, b); }
  __forceinline vllong2 operator *(const vllong2& a, long long      b) { return a * vllong2(b); }
  __forceinline vllong2 operator *(long long      a, const vllong2& b) { return vllong2(a) * b; }

  __forceinline vllong2 operator &(const vllong2& a, const vllong2& b) { return _mm_and_si128(a, b); }
  __forceinline vllong2 operator &(const vllong2& a, long long      b) { return a & vllong2(b); }
  __forceinline vllong2 operator &(long long      a, const vllong2& b) { return vllong2(a) & b; }

  __forceinline vllong2 operator |(const vllong2& a, const vllong2& b) { return _mm_or_si128(a, b); }
  __forceinline vllong2 operator |(const vllong2& a, long long      b) { return a | vllong2(b); }
  __forceinline vllong2 operator |(long long      a, const vllong2& b) { return vllong2(a) | b; }

  __forceinline vllong2 operator ^(const vllong2& a, const vllong2& b) { return _mm_xor_si128(a, b); }
  __forceinline vllong2 operator ^(const vllong2& a, long long      b) { return a ^ vllong2(b); }
  __forceinline vllong2 operator ^(long long      a, const vllong2& b) { return vllong2(a) ^ b; }

  __forceinline vllong2 operator <<(const vllong2& a, long long n) { return _mm_slli_epi64(a, (int)n); }

#if defined(ESIMD_ARM64)
  __forceinline vllong2 operator <<(const vllong2& a, const vllong2& n) { return vreinterpretq_m128i_s64(vshlq_s64(vreinterpretq_s64_m128i(a), vreinterpretq_s64_m128i(n))); }
#elif defined(__AVX2__)
  __forceinline vllong2 operator <<(const vllong2& a, const vllong2& n) { return _mm_sllv_epi64(a, n); }
#endif

  __forceinline vllong2 srl(const vllong2& a, long long b) { return _mm_srli_epi64(a, (int)b); }

#if defined(__AVX512VL__)
  __forceinline vllong2 mask_and(const vboold2& m, const vllong2& c, const vllong2& a, const vllong2& b) { return _mm_mask_and_epi64(c,m,a,b); }
  __forceinline vllong2 mask_or (const vboold2& m, const vllong2& c, const vllong2& a, const vllong2& b) { return _mm_mask_or_epi64(c,m,a,b); }
#else
  __forceinline vllong2 mask_and(const vboold2& m, const vllong2& c, const vllong2& a, const vllong2& b) { return select(m, a & b, c); }
  __forceinline vllong2 mask_or (const vboold2& m, const vllong2& c, const vllong2& a, const vllong2& b) { return select(m, a | b, c); }
#endif

  ////////////////////////////////////////////////////////////////////////////////
  /// Assignment Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vllong2& operator +=(vllong2& a, const vllong2& b) { return a = a + b; }
  __forceinline vllong2& operator +=(vllong2& a, long long      b) { return a = a + b; }

  __forceinline vllong2& operator -=(vllong2& a, const vllong2& b) { return a = a - b; }
  __forceinline vllong2& operator -=(vllong2& a, long long      b) { return a = a - b; }

  __forceinline vllong2& operator *=(vllong2& a, const vllong2& b) { return a = a * b; }
  __forceinline vllong2& operator *=(vllong2& a, long long      b) { return a = a * b; }

  __forceinline vllong2& operator &=(vllong2& a, const vllong2& b) { return a = a & b; }
  __forceinline vllong2& operator &=(vllong2& a, long long      b) { return a = a & b; }

  __forceinline vllong2& operator |=(vllong2& a, const vllong2& b) { return a = a | b; }
  __forceinline vllong2& operator |=(vllong2& a, long long      b) { return a = a | b; }

  __forceinline vllong2& operator <<=(vllong2& a, long long      b) { return a = a << b; }

  ////////////////////////////////////////////////////////////////////////////////
  /// Comparison Operators
  ////////////////////////////////////////////////////////////////////////////////

#if defined(__AVX512VL__)
  __forceinline vboold2 operator ==(const vllong2& a, const vllong2& b) { return _mm_cmp_epi64_mask(a,b,_MM_CMPINT_EQ); }
  __forceinline vboold2 operator !=(const vllong2& a, const vllong2& b) { return _mm_cmp_epi64_mask(a,b,_MM_CMPINT_NE); }
  __forceinline vboold2 operator < (const vllong2& a, const vllong2& b) { return _mm_cmp_epi64_mask(a,b,_MM_CMPINT_LT); }
  __forceinline vboold2 operator >=(const vllong2& a, const vllong2& b) { return _mm_cmp_epi64_mask(a,b,_MM_CMPINT_GE); }
  __forceinline vboold2 operator > (const vllong2& a, const vllong2& b) { return _mm_cmp_epi64_mask(a,b,_MM_CMPINT_GT); }
  __forceinline vboold2 operator <=(const vllong2& a, const vllong2& b) { return _mm_cmp_epi64_mask(a,b,_MM_CMPINT_LE); }
#else
  __forceinline vboold2 operator ==(const vllong2& a, const vllong2& b) { return _mm_cmpeq_epi64(a,b); }
  __forceinline vboold2 operator !=(const vllong2& a, const vllong2& b) { return !(a == b); }
  __forceinline vboold2 operator > (const vllong2& a, const vllong2& b) { return _mm_cmpgt_epi64(a,b); }
  __forceinline vboold2 operator < (const vllong2& a, const vllong2& b) { return _mm_cmpgt_epi64(b,a); }
  __forceinline vboold2 operator >=(const vllong2& a, const vllong2& b) { return !(a < b); }
  __forceinline vboold2 operator <=(const vllong2& a, const vllong2& b) { return !(a > b); }
#endif

  __forceinline vboold2 operator ==(const vllong2& a, long long      b) { return a == vllong2(b); }
  __forceinline vboold2 operator ==(long long      a, const vllong2& b) { return vllong2(a) == b; }

  __forceinline vboold2 operator !=(const vllong2& a, long long      b) { return a != vllong2(b); }
  __forceinline vboold2 operator !=(long long      a, const vllong2& b) { return vllong2(a) != b; }

  __forceinline vboold2 operator > (const vllong2& a, long long      b) { return a >  vllong2(b); }
  __forceinline vboold2 operator > (long long      a, const vllong2& b) { return vllong2(a) >  b; }

  __forceinline vboold2 operator < (const vllong2& a, long long      b) { return a <  vllong2(b); }
  __forceinline vboold2 operator < (long long      a, const vllong2& b) { return vllong2(a) <  b; }

  __forceinline vboold2 operator >=(const vllong2& a, long long      b) { return a >= vllong2(b); }
  __forceinline vboold2 operator >=(long long      a, const vllong2& b) { return vllong2(a) >= b; }

  __forceinline vboold2 operator <=(const vllong2& a, long long      b) { return a <= vllong2(b); }
  __forceinline vboold2 operator <=(long long      a, const vllong2& b) { return vllong2(a) <= b; }

  __forceinline vboold2 eq(const vllong2& a, const vllong2& b) { return a == b; }
  __forceinline vboold2 ne(const vllong2& a, const vllong2& b) { return a != b; }
  __forceinline vboold2 lt(const vllong2& a, const vllong2& b) { return a <  b; }
  __forceinline vboold2 ge(const vllong2& a, const vllong2& b) { return a >= b; }
  __forceinline vboold2 gt(const vllong2& a, const vllong2& b) { return a >  b; }
  __forceinline vboold2 le(const vllong2& a, const vllong2& b) { return a <= b; }

#if defined(__AVX512VL__)
  __forceinline vboold2 eq(const vboold2& mask, const vllong2& a, const vllong2& b) { return _mm_mask_cmp_epi64_mask(mask, a, b, _MM_CMPINT_EQ); }
  __forceinline vboold2 ne(const vboold2& mask, const vllong2& a, const vllong2& b) { return _mm_mask_cmp_epi64_mask(mask, a, b, _MM_CMPINT_NE); }
  __forceinline vboold2 lt(const vboold2& mask, const vllong2& a, const vllong2& b) { return _mm_mask_cmp_epi64_mask(mask, a, b, _MM_CMPINT_LT); }
  __forceinline vboold2 ge(const vboold2& mask, const vllong2& a, const vllong2& b) { return _mm_mask_cmp_epi64_mask(mask, a, b, _MM_CMPINT_GE); }
  __forceinline vboold2 gt(const vboold2& mask, const vllong2& a, const vllong2& b) { return _mm_mask_cmp_epi64_mask(mask, a, b, _MM_CMPINT_GT); }
  __forceinline vboold2 le(const vboold2& mask, const vllong2& a, const vllong2& b) { return _mm_mask_cmp_epi64_mask(mask, a, b, _MM_CMPINT_LE); }
#else
  __forceinline vboold2 eq(const vboold2& mask, const vllong2& a, const vllong2& b) { return mask & (a == b); }
  __forceinline vboold2 ne(const vboold2& mask, const vllong2& a, const vllong2& b) { return mask & (a != b); }
  __forceinline vboold2 lt(const vboold2& mask, const vllong2& a, const vllong2& b) { return mask & (a <  b); }
  __forceinline vboold2 ge(const vboold2& mask, const vllong2& a, const vllong2& b) { return mask & (a >= b); }
  __forceinline vboold2 gt(const vboold2& mask, const vllong2& a, const vllong2& b) { return mask & (a >  b); }
  __forceinline vboold2 le(const vboold2& mask, const vllong2& a, const vllong2& b) { return mask & (a <= b); }
#endif

  ////////////////////////////////////////////////////////////////////////////////
  // Movement/Shifting/Shuffling Functions
  ////////////////////////////////////////////////////////////////////////////////

  template<int i0, int i1>
  __forceinline vllong2 shuffle(const vllong2& v) {
    return _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(v), _mm_castsi128_pd(v), (i1 << 1) | i0));
  }

  template<int i>
  __forceinline vllong2 shuffle(const vllong2& v) {
    return shuffle<i, i>(v);
  }

  __forceinline long long toScalar(const vllong2& v) {
    return _mm_cvtsi128_si64(v);
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Reductions
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vllong2 vreduce_and(const vllong2& x) { return x & shuffle<1,0>(x); }
  __forceinline vllong2 vreduce_or (const vllong2& x) { return x | shuffle<1,0>(x); }
  __forceinline vllong2 vreduce_add(const vllong2& x) { return x + shuffle<1,0>(x); }

  __forceinline long long reduce_add(const vllong2& a) { return toScalar(vreduce_add(a)); }
  __forceinline long long reduce_or (const vllong2& a) { return toScalar(vreduce_or(a)); }
  __forceinline long long reduce_and(const vllong2& a) { return toScalar(vreduce_and(a)); }
}

#undef vboolf
#undef vboold
#undef vint
#undef vuint
#undef vllong
#undef vfloat
#undef vdouble
