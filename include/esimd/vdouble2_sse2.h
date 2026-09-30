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
  /* 2-wide SSE 64-bit double type */
  template<>
  struct vdouble<2>
  {
    ALIGNED_STRUCT_(16);

    typedef vboold2 Bool;

    enum  { size = 2 }; // number of SIMD elements
    union {             // data
      __m128d v;
      double i[2];
    };

    ////////////////////////////////////////////////////////////////////////////////
    /// Constructors, Assignment & Cast Operators
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline vdouble() {}
    __forceinline vdouble(const vdouble2& t) { v = t.v; }
    __forceinline vdouble2& operator =(const vdouble2& f) { v = f.v; return *this; }

    __forceinline vdouble(const __m128d& t) { v = t; }
    __forceinline operator __m128d() const { return v; }

    __forceinline vdouble(double i) {
      v = _mm_set1_pd(i);
    }

    __forceinline vdouble(double a, double b) {
      v = _mm_set_pd(b,a);
    }

    ////////////////////////////////////////////////////////////////////////////////
    /// Constants
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline vdouble(ZeroTy) : v(_mm_setzero_pd()) {}
    __forceinline vdouble(OneTy)  : v(_mm_set1_pd(1)) {}
    __forceinline vdouble(StepTy) : v(_mm_set_pd(1.0,0.0)) {}
    __forceinline vdouble(ReverseStepTy) : v(_mm_setr_pd(1.0,0.0)) {}

    ////////////////////////////////////////////////////////////////////////////////
    /// Loads and Stores
    ////////////////////////////////////////////////////////////////////////////////

    static __forceinline void store_nt(double *__restrict__ ptr, const vdouble2& a) {
      _mm_stream_pd(ptr, a);
    }

    static __forceinline vdouble2 load (const vdouble2* addr) { return _mm_load_pd((double*)addr); }
    static __forceinline vdouble2 load (const double* addr)   { return _mm_load_pd(addr); }
    static __forceinline vdouble2 loadu(const double* addr)   { return _mm_loadu_pd(addr); }

    static __forceinline void store (double* ptr, const vdouble2& v) { _mm_store_pd(ptr,v); }
    static __forceinline void storeu(double* ptr, const vdouble2& v) { _mm_storeu_pd(ptr,v); }

#if defined(__AVX512VL__)
    static __forceinline vdouble2 load (const vboold2& mask, const double* ptr) { return _mm_mask_load_pd (_mm_setzero_pd(),mask,ptr); }
    static __forceinline vdouble2 loadu(const vboold2& mask, const double* ptr) { return _mm_mask_loadu_pd(_mm_setzero_pd(),mask,ptr); }

    static __forceinline void store (const vboold2& mask, double* ptr, const vdouble2& v) { _mm_mask_store_pd (ptr,mask,v); }
    static __forceinline void storeu(const vboold2& mask, double* ptr, const vdouble2& v) { _mm_mask_storeu_pd(ptr,mask,v); }
#elif defined(__AVX__) && !defined(ESIMD_ARM64)
    static __forceinline vdouble2 load (const vboold2& mask, const double* ptr) { return _mm_maskload_pd(ptr,mask.m128i()); }
    static __forceinline vdouble2 loadu(const vboold2& mask, const double* ptr) { return _mm_maskload_pd(ptr,mask.m128i()); }

    static __forceinline void store (const vboold2& mask, double* ptr, const vdouble2& v) { _mm_maskstore_pd(ptr,mask.m128i(),v); }
    static __forceinline void storeu(const vboold2& mask, double* ptr, const vdouble2& v) { _mm_maskstore_pd(ptr,mask.m128i(),v); }
#else
    // Per lane, so masked-off lanes are never touched.
    static __forceinline vdouble2 loadu(const vboold2& mask, const double* ptr) {
      const size_t m = _mm_movemask_pd(mask);
      return vdouble2((m & 1) ? ptr[0] : 0.0, (m & 2) ? ptr[1] : 0.0);
    }
    static __forceinline vdouble2 load(const vboold2& mask, const double* ptr) { return loadu(mask,ptr); }

    static __forceinline void storeu(const vboold2& mask, double* ptr, const vdouble2& v) {
      const size_t m = _mm_movemask_pd(mask);
      if (m & 1) ptr[0] = v[0];
      if (m & 2) ptr[1] = v[1];
    }
    static __forceinline void store(const vboold2& mask, double* ptr, const vdouble2& v) { storeu(mask,ptr,v); }
#endif

    static __forceinline vdouble2 broadcast(const void* a) { return _mm_set1_pd(*(double*)a); }

    ////////////////////////////////////////////////////////////////////////////////
    /// Array Access
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline       double& operator [](size_t index)       { assert(index < 2); return i[index]; }
    __forceinline const double& operator [](size_t index) const { assert(index < 2); return i[index]; }
  };

  ////////////////////////////////////////////////////////////////////////////////
  /// Unary Operators
  ////////////////////////////////////////////////////////////////////////////////

#if ESIMD_HAS_VLLONG2
  __forceinline vdouble2 asDouble(const vllong2&  a) { return _mm_castsi128_pd(a); }
  __forceinline vllong2  asLLong (const vdouble2& a) { return _mm_castpd_si128(a); }
#endif

  __forceinline vdouble2 operator +(const vdouble2& a) { return a; }
  __forceinline vdouble2 operator -(const vdouble2& a) { return _mm_sub_pd(_mm_setzero_pd(), a); }
  __forceinline vdouble2 sqrt (const vdouble2& a) { return _mm_sqrt_pd(a); }
  __forceinline vdouble2 rsqrt(const vdouble2& a)
  {
#if defined(__AVX512VL__)
    const vdouble2 r = _mm_rsqrt14_pd(a);
    return _mm_fmadd_pd(_mm_set1_pd(1.5), r,
                        _mm_mul_pd(_mm_mul_pd(_mm_mul_pd(a, _mm_set1_pd(-0.5)), r), _mm_mul_pd(r, r)));
#else
    return _mm_div_pd(_mm_set1_pd(1.0), _mm_sqrt_pd(a));
#endif
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Binary Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vdouble2 operator +(const vdouble2& a, const vdouble2& b) { return _mm_add_pd(a, b); }
  __forceinline vdouble2 operator +(const vdouble2& a, double          b) { return a + vdouble2(b); }
  __forceinline vdouble2 operator +(double          a, const vdouble2& b) { return vdouble2(a) + b; }

  __forceinline vdouble2 operator -(const vdouble2& a, const vdouble2& b) { return _mm_sub_pd(a, b); }
  __forceinline vdouble2 operator -(const vdouble2& a, double          b) { return a - vdouble2(b); }
  __forceinline vdouble2 operator -(double          a, const vdouble2& b) { return vdouble2(a) - b; }

  __forceinline vdouble2 operator *(const vdouble2& a, const vdouble2& b) { return _mm_mul_pd(a, b); }
  __forceinline vdouble2 operator *(const vdouble2& a, double          b) { return a * vdouble2(b); }
  __forceinline vdouble2 operator *(double          a, const vdouble2& b) { return vdouble2(a) * b; }

  __forceinline vdouble2 operator /(const vdouble2& a, const vdouble2& b) { return _mm_div_pd(a, b); }
  __forceinline vdouble2 operator /(const vdouble2& a, double          b) { return a / vdouble2(b); }
  __forceinline vdouble2 operator /(double          a, const vdouble2& b) { return vdouble2(a) / b; }

  __forceinline vdouble2 operator &(const vdouble2& a, const vdouble2& b) { return _mm_and_pd(a, b); }
  __forceinline vdouble2 operator &(const vdouble2& a, double          b) { return a & vdouble2(b); }
  __forceinline vdouble2 operator &(double          a, const vdouble2& b) { return vdouble2(a) & b; }

  __forceinline vdouble2 operator |(const vdouble2& a, const vdouble2& b) { return _mm_or_pd(a, b); }
  __forceinline vdouble2 operator |(const vdouble2& a, double          b) { return a | vdouble2(b); }
  __forceinline vdouble2 operator |(double          a, const vdouble2& b) { return vdouble2(a) | b; }

  __forceinline vdouble2 operator ^(const vdouble2& a, const vdouble2& b) { return _mm_xor_pd(a, b); }
  __forceinline vdouble2 operator ^(const vdouble2& a, double          b) { return a ^ vdouble2(b); }
  __forceinline vdouble2 operator ^(double          a, const vdouble2& b) { return vdouble2(a) ^ b; }

  __forceinline vdouble2 min(const vdouble2& a, const vdouble2& b) { return _mm_min_pd(a, b); }
  __forceinline vdouble2 min(const vdouble2& a, double          b) { return min(a,vdouble2(b)); }
  __forceinline vdouble2 min(double          a, const vdouble2& b) { return min(vdouble2(a),b); }

  __forceinline vdouble2 max(const vdouble2& a, const vdouble2& b) { return _mm_max_pd(a, b); }
  __forceinline vdouble2 max(const vdouble2& a, double          b) { return max(a,vdouble2(b)); }
  __forceinline vdouble2 max(double          a, const vdouble2& b) { return max(vdouble2(a),b); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Ternary Operators
  ////////////////////////////////////////////////////////////////////////////////

#if defined(__FMA__) || defined(ESIMD_ARM64)
  __forceinline vdouble2 madd (const vdouble2& a, const vdouble2& b, const vdouble2& c) { return _mm_fmadd_pd(a,b,c); }
  __forceinline vdouble2 msub (const vdouble2& a, const vdouble2& b, const vdouble2& c) { return _mm_fmsub_pd(a,b,c); }
  __forceinline vdouble2 nmadd(const vdouble2& a, const vdouble2& b, const vdouble2& c) { return _mm_fnmadd_pd(a,b,c); }
  __forceinline vdouble2 nmsub(const vdouble2& a, const vdouble2& b, const vdouble2& c) { return _mm_fnmsub_pd(a,b,c); }
#else
  __forceinline vdouble2 madd (const vdouble2& a, const vdouble2& b, const vdouble2& c) { return a*b+c; }
  __forceinline vdouble2 msub (const vdouble2& a, const vdouble2& b, const vdouble2& c) { return a*b-c; }
  __forceinline vdouble2 nmadd(const vdouble2& a, const vdouble2& b, const vdouble2& c) { return -a*b+c;}
  __forceinline vdouble2 nmsub(const vdouble2& a, const vdouble2& b, const vdouble2& c) { return -a*b-c; }
#endif

  ////////////////////////////////////////////////////////////////////////////////
  /// Assignment Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vdouble2& operator +=(vdouble2& a, const vdouble2& b) { return a = a + b; }
  __forceinline vdouble2& operator +=(vdouble2& a, double          b) { return a = a + b; }

  __forceinline vdouble2& operator -=(vdouble2& a, const vdouble2& b) { return a = a - b; }
  __forceinline vdouble2& operator -=(vdouble2& a, double          b) { return a = a - b; }

  __forceinline vdouble2& operator *=(vdouble2& a, const vdouble2& b) { return a = a * b; }
  __forceinline vdouble2& operator *=(vdouble2& a, double          b) { return a = a * b; }

  __forceinline vdouble2& operator /=(vdouble2& a, const vdouble2& b) { return a = a / b; }
  __forceinline vdouble2& operator /=(vdouble2& a, double          b) { return a = a / b; }

  __forceinline vdouble2& operator &=(vdouble2& a, const vdouble2& b) { return a = a & b; }
  __forceinline vdouble2& operator &=(vdouble2& a, double          b) { return a = a & b; }

  __forceinline vdouble2& operator |=(vdouble2& a, const vdouble2& b) { return a = a | b; }
  __forceinline vdouble2& operator |=(vdouble2& a, double          b) { return a = a | b; }

  ////////////////////////////////////////////////////////////////////////////////
  /// Comparison Operators + Select
  ////////////////////////////////////////////////////////////////////////////////

#if defined(__AVX512VL__)
  __forceinline vboold2 operator ==(const vdouble2& a, const vdouble2& b) { return _mm_cmp_pd_mask(a, b, _MM_CMPINT_EQ); }
  __forceinline vboold2 operator !=(const vdouble2& a, const vdouble2& b) { return _mm_cmp_pd_mask(a, b, _MM_CMPINT_NE); }
  __forceinline vboold2 operator < (const vdouble2& a, const vdouble2& b) { return _mm_cmp_pd_mask(a, b, _MM_CMPINT_LT); }
  __forceinline vboold2 operator >=(const vdouble2& a, const vdouble2& b) { return _mm_cmp_pd_mask(a, b, _MM_CMPINT_GE); }
  __forceinline vboold2 operator > (const vdouble2& a, const vdouble2& b) { return _mm_cmp_pd_mask(a, b, _MM_CMPINT_GT); }
  __forceinline vboold2 operator <=(const vdouble2& a, const vdouble2& b) { return _mm_cmp_pd_mask(a, b, _MM_CMPINT_LE); }
#elif defined(__AVX__) && !defined(ESIMD_ARM64)
  __forceinline vboold2 operator ==(const vdouble2& a, const vdouble2& b) { return _mm_cmp_pd(a, b, _CMP_EQ_OQ);  }
  __forceinline vboold2 operator !=(const vdouble2& a, const vdouble2& b) { return _mm_cmp_pd(a, b, _CMP_NEQ_UQ); }
  __forceinline vboold2 operator < (const vdouble2& a, const vdouble2& b) { return _mm_cmp_pd(a, b, _CMP_LT_OS);  }
  __forceinline vboold2 operator >=(const vdouble2& a, const vdouble2& b) { return _mm_cmp_pd(a, b, _CMP_NLT_US); }
  __forceinline vboold2 operator > (const vdouble2& a, const vdouble2& b) { return _mm_cmp_pd(a, b, _CMP_NLE_US); }
  __forceinline vboold2 operator <=(const vdouble2& a, const vdouble2& b) { return _mm_cmp_pd(a, b, _CMP_LE_OS);  }
#else
  __forceinline vboold2 operator ==(const vdouble2& a, const vdouble2& b) { return _mm_cmpeq_pd(a, b);  }
  __forceinline vboold2 operator !=(const vdouble2& a, const vdouble2& b) { return _mm_cmpneq_pd(a, b); }
  __forceinline vboold2 operator < (const vdouble2& a, const vdouble2& b) { return _mm_cmplt_pd(a, b);  }
  __forceinline vboold2 operator >=(const vdouble2& a, const vdouble2& b) { return _mm_cmpnlt_pd(a, b); }
  __forceinline vboold2 operator > (const vdouble2& a, const vdouble2& b) { return _mm_cmpnle_pd(a, b); }
  __forceinline vboold2 operator <=(const vdouble2& a, const vdouble2& b) { return _mm_cmple_pd(a, b);  }
#endif

  __forceinline vboold2 operator ==(const vdouble2& a, double          b) { return a == vdouble2(b); }
  __forceinline vboold2 operator ==(double          a, const vdouble2& b) { return vdouble2(a) == b; }

  __forceinline vboold2 operator !=(const vdouble2& a, double          b) { return a != vdouble2(b); }
  __forceinline vboold2 operator !=(double          a, const vdouble2& b) { return vdouble2(a) != b; }

  __forceinline vboold2 operator < (const vdouble2& a, double          b) { return a <  vdouble2(b); }
  __forceinline vboold2 operator < (double          a, const vdouble2& b) { return vdouble2(a) <  b; }

  __forceinline vboold2 operator >=(const vdouble2& a, double          b) { return a >= vdouble2(b); }
  __forceinline vboold2 operator >=(double          a, const vdouble2& b) { return vdouble2(a) >= b; }

  __forceinline vboold2 operator > (const vdouble2& a, double          b) { return a >  vdouble2(b); }
  __forceinline vboold2 operator > (double          a, const vdouble2& b) { return vdouble2(a) >  b; }

  __forceinline vboold2 operator <=(const vdouble2& a, double          b) { return a <= vdouble2(b); }
  __forceinline vboold2 operator <=(double          a, const vdouble2& b) { return vdouble2(a) <= b; }

  __forceinline vboold2 eq(const vdouble2& a, const vdouble2& b) { return a == b; }
  __forceinline vboold2 ne(const vdouble2& a, const vdouble2& b) { return a != b; }
  __forceinline vboold2 lt(const vdouble2& a, const vdouble2& b) { return a <  b; }
  __forceinline vboold2 ge(const vdouble2& a, const vdouble2& b) { return a >= b; }
  __forceinline vboold2 gt(const vdouble2& a, const vdouble2& b) { return a >  b; }
  __forceinline vboold2 le(const vdouble2& a, const vdouble2& b) { return a <= b; }

#if defined(__AVX512VL__)
  __forceinline vboold2 eq(const vboold2& mask, const vdouble2& a, const vdouble2& b) { return _mm_mask_cmp_pd_mask(mask, a, b, _MM_CMPINT_EQ); }
  __forceinline vboold2 ne(const vboold2& mask, const vdouble2& a, const vdouble2& b) { return _mm_mask_cmp_pd_mask(mask, a, b, _MM_CMPINT_NE); }
  __forceinline vboold2 lt(const vboold2& mask, const vdouble2& a, const vdouble2& b) { return _mm_mask_cmp_pd_mask(mask, a, b, _MM_CMPINT_LT); }
  __forceinline vboold2 ge(const vboold2& mask, const vdouble2& a, const vdouble2& b) { return _mm_mask_cmp_pd_mask(mask, a, b, _MM_CMPINT_GE); }
  __forceinline vboold2 gt(const vboold2& mask, const vdouble2& a, const vdouble2& b) { return _mm_mask_cmp_pd_mask(mask, a, b, _MM_CMPINT_GT); }
  __forceinline vboold2 le(const vboold2& mask, const vdouble2& a, const vdouble2& b) { return _mm_mask_cmp_pd_mask(mask, a, b, _MM_CMPINT_LE); }
#else
  __forceinline vboold2 eq(const vboold2& mask, const vdouble2& a, const vdouble2& b) { return mask & (a == b); }
  __forceinline vboold2 ne(const vboold2& mask, const vdouble2& a, const vdouble2& b) { return mask & (a != b); }
  __forceinline vboold2 lt(const vboold2& mask, const vdouble2& a, const vdouble2& b) { return mask & (a <  b); }
  __forceinline vboold2 ge(const vboold2& mask, const vdouble2& a, const vdouble2& b) { return mask & (a >= b); }
  __forceinline vboold2 gt(const vboold2& mask, const vdouble2& a, const vdouble2& b) { return mask & (a >  b); }
  __forceinline vboold2 le(const vboold2& mask, const vdouble2& a, const vdouble2& b) { return mask & (a <= b); }
#endif

  __forceinline vdouble2 select(const vboold2& m, const vdouble2& t, const vdouble2& f) {
#if defined(__AVX512VL__)
    return _mm_mask_blend_pd(m, f, t);
#elif defined(__SSE4_1__) || defined(ESIMD_ARM64)
    return _mm_blendv_pd(f, t, m);
#else
    return _mm_or_pd(_mm_and_pd(m, t), _mm_andnot_pd(m, f));
#endif
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Movement/Shifting/Shuffling Functions
  ////////////////////////////////////////////////////////////////////////////////

  template<int i0, int i1>
  __forceinline vdouble2 shuffle(const vdouble2& v) {
    return _mm_shuffle_pd(v, v, (i1 << 1) | i0);
  }

  template<int i>
  __forceinline vdouble2 shuffle(const vdouble2& v) {
    return shuffle<i, i>(v);
  }

  __forceinline double toScalar(const vdouble2& v) {
    return _mm_cvtsd_f64(v);
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Reductions
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vdouble2 vreduce_min(const vdouble2& x) { return min(x, shuffle<1,0>(x)); }
  __forceinline vdouble2 vreduce_max(const vdouble2& x) { return max(x, shuffle<1,0>(x)); }
  __forceinline vdouble2 vreduce_and(const vdouble2& x) { return x & shuffle<1,0>(x); }
  __forceinline vdouble2 vreduce_or (const vdouble2& x) { return x | shuffle<1,0>(x); }
  __forceinline vdouble2 vreduce_add(const vdouble2& x) { return x + shuffle<1,0>(x); }

  __forceinline double reduce_add(const vdouble2& a) { return toScalar(vreduce_add(a)); }
  __forceinline double reduce_min(const vdouble2& a) { return toScalar(vreduce_min(a)); }
  __forceinline double reduce_max(const vdouble2& a) { return toScalar(vreduce_max(a)); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Memory load and store operations
  ////////////////////////////////////////////////////////////////////////////////

  // Reads 4 doubles, no alignment required.
  __forceinline void deinterleave(const double* xy, vdouble2& x, vdouble2& y) {
    const __m128d a = _mm_loadu_pd(xy), b = _mm_loadu_pd(xy + 2);
    x = _mm_unpacklo_pd(a, b);
    y = _mm_unpackhi_pd(a, b);
  }

  // Two lanes have no cross-lane order to give up, so this is deinterleave.
  __forceinline void deinterleave_unordered(const double* xy, vdouble2& x, vdouble2& y) {
    deinterleave(xy, x, y);
  }
}

#undef vboolf
#undef vboold
#undef vint
#undef vuint
#undef vllong
#undef vfloat
#undef vdouble
