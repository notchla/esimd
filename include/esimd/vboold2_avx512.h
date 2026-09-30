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
  /* 2-wide AVX-512 bool type for 64bit data types */
  template<>
  struct vboold<2>
  {
    typedef vboold2 Bool;

    enum { size = 2 }; // number of SIMD elements
    __mmask8 v;        // data

    ////////////////////////////////////////////////////////////////////////////////
    /// Constructors, Assignment & Cast Operators
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline vboold() {}
    __forceinline vboold(const vboold2& t) { v = t.v; }
    __forceinline vboold2& operator =(const vboold2& f) { v = f.v; return *this; }

    __forceinline vboold(const __mmask8 &t) { v = t; }
    __forceinline operator __mmask8() const { return v; }

    __forceinline vboold(bool b) { v = b ? 0x3 : 0x0; }
    __forceinline vboold(int t)  { v = (__mmask8)t; }
    __forceinline vboold(unsigned int t) { v = (__mmask8)t; }

    /* return int64 mask */
    __forceinline __m128i mask64() const {
      return _mm_movm_epi64(v);
    }

    ////////////////////////////////////////////////////////////////////////////////
    /// Constants
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline vboold(FalseTy) : v(0x0) {}
    __forceinline vboold(TrueTy)  : v(0x3) {}

    ////////////////////////////////////////////////////////////////////////////////
    /// Array Access
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline bool operator [](size_t index) const {
      assert(index < 2); return (mm512_mask2int(v) >> index) & 1;
    }
  };

  ////////////////////////////////////////////////////////////////////////////////
  /// Unary Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vboold2 operator !(const vboold2& a) { return mm512_kandn(a, 0x3); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Binary Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vboold2 operator &(const vboold2& a, const vboold2& b) { return mm512_kand(a, b); }
  __forceinline vboold2 operator |(const vboold2& a, const vboold2& b) { return mm512_kor(a, b); }
  __forceinline vboold2 operator ^(const vboold2& a, const vboold2& b) { return mm512_kxor(a, b); }

  __forceinline vboold2 andn(const vboold2& a, const vboold2& b) { return mm512_kandn(b, a); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Assignment Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vboold2& operator &=(vboold2& a, const vboold2& b) { return a = a & b; }
  __forceinline vboold2& operator |=(vboold2& a, const vboold2& b) { return a = a | b; }
  __forceinline vboold2& operator ^=(vboold2& a, const vboold2& b) { return a = a ^ b; }

  ////////////////////////////////////////////////////////////////////////////////
  /// Comparison Operators + Select
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vboold2 operator !=(const vboold2& a, const vboold2& b) { return mm512_kxor(a, b); }
  __forceinline vboold2 operator ==(const vboold2& a, const vboold2& b) { return mm512_kand(mm512_kxnor(a, b), 0x3); }

  __forceinline vboold2 select(const vboold2& s, const vboold2& a, const vboold2& b) {
    return mm512_kor(mm512_kand(s, a), mm512_kandn(s, b));
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Reduction Operations
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline bool reduce_and(const vboold2& a) { return a.v == 0x3; }
  __forceinline bool reduce_or (const vboold2& a) { return mm512_kortestz(a, a) == 0; }

  __forceinline int all (const vboold2& a) { return a.v == 0x3; }
  __forceinline int any (const vboold2& a) { return mm512_kortestz(a, a) == 0; }
  __forceinline int none(const vboold2& a) { return mm512_kortestz(a, a) != 0; }

  __forceinline int all (const vboold2& valid, const vboold2& b) { return all((!valid) | b); }
  __forceinline int any (const vboold2& valid, const vboold2& b) { return any(valid & b); }
  __forceinline int none(const vboold2& valid, const vboold2& b) { return none(valid & b); }

  __forceinline size_t movemask(const vboold2& a) { return _mm512_kmov(a); }
  __forceinline size_t popcnt  (const vboold2& a) { return popcnt(a.v); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Conversion Operations
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline unsigned int toInt(const vboold2& a) { return mm512_mask2int(a); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Get/Set Functions
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline bool get(const vboold2& a, size_t index) { assert(index < 2); return (toInt(a) >> index) & 1; }
  __forceinline void set(vboold2& a, size_t index)       { assert(index < 2); a |= 1 << index; }
  __forceinline void clear(vboold2& a, size_t index)     { assert(index < 2); a = andn(a, 1 << index); }
}

#undef vboolf
#undef vboold
#undef vint
#undef vuint
#undef vllong
#undef vfloat
#undef vdouble
