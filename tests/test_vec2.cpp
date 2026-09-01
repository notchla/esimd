// Copyright 2026 notchla liso.lorenzo@gmail.com
// SPDX-License-Identifier: Apache-2.0
//
// Correctness tests for the optional data-types layer (<esimd/types.h>): the
// generic Vec2<T> and the SSE-backed Vec2fa. Vec2<T> is component-wise plumbing
// over T, so it is checked both at T = scalar (float/int/bool) and at every
// vector width the active ISA defines, lane-by-lane against a scalar reference.
// Compiled once per ISA, like the other suites.

#include "test_helpers.h"
#include <esimd/types.h>

using namespace esimd;
using namespace esimd_test;

namespace {

////////////////////////////////////////////////////////////////////////////////
// Helpers for the vector instantiations
////////////////////////////////////////////////////////////////////////////////

// Lane i = base + i, so a mixed-up lane cannot pass unnoticed.
template <typename V> V ramp(float base) {
  float tmp[V::size];
  for (int i = 0; i < V::size; ++i)
    tmp[i] = base + float(i);
  return V::loadu(tmp);
}

// Element-wise check of a Vec2<V> against a per-lane scalar reference, given as
// two functions of the lane index.
template <typename V, typename FX, typename FY>
void expect_vec2(const Vec2<V> &v, FX rx, FY ry, float tol) {
  for (int i = 0; i < V::size; ++i) {
    EXPECT_NEAR(v.x[i], rx(i), tol) << "at x lane " << i;
    EXPECT_NEAR(v.y[i], ry(i), tol) << "at y lane " << i;
  }
}

// Same, for an operation reducing a Vec2<V> back to a V.
template <typename V, typename F>
void expect_lanes(const V &v, F ref, float tol) {
  for (int i = 0; i < V::size; ++i)
    EXPECT_NEAR(v[i], ref(i), tol) << "at lane " << i;
}

} // namespace

////////////////////////////////////////////////////////////////////////////////
// Vec2f -- the scalar float instantiation
////////////////////////////////////////////////////////////////////////////////

TEST(Vec2f, constructors_and_constants) {
  const Vec2f a(1.f, 2.f);
  EXPECT_EQ(a.x, 1.f);
  EXPECT_EQ(a.y, 2.f);
  EXPECT_EQ(Vec2f(3.f).x, 3.f); // explicit broadcast
  EXPECT_EQ(Vec2f(3.f).y, 3.f);
  EXPECT_EQ(Vec2f(a).y, 2.f);           // copy
  EXPECT_EQ(Vec2f(Vec2i(4, 5)).x, 4.f); // converting

  EXPECT_EQ(Vec2f(zero).x, 0.f);
  EXPECT_EQ(Vec2f(one).y, 1.f);
  EXPECT_EQ(Vec2f(pos_inf).x, std::numeric_limits<float>::infinity());
  EXPECT_EQ(Vec2f(neg_inf).y, -std::numeric_limits<float>::infinity());

  EXPECT_EQ(Vec2f::N, 2);
  EXPECT_EQ(a[0], 1.f);
  EXPECT_EQ(a[1], 2.f);
  Vec2f m(0.f, 0.f);
  m[1] = 7.f;
  EXPECT_EQ(m.y, 7.f);
}

TEST(Vec2f, unary) {
  const Vec2f a(-1.5f, 2.5f);
  EXPECT_EQ((+a).x, -1.5f);
  EXPECT_EQ((-a).x, 1.5f);
  EXPECT_EQ((-a).y, -2.5f);
  EXPECT_EQ(abs(a).x, 1.5f);
  EXPECT_NEAR(rcp(Vec2f(2.f, 4.f)).y, 0.25f, 1e-6f);
  EXPECT_NEAR(rsqrt(Vec2f(4.f, 16.f)).y, 0.25f, 1e-5f);
  EXPECT_NEAR(sqrt(Vec2f(4.f, 9.f)).y, 3.f, 1e-6f);
  EXPECT_NEAR(frac(Vec2f(1.25f, -1.25f)).x, 0.25f, 1e-6f);
  EXPECT_NEAR(frac(Vec2f(1.25f, -1.25f)).y, 0.75f, 1e-6f);
}

TEST(Vec2f, binary) {
  const Vec2f a(1.f, 2.f), b(4.f, 8.f);
  EXPECT_EQ((a + b).x, 5.f);
  EXPECT_EQ((a + 1.f).y, 3.f);
  EXPECT_EQ((1.f + a).x, 2.f);
  EXPECT_EQ((a - b).y, -6.f);
  EXPECT_EQ((a - 1.f).x, 0.f);
  EXPECT_EQ((1.f - a).y, -1.f);
  EXPECT_EQ((a * b).y, 16.f);
  EXPECT_EQ((a * 2.f).x, 2.f);
  EXPECT_EQ((2.f * a).y, 4.f);
  EXPECT_EQ((b / a).y, 4.f);
  EXPECT_EQ((b / 2.f).x, 2.f);
  EXPECT_EQ((8.f / b).y, 1.f);
  EXPECT_EQ(min(a, b).x, 1.f);
  EXPECT_EQ(max(a, b).x, 4.f);
}

TEST(Vec2f, ternary_fma) {
  const Vec2f a(1.f, 2.f), b(3.f, 4.f), c(5.f, 6.f);
  EXPECT_EQ(madd(a, b, c).x, 1.f * 3.f + 5.f);
  EXPECT_EQ(msub(a, b, c).y, 2.f * 4.f - 6.f);
  EXPECT_EQ(nmadd(a, b, c).x, -1.f * 3.f + 5.f);
  EXPECT_EQ(nmsub(a, b, c).y, -2.f * 4.f - 6.f);
  // scalar-first overloads
  EXPECT_EQ(madd(2.f, b, c).x, 2.f * 3.f + 5.f);
  EXPECT_EQ(msub(2.f, b, c).y, 2.f * 4.f - 6.f);
  EXPECT_EQ(nmadd(2.f, b, c).x, -2.f * 3.f + 5.f);
  EXPECT_EQ(nmsub(2.f, b, c).y, -2.f * 4.f - 6.f);
}

TEST(Vec2f, assignment) {
  Vec2f a(1.f, 2.f);
  a += Vec2f(1.f, 1.f);
  EXPECT_EQ(a.y, 3.f);
  a -= Vec2f(1.f, 1.f);
  EXPECT_EQ(a.x, 1.f);
  a *= 3.f;
  EXPECT_EQ(a.y, 6.f);
  a /= 3.f;
  EXPECT_EQ(a.y, 2.f);
  a = Vec2i(7, 8);
  EXPECT_EQ(a.x, 7.f); // templated assignment
}

TEST(Vec2f, reductions) {
  const Vec2f a(3.f, 5.f);
  EXPECT_EQ(reduce_add(a), 8.f);
  EXPECT_EQ(reduce_mul(a), 15.f);
  EXPECT_EQ(reduce_min(a), 3.f);
  EXPECT_EQ(reduce_max(a), 5.f);
}

TEST(Vec2f, comparison) {
  EXPECT_TRUE(Vec2f(1.f, 2.f) == Vec2f(1.f, 2.f));
  EXPECT_FALSE(Vec2f(1.f, 2.f) == Vec2f(1.f, 3.f));
  EXPECT_TRUE(Vec2f(1.f, 2.f) != Vec2f(1.f, 3.f));
  // lexicographic: x first, then y, and never strictly less than itself
  EXPECT_TRUE(Vec2f(1.f, 9.f) < Vec2f(2.f, 0.f));
  EXPECT_TRUE(Vec2f(1.f, 2.f) < Vec2f(1.f, 3.f));
  EXPECT_FALSE(Vec2f(1.f, 2.f) < Vec2f(1.f, 2.f));
}

TEST(Vec2f, euclidean) {
  const Vec2f a(3.f, 4.f), b(1.f, 2.f);
  EXPECT_NEAR(dot(a, b), 11.f, 1e-5f);
  EXPECT_EQ(cross(a).x, -4.f); // (-y, x)
  EXPECT_EQ(cross(a).y, 3.f);
  EXPECT_NEAR(length(a), 5.f, 1e-5f);
  EXPECT_NEAR(normalize(a).x, 0.6f, 1e-3f);
  EXPECT_NEAR(normalize(a).y, 0.8f, 1e-3f);
  EXPECT_NEAR(distance(a, b), std::sqrt(2.f * 2.f + 2.f * 2.f), 1e-5f);
  EXPECT_NEAR(det(a, b), 3.f * 2.f - 4.f * 1.f, 1e-5f);
}

TEST(Vec2f, normalize_safe) {
  const Vec2f a(3.f, 4.f);
  EXPECT_NEAR(normalize_safe(a).x, 0.6f, 1e-3f);
  // a zero vector must come back unchanged rather than as NaN
  const Vec2f z(0.f, 0.f);
  EXPECT_EQ(normalize_safe(z).x, 0.f);
  EXPECT_EQ(normalize_safe(z).y, 0.f);
}

TEST(Vec2f, select_lerp_maxDim) {
  const Vec2f t(1.f, 2.f), f(3.f, 4.f);
  EXPECT_EQ(select(true, t, f).x, 1.f);
  EXPECT_EQ(select(false, t, f).y, 4.f);
  EXPECT_EQ(select(Vec2b(true, false), t, f).x, 1.f);
  EXPECT_EQ(select(Vec2b(true, false), t, f).y, 4.f);

  EXPECT_NEAR(lerp(t, f, 0.5f).x, 2.f, 1e-5f);
  EXPECT_NEAR(lerp(t, f, 0.0f).y, 2.f, 1e-5f);
  EXPECT_NEAR(lerp(t, f, 1.0f).y, 4.f, 1e-5f);

  EXPECT_EQ(maxDim(Vec2f(-3.f, 2.f)), 0);
  EXPECT_EQ(maxDim(Vec2f(1.f, -2.f)), 1);
}

////////////////////////////////////////////////////////////////////////////////
// Vec2i / Vec2b -- the other scalar instantiations
////////////////////////////////////////////////////////////////////////////////

TEST(Vec2i, arithmetic_and_reductions) {
  const Vec2i a(1, 2), b(3, 4);
  EXPECT_EQ((a + b).x, 4);
  EXPECT_EQ((a - b).y, -2);
  EXPECT_EQ((a * b).y, 8);
  EXPECT_EQ((b / a).y, 2);
  EXPECT_EQ(min(a, b).y, 2);
  EXPECT_EQ(max(a, b).x, 3);
  EXPECT_EQ(reduce_add(a), 3);
  EXPECT_EQ(reduce_mul(b), 12);
  EXPECT_EQ(reduce_min(b), 3);
  EXPECT_EQ(reduce_max(b), 4);
  EXPECT_EQ(Vec2i(zero).x, 0);
  EXPECT_EQ(Vec2i(one).y, 1);
  EXPECT_TRUE(a != b);
  EXPECT_TRUE(a < b);
  EXPECT_EQ(select(true, a, b).x, 1);
  EXPECT_EQ(det(a, b), 1 * 4 - 2 * 3);
}

TEST(Vec2b, construction_and_select) {
  const Vec2b m(true, false);
  EXPECT_TRUE(m.x);
  EXPECT_FALSE(m.y);
  EXPECT_TRUE(m[0]);
  EXPECT_TRUE(m == Vec2b(true, false));
  EXPECT_TRUE(m != Vec2b(false, false));
  EXPECT_EQ(select(true, m, Vec2b(false, true)).y, false);
}

////////////////////////////////////////////////////////////////////////////////
// Vec2<V> -- the vector instantiations, one macro per width the ISA defines
////////////////////////////////////////////////////////////////////////////////

#define ESIMD_VEC2_TESTS(suite, V)                                             \
                                                                               \
  TEST(suite, constructors_and_constants) {                                    \
    const Vec2<V> a(ramp<V>(1.f), ramp<V>(10.f));                              \
    expect_vec2(                                                               \
        a, [](int i) { return 1.f + i; }, [](int i) { return 10.f + i; },      \
        0.f);                                                                  \
    const Vec2<V> b(ramp<V>(2.f)); /* explicit broadcast */                    \
    expect_vec2(                                                               \
        b, [](int i) { return 2.f + i; }, [](int i) { return 2.f + i; }, 0.f); \
    const Vec2<V> c(a); /* copy */                                             \
    expect_vec2(                                                               \
        c, [](int i) { return 1.f + i; }, [](int i) { return 10.f + i; },      \
        0.f);                                                                  \
    expect_lanes(Vec2<V>(zero).x, [](int) { return 0.f; }, 0.f);               \
    expect_lanes(Vec2<V>(one).y, [](int) { return 1.f; }, 0.f);                \
    expect_lanes(a[1], [](int i) { return 10.f + i; }, 0.f); /* operator[] */  \
  }                                                                            \
                                                                               \
  TEST(suite, unary_and_binary) {                                              \
    const Vec2<V> a(ramp<V>(1.f), ramp<V>(10.f));                              \
    const Vec2<V> b(ramp<V>(3.f), ramp<V>(5.f));                               \
    expect_vec2(                                                               \
        -a, [](int i) { return -(1.f + i); },                                  \
        [](int i) { return -(10.f + i); }, 0.f);                               \
    expect_vec2(                                                               \
        abs(-a), [](int i) { return 1.f + i; },                                \
        [](int i) { return 10.f + i; }, 0.f);                                  \
    expect_vec2(                                                               \
        sqrt(a), [](int i) { return std::sqrt(1.f + i); },                     \
        [](int i) { return std::sqrt(10.f + i); }, 1e-5f);                     \
    expect_vec2(                                                               \
        rcp(a), [](int i) { return 1.f / (1.f + i); },                         \
        [](int i) { return 1.f / (10.f + i); }, 1e-5f);                        \
    expect_vec2(                                                               \
        rsqrt(a), [](int i) { return 1.f / std::sqrt(1.f + i); },              \
        [](int i) { return 1.f / std::sqrt(10.f + i); }, 1e-4f);               \
    expect_vec2(                                                               \
        a + b, [](int i) { return 4.f + 2 * i; },                              \
        [](int i) { return 15.f + 2 * i; }, 0.f);                              \
    expect_vec2(                                                               \
        a - b, [](int) { return -2.f; }, [](int) { return 5.f; }, 0.f);        \
    expect_vec2(                                                               \
        a *b, [](int i) { return (1.f + i) * (3.f + i); },                     \
        [](int i) { return (10.f + i) * (5.f + i); }, 1e-4f);                  \
    expect_vec2(                                                               \
        b / a, [](int i) { return (3.f + i) / (1.f + i); },                    \
        [](int i) { return (5.f + i) / (10.f + i); }, 1e-5f);                  \
    expect_vec2(                                                               \
        min(a, b), [](int i) { return std::min(1.f + i, 3.f + i); },           \
        [](int i) { return std::min(10.f + i, 5.f + i); }, 0.f);               \
    expect_vec2(                                                               \
        max(a, b), [](int i) { return std::max(1.f + i, 3.f + i); },           \
        [](int i) { return std::max(10.f + i, 5.f + i); }, 0.f);               \
  }                                                                            \
                                                                               \
  TEST(suite, ternary_fma) {                                                   \
    const Vec2<V> a(ramp<V>(1.f), ramp<V>(2.f));                               \
    const Vec2<V> b(ramp<V>(3.f), ramp<V>(4.f));                               \
    const Vec2<V> c(ramp<V>(5.f), ramp<V>(6.f));                               \
    expect_vec2(                                                               \
        madd(a, b, c),                                                         \
        [](int i) { return (1.f + i) * (3.f + i) + (5.f + i); },               \
        [](int i) { return (2.f + i) * (4.f + i) + (6.f + i); }, 1e-4f);       \
    expect_vec2(                                                               \
        msub(a, b, c),                                                         \
        [](int i) { return (1.f + i) * (3.f + i) - (5.f + i); },               \
        [](int i) { return (2.f + i) * (4.f + i) - (6.f + i); }, 1e-4f);       \
    expect_vec2(                                                               \
        nmadd(a, b, c),                                                        \
        [](int i) { return -(1.f + i) * (3.f + i) + (5.f + i); },              \
        [](int i) { return -(2.f + i) * (4.f + i) + (6.f + i); }, 1e-4f);      \
    expect_vec2(                                                               \
        nmsub(a, b, c),                                                        \
        [](int i) { return -(1.f + i) * (3.f + i) - (5.f + i); },              \
        [](int i) { return -(2.f + i) * (4.f + i) - (6.f + i); }, 1e-4f);      \
    /* scalar-first overloads take a V, not a float */                         \
    const V s(2.f);                                                            \
    expect_vec2(                                                               \
        madd(s, b, c), [](int i) { return 2.f * (3.f + i) + (5.f + i); },      \
        [](int i) { return 2.f * (4.f + i) + (6.f + i); }, 1e-4f);             \
  }                                                                            \
                                                                               \
  TEST(suite, assignment_and_reductions) {                                     \
    Vec2<V> a(ramp<V>(1.f), ramp<V>(2.f));                                     \
    a += Vec2<V>(V(1.f));                                                      \
    expect_vec2(                                                               \
        a, [](int i) { return 2.f + i; }, [](int i) { return 3.f + i; }, 0.f); \
    a -= Vec2<V>(V(1.f));                                                      \
    expect_vec2(                                                               \
        a, [](int i) { return 1.f + i; }, [](int i) { return 2.f + i; }, 0.f); \
    a *= V(2.f);                                                               \
    expect_vec2(                                                               \
        a, [](int i) { return 2.f * (1.f + i); },                              \
        [](int i) { return 2.f * (2.f + i); }, 0.f);                           \
    a /= V(2.f);                                                               \
    expect_vec2(                                                               \
        a, [](int i) { return 1.f + i; }, [](int i) { return 2.f + i; },       \
        1e-5f);                                                                \
    expect_lanes(                                                              \
        reduce_add(a), [](int i) { return (1.f + i) + (2.f + i); }, 0.f);      \
    expect_lanes(                                                              \
        reduce_mul(a), [](int i) { return (1.f + i) * (2.f + i); }, 1e-4f);    \
    expect_lanes(reduce_min(a), [](int i) { return 1.f + i; }, 0.f);           \
    expect_lanes(reduce_max(a), [](int i) { return 2.f + i; }, 0.f);           \
  }                                                                            \
                                                                               \
  TEST(suite, euclidean) {                                                     \
    const Vec2<V> a(V(3.f), V(4.f));                                           \
    const Vec2<V> b(ramp<V>(1.f), ramp<V>(2.f));                               \
    expect_lanes(                                                              \
        dot(a, b), [](int i) { return 3.f * (1.f + i) + 4.f * (2.f + i); },    \
        1e-4f);                                                                \
    expect_vec2(                                                               \
        cross(b), [](int i) { return -(2.f + i); },                            \
        [](int i) { return (1.f + i); }, 0.f);                                 \
    expect_lanes(length(a), [](int) { return 5.f; }, 1e-4f);                   \
    expect_vec2(                                                               \
        normalize(a), [](int) { return 0.6f; }, [](int) { return 0.8f; },      \
        1e-3f);                                                                \
    expect_lanes(                                                              \
        distance(a, Vec2<V>(V(0.f))), [](int) { return 5.f; }, 1e-4f);         \
    expect_lanes(                                                              \
        det(a, b), [](int i) { return 3.f * (2.f + i) - 4.f * (1.f + i); },    \
        1e-4f);                                                                \
  }                                                                            \
                                                                               \
  TEST(suite, select_and_lerp) {                                               \
    const Vec2<V> t(V(1.f), V(2.f)), f(V(3.f), V(4.f));                        \
    /* select(bool, ...) */                                                    \
    expect_vec2(                                                               \
        select(true, t, f), [](int) { return 1.f; }, [](int) { return 2.f; },  \
        0.f);                                                                  \
    expect_vec2(                                                               \
        select(false, t, f), [](int) { return 3.f; }, [](int) { return 4.f; }, \
        0.f);                                                                  \
    /* select(typename T::Bool, ...): even lanes take t, odd lanes take f */   \
    const typename V::Bool m =                                                 \
        (ramp<V>(0.f) - V(2.f) * floor(ramp<V>(0.f) * V(0.5f))) < V(0.5f);     \
    const Vec2<V> s = select(m, t, f);                                         \
    expect_vec2(                                                               \
        s, [](int i) { return (i % 2 == 0) ? 1.f : 3.f; },                     \
        [](int i) { return (i % 2 == 0) ? 2.f : 4.f; }, 0.f);                  \
    expect_vec2(                                                               \
        lerp(t, f, V(0.5f)), [](int) { return 2.f; }, [](int) { return 3.f; }, \
        1e-5f);                                                                \
    /* a zero vector survives normalize_safe unchanged */                      \
    expect_vec2(                                                               \
        normalize_safe(Vec2<V>(V(0.f))), [](int) { return 0.f; },              \
        [](int) { return 0.f; }, 0.f);                                         \
    expect_vec2(                                                               \
        normalize_safe(Vec2<V>(V(3.f), V(4.f))), [](int) { return 0.6f; },     \
        [](int) { return 0.8f; }, 1e-3f);                                      \
  }                                                                            \
                                                                               \
  TEST(suite, shift_right_1) {                                                 \
    /* lane i takes lane i+1; the top lane is left undefined by the simd layer \
     */                                                                        \
    const Vec2<V> a(ramp<V>(1.f), ramp<V>(10.f));                              \
    const Vec2<V> s = shift_right_1(a);                                        \
    for (int i = 0; i < V::size - 1; ++i) {                                    \
      EXPECT_NEAR(s.x[i], 2.f + i, 0.f) << "at x lane " << i;                  \
      EXPECT_NEAR(s.y[i], 11.f + i, 0.f) << "at y lane " << i;                 \
    }                                                                          \
  }

// vfloat4 is the only width every backend has; the rest follow the ISA.
ESIMD_VEC2_TESTS(Vec2_vfloat4, vfloat4)

#if defined(__AVX__) // AVX, AVX2, AVX512 and NEON2X all reach the 8-wide types
ESIMD_VEC2_TESTS(Vec2_vfloat8, vfloat8)
#endif

#if defined(__AVX512F__)
ESIMD_VEC2_TESTS(Vec2_vfloat16, vfloat16)
#endif

// frac() exists for vfloat4 and vfloat8 only -- upstream parity, so
// Vec2<vfloat16> has no frac either.
TEST(Vec2_vfloat4, frac) {
  const Vec2<vfloat4> a(vfloat4(1.25f), vfloat4(-1.25f));
  expect_lanes(frac(a).x, [](int) { return 0.25f; }, 1e-6f);
  expect_lanes(frac(a).y, [](int) { return 0.75f; }, 1e-6f);
}

////////////////////////////////////////////////////////////////////////////////
// Vec2fa -- the SSE-backed 2-float type
////////////////////////////////////////////////////////////////////////////////

TEST(Vec2fa, constructors_and_constants) {
  const Vec2fa a(1.f, 2.f);
  EXPECT_EQ(a.x, 1.f);
  EXPECT_EQ(a.y, 2.f);
  EXPECT_EQ(a[0], 1.f);
  EXPECT_EQ(a[1], 2.f);
  EXPECT_EQ(Vec2fa(3.f).y, 3.f); // explicit broadcast
  EXPECT_EQ(Vec2fa(a).x, 1.f);   // copy
  EXPECT_EQ(Vec2fa(zero).x, 0.f);
  EXPECT_EQ(Vec2fa(one).y, 1.f);
  EXPECT_EQ(Vec2fa(pos_inf).x, std::numeric_limits<float>::infinity());
  EXPECT_EQ(Vec2fa(neg_inf).y, -std::numeric_limits<float>::infinity());
  Vec2fa m(0.f, 0.f);
  m[1] = 7.f;
  EXPECT_EQ(m.y, 7.f);
}

TEST(Vec2fa, conversion_roundtrip) {
  // The reason vec2.h forward-declares Vec2fa: Vec2<float> <-> Vec2fa both
  // ways.
  const Vec2f s(1.5f, -2.5f);
  const Vec2fa fa(s);
  EXPECT_EQ(fa.x, 1.5f);
  EXPECT_EQ(fa.y, -2.5f);
  const Vec2f back(fa);
  EXPECT_EQ(back.x, 1.5f);
  EXPECT_EQ(back.y, -2.5f);
  Vec2fa assigned(0.f, 0.f);
  assigned = s;
  EXPECT_EQ(assigned.y, -2.5f);
}

TEST(Vec2fa, load_store) {
  __aligned(16) const float src[4] = {1.f, 2.f, 3.f, 4.f};
  const Vec2fa a = Vec2fa::load(src);
  EXPECT_EQ(a.x, 1.f);
  EXPECT_EQ(a.y, 2.f);
  EXPECT_EQ(a.az, 0.f); // load()/loadu() mask the unused upper lanes off
  EXPECT_EQ(a.aw, 0.f);

  const float usrc[5] = {0.f, 5.f, 6.f, 7.f, 8.f};
  const Vec2fa b = Vec2fa::loadu(usrc + 1);
  EXPECT_EQ(b.x, 5.f);
  EXPECT_EQ(b.y, 6.f);

  float dst[4] = {0.f, 0.f, 0.f, 0.f};
  Vec2fa::storeu(dst, Vec2fa(9.f, 10.f));
  EXPECT_EQ(dst[0], 9.f);
  EXPECT_EQ(dst[1], 10.f);
}

TEST(Vec2fa, unary) {
  const Vec2fa a(-1.5f, 2.5f);
  EXPECT_EQ((+a).x, -1.5f);
  EXPECT_EQ((-a).x, 1.5f);
  EXPECT_EQ(abs(a).x, 1.5f);
  EXPECT_EQ(sign(a).x, -1.f);
  EXPECT_EQ(sign(a).y, 1.f);
  EXPECT_NEAR(sqrt(Vec2fa(4.f, 9.f)).y, 3.f, 1e-5f);
  EXPECT_NEAR(sqr(Vec2fa(3.f, 4.f)).y, 16.f, 1e-5f);
  EXPECT_NEAR(rcp(Vec2fa(2.f, 4.f)).y, 0.25f, 1e-5f);
  EXPECT_NEAR(rsqrt(Vec2fa(4.f, 16.f)).y, 0.25f, 1e-4f);
  EXPECT_NEAR(log(Vec2fa(1.f, float(M_E))).y, 1.f, 1e-5f);
  EXPECT_NEAR(exp(Vec2fa(0.f, 1.f)).y, float(M_E), 1e-5f);
  // rcp_safe clamps a zero input instead of returning inf
  EXPECT_TRUE(std::isfinite(rcp_safe(Vec2fa(0.f, 2.f)).x));
  EXPECT_NEAR(rcp_safe(Vec2fa(0.f, 2.f)).y, 0.5f, 1e-5f);
}

TEST(Vec2fa, binary_and_assignment) {
  const Vec2fa a(1.f, 2.f), b(4.f, 8.f);
  EXPECT_EQ((a + b).x, 5.f);
  EXPECT_EQ((a - b).y, -6.f);
  EXPECT_EQ((a * b).y, 16.f);
  EXPECT_EQ((a * 2.f).x, 2.f);
  EXPECT_EQ((2.f * a).y, 4.f);
  EXPECT_EQ((b / a).y, 4.f);
  EXPECT_EQ((b / 2.f).x, 2.f);
  EXPECT_EQ((8.f / b).y, 1.f);
  EXPECT_EQ(min(a, b).x, 1.f);
  EXPECT_EQ(max(a, b).x, 4.f);
  EXPECT_NEAR(pow(Vec2fa(2.f, 3.f), 2.f).y, 9.f, 1e-5f);

  Vec2fa m(1.f, 2.f);
  m += Vec2fa(1.f, 1.f);
  EXPECT_EQ(m.y, 3.f);
  m -= Vec2fa(1.f, 1.f);
  EXPECT_EQ(m.x, 1.f);
  m *= Vec2fa(2.f, 2.f);
  EXPECT_EQ(m.y, 4.f);
  m *= 2.f;
  EXPECT_EQ(m.x, 4.f);
  m /= Vec2fa(2.f, 2.f);
  EXPECT_EQ(m.y, 4.f);
  m /= 2.f;
  EXPECT_EQ(m.x, 1.f);
}

TEST(Vec2fa, ternary_fma) {
  const Vec2fa a(1.f, 2.f), b(3.f, 4.f), c(5.f, 6.f);
  EXPECT_NEAR(madd(a, b, c).x, 1.f * 3.f + 5.f, 1e-5f);
  EXPECT_NEAR(msub(a, b, c).y, 2.f * 4.f - 6.f, 1e-5f);
  EXPECT_NEAR(nmadd(a, b, c).x, -1.f * 3.f + 5.f, 1e-5f);
  EXPECT_NEAR(nmsub(a, b, c).y, -2.f * 4.f - 6.f, 1e-5f);
  EXPECT_NEAR(madd(2.f, b, c).x, 2.f * 3.f + 5.f, 1e-5f);
  EXPECT_NEAR(nmsub(2.f, b, c).y, -2.f * 4.f - 6.f, 1e-5f);
}

TEST(Vec2fa, reductions_and_comparison) {
  const Vec2fa a(3.f, 5.f);
  EXPECT_EQ(reduce_add(a), 8.f);
  EXPECT_EQ(reduce_mul(a), 15.f);
  EXPECT_EQ(reduce_min(a), 3.f);
  EXPECT_EQ(reduce_max(a), 5.f);
  // only x and y take part: the padding lanes must not decide the comparison
  EXPECT_TRUE(Vec2fa(1.f, 2.f) == Vec2fa(1.f, 2.f));
  EXPECT_FALSE(Vec2fa(1.f, 2.f) == Vec2fa(1.f, 3.f));
  EXPECT_TRUE(Vec2fa(1.f, 2.f) != Vec2fa(1.f, 3.f));
  EXPECT_FALSE(Vec2fa(1.f, 2.f) != Vec2fa(1.f, 2.f));
}

TEST(Vec2fa, euclidean) {
  const Vec2fa a(3.f, 4.f), b(1.f, 2.f);
  EXPECT_NEAR(dot(a, b), 11.f, 1e-4f);
  EXPECT_EQ(cross(a).x, -4.f);
  EXPECT_EQ(cross(a).y, 3.f);
  EXPECT_NEAR(sqr_length(a), 25.f, 1e-4f);
  EXPECT_NEAR(rcp_length(a), 0.2f, 1e-4f);
  EXPECT_NEAR(rcp_length2(a), 0.04f, 1e-4f);
  EXPECT_NEAR(length(a), 5.f, 1e-4f);
  EXPECT_NEAR(normalize(a).x, 0.6f, 1e-3f);
  EXPECT_NEAR(normalize(a).y, 0.8f, 1e-3f);
  EXPECT_NEAR(distance(a, b), std::sqrt(2.f * 2.f + 2.f * 2.f), 1e-4f);
}

TEST(Vec2fa, select_lerp_maxDim_rounding) {
  const Vec2fa t(1.f, 2.f), f(3.f, 4.f);
  EXPECT_EQ(select(true, t, f).x, 1.f);
  EXPECT_EQ(select(false, t, f).y, 4.f);
  EXPECT_NEAR(lerp(t, f, 0.5f).x, 2.f, 1e-5f);
  EXPECT_EQ(maxDim(Vec2fa(-3.f, 2.f)), 0);
  EXPECT_EQ(maxDim(Vec2fa(1.f, -2.f)), 1);
  const Vec2fa r(1.4f, -1.4f);
  EXPECT_EQ(floor(r).x, 1.f);
  EXPECT_EQ(floor(r).y, -2.f);
  EXPECT_EQ(ceil(r).x, 2.f);
  EXPECT_EQ(ceil(r).y, -1.f);
}

#if defined(ESIMD_ARM64) || defined(__SSE4_1__)
TEST(Vec2fa, mini_maxi) {
  // integer min/max on the float bit patterns; for positive floats it agrees
  // with the ordinary min/max
  const Vec2fa a(1.f, 8.f), b(4.f, 2.f);
  EXPECT_EQ(mini(a, b).x, 1.f);
  EXPECT_EQ(mini(a, b).y, 2.f);
  EXPECT_EQ(maxi(a, b).x, 4.f);
  EXPECT_EQ(maxi(a, b).y, 8.f);
}
#endif
