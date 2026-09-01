// Copyright 2026 notchla liso.lorenzo@gmail.com
// SPDX-License-Identifier: Apache-2.0
//
// Correctness tests for Vec4<T> from the optional data-types layer
// (<esimd/types.h>). Vec4<T> is component-wise plumbing over T with no SSE
// companion of its own, so it is checked at T = scalar (float/int/bool/unsigned
// char) and at every vector width the active ISA defines, lane-by-lane against a
// scalar reference. Compiled once per ISA.

#include "test_helpers.h"
#include <esimd/types.h>

using namespace esimd;
using namespace esimd_test;

namespace {

// Lane i = base + i, so a mixed-up lane cannot pass unnoticed.
template <typename V> V ramp(float base) {
  float tmp[V::size];
  for (int i = 0; i < V::size; ++i)
    tmp[i] = base + float(i);
  return V::loadu(tmp);
}

// Element-wise check of a Vec4<V> against a per-lane scalar reference, given as
// four functions of the lane index.
template <typename V, typename FX, typename FY, typename FZ, typename FW>
void expect_vec4(const Vec4<V> &v, FX rx, FY ry, FZ rz, FW rw, float tol) {
  for (int i = 0; i < V::size; ++i) {
    EXPECT_NEAR(v.x[i], rx(i), tol) << "at x lane " << i;
    EXPECT_NEAR(v.y[i], ry(i), tol) << "at y lane " << i;
    EXPECT_NEAR(v.z[i], rz(i), tol) << "at z lane " << i;
    EXPECT_NEAR(v.w[i], rw(i), tol) << "at w lane " << i;
  }
}

// Same, for an operation reducing a Vec4<V> back to a V.
template <typename V, typename F>
void expect_lanes(const V &v, F ref, float tol) {
  for (int i = 0; i < V::size; ++i)
    EXPECT_NEAR(v[i], ref(i), tol) << "at lane " << i;
}

} // namespace

////////////////////////////////////////////////////////////////////////////////
// Vec4f -- the scalar float instantiation
////////////////////////////////////////////////////////////////////////////////

TEST(Vec4f, constructors_and_constants) {
  const Vec4f a(1.f, 2.f, 3.f, 4.f);
  EXPECT_EQ(a.x, 1.f);
  EXPECT_EQ(a.w, 4.f);
  EXPECT_EQ(Vec4f(5.f).w, 5.f);            // explicit broadcast
  EXPECT_EQ(Vec4f(a).z, 3.f);              // copy
  EXPECT_EQ(Vec4f(Vec4i(4, 5, 6, 7)).w, 7.f); // converting
  EXPECT_EQ(Vec4f(zero).w, 0.f);
  EXPECT_EQ(Vec4f(one).w, 1.f);
  EXPECT_TRUE(std::isinf(Vec4f(pos_inf).w));
  EXPECT_LT(Vec4f(neg_inf).x, 0.f);
  EXPECT_EQ(a[0], 1.f); // operator[]
  EXPECT_EQ(a[3], 4.f);
  Vec4f m(0.f);
  m[2] = 7.f;
  EXPECT_EQ(m.z, 7.f);
}

TEST(Vec4f, vec3_interop) {
  // Vec3 + w
  const Vec4f a(Vec3f(1.f, 2.f, 3.f), 4.f);
  EXPECT_EQ(a.z, 3.f);
  EXPECT_EQ(a.w, 4.f);
  // xyz() swizzle and the implicit conversion both drop w
  EXPECT_TRUE(a.xyz() == Vec3f(1.f, 2.f, 3.f));
  const Vec3f b = a;
  EXPECT_TRUE(b == Vec3f(1.f, 2.f, 3.f));
  // Vec3fx carries its fourth lane into w
  const Vec4f c(Vec3fx(1.f, 2.f, 3.f, 9.f));
  EXPECT_EQ(c.z, 3.f);
  EXPECT_EQ(c.w, 9.f);
}

TEST(Vec4f, unary_and_binary) {
  const Vec4f a(1.f, 2.f, 3.f, 4.f), b(4.f, 8.f, 12.f, 16.f);
  EXPECT_EQ((+a).x, 1.f);
  EXPECT_EQ((-a).w, -4.f);
  EXPECT_EQ(abs(Vec4f(-1.f, 2.f, -3.f, 4.f)).z, 3.f);
  EXPECT_NEAR(sqrt(Vec4f(4.f, 9.f, 16.f, 25.f)).w, 5.f, 1e-5f);
  EXPECT_NEAR(rcp(Vec4f(2.f, 4.f, 8.f, 16.f)).w, 0.0625f, 1e-5f);
  EXPECT_NEAR(rsqrt(Vec4f(4.f, 16.f, 64.f, 256.f)).w, 0.0625f, 1e-4f);
  EXPECT_EQ((a + b).w, 20.f);
  EXPECT_EQ((a - b).w, -12.f);
  EXPECT_EQ((a * b).w, 64.f);
  EXPECT_EQ((a * 2.f).w, 8.f);
  EXPECT_EQ((2.f * a).w, 8.f);
  EXPECT_EQ((b / a).w, 4.f);
  EXPECT_EQ((b / 2.f).w, 8.f);
  EXPECT_EQ((16.f / b).w, 1.f);
  EXPECT_EQ(min(a, b).w, 4.f);
  EXPECT_EQ(max(a, b).w, 16.f);
}

TEST(Vec4f, ternary_fma) {
  const Vec4f a(1.f, 2.f, 3.f, 4.f), b(5.f, 6.f, 7.f, 8.f),
      c(9.f, 10.f, 11.f, 12.f);
  EXPECT_NEAR(madd(a, b, c).w, 4.f * 8.f + 12.f, 1e-5f);
  EXPECT_NEAR(msub(a, b, c).w, 4.f * 8.f - 12.f, 1e-5f);
  EXPECT_NEAR(nmadd(a, b, c).w, -4.f * 8.f + 12.f, 1e-5f);
  EXPECT_NEAR(nmsub(a, b, c).w, -4.f * 8.f - 12.f, 1e-5f);
  EXPECT_NEAR(madd(2.f, b, c).x, 2.f * 5.f + 9.f, 1e-5f);
  EXPECT_NEAR(lerp(a, b, 0.5f).w, 6.f, 1e-5f);
}

TEST(Vec4f, assignment_and_reductions) {
  Vec4f a(1.f, 2.f, 3.f, 4.f);
  a += Vec4f(one);
  EXPECT_EQ(a.w, 5.f);
  a -= Vec4f(one);
  EXPECT_EQ(a.w, 4.f);
  a *= 2.f;
  EXPECT_EQ(a.w, 8.f);
  a /= 2.f;
  EXPECT_EQ(a.w, 4.f);

  const Vec4f b(1.f, 2.f, 4.f, 8.f);
  EXPECT_EQ(reduce_add(b), 15.f);
  EXPECT_EQ(reduce_mul(b), 64.f);
  EXPECT_EQ(reduce_min(b), 1.f);
  EXPECT_EQ(reduce_max(b), 8.f);
}

TEST(Vec4f, comparison) {
  const Vec4f a(1.f, 2.f, 3.f, 4.f), b(1.f, 2.f, 3.f, 9.f);
  EXPECT_TRUE(a == Vec4f(1.f, 2.f, 3.f, 4.f));
  EXPECT_TRUE(a != b);
  EXPECT_TRUE(a < b); // lexicographic: w decides
  EXPECT_FALSE(b < a);
  EXPECT_FALSE(a < a);
}

TEST(Vec4f, euclidean_and_select) {
  const Vec4f a(1.f, 2.f, 2.f, 4.f); // length 5
  EXPECT_NEAR(dot(a, a), 25.f, 1e-4f);
  EXPECT_NEAR(length(a), 5.f, 1e-4f);
  EXPECT_NEAR(distance(a, Vec4f(1.f, 2.f, 2.f, 7.f)), 3.f, 1e-4f);
  EXPECT_NEAR(normalize(a).w, 0.8f, 1e-3f);

  const Vec4f t(1.f, 2.f, 3.f, 4.f), f(5.f, 6.f, 7.f, 8.f);
  EXPECT_EQ(select(true, t, f).w, 4.f);
  EXPECT_EQ(select(false, t, f).w, 8.f);
  const Vec4f m = select(Vec4b(true, false, true, false), t, f);
  EXPECT_EQ(m.x, 1.f);
  EXPECT_EQ(m.y, 6.f);
  EXPECT_EQ(m.z, 3.f);
  EXPECT_EQ(m.w, 8.f);
}

////////////////////////////////////////////////////////////////////////////////
// Vec4i / Vec4b / Vec4uc -- the other scalar instantiations
////////////////////////////////////////////////////////////////////////////////

TEST(Vec4i, arithmetic_and_reductions) {
  const Vec4i a(1, 2, 3, 4), b(4, 8, 12, 16);
  EXPECT_EQ((a + b).w, 20);
  EXPECT_EQ((a - b).w, -12);
  EXPECT_EQ((a * b).w, 64);
  EXPECT_EQ((b / a).w, 4);
  EXPECT_EQ(min(a, b).w, 4);
  EXPECT_EQ(max(a, b).w, 16);
  EXPECT_EQ(reduce_add(a), 10);
  EXPECT_EQ(reduce_mul(a), 24);
  EXPECT_EQ(reduce_min(a), 1);
  EXPECT_EQ(reduce_max(a), 4);
  EXPECT_TRUE(a < b);
  EXPECT_EQ(Vec4i(zero).w, 0);
  EXPECT_EQ(Vec4i(one).w, 1);
  EXPECT_TRUE(a.xyz() == Vec3i(1, 2, 3));
}

TEST(Vec4b, logic) {
  const Vec4b a(true, false, true, false);
  EXPECT_TRUE(a.x);
  EXPECT_FALSE(a.w);
  EXPECT_TRUE(a[2]);
  EXPECT_TRUE(a == Vec4b(true, false, true, false));
  EXPECT_TRUE(a != Vec4b(true, true, true, true));
  EXPECT_TRUE(Vec4b(true).w);
}

TEST(Vec4uc, bytes) {
  // the unsigned-char instantiation exists mostly for packed colours
  const Vec4uc a(1, 2, 3, 4);
  EXPECT_EQ(int(a.x), 1);
  EXPECT_EQ(int(a.w), 4);
  EXPECT_EQ(int(a[2]), 3);
  EXPECT_EQ(int((a + a).w), 8);
  EXPECT_EQ(int(min(a, Vec4uc(2)).w), 2);
  EXPECT_TRUE(a == Vec4uc(1, 2, 3, 4));
  EXPECT_EQ(int(Vec4uc(zero).w), 0);
}

////////////////////////////////////////////////////////////////////////////////
// Vec4<V> -- one instantiation per vector width the ISA defines
////////////////////////////////////////////////////////////////////////////////

#define ESIMD_VEC4_TESTS(suite, V)                                             \
                                                                               \
  TEST(suite, constructors_and_constants) {                                    \
    const Vec4<V> a(ramp<V>(1.f), ramp<V>(10.f), ramp<V>(20.f), ramp<V>(30.f)); \
    expect_vec4(                                                               \
        a, [](int i) { return 1.f + i; }, [](int i) { return 10.f + i; },      \
        [](int i) { return 20.f + i; }, [](int i) { return 30.f + i; }, 0.f);  \
    const Vec4<V> b(ramp<V>(2.f)); /* explicit broadcast */                    \
    expect_vec4(                                                               \
        b, [](int i) { return 2.f + i; }, [](int i) { return 2.f + i; },       \
        [](int i) { return 2.f + i; }, [](int i) { return 2.f + i; }, 0.f);    \
    const Vec4<V> c(a); /* copy */                                             \
    expect_vec4(                                                               \
        c, [](int i) { return 1.f + i; }, [](int i) { return 10.f + i; },      \
        [](int i) { return 20.f + i; }, [](int i) { return 30.f + i; }, 0.f);  \
    expect_lanes(Vec4<V>(zero).w, [](int) { return 0.f; }, 0.f);               \
    expect_lanes(Vec4<V>(one).w, [](int) { return 1.f; }, 0.f);                \
    expect_lanes(a[3], [](int i) { return 30.f + i; }, 0.f); /* operator[] */  \
  }                                                                            \
                                                                               \
  TEST(suite, vec3_interop) {                                                  \
    const Vec4<V> a(Vec3<V>(V(1.f), V(2.f), V(3.f)), V(4.f));                  \
    expect_lanes(a.w, [](int) { return 4.f; }, 0.f);                           \
    expect_lanes(a.xyz().z, [](int) { return 3.f; }, 0.f);                     \
    const Vec3<V> b = a; /* implicit conversion drops w */                     \
    expect_lanes(b.z, [](int) { return 3.f; }, 0.f);                           \
    /* Vec3fx broadcasts its four floats across every lane */                  \
    const Vec4<V> c(Vec3fx(1.f, 2.f, 3.f, 9.f));                               \
    expect_vec4(                                                               \
        c, [](int) { return 1.f; }, [](int) { return 2.f; },                   \
        [](int) { return 3.f; }, [](int) { return 9.f; }, 0.f);                \
  }                                                                            \
                                                                               \
  TEST(suite, unary_and_binary) {                                              \
    const Vec4<V> a(ramp<V>(1.f), ramp<V>(10.f), ramp<V>(20.f), ramp<V>(30.f)); \
    const Vec4<V> b(ramp<V>(3.f), ramp<V>(5.f), ramp<V>(7.f), ramp<V>(9.f));   \
    expect_vec4(                                                               \
        -a, [](int i) { return -(1.f + i); },                                  \
        [](int i) { return -(10.f + i); }, [](int i) { return -(20.f + i); },  \
        [](int i) { return -(30.f + i); }, 0.f);                               \
    expect_vec4(                                                               \
        abs(-a), [](int i) { return 1.f + i; },                                \
        [](int i) { return 10.f + i; }, [](int i) { return 20.f + i; },        \
        [](int i) { return 30.f + i; }, 0.f);                                  \
    expect_vec4(                                                               \
        sqrt(a), [](int i) { return std::sqrt(1.f + i); },                     \
        [](int i) { return std::sqrt(10.f + i); },                             \
        [](int i) { return std::sqrt(20.f + i); },                             \
        [](int i) { return std::sqrt(30.f + i); }, 1e-5f);                     \
    expect_vec4(                                                               \
        rcp(a), [](int i) { return 1.f / (1.f + i); },                         \
        [](int i) { return 1.f / (10.f + i); },                                \
        [](int i) { return 1.f / (20.f + i); },                                \
        [](int i) { return 1.f / (30.f + i); }, 1e-5f);                        \
    expect_vec4(                                                               \
        rsqrt(a), [](int i) { return 1.f / std::sqrt(1.f + i); },              \
        [](int i) { return 1.f / std::sqrt(10.f + i); },                       \
        [](int i) { return 1.f / std::sqrt(20.f + i); },                       \
        [](int i) { return 1.f / std::sqrt(30.f + i); }, 1e-4f);               \
    expect_vec4(                                                               \
        a + b, [](int i) { return 4.f + 2 * i; },                              \
        [](int i) { return 15.f + 2 * i; }, [](int i) { return 27.f + 2 * i; }, \
        [](int i) { return 39.f + 2 * i; }, 0.f);                              \
    expect_vec4(                                                               \
        a - b, [](int) { return -2.f; }, [](int) { return 5.f; },              \
        [](int) { return 13.f; }, [](int) { return 21.f; }, 0.f);              \
    expect_vec4(                                                               \
        a *b, [](int i) { return (1.f + i) * (3.f + i); },                     \
        [](int i) { return (10.f + i) * (5.f + i); },                          \
        [](int i) { return (20.f + i) * (7.f + i); },                          \
        [](int i) { return (30.f + i) * (9.f + i); }, 1e-3f);                  \
    expect_vec4(                                                               \
        b / a, [](int i) { return (3.f + i) / (1.f + i); },                    \
        [](int i) { return (5.f + i) / (10.f + i); },                          \
        [](int i) { return (7.f + i) / (20.f + i); },                          \
        [](int i) { return (9.f + i) / (30.f + i); }, 1e-5f);                  \
    expect_vec4(                                                               \
        min(a, b), [](int i) { return std::min(1.f + i, 3.f + i); },           \
        [](int i) { return std::min(10.f + i, 5.f + i); },                     \
        [](int i) { return std::min(20.f + i, 7.f + i); },                     \
        [](int i) { return std::min(30.f + i, 9.f + i); }, 0.f);               \
    expect_vec4(                                                               \
        max(a, b), [](int i) { return std::max(1.f + i, 3.f + i); },           \
        [](int i) { return std::max(10.f + i, 5.f + i); },                     \
        [](int i) { return std::max(20.f + i, 7.f + i); },                     \
        [](int i) { return std::max(30.f + i, 9.f + i); }, 0.f);               \
  }                                                                            \
                                                                               \
  TEST(suite, ternary_fma) {                                                   \
    const Vec4<V> a(ramp<V>(1.f), ramp<V>(2.f), ramp<V>(3.f), ramp<V>(4.f));   \
    const Vec4<V> b(ramp<V>(5.f), ramp<V>(6.f), ramp<V>(7.f), ramp<V>(8.f));   \
    const Vec4<V> c(ramp<V>(9.f), ramp<V>(10.f), ramp<V>(11.f), ramp<V>(12.f)); \
    expect_vec4(                                                               \
        madd(a, b, c),                                                         \
        [](int i) { return (1.f + i) * (5.f + i) + (9.f + i); },               \
        [](int i) { return (2.f + i) * (6.f + i) + (10.f + i); },              \
        [](int i) { return (3.f + i) * (7.f + i) + (11.f + i); },              \
        [](int i) { return (4.f + i) * (8.f + i) + (12.f + i); }, 1e-4f);      \
    expect_vec4(                                                               \
        msub(a, b, c),                                                         \
        [](int i) { return (1.f + i) * (5.f + i) - (9.f + i); },               \
        [](int i) { return (2.f + i) * (6.f + i) - (10.f + i); },              \
        [](int i) { return (3.f + i) * (7.f + i) - (11.f + i); },              \
        [](int i) { return (4.f + i) * (8.f + i) - (12.f + i); }, 1e-4f);      \
    expect_vec4(                                                               \
        nmadd(a, b, c),                                                        \
        [](int i) { return -(1.f + i) * (5.f + i) + (9.f + i); },              \
        [](int i) { return -(2.f + i) * (6.f + i) + (10.f + i); },             \
        [](int i) { return -(3.f + i) * (7.f + i) + (11.f + i); },             \
        [](int i) { return -(4.f + i) * (8.f + i) + (12.f + i); }, 1e-4f);     \
    expect_vec4(                                                               \
        nmsub(a, b, c),                                                        \
        [](int i) { return -(1.f + i) * (5.f + i) - (9.f + i); },              \
        [](int i) { return -(2.f + i) * (6.f + i) - (10.f + i); },             \
        [](int i) { return -(3.f + i) * (7.f + i) - (11.f + i); },             \
        [](int i) { return -(4.f + i) * (8.f + i) - (12.f + i); }, 1e-4f);     \
    /* scalar-first overloads take a V, not a float */                         \
    const V s(2.f);                                                            \
    expect_vec4(                                                               \
        madd(s, b, c), [](int i) { return 2.f * (5.f + i) + (9.f + i); },      \
        [](int i) { return 2.f * (6.f + i) + (10.f + i); },                    \
        [](int i) { return 2.f * (7.f + i) + (11.f + i); },                    \
        [](int i) { return 2.f * (8.f + i) + (12.f + i); }, 1e-4f);            \
    expect_vec4(                                                               \
        lerp(a, b, V(0.5f)),                                                   \
        [](int i) { return 0.5f * ((1.f + i) + (5.f + i)); },                  \
        [](int i) { return 0.5f * ((2.f + i) + (6.f + i)); },                  \
        [](int i) { return 0.5f * ((3.f + i) + (7.f + i)); },                  \
        [](int i) { return 0.5f * ((4.f + i) + (8.f + i)); }, 1e-4f);          \
  }                                                                            \
                                                                               \
  TEST(suite, assignment_and_reductions) {                                     \
    Vec4<V> a(ramp<V>(1.f), ramp<V>(2.f), ramp<V>(3.f), ramp<V>(4.f));         \
    a += Vec4<V>(V(1.f));                                                      \
    expect_vec4(                                                               \
        a, [](int i) { return 2.f + i; }, [](int i) { return 3.f + i; },       \
        [](int i) { return 4.f + i; }, [](int i) { return 5.f + i; }, 0.f);    \
    a -= Vec4<V>(V(1.f));                                                      \
    expect_vec4(                                                               \
        a, [](int i) { return 1.f + i; }, [](int i) { return 2.f + i; },       \
        [](int i) { return 3.f + i; }, [](int i) { return 4.f + i; }, 0.f);    \
    a *= V(2.f);                                                               \
    expect_lanes(a.w, [](int i) { return 2.f * (4.f + i); }, 0.f);             \
    a /= V(2.f);                                                               \
    expect_lanes(a.w, [](int i) { return 4.f + i; }, 1e-5f);                   \
    expect_lanes(                                                              \
        reduce_add(a),                                                         \
        [](int i) { return (1.f + i) + (2.f + i) + (3.f + i) + (4.f + i); },   \
        1e-5f);                                                                \
    expect_lanes(                                                              \
        reduce_mul(a),                                                         \
        [](int i) { return (1.f + i) * (2.f + i) * (3.f + i) * (4.f + i); },   \
        1e-2f);                                                                \
    expect_lanes(reduce_min(a), [](int i) { return 1.f + i; }, 0.f);           \
    expect_lanes(reduce_max(a), [](int i) { return 4.f + i; }, 0.f);           \
  }                                                                            \
                                                                               \
  TEST(suite, euclidean) {                                                     \
    const Vec4<V> a(V(1.f), V(2.f), V(2.f), V(4.f)); /* length 5 */            \
    expect_lanes(dot(a, a), [](int) { return 25.f; }, 1e-4f);                  \
    expect_lanes(length(a), [](int) { return 5.f; }, 1e-4f);                   \
    expect_lanes(distance(a, a), [](int) { return 0.f; }, 1e-5f);              \
    expect_vec4(                                                               \
        normalize(a), [](int) { return 0.2f; }, [](int) { return 0.4f; },      \
        [](int) { return 0.4f; }, [](int) { return 0.8f; }, 1e-3f);            \
  }                                                                            \
                                                                               \
  TEST(suite, select_and_shift) {                                              \
    const Vec4<V> t(V(1.f), V(2.f), V(3.f), V(4.f));                           \
    const Vec4<V> f(V(5.f), V(6.f), V(7.f), V(8.f));                           \
    expect_vec4(                                                               \
        select(true, t, f), [](int) { return 1.f; }, [](int) { return 2.f; },  \
        [](int) { return 3.f; }, [](int) { return 4.f; }, 0.f);                \
    expect_vec4(                                                               \
        select(Vec4b(true, false, true, false), t, f),                         \
        [](int) { return 1.f; }, [](int) { return 6.f; },                      \
        [](int) { return 3.f; }, [](int) { return 8.f; }, 0.f);                \
    /* lane mask: even lanes take t, odd lanes take f */                       \
    float sel[V::size];                                                        \
    for (int i = 0; i < V::size; ++i)                                          \
      sel[i] = (i % 2 == 0) ? 0.f : 1.f;                                       \
    const typename V::Bool m = V::loadu(sel) < V(0.5f);                        \
    const Vec4<V> s = select(m, t, f);                                         \
    for (int i = 0; i < V::size; ++i)                                          \
      EXPECT_EQ(s.w[i], (i % 2 == 0) ? 4.f : 8.f) << "at lane " << i;          \
    /* shift_right_1 drops lane 0 and shifts the rest down */                  \
    const Vec4<V> sr = shift_right_1(Vec4<V>(ramp<V>(1.f)));                   \
    for (int i = 0; i + 1 < V::size; ++i)                                      \
      EXPECT_EQ(sr.w[i], 2.f + i) << "at lane " << i;                          \
  }

ESIMD_VEC4_TESTS(Vec4_vfloat4, vfloat4)
#if defined(__AVX__)
ESIMD_VEC4_TESTS(Vec4_vfloat8, vfloat8)
#endif
#if defined(__AVX512F__)
ESIMD_VEC4_TESTS(Vec4_vfloat16, vfloat16)
#endif
