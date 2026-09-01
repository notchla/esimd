// Copyright 2026 notchla liso.lorenzo@gmail.com
// SPDX-License-Identifier: Apache-2.0
//
// Correctness tests for the Vec3 half of the optional data-types layer
// (<esimd/types.h>): the generic Vec3<T> plus the three SSE-backed companions
// vec3.h drags in -- Vec3ba (mask), Vec3ia (int) and Vec3fa/Vec3fx (float).
// Vec3<T> is component-wise plumbing over T, so it is checked both at
// T = scalar (float/int/bool) and at every vector width the active ISA defines,
// lane-by-lane against a scalar reference. Compiled once per ISA.

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

// Element-wise check of a Vec3<V> against a per-lane scalar reference, given as
// three functions of the lane index.
template <typename V, typename FX, typename FY, typename FZ>
void expect_vec3(const Vec3<V> &v, FX rx, FY ry, FZ rz, float tol) {
  for (int i = 0; i < V::size; ++i) {
    EXPECT_NEAR(v.x[i], rx(i), tol) << "at x lane " << i;
    EXPECT_NEAR(v.y[i], ry(i), tol) << "at y lane " << i;
    EXPECT_NEAR(v.z[i], rz(i), tol) << "at z lane " << i;
  }
}

// Same, for an operation reducing a Vec3<V> back to a V.
template <typename V, typename F>
void expect_lanes(const V &v, F ref, float tol) {
  for (int i = 0; i < V::size; ++i)
    EXPECT_NEAR(v[i], ref(i), tol) << "at lane " << i;
}

} // namespace

////////////////////////////////////////////////////////////////////////////////
// Vec3f -- the scalar float instantiation
////////////////////////////////////////////////////////////////////////////////

TEST(Vec3f, constructors_and_constants) {
  const Vec3f a(1.f, 2.f, 3.f);
  EXPECT_EQ(a.x, 1.f);
  EXPECT_EQ(a.y, 2.f);
  EXPECT_EQ(a.z, 3.f);
  EXPECT_EQ(Vec3f(4.f).z, 4.f);            // explicit broadcast
  EXPECT_EQ(Vec3f(a).y, 2.f);              // copy
  EXPECT_EQ(Vec3f(Vec3i(4, 5, 6)).x, 4.f); // converting
  EXPECT_EQ(Vec3f(zero).x, 0.f);
  EXPECT_EQ(Vec3f(one).y, 1.f);
  EXPECT_TRUE(std::isinf(Vec3f(pos_inf).z));
  EXPECT_LT(Vec3f(neg_inf).x, 0.f);
  EXPECT_EQ(a[0], 1.f); // operator[]
  EXPECT_EQ(a[2], 3.f);
  Vec3f m(0.f);
  m[1] = 7.f;
  EXPECT_EQ(m.y, 7.f);
}

TEST(Vec3f, unary) {
  const Vec3f a(-1.5f, 2.5f, -4.f);
  EXPECT_EQ((+a).x, -1.5f);
  EXPECT_EQ((-a).x, 1.5f);
  EXPECT_EQ(abs(a).z, 4.f);
  EXPECT_NEAR(sqrt(Vec3f(4.f, 9.f, 16.f)).z, 4.f, 1e-5f);
  EXPECT_NEAR(rcp(Vec3f(2.f, 4.f, 8.f)).z, 0.125f, 1e-5f);
  EXPECT_NEAR(rsqrt(Vec3f(4.f, 16.f, 64.f)).z, 0.125f, 1e-4f);
  // zero_fix clamps a zero component away from zero, so rcp_safe stays finite
  EXPECT_NE(zero_fix(Vec3f(0.f, 1.f, 2.f)).x, 0.f);
  EXPECT_TRUE(std::isfinite(rcp_safe(Vec3f(0.f, 2.f, 4.f)).x));
  EXPECT_NEAR(rcp_safe(Vec3f(0.f, 2.f, 4.f)).z, 0.25f, 1e-5f);
}

TEST(Vec3f, binary) {
  const Vec3f a(1.f, 2.f, 3.f), b(4.f, 8.f, 12.f);
  EXPECT_EQ((a + b).x, 5.f);
  EXPECT_EQ((a - b).y, -6.f);
  EXPECT_EQ((a * b).z, 36.f);
  EXPECT_EQ((a * 2.f).x, 2.f);
  EXPECT_EQ((2.f * a).y, 4.f);
  EXPECT_EQ((b / a).z, 4.f);
  EXPECT_EQ((b / 2.f).x, 2.f);
  EXPECT_EQ((12.f / b).z, 1.f);
  EXPECT_EQ(min(a, b).z, 3.f);
  EXPECT_EQ(max(a, b).z, 12.f);
}

TEST(Vec3f, ternary_fma) {
  const Vec3f a(1.f, 2.f, 3.f), b(4.f, 5.f, 6.f), c(7.f, 8.f, 9.f);
  EXPECT_NEAR(madd(a, b, c).z, 3.f * 6.f + 9.f, 1e-5f);
  EXPECT_NEAR(msub(a, b, c).z, 3.f * 6.f - 9.f, 1e-5f);
  EXPECT_NEAR(nmadd(a, b, c).z, -3.f * 6.f + 9.f, 1e-5f);
  EXPECT_NEAR(nmsub(a, b, c).z, -3.f * 6.f - 9.f, 1e-5f);
  EXPECT_NEAR(madd(2.f, b, c).x, 2.f * 4.f + 7.f, 1e-5f);
  EXPECT_NEAR(lerp(a, b, 0.5f).z, 4.5f, 1e-5f);
}

TEST(Vec3f, assignment_and_reductions) {
  Vec3f a(1.f, 2.f, 3.f);
  a += 1.f; // scalar broadcast form
  EXPECT_EQ(a.z, 4.f);
  a += Vec3f(1.f, 1.f, 1.f);
  EXPECT_EQ(a.z, 5.f);
  a -= Vec3f(1.f, 1.f, 1.f);
  EXPECT_EQ(a.x, 2.f);
  a *= 2.f;
  EXPECT_EQ(a.x, 4.f);
  a /= 2.f;
  EXPECT_EQ(a.x, 2.f);

  const Vec3f b(1.f, 2.f, 4.f);
  EXPECT_EQ(reduce_add(b), 7.f);
  EXPECT_EQ(reduce_mul(b), 8.f);
  EXPECT_EQ(reduce_min(b), 1.f);
  EXPECT_EQ(reduce_max(b), 4.f);
  EXPECT_EQ(sum(b), 7.f);
  // halfArea(d) = dx*dy + dx*dz + dy*dz, area = 2*halfArea
  EXPECT_NEAR(halfArea(b), 1.f * 2.f + 1.f * 4.f + 2.f * 4.f, 1e-5f);
  EXPECT_NEAR(area(b), 2.f * halfArea(b), 1e-5f);
}

TEST(Vec3f, comparison) {
  const Vec3f a(1.f, 2.f, 3.f), b(1.f, 5.f, 3.f);
  EXPECT_TRUE(a == Vec3f(1.f, 2.f, 3.f));
  EXPECT_TRUE(a != b);
  EXPECT_TRUE(a < b); // lexicographic: y decides
  EXPECT_FALSE(b < a);
  EXPECT_FALSE(a < a);
  EXPECT_TRUE(eq_mask(a, b).x);
  EXPECT_FALSE(eq_mask(a, b).y);
  EXPECT_TRUE(neq_mask(a, b).y);
  EXPECT_TRUE(lt_mask(a, b).y);
  EXPECT_TRUE(le_mask(a, b).z);
  EXPECT_TRUE(gt_mask(b, a).y);
  EXPECT_TRUE(ge_mask(b, a).z);
}

TEST(Vec3f, euclidean) {
  const Vec3f a(1.f, 2.f, 2.f); // length 3
  const Vec3f x(1.f, 0.f, 0.f), y(0.f, 1.f, 0.f);
  EXPECT_NEAR(dot(a, a), 9.f, 1e-5f);
  EXPECT_NEAR(sqr(a), 9.f, 1e-5f);
  EXPECT_NEAR(length(a), 3.f, 1e-5f);
  EXPECT_NEAR(rcp_length(a), 1.f / 3.f, 1e-4f);
  EXPECT_NEAR(distance(a, Vec3f(1.f, 2.f, 5.f)), 3.f, 1e-5f);
  EXPECT_NEAR(normalize(a).z, 2.f / 3.f, 1e-4f);
  // right-handed cross product
  const Vec3f z = cross(x, y);
  EXPECT_NEAR(z.x, 0.f, 1e-6f);
  EXPECT_NEAR(z.y, 0.f, 1e-6f);
  EXPECT_NEAR(z.z, 1.f, 1e-6f);
  // squared distance from (0,2,0) to the x axis is 4
  EXPECT_NEAR(sqr_point_to_line_distance(Vec3f(0.f, 2.f, 0.f), Vec3f(zero), x),
              4.f, 1e-4f);
  EXPECT_NEAR(sqr_point_to_line_distance(Vec3f(0.f, 2.f, 0.f), x), 4.f, 1e-4f);
  // stable_triangle_normal takes the three edge vectors of a triangle (which
  // sum to zero) and picks, per component, the better-conditioned of two cross
  // products. For the unit right triangle in the xy plane that is +z.
  const Vec3f n = stable_triangle_normal(x, y, Vec3f(-1.f, -1.f, 0.f));
  EXPECT_NEAR(n.x, 0.f, 1e-5f);
  EXPECT_NEAR(n.y, 0.f, 1e-5f);
  EXPECT_NEAR(n.z, 1.f, 1e-5f);
}

TEST(Vec3f, normalize_safe) {
  EXPECT_NEAR(normalize_safe(Vec3f(0.f, 3.f, 4.f)).z, 0.8f, 1e-3f);
  // a zero vector is returned unchanged instead of producing NaNs
  const Vec3f z = normalize_safe(Vec3f(zero));
  EXPECT_EQ(z.x, 0.f);
  EXPECT_EQ(z.y, 0.f);
  EXPECT_EQ(z.z, 0.f);
}

TEST(Vec3f, select_and_maxDim) {
  const Vec3f t(1.f, 2.f, 3.f), f(4.f, 5.f, 6.f);
  EXPECT_EQ(select(true, t, f).z, 3.f);
  EXPECT_EQ(select(false, t, f).z, 6.f);
  const Vec3f m = select(Vec3b(true, false, true), t, f);
  EXPECT_EQ(m.x, 1.f);
  EXPECT_EQ(m.y, 5.f);
  EXPECT_EQ(m.z, 3.f);
  // maxDim branches on scalar comparisons, so it is meaningful for scalar T
  // only
  EXPECT_EQ(maxDim(Vec3f(1.f, -5.f, 3.f)), 1);
  EXPECT_EQ(maxDim(Vec3f(9.f, -5.f, 3.f)), 0);
  EXPECT_EQ(maxDim(Vec3f(1.f, -5.f, 7.f)), 2);
}

////////////////////////////////////////////////////////////////////////////////
// Vec3i / Vec3b -- the other scalar instantiations
////////////////////////////////////////////////////////////////////////////////

TEST(Vec3i, arithmetic_and_reductions) {
  const Vec3i a(1, 2, 3), b(4, 8, 12);
  EXPECT_EQ((a + b).z, 15);
  EXPECT_EQ((a - b).y, -6);
  EXPECT_EQ((a * b).z, 36);
  EXPECT_EQ((b / a).z, 4);
  EXPECT_EQ(min(a, b).z, 3);
  EXPECT_EQ(max(a, b).z, 12);
  EXPECT_EQ((a << 2).z, 12);
  EXPECT_EQ((b >> 2).z, 3);
  EXPECT_EQ(reduce_add(a), 6);
  EXPECT_EQ(reduce_mul(a), 6);
  EXPECT_EQ(reduce_min(a), 1);
  EXPECT_EQ(reduce_max(a), 3);
  EXPECT_TRUE(a < b);
  EXPECT_EQ(Vec3i(zero).z, 0);
  EXPECT_EQ(Vec3i(one).z, 1);
}

TEST(Vec3b, logic) {
  const Vec3b a(true, false, true);
  EXPECT_TRUE(a.x);
  EXPECT_FALSE(a.y);
  EXPECT_TRUE(a[2]);
  EXPECT_TRUE(a == Vec3b(true, false, true));
  EXPECT_TRUE(a != Vec3b(true, true, true));
  EXPECT_TRUE(Vec3b(true).z);
  EXPECT_TRUE(eq_mask(Vec3i(1, 2, 3), Vec3i(1, 9, 3)) ==
              Vec3b(true, false, true));
}

////////////////////////////////////////////////////////////////////////////////
// Vec3<V> -- one instantiation per vector width the ISA defines
////////////////////////////////////////////////////////////////////////////////

#define ESIMD_VEC3_TESTS(suite, V)                                             \
                                                                               \
  TEST(suite, constructors_and_constants) {                                    \
    const Vec3<V> a(ramp<V>(1.f), ramp<V>(10.f), ramp<V>(20.f));               \
    expect_vec3(                                                               \
        a, [](int i) { return 1.f + i; }, [](int i) { return 10.f + i; },      \
        [](int i) { return 20.f + i; }, 0.f);                                  \
    const Vec3<V> b(ramp<V>(2.f)); /* explicit broadcast */                    \
    expect_vec3(                                                               \
        b, [](int i) { return 2.f + i; }, [](int i) { return 2.f + i; },       \
        [](int i) { return 2.f + i; }, 0.f);                                   \
    const Vec3<V> c(a); /* copy */                                             \
    expect_vec3(                                                               \
        c, [](int i) { return 1.f + i; }, [](int i) { return 10.f + i; },      \
        [](int i) { return 20.f + i; }, 0.f);                                  \
    expect_lanes(Vec3<V>(zero).x, [](int) { return 0.f; }, 0.f);               \
    expect_lanes(Vec3<V>(one).z, [](int) { return 1.f; }, 0.f);                \
    expect_lanes(a[2], [](int i) { return 20.f + i; }, 0.f); /* operator[] */  \
  }                                                                            \
                                                                               \
  TEST(suite, unary_and_binary) {                                              \
    const Vec3<V> a(ramp<V>(1.f), ramp<V>(10.f), ramp<V>(20.f));               \
    const Vec3<V> b(ramp<V>(3.f), ramp<V>(5.f), ramp<V>(7.f));                 \
    expect_vec3(                                                               \
        -a, [](int i) { return -(1.f + i); },                                  \
        [](int i) { return -(10.f + i); }, [](int i) { return -(20.f + i); },  \
        0.f);                                                                  \
    expect_vec3(                                                               \
        abs(-a), [](int i) { return 1.f + i; },                                \
        [](int i) { return 10.f + i; }, [](int i) { return 20.f + i; }, 0.f);  \
    expect_vec3(                                                               \
        sqrt(a), [](int i) { return std::sqrt(1.f + i); },                     \
        [](int i) { return std::sqrt(10.f + i); },                             \
        [](int i) { return std::sqrt(20.f + i); }, 1e-5f);                     \
    expect_vec3(                                                               \
        rcp(a), [](int i) { return 1.f / (1.f + i); },                         \
        [](int i) { return 1.f / (10.f + i); },                                \
        [](int i) { return 1.f / (20.f + i); }, 1e-5f);                        \
    expect_vec3(                                                               \
        rsqrt(a), [](int i) { return 1.f / std::sqrt(1.f + i); },              \
        [](int i) { return 1.f / std::sqrt(10.f + i); },                       \
        [](int i) { return 1.f / std::sqrt(20.f + i); }, 1e-4f);               \
    expect_vec3(                                                               \
        a + b, [](int i) { return 4.f + 2 * i; },                              \
        [](int i) { return 15.f + 2 * i; },                                    \
        [](int i) { return 27.f + 2 * i; }, 0.f);                              \
    expect_vec3(                                                               \
        a - b, [](int) { return -2.f; }, [](int) { return 5.f; },              \
        [](int) { return 13.f; }, 0.f);                                        \
    expect_vec3(                                                               \
        a *b, [](int i) { return (1.f + i) * (3.f + i); },                     \
        [](int i) { return (10.f + i) * (5.f + i); },                          \
        [](int i) { return (20.f + i) * (7.f + i); }, 1e-3f);                  \
    expect_vec3(                                                               \
        b / a, [](int i) { return (3.f + i) / (1.f + i); },                    \
        [](int i) { return (5.f + i) / (10.f + i); },                          \
        [](int i) { return (7.f + i) / (20.f + i); }, 1e-5f);                  \
    expect_vec3(                                                               \
        min(a, b), [](int i) { return std::min(1.f + i, 3.f + i); },           \
        [](int i) { return std::min(10.f + i, 5.f + i); },                     \
        [](int i) { return std::min(20.f + i, 7.f + i); }, 0.f);               \
    expect_vec3(                                                               \
        max(a, b), [](int i) { return std::max(1.f + i, 3.f + i); },           \
        [](int i) { return std::max(10.f + i, 5.f + i); },                     \
        [](int i) { return std::max(20.f + i, 7.f + i); }, 0.f);               \
  }                                                                            \
                                                                               \
  TEST(suite, ternary_fma) {                                                   \
    const Vec3<V> a(ramp<V>(1.f), ramp<V>(2.f), ramp<V>(3.f));                 \
    const Vec3<V> b(ramp<V>(4.f), ramp<V>(5.f), ramp<V>(6.f));                 \
    const Vec3<V> c(ramp<V>(7.f), ramp<V>(8.f), ramp<V>(9.f));                 \
    expect_vec3(                                                               \
        madd(a, b, c),                                                         \
        [](int i) { return (1.f + i) * (4.f + i) + (7.f + i); },               \
        [](int i) { return (2.f + i) * (5.f + i) + (8.f + i); },               \
        [](int i) { return (3.f + i) * (6.f + i) + (9.f + i); }, 1e-4f);       \
    expect_vec3(                                                               \
        msub(a, b, c),                                                         \
        [](int i) { return (1.f + i) * (4.f + i) - (7.f + i); },               \
        [](int i) { return (2.f + i) * (5.f + i) - (8.f + i); },               \
        [](int i) { return (3.f + i) * (6.f + i) - (9.f + i); }, 1e-4f);       \
    expect_vec3(                                                               \
        nmadd(a, b, c),                                                        \
        [](int i) { return -(1.f + i) * (4.f + i) + (7.f + i); },              \
        [](int i) { return -(2.f + i) * (5.f + i) + (8.f + i); },              \
        [](int i) { return -(3.f + i) * (6.f + i) + (9.f + i); }, 1e-4f);      \
    expect_vec3(                                                               \
        nmsub(a, b, c),                                                        \
        [](int i) { return -(1.f + i) * (4.f + i) - (7.f + i); },              \
        [](int i) { return -(2.f + i) * (5.f + i) - (8.f + i); },              \
        [](int i) { return -(3.f + i) * (6.f + i) - (9.f + i); }, 1e-4f);      \
    /* scalar-first overloads take a V, not a float */                         \
    const V s(2.f);                                                            \
    expect_vec3(                                                               \
        madd(s, b, c), [](int i) { return 2.f * (4.f + i) + (7.f + i); },      \
        [](int i) { return 2.f * (5.f + i) + (8.f + i); },                     \
        [](int i) { return 2.f * (6.f + i) + (9.f + i); }, 1e-4f);             \
    expect_vec3(                                                               \
        lerp(a, b, V(0.5f)),                                                   \
        [](int i) { return 0.5f * ((1.f + i) + (4.f + i)); },                  \
        [](int i) { return 0.5f * ((2.f + i) + (5.f + i)); },                  \
        [](int i) { return 0.5f * ((3.f + i) + (6.f + i)); }, 1e-4f);          \
  }                                                                            \
                                                                               \
  TEST(suite, assignment_and_reductions) {                                     \
    Vec3<V> a(ramp<V>(1.f), ramp<V>(2.f), ramp<V>(3.f));                       \
    a += Vec3<V>(V(1.f));                                                      \
    expect_vec3(                                                               \
        a, [](int i) { return 2.f + i; }, [](int i) { return 3.f + i; },       \
        [](int i) { return 4.f + i; }, 0.f);                                   \
    a -= Vec3<V>(V(1.f));                                                      \
    expect_vec3(                                                               \
        a, [](int i) { return 1.f + i; }, [](int i) { return 2.f + i; },       \
        [](int i) { return 3.f + i; }, 0.f);                                   \
    a *= V(2.f);                                                               \
    expect_vec3(                                                               \
        a, [](int i) { return 2.f * (1.f + i); },                              \
        [](int i) { return 2.f * (2.f + i); },                                 \
        [](int i) { return 2.f * (3.f + i); }, 0.f);                           \
    a /= V(2.f);                                                               \
    expect_vec3(                                                               \
        a, [](int i) { return 1.f + i; }, [](int i) { return 2.f + i; },       \
        [](int i) { return 3.f + i; }, 1e-5f);                                 \
    expect_lanes(                                                              \
        reduce_add(a),                                                         \
        [](int i) { return (1.f + i) + (2.f + i) + (3.f + i); }, 1e-5f);       \
    expect_lanes(                                                              \
        reduce_mul(a),                                                         \
        [](int i) { return (1.f + i) * (2.f + i) * (3.f + i); }, 1e-3f);       \
    expect_lanes(reduce_min(a), [](int i) { return 1.f + i; }, 0.f);           \
    expect_lanes(reduce_max(a), [](int i) { return 3.f + i; }, 0.f);           \
    expect_lanes(sum(a), [](int i) { return 3.f * (2.f + i); }, 1e-5f);        \
  }                                                                            \
                                                                               \
  TEST(suite, euclidean) {                                                     \
    const Vec3<V> a(V(1.f), V(2.f), V(2.f)); /* length 3 */                    \
    const Vec3<V> b(ramp<V>(1.f), ramp<V>(2.f), ramp<V>(3.f));                 \
    expect_lanes(dot(a, a), [](int) { return 9.f; }, 1e-4f);                   \
    expect_lanes(sqr(a), [](int) { return 9.f; }, 1e-4f);                      \
    expect_lanes(length(a), [](int) { return 3.f; }, 1e-4f);                   \
    expect_lanes(rcp_length(a), [](int) { return 1.f / 3.f; }, 1e-4f);         \
    expect_lanes(distance(a, a), [](int) { return 0.f; }, 1e-5f);              \
    expect_vec3(                                                               \
        normalize(a), [](int) { return 1.f / 3.f; },                           \
        [](int) { return 2.f / 3.f; }, [](int) { return 2.f / 3.f; }, 1e-3f);  \
    /* cross(b,b) is zero regardless of lane */                                \
    expect_vec3(                                                               \
        cross(b, b), [](int) { return 0.f; }, [](int) { return 0.f; },         \
        [](int) { return 0.f; }, 1e-3f);                                       \
    /* x cross y = z, per lane */                                              \
    expect_vec3(                                                               \
        cross(Vec3<V>(V(1.f), V(0.f), V(0.f)),                                 \
              Vec3<V>(V(0.f), V(1.f), V(0.f))),                                \
        [](int) { return 0.f; }, [](int) { return 0.f; },                      \
        [](int) { return 1.f; }, 1e-6f);                                       \
    expect_lanes(                                                              \
        halfArea(b),                                                           \
        [](int i) {                                                            \
          const float x = 1.f + i, y = 2.f + i, z = 3.f + i;                   \
          return x * y + x * z + y * z;                                        \
        },                                                                     \
        1e-3f);                                                                \
  }                                                                            \
                                                                               \
  TEST(suite, select_and_masks) {                                              \
    const Vec3<V> t(V(1.f), V(2.f), V(3.f));                                   \
    const Vec3<V> f(V(4.f), V(5.f), V(6.f));                                   \
    expect_vec3(                                                               \
        select(true, t, f), [](int) { return 1.f; }, [](int) { return 2.f; },  \
        [](int) { return 3.f; }, 0.f);                                         \
    expect_vec3(                                                               \
        select(Vec3b(true, false, true), t, f), [](int) { return 1.f; },       \
        [](int) { return 5.f; }, [](int) { return 3.f; }, 0.f);                \
    /* lane mask: even lanes take t, odd lanes take f */                       \
    float sel[V::size];                                                        \
    for (int i = 0; i < V::size; ++i)                                          \
      sel[i] = (i % 2 == 0) ? 0.f : 1.f;                                       \
    const typename V::Bool m = V::loadu(sel) < V(0.5f);                        \
    const Vec3<V> s = select(m, t, f);                                         \
    for (int i = 0; i < V::size; ++i)                                          \
      EXPECT_EQ(s.z[i], (i % 2 == 0) ? 3.f : 6.f) << "at lane " << i;          \
    /* shift_right_1 drops lane 0 and shifts the rest down */                  \
    const Vec3<V> sr = shift_right_1(Vec3<V>(ramp<V>(1.f)));                   \
    for (int i = 0; i + 1 < V::size; ++i)                                      \
      EXPECT_EQ(sr.x[i], 2.f + i) << "at lane " << i;                          \
  }                                                                            \
                                                                               \
  TEST(suite, normalize_safe_and_zero_fix) {                                   \
    expect_vec3(                                                               \
        normalize_safe(Vec3<V>(V(0.f), V(3.f), V(4.f))),                       \
        [](int) { return 0.f; }, [](int) { return 0.6f; },                     \
        [](int) { return 0.8f; }, 1e-3f);                                      \
    /* the all-zero vector comes back unchanged rather than as NaNs */         \
    expect_vec3(                                                               \
        normalize_safe(Vec3<V>(zero)), [](int) { return 0.f; },                \
        [](int) { return 0.f; }, [](int) { return 0.f; }, 0.f);                \
    for (int i = 0; i < V::size; ++i)                                          \
      EXPECT_TRUE(std::isfinite(rcp_safe(Vec3<V>(zero)).x[i]))                 \
          << "at lane " << i;                                                  \
  }

ESIMD_VEC3_TESTS(Vec3_vfloat4, vfloat4)
#if defined(__AVX__)
ESIMD_VEC3_TESTS(Vec3_vfloat8, vfloat8)
#endif
#if defined(__AVX512F__)
ESIMD_VEC3_TESTS(Vec3_vfloat16, vfloat16)
#endif

// vec3.h only defines the Vec3-level shuffle<i0,i1,i2,i3> and the broadcast
// specializations at the 4- and 8-wide widths, so this second macro stops there
// rather than following the widths above.
#define ESIMD_VEC3_SHUFFLE_TESTS(suite, V)                                     \
  TEST(suite, broadcast_and_shuffle) {                                         \
    const Vec3<V> a(ramp<V>(1.f), ramp<V>(10.f), ramp<V>(20.f));               \
    /* broadcast<V,V> splats lane k across every lane */                       \
    const Vec3<V> b = broadcast<V, V>(a, 1);                                   \
    expect_vec3(                                                               \
        b, [](int) { return 2.f; }, [](int) { return 11.f; },                  \
        [](int) { return 21.f; }, 0.f);                                        \
    /* shuffle acts on each component independently */                         \
    const Vec3<V> s = shuffle<1, 0, 3, 2>(a);                                  \
    for (int i = 0; i < V::size; i += 4) {                                     \
      EXPECT_EQ(s.x[i + 0], a.x[i + 1]) << "at lane " << i;                    \
      EXPECT_EQ(s.z[i + 3], a.z[i + 2]) << "at lane " << i;                    \
    }                                                                          \
  }

ESIMD_VEC3_SHUFFLE_TESTS(Vec3_vfloat4, vfloat4)
#if defined(__AVX__)
ESIMD_VEC3_SHUFFLE_TESTS(Vec3_vfloat8, vfloat8)
#endif

////////////////////////////////////////////////////////////////////////////////
// Vec3ba -- the SSE mask companion
////////////////////////////////////////////////////////////////////////////////

TEST(Vec3ba, construction_and_logic) {
  const Vec3ba t(True), f(False);
  EXPECT_TRUE(all(t));
  EXPECT_FALSE(any(f));
  EXPECT_TRUE(none(f));
  const Vec3ba m(true, false, true);
  EXPECT_TRUE(any(m));
  EXPECT_FALSE(all(m));
  EXPECT_EQ(movemask(m), size_t(0x5)); // only the low three lanes count
  EXPECT_TRUE(all(!f));
  EXPECT_TRUE(all(m | !m));
  EXPECT_TRUE(none(m & !m));
  EXPECT_TRUE(all(m ^ !m));
  EXPECT_TRUE(reduce_or(m));
  EXPECT_FALSE(reduce_and(m));
  EXPECT_TRUE(m == Vec3ba(true, false, true));
  EXPECT_TRUE(m != t);
}

////////////////////////////////////////////////////////////////////////////////
// Vec3ia -- the SSE int companion
////////////////////////////////////////////////////////////////////////////////

TEST(Vec3ia, arithmetic) {
  const Vec3ia a(1, 2, 3), b(4, 8, 12);
  EXPECT_EQ((a + b).z, 15);
  EXPECT_EQ((a - b).y, -6);
  EXPECT_EQ((-a).x, -1);
  EXPECT_EQ(abs(Vec3ia(-1, 2, -3)).z, 3);
  EXPECT_EQ((a + 1).z, 4);
  EXPECT_EQ((b - 1).x, 3);
  EXPECT_EQ(Vec3ia(zero).z, 0);
  EXPECT_EQ(Vec3ia(one).z, 1);
  EXPECT_EQ(a[2], 3);
#if defined(ESIMD_ARM64) || defined(__SSE4_1__)
  EXPECT_EQ((a * b).z, 36);
  Vec3ia m(1, 2, 3);
  m *= 2;
  EXPECT_EQ(m.z, 6);
#endif
}

TEST(Vec3ia, bitwise_and_shifts) {
  const Vec3ia a(0xF0, 0x0F, 0xFF);
  EXPECT_EQ((a & Vec3ia(0x11, 0x11, 0x11)).x, 0x10);
  EXPECT_EQ((a | Vec3ia(0x01, 0x01, 0x01)).x, 0xF1);
  EXPECT_EQ((a ^ a).z, 0);
  EXPECT_EQ(sll(Vec3ia(1, 2, 3), 2).z, 12);
  EXPECT_EQ(srl(Vec3ia(4, 8, 12), 2).z, 3);
  EXPECT_EQ(sra(Vec3ia(-8, 8, 16), 2).x, -2);
  Vec3ia m(0xF0, 0x0F, 0xFF);
  m &= Vec3ia(0x0F, 0x0F, 0x0F);
  EXPECT_EQ(m.z, 0x0F);
  m |= 0xF0;
  EXPECT_EQ(m.z, 0xFF);
}

TEST(Vec3ia, reductions_and_comparison) {
  const Vec3ia a(1, 2, 3), b(1, 5, 3);
  EXPECT_EQ(reduce_add(a), 6);
  EXPECT_EQ(reduce_mul(a), 6);
  EXPECT_EQ(reduce_min(a), 1);
  EXPECT_EQ(reduce_max(a), 3);
  EXPECT_TRUE(a == Vec3ia(1, 2, 3));
  EXPECT_TRUE(a != b);
  EXPECT_TRUE(a < b); // lexicographic
  EXPECT_TRUE(all(eq_mask(a, Vec3ia(1, 2, 3))));
  EXPECT_TRUE(any(lt_mask(a, b)));
  EXPECT_TRUE(any(gt_mask(b, a)));
  EXPECT_EQ(min(a, b).y, 2);
  EXPECT_EQ(max(a, b).y, 5);
  EXPECT_EQ(select(Vec3ba(true, false, true), a, b).y, 5);
}

////////////////////////////////////////////////////////////////////////////////
// Vec3fa -- the SSE float companion (three floats in one __m128, lane 3
// padding)
////////////////////////////////////////////////////////////////////////////////

TEST(Vec3fa, construction_and_constants) {
  const Vec3fa a(1.f, 2.f, 3.f);
  EXPECT_EQ(a.x, 1.f);
  EXPECT_EQ(a.z, 3.f);
  EXPECT_EQ(Vec3fa(4.f).z, 4.f); // explicit broadcast
  EXPECT_EQ(Vec3fa(a).y, 2.f);   // copy
  EXPECT_EQ(Vec3fa(zero).z, 0.f);
  EXPECT_EQ(Vec3fa(one).z, 1.f);
  EXPECT_TRUE(std::isinf(Vec3fa(pos_inf).z));
  EXPECT_EQ(a[0], 1.f);
  EXPECT_EQ(a[2], 3.f);
  Vec3fa m(zero);
  m[1] = 7.f;
  EXPECT_EQ(m.y, 7.f);
}

TEST(Vec3fa, conversion_roundtrip) {
  const Vec3f s(1.f, 2.f, 3.f);
  const Vec3fa a(s); // Vec3<float> -> Vec3fa
  EXPECT_EQ(a.x, 1.f);
  EXPECT_EQ(a.z, 3.f);
  const Vec3f back(a); // Vec3fa -> Vec3<float>
  EXPECT_EQ(back.y, 2.f);
  EXPECT_TRUE(back == s);
  // broadcast into the vector instantiations
  const Vec3<vfloat4> v(a);
  for (int i = 0; i < vfloat4::size; ++i)
    EXPECT_EQ(v.z[i], 3.f) << "at lane " << i;
  // explicit casts to the neighbouring types
  EXPECT_EQ(Vec3ia(a).z, 3);
  EXPECT_EQ(Vec2fa(a).y, 2.f);
  EXPECT_EQ(vfloat4(a)[2], 3.f);
}

TEST(Vec3fa, load_store) {
  __aligned(16) const float src[4] = {1.f, 2.f, 3.f, 99.f};
  const Vec3fa a = Vec3fa::load(src);
  EXPECT_EQ(a.x, 1.f);
  EXPECT_EQ(a.z, 3.f);
  // load masks the padding lane away; loadu keeps whatever is there
  EXPECT_EQ(vfloat4(a)[3], 0.f);
  const Vec3fa b = Vec3fa::loadu(src);
  EXPECT_EQ(b.z, 3.f);

  float dst[4] = {0.f, 0.f, 0.f, 0.f};
  Vec3fa::storeu(dst, Vec3fa(7.f, 8.f, 9.f));
  EXPECT_EQ(dst[0], 7.f);
  EXPECT_EQ(dst[2], 9.f);
}

TEST(Vec3fa, unary) {
  const Vec3fa a(-1.5f, 2.5f, -4.f);
  EXPECT_EQ((+a).x, -1.5f);
  EXPECT_EQ((-a).z, 4.f);
  EXPECT_EQ(abs(a).z, 4.f);
  EXPECT_EQ(sign(a).x, -1.f);
  EXPECT_EQ(sign(a).y, 1.f);
  EXPECT_NEAR(sqrt(Vec3fa(4.f, 9.f, 16.f)).z, 4.f, 1e-5f);
  /* NB: sqr(Vec3fa) is component-wise, while sqr(Vec3<T>) is dot(a,a) */
  EXPECT_NEAR(sqr(Vec3fa(2.f, 3.f, 4.f)).z, 16.f, 1e-4f);
  EXPECT_NEAR(rcp(Vec3fa(2.f, 4.f, 8.f)).z, 0.125f, 1e-5f);
  EXPECT_NEAR(rsqrt(Vec3fa(4.f, 16.f, 64.f)).z, 0.125f, 1e-4f);
  EXPECT_NEAR(log(Vec3fa(1.f, float(M_E), 1.f)).y, 1.f, 1e-5f);
  EXPECT_NEAR(exp(Vec3fa(0.f, 1.f, 0.f)).y, float(M_E), 1e-5f);
  EXPECT_TRUE(std::isfinite(rcp_safe(Vec3fa(0.f, 2.f, 4.f)).x));
  EXPECT_NEAR(rcp_safe(Vec3fa(0.f, 2.f, 4.f)).z, 0.25f, 1e-5f);
  EXPECT_NE(zero_fix(Vec3fa(0.f, 1.f, 2.f)).x, 0.f);
}

TEST(Vec3fa, binary_and_assignment) {
  const Vec3fa a(1.f, 2.f, 3.f), b(4.f, 8.f, 12.f);
  EXPECT_EQ((a + b).x, 5.f);
  EXPECT_EQ((a - b).y, -6.f);
  EXPECT_EQ((a * b).z, 36.f);
  EXPECT_EQ((a * 2.f).x, 2.f);
  EXPECT_EQ((2.f * a).y, 4.f);
  EXPECT_EQ((b / a).z, 4.f);
  EXPECT_EQ((b / 2.f).x, 2.f);
  EXPECT_EQ((12.f / b).z, 1.f);
  EXPECT_EQ(min(a, b).z, 3.f);
  EXPECT_EQ(max(a, b).z, 12.f);
  EXPECT_NEAR(pow(Vec3fa(2.f, 3.f, 4.f), 2.f).z, 16.f, 1e-5f);

  Vec3fa m(1.f, 2.f, 3.f);
  m += Vec3fa(1.f, 1.f, 1.f);
  EXPECT_EQ(m.z, 4.f);
  m -= Vec3fa(1.f, 1.f, 1.f);
  EXPECT_EQ(m.z, 3.f);
  m *= 2.f;
  EXPECT_EQ(m.z, 6.f);
  m /= 2.f;
  EXPECT_EQ(m.z, 3.f);
}

TEST(Vec3fa, ternary_and_reductions) {
  const Vec3fa a(1.f, 2.f, 3.f), b(4.f, 5.f, 6.f), c(7.f, 8.f, 9.f);
  EXPECT_NEAR(madd(a, b, c).z, 3.f * 6.f + 9.f, 1e-5f);
  EXPECT_NEAR(msub(a, b, c).z, 3.f * 6.f - 9.f, 1e-5f);
  EXPECT_NEAR(nmadd(a, b, c).z, -3.f * 6.f + 9.f, 1e-5f);
  EXPECT_NEAR(nmsub(a, b, c).z, -3.f * 6.f - 9.f, 1e-5f);
  EXPECT_NEAR(madd(2.f, b, c).x, 2.f * 4.f + 7.f, 1e-5f);
  EXPECT_NEAR(lerp(a, b, 0.5f).z, 4.5f, 1e-5f);

  const Vec3fa d(1.f, 2.f, 4.f);
  // the reductions must ignore the padding lane
  EXPECT_NEAR(reduce_add(d), 7.f, 1e-5f);
  EXPECT_EQ(reduce_mul(d), 8.f);
  EXPECT_EQ(reduce_min(d), 1.f);
  EXPECT_EQ(reduce_max(d), 4.f);
  EXPECT_NEAR(halfArea(d), 1.f * 2.f + 1.f * 4.f + 2.f * 4.f, 1e-5f);
  EXPECT_NEAR(area(d), 2.f * halfArea(d), 1e-5f);
}

TEST(Vec3fa, comparison) {
  const Vec3fa a(1.f, 2.f, 3.f), b(1.f, 5.f, 3.f);
  EXPECT_TRUE(a == Vec3fa(1.f, 2.f, 3.f));
  EXPECT_TRUE(a != b);
  EXPECT_TRUE(all(eq_mask(a, Vec3fa(1.f, 2.f, 3.f))));
  EXPECT_TRUE(any(neq_mask(a, b)));
  EXPECT_TRUE(any(lt_mask(a, b)));
  EXPECT_TRUE(all(le_mask(a, b)));
  EXPECT_TRUE(any(gt_mask(b, a)));
  EXPECT_TRUE(all(ge_mask(b, a)));
  EXPECT_TRUE(isvalid(a));
  EXPECT_TRUE(is_finite(a));
  EXPECT_TRUE(isvalid4(a));
  EXPECT_TRUE(is_finite4(a));
  EXPECT_FALSE(is_finite(Vec3fa(pos_inf)));
}

TEST(Vec3fa, euclidean) {
  const Vec3fa a(1.f, 2.f, 2.f); // length 3
  const Vec3fa x(1.f, 0.f, 0.f), y(0.f, 1.f, 0.f);
  EXPECT_NEAR(dot(a, a), 9.f, 1e-4f);
  EXPECT_NEAR(sqr_length(a), 9.f, 1e-4f);
  EXPECT_NEAR(length(a), 3.f, 1e-4f);
  EXPECT_NEAR(rcp_length(a), 1.f / 3.f, 1e-4f);
  EXPECT_NEAR(rcp_length2(a), 1.f / 9.f, 1e-4f);
  EXPECT_NEAR(distance(a, Vec3fa(1.f, 2.f, 5.f)), 3.f, 1e-4f);
  EXPECT_NEAR(normalize(a).z, 2.f / 3.f, 1e-3f);
  const Vec3fa z = cross(x, y);
  EXPECT_NEAR(z.x, 0.f, 1e-6f);
  EXPECT_NEAR(z.z, 1.f, 1e-6f);
  EXPECT_NEAR(normalize_safe(Vec3fa(0.f, 3.f, 4.f)).z, 0.8f, 1e-3f);
  EXPECT_TRUE(normalize_safe(Vec3fa(zero)) == Vec3fa(zero));
  // dnormalize is the derivative of normalize; it is orthogonal to p
  EXPECT_NEAR(dot(dnormalize(a, Vec3fa(1.f, 0.f, 0.f)), a), 0.f, 1e-4f);
}

TEST(Vec3fa, select_rounding_maxDim) {
  const Vec3fa t(1.f, 2.f, 3.f), f(4.f, 5.f, 6.f);
  EXPECT_EQ(select(true, t, f).z, 3.f);
  EXPECT_EQ(select(false, t, f).z, 6.f);
  const Vec3fa m = select(Vec3ba(true, false, true), t, f);
  EXPECT_EQ(m.x, 1.f);
  EXPECT_EQ(m.y, 5.f);
  EXPECT_EQ(m.z, 3.f);
  EXPECT_EQ(floor(Vec3fa(1.7f, -1.2f, 2.5f)).x, 1.f);
  EXPECT_EQ(floor(Vec3fa(1.7f, -1.2f, 2.5f)).y, -2.f);
  EXPECT_EQ(ceil(Vec3fa(1.2f, -1.7f, 2.5f)).x, 2.f);
  EXPECT_EQ(ceil(Vec3fa(1.2f, -1.7f, 2.5f)).y, -1.f);
  // trunc is inconsistent upstream and the port keeps that: on ARM it is
  // vrndq_f32 (round toward zero) but on x86 it is _mm_round_ps with
  // _MM_FROUND_TO_NEAREST_INT, which rounds. Only the fractions where the two
  // agree are asserted here; see the README.
  EXPECT_EQ(trunc(Vec3fa(1.4f, -1.4f, 0.25f)).x, 1.f);
  EXPECT_EQ(trunc(Vec3fa(1.4f, -1.4f, 0.25f)).y, -1.f);
  EXPECT_EQ(trunc(Vec3fa(1.4f, -1.4f, 0.25f)).z, 0.f);
  EXPECT_EQ(maxDim(Vec3fa(1.f, -5.f, 3.f)), 1);
  EXPECT_EQ(maxDim(Vec3fa(9.f, -5.f, 3.f)), 0);
  EXPECT_EQ(maxDim(Vec3fa(1.f, -5.f, 7.f)), 2);
}

#if defined(ESIMD_ARM64) || defined(__SSE4_1__)
TEST(Vec3fa, mini_maxi) {
  // integer min/max on the float bit patterns: for positive floats the ordering
  // agrees with the float ordering
  const Vec3fa a(1.f, 8.f, 3.f), b(4.f, 2.f, 3.f);
  EXPECT_EQ(mini(a, b).x, 1.f);
  EXPECT_EQ(mini(a, b).y, 2.f);
  EXPECT_EQ(maxi(a, b).x, 4.f);
  EXPECT_EQ(maxi(a, b).y, 8.f);
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Vec3fx / Vec3ff -- Vec3fa with the fourth lane exposed as w / a / u
////////////////////////////////////////////////////////////////////////////////

TEST(Vec3fx, fourth_lane) {
  const Vec3fx a(1.f, 2.f, 3.f, 4.f);
  EXPECT_EQ(a.x, 1.f);
  EXPECT_EQ(a.z, 3.f);
  EXPECT_EQ(a.w, 4.f);
  // the same storage reinterpreted as int / unsigned
  const Vec3fx b(Vec3fa(1.f, 2.f, 3.f), 42);
  EXPECT_EQ(b.a, 42);
  const Vec3fx c(Vec3fa(1.f, 2.f, 3.f), 7u);
  EXPECT_EQ(c.u, 7u);
  const Vec3fx d(Vec3fa(1.f, 2.f, 3.f), 5.f);
  EXPECT_EQ(d.w, 5.f);
  EXPECT_EQ(d.z, 3.f);
  // Vec3fx converts back to Vec3fa, dropping the payload
  const Vec3fa e = d;
  EXPECT_EQ(e.z, 3.f);
  // Vec3ff is the documented alias
  const Vec3ff g(1.f, 2.f, 3.f, 4.f);
  EXPECT_EQ(g.w, 4.f);
  EXPECT_EQ(Vec3fx(zero).w, 0.f);
  EXPECT_EQ(a[2], 3.f);
}

TEST(Vec3fx, arithmetic) {
  const Vec3fx a(1.f, 2.f, 3.f, 10.f), b(4.f, 8.f, 12.f, 20.f);
  EXPECT_EQ((a + b).z, 15.f);
  EXPECT_EQ((a - b).y, -6.f);
  EXPECT_EQ((a * b).z, 36.f);
  EXPECT_EQ((b / a).z, 4.f);
  EXPECT_EQ(min(a, b).z, 3.f);
  EXPECT_EQ(max(a, b).z, 12.f);
  EXPECT_EQ(abs(Vec3fx(-1.f, 2.f, -3.f, 0.f)).z, 3.f);
  EXPECT_EQ(sign(Vec3fx(-1.f, 2.f, -3.f, 0.f)).x, -1.f);
  EXPECT_NEAR(dot(a, b), 1.f * 4.f + 2.f * 8.f + 3.f * 12.f, 1e-3f);
  EXPECT_NEAR(length(Vec3fx(1.f, 2.f, 2.f, 99.f)), 3.f, 1e-4f);
  EXPECT_EQ(reduce_add(Vec3fx(1.f, 2.f, 4.f, 99.f)), 7.f);
  EXPECT_TRUE(a == Vec3fx(1.f, 2.f, 3.f, 10.f));
  EXPECT_EQ(select(Vec3ba(true, false, true), a, b).y, 8.f);
  EXPECT_EQ(maxDim(Vec3fx(1.f, -5.f, 3.f, 0.f)), 1);
}
