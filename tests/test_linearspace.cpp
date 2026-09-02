// Copyright 2026 notchla liso.lorenzo@gmail.com
// SPDX-License-Identifier: Apache-2.0
//
// Correctness tests for LinearSpace2<T> / LinearSpace3<T> from the optional
// data-types layer (<esimd/types.h>) -- the 2x2 and 3x3 matrices embree builds
// out of Vec2/Vec3 column vectors. Checked at T = Vec2f/Vec3f, at the SSE-backed
// Vec2fa/Vec3fa (LinearSpace3<Vec3fa> has its own transposed() specialization),
// and at every vector width the active ISA defines. Compiled once per ISA.

#include "test_helpers.h"
#include <esimd/types.h>

using namespace esimd;
using namespace esimd_test;

namespace {

template <typename V> V ramp(float base) {
  float tmp[V::size];
  for (int i = 0; i < V::size; ++i)
    tmp[i] = base + float(i);
  return V::loadu(tmp);
}

template <typename V, typename F>
void expect_lanes(const V &v, F ref, float tol) {
  for (int i = 0; i < V::size; ++i)
    EXPECT_NEAR(v[i], ref(i), tol) << "at lane " << i;
}

// A 3x3 identity check that tolerates rounding, for inverse()/orthogonal().
void expect_identity3(const LinearSpace3f &m, float tol) {
  EXPECT_NEAR(m.vx.x, 1.f, tol);  EXPECT_NEAR(m.vx.y, 0.f, tol);
  EXPECT_NEAR(m.vx.z, 0.f, tol);  EXPECT_NEAR(m.vy.x, 0.f, tol);
  EXPECT_NEAR(m.vy.y, 1.f, tol);  EXPECT_NEAR(m.vy.z, 0.f, tol);
  EXPECT_NEAR(m.vz.x, 0.f, tol);  EXPECT_NEAR(m.vz.y, 0.f, tol);
  EXPECT_NEAR(m.vz.z, 1.f, tol);
}

} // namespace

////////////////////////////////////////////////////////////////////////////////
// LinearSpace2f
////////////////////////////////////////////////////////////////////////////////

TEST(LinearSpace2f, construction_and_accessors) {
  // column vectors
  const LinearSpace2f a(Vec2f(1.f, 2.f), Vec2f(3.f, 4.f));
  EXPECT_TRUE(a.vx == Vec2f(1.f, 2.f));
  EXPECT_TRUE(a.vy == Vec2f(3.f, 4.f));
  // row-major data: (m00 m01 / m10 m11) -> columns (m00,m10) and (m01,m11)
  const LinearSpace2f b(1.f, 3.f, 2.f, 4.f);
  EXPECT_TRUE(b == a);
  EXPECT_TRUE(LinearSpace2f(a) == a); // copy
  // rows read back the row-major order
  EXPECT_TRUE(a.row0() == Vec2f(1.f, 3.f));
  EXPECT_TRUE(a.row1() == Vec2f(2.f, 4.f));
  // constants
  EXPECT_TRUE(LinearSpace2f(zero).vx == Vec2f(0.f, 0.f));
  const LinearSpace2f id(one);
  EXPECT_TRUE(id.vx == Vec2f(1.f, 0.f));
  EXPECT_TRUE(id.vy == Vec2f(0.f, 1.f));
}

TEST(LinearSpace2f, det_inverse_transpose) {
  const LinearSpace2f a(Vec2f(1.f, 2.f), Vec2f(3.f, 4.f));
  EXPECT_NEAR(a.det(), 1.f * 4.f - 3.f * 2.f, 1e-5f); // -2
  // adjoint / det == inverse
  const LinearSpace2f inv = a.inverse();
  const LinearSpace2f id = a * inv;
  EXPECT_NEAR(id.vx.x, 1.f, 1e-5f);
  EXPECT_NEAR(id.vx.y, 0.f, 1e-5f);
  EXPECT_NEAR(id.vy.x, 0.f, 1e-5f);
  EXPECT_NEAR(id.vy.y, 1.f, 1e-5f);
  EXPECT_TRUE(rcp(a) == inv); // rcp is a synonym for inverse
  // transposing swaps the off-diagonal
  const LinearSpace2f t = a.transposed();
  EXPECT_TRUE(t.vx == Vec2f(1.f, 3.f));
  EXPECT_TRUE(t.vy == Vec2f(2.f, 4.f));
  EXPECT_TRUE(t.transposed() == a);
}

TEST(LinearSpace2f, scale_rotate_orthogonal) {
  const LinearSpace2f s = LinearSpace2f::scale(Vec2f(2.f, 3.f));
  EXPECT_TRUE(s * Vec2f(1.f, 1.f) == Vec2f(2.f, 3.f));
  EXPECT_NEAR(s.det(), 6.f, 1e-5f);
  // a quarter turn maps +x to +y
  const LinearSpace2f r = LinearSpace2f::rotate(float(M_PI) * 0.5f);
  const Vec2f v = r * Vec2f(1.f, 0.f);
  EXPECT_NEAR(v.x, 0.f, 1e-6f);
  EXPECT_NEAR(v.y, 1.f, 1e-6f);
  EXPECT_NEAR(r.det(), 1.f, 1e-5f); // rotations preserve area
  // orthogonal() of a rotation is that rotation again
  const LinearSpace2f o = r.orthogonal();
  EXPECT_NEAR(o.vx.x, r.vx.x, 1e-4f);
  EXPECT_NEAR(o.vy.y, r.vy.y, 1e-4f);
  // and of a scaled rotation it strips the scale
  EXPECT_NEAR((2.f * r).orthogonal().det(), 1.f, 1e-4f);
}

TEST(LinearSpace2f, operators) {
  const LinearSpace2f a(Vec2f(1.f, 2.f), Vec2f(3.f, 4.f));
  const LinearSpace2f b(Vec2f(5.f, 6.f), Vec2f(7.f, 8.f));
  EXPECT_TRUE((-a).vx == Vec2f(-1.f, -2.f));
  EXPECT_TRUE((+a) == a);
  EXPECT_TRUE((a + b).vx == Vec2f(6.f, 8.f));
  EXPECT_TRUE((a - b).vy == Vec2f(-4.f, -4.f));
  EXPECT_TRUE((2.f * a).vx == Vec2f(2.f, 4.f));
  EXPECT_TRUE((a / 2.f).vy == Vec2f(1.5f, 2.f));
  // matrix * vector applies the columns
  EXPECT_TRUE(a * Vec2f(1.f, 0.f) == Vec2f(1.f, 2.f));
  EXPECT_TRUE(a * Vec2f(0.f, 1.f) == Vec2f(3.f, 4.f));
  // matrix * matrix composes: (a*b) applied to v == a applied to (b applied to v)
  const Vec2f v(2.f, -1.f);
  const Vec2f lhs = (a * b) * v;
  const Vec2f rhs = a * (b * v);
  EXPECT_NEAR(lhs.x, rhs.x, 1e-4f);
  EXPECT_NEAR(lhs.y, rhs.y, 1e-4f);
  // identity is the multiplicative unit
  EXPECT_TRUE(a * LinearSpace2f(one) == a);

  LinearSpace2f m = a;
  m *= LinearSpace2f(one);
  EXPECT_TRUE(m == a);
  m /= LinearSpace2f(one);
  EXPECT_TRUE(m == a);
  EXPECT_TRUE(a != b);
}

TEST(LinearSpace2fa, sse_backed) {
  const LinearSpace2fa a(Vec2fa(1.f, 2.f), Vec2fa(3.f, 4.f));
  EXPECT_NEAR(a.det(), -2.f, 1e-5f);
  EXPECT_TRUE(a.transposed().vx == Vec2fa(1.f, 3.f));
  const Vec2fa v = a * Vec2fa(1.f, 0.f);
  EXPECT_EQ(v.x, 1.f);
  EXPECT_EQ(v.y, 2.f);
  EXPECT_TRUE(LinearSpace2fa(one).vy == Vec2fa(0.f, 1.f));
  // the converting constructor bridges the two backings
  const LinearSpace2f b(a);
  EXPECT_NEAR(b.det(), -2.f, 1e-5f);
}

////////////////////////////////////////////////////////////////////////////////
// LinearSpace3f
////////////////////////////////////////////////////////////////////////////////

TEST(LinearSpace3f, construction_and_accessors) {
  const LinearSpace3f a(Vec3f(1.f, 2.f, 3.f), Vec3f(4.f, 5.f, 6.f),
                        Vec3f(7.f, 8.f, 9.f));
  EXPECT_TRUE(a.vz == Vec3f(7.f, 8.f, 9.f));
  // row-major data
  const LinearSpace3f b(1.f, 4.f, 7.f, 2.f, 5.f, 8.f, 3.f, 6.f, 9.f);
  EXPECT_TRUE(b == a);
  EXPECT_TRUE(a.row0() == Vec3f(1.f, 4.f, 7.f));
  EXPECT_TRUE(a.row1() == Vec3f(2.f, 5.f, 8.f));
  EXPECT_TRUE(a.row2() == Vec3f(3.f, 6.f, 9.f));
  EXPECT_TRUE(LinearSpace3f(zero).vz == Vec3f(zero));
  const LinearSpace3f id(one);
  EXPECT_TRUE(id.vx == Vec3f(1.f, 0.f, 0.f));
  EXPECT_TRUE(id.vz == Vec3f(0.f, 0.f, 1.f));
}

TEST(LinearSpace3f, det_inverse_transpose) {
  const LinearSpace3f a(Vec3f(2.f, 0.f, 0.f), Vec3f(0.f, 3.f, 0.f),
                        Vec3f(0.f, 0.f, 4.f));
  EXPECT_NEAR(a.det(), 24.f, 1e-4f);
  expect_identity3(a * a.inverse(), 1e-5f);
  expect_identity3(a * rcp(a), 1e-5f);

  const LinearSpace3f b(Vec3f(1.f, 2.f, 3.f), Vec3f(4.f, 5.f, 6.f),
                        Vec3f(7.f, 8.f, 10.f));
  EXPECT_NEAR(b.det(), -3.f, 1e-3f);
  expect_identity3(b * b.inverse(), 1e-4f);
  // transposed() and the free transposed() agree, and it is an involution
  EXPECT_TRUE(b.transposed() == transposed(b));
  EXPECT_TRUE(b.transposed().vx == Vec3f(1.f, 4.f, 7.f));
  EXPECT_TRUE(b.transposed().transposed() == b);
}

TEST(LinearSpace3f, scale_rotate) {
  const LinearSpace3f s = LinearSpace3f::scale(Vec3f(2.f, 3.f, 4.f));
  EXPECT_TRUE(s * Vec3f(1.f, 1.f, 1.f) == Vec3f(2.f, 3.f, 4.f));
  EXPECT_NEAR(s.det(), 24.f, 1e-4f);
  // a quarter turn about +z maps +x to +y
  const LinearSpace3f r =
      LinearSpace3f::rotate(Vec3f(0.f, 0.f, 1.f), float(M_PI) * 0.5f);
  const Vec3f v = r * Vec3f(1.f, 0.f, 0.f);
  EXPECT_NEAR(v.x, 0.f, 1e-6f);
  EXPECT_NEAR(v.y, 1.f, 1e-6f);
  EXPECT_NEAR(v.z, 0.f, 1e-6f);
  EXPECT_NEAR(r.det(), 1.f, 1e-5f);
  // a rotation's inverse is its transpose
  expect_identity3(r * r.transposed(), 1e-5f);
  // the rotation axis is unchanged, and the axis need not be normalized
  const LinearSpace3f r2 = LinearSpace3f::rotate(Vec3f(0.f, 0.f, 5.f), 1.f);
  const Vec3f axis = r2 * Vec3f(0.f, 0.f, 1.f);
  EXPECT_NEAR(axis.z, 1.f, 1e-5f);
}

TEST(LinearSpace3f, from_quaternion) {
  // the identity quaternion gives the identity matrix
  expect_identity3(LinearSpace3f(Quaternion3f(one)), 1e-6f);
  // and a quarter turn about +z matches the equivalent rotate()
  const Quaternion3f q =
      Quaternion3f::rotate(Vec3f(0.f, 0.f, 1.f), float(M_PI) * 0.5f);
  const LinearSpace3f fromq(q);
  const LinearSpace3f r =
      LinearSpace3f::rotate(Vec3f(0.f, 0.f, 1.f), float(M_PI) * 0.5f);
  EXPECT_NEAR(fromq.vx.x, r.vx.x, 1e-5f);
  EXPECT_NEAR(fromq.vx.y, r.vx.y, 1e-5f);
  EXPECT_NEAR(fromq.vy.x, r.vy.x, 1e-5f);
  EXPECT_NEAR(fromq.vz.z, r.vz.z, 1e-5f);
}

TEST(LinearSpace3f, frame_and_clamp) {
  // frame(N) builds an orthonormal basis whose third column is N
  const Vec3f n = normalize(Vec3f(1.f, 2.f, 3.f));
  const LinearSpace3f f = frame(n);
  EXPECT_NEAR(f.vz.x, n.x, 1e-5f);
  EXPECT_NEAR(f.vz.z, n.z, 1e-5f);
  EXPECT_NEAR(dot(f.vx, f.vy), 0.f, 1e-5f);
  EXPECT_NEAR(dot(f.vx, f.vz), 0.f, 1e-5f);
  EXPECT_NEAR(dot(f.vy, f.vz), 0.f, 1e-5f);
  EXPECT_NEAR(length(f.vx), 1.f, 1e-5f);
  EXPECT_NEAR(length(f.vy), 1.f, 1e-5f);
  // the two-argument form takes a hint for the x direction
  const LinearSpace3f g = frame(Vec3f(0.f, 0.f, 1.f), Vec3f(0.f, 1.f, 0.f));
  EXPECT_NEAR(dot(g.vx, g.vz), 0.f, 1e-5f);
  EXPECT_NEAR(length(g.vx), 1.f, 1e-5f);
  // a hint nearly parallel to N falls back to the one-argument form
  const LinearSpace3f h = frame(Vec3f(0.f, 0.f, 1.f), Vec3f(0.f, 0.f, 1.f));
  EXPECT_NEAR(length(h.vx), 1.f, 1e-5f);
  EXPECT_NEAR(dot(h.vx, h.vz), 0.f, 1e-5f);
  // clamp squashes every component into [-1,1]
  const LinearSpace3f c =
      clamp(LinearSpace3f(Vec3f(5.f, -5.f, 0.5f), Vec3f(zero), Vec3f(zero)));
  EXPECT_EQ(c.vx.x, 1.f);
  EXPECT_EQ(c.vx.y, -1.f);
  EXPECT_EQ(c.vx.z, 0.5f);
}

TEST(LinearSpace3f, operators_and_xfm) {
  const LinearSpace3f a(Vec3f(1.f, 2.f, 3.f), Vec3f(4.f, 5.f, 6.f),
                        Vec3f(7.f, 8.f, 10.f));
  const LinearSpace3f b(one);
  EXPECT_TRUE((-a).vx == Vec3f(-1.f, -2.f, -3.f));
  EXPECT_TRUE((+a) == a);
  EXPECT_TRUE((a + a).vx == Vec3f(2.f, 4.f, 6.f));
  EXPECT_TRUE((a - a).vz == Vec3f(zero));
  EXPECT_TRUE((2.f * a).vx == Vec3f(2.f, 4.f, 6.f));
  EXPECT_TRUE((a / 2.f).vx == Vec3f(0.5f, 1.f, 1.5f));
  EXPECT_TRUE(a * b == a);
  // matrix * vector picks out columns
  EXPECT_TRUE(a * Vec3f(0.f, 1.f, 0.f) == Vec3f(4.f, 5.f, 6.f));
  // xfmPoint / xfmVector are the same thing for a linear (non-affine) space
  EXPECT_TRUE(xfmPoint(a, Vec3f(0.f, 0.f, 1.f)) == Vec3f(7.f, 8.f, 10.f));
  EXPECT_TRUE(xfmVector(a, Vec3f(0.f, 0.f, 1.f)) == Vec3f(7.f, 8.f, 10.f));
  // xfmNormal uses the inverse transpose, so normals stay perpendicular
  const LinearSpace3f s = LinearSpace3f::scale(Vec3f(2.f, 1.f, 1.f));
  const Vec3f tangent = xfmVector(s, Vec3f(0.f, 1.f, 0.f));
  const Vec3f normal = xfmNormal(s, Vec3f(0.f, 1.f, 0.f));
  EXPECT_NEAR(dot(cross(tangent, normal), cross(tangent, normal)), 0.f, 1e-4f);

  LinearSpace3f m = a;
  m *= b;
  EXPECT_TRUE(m == a);
  m /= b;
  EXPECT_TRUE(m == a);
  EXPECT_TRUE(a != b);
  // lerp blends column-wise
  EXPECT_TRUE(lerp(LinearSpace3f(zero), a, 0.5f).vx == Vec3f(0.5f, 1.f, 1.5f));
}

TEST(LinearSpace3fa, sse_backed) {
  const LinearSpace3fa a(Vec3fa(1.f, 2.f, 3.f), Vec3fa(4.f, 5.f, 6.f),
                         Vec3fa(7.f, 8.f, 10.f));
  EXPECT_NEAR(a.det(), -3.f, 1e-3f);
  // LinearSpace3<Vec3fa> has its own transposed(), routed through the 4x4
  // vfloat4 transpose rather than the generic component shuffle
  const LinearSpace3fa t = a.transposed();
  EXPECT_EQ(t.vx.x, 1.f);
  EXPECT_EQ(t.vx.y, 4.f);
  EXPECT_EQ(t.vx.z, 7.f);
  EXPECT_EQ(t.vy.x, 2.f);
  EXPECT_EQ(t.vz.z, 10.f);
  EXPECT_TRUE(t.transposed() == a);
  const Vec3fa v = a * Vec3fa(0.f, 1.f, 0.f);
  EXPECT_EQ(v.x, 4.f);
  EXPECT_EQ(v.z, 6.f);
  // Vec3fx-backed alias
  const LinearSpace3fx b(Vec3fx(1.f, 0.f, 0.f, 0.f), Vec3fx(0.f, 1.f, 0.f, 0.f),
                         Vec3fx(0.f, 0.f, 1.f, 0.f));
  EXPECT_NEAR(b.det(), 1.f, 1e-5f);
}

////////////////////////////////////////////////////////////////////////////////
// LinearSpace2<Vec2<V>> / LinearSpace3<Vec3<V>> -- one per vector width
////////////////////////////////////////////////////////////////////////////////

#define ESIMD_LINEARSPACE_TESTS(suite, V)                                      \
                                                                               \
  TEST(suite, linearspace2) {                                                  \
    typedef LinearSpace2<Vec2<V>> L2;                                          \
    const L2 a(Vec2<V>(ramp<V>(1.f), ramp<V>(2.f)),                            \
               Vec2<V>(ramp<V>(3.f), ramp<V>(4.f)));                           \
    /* det = vx.x*vy.y - vx.y*vy.x, per lane */                                \
    expect_lanes(                                                              \
        a.det(),                                                               \
        [](int i) {                                                            \
          return (1.f + i) * (4.f + i) - (2.f + i) * (3.f + i);                \
        },                                                                     \
        1e-4f);                                                                \
    expect_lanes(a.transposed().vx.y, [](int i) { return 3.f + i; }, 0.f);     \
    expect_lanes(a.row0().y, [](int i) { return 3.f + i; }, 0.f);              \
    /* identity is the unit, per lane */                                       \
    const L2 id(one);                                                          \
    expect_lanes((a * id).vx.x, [](int i) { return 1.f + i; }, 1e-5f);         \
    /* inverse, checked by composing back to the identity */                   \
    const L2 inv = a.inverse();                                                \
    expect_lanes((a * inv).vx.x, [](int) { return 1.f; }, 1e-3f);              \
    expect_lanes((a * inv).vx.y, [](int) { return 0.f; }, 1e-3f);              \
    expect_lanes((a * inv).vy.y, [](int) { return 1.f; }, 1e-3f);              \
    /* matrix * vector */                                                      \
    const Vec2<V> v = a * Vec2<V>(V(1.f), V(0.f));                             \
    expect_lanes(v.x, [](int i) { return 1.f + i; }, 0.f);                     \
    expect_lanes(v.y, [](int i) { return 2.f + i; }, 0.f);                     \
    expect_lanes((-a).vx.x, [](int i) { return -(1.f + i); }, 0.f);            \
    expect_lanes((a + a).vy.x, [](int i) { return 2.f * (3.f + i); }, 0.f);    \
    expect_lanes((V(2.f) * a).vx.y, [](int i) { return 2.f * (2.f + i); },     \
                 0.f);                                                         \
    expect_lanes((a / V(2.f)).vx.y, [](int i) { return 0.5f * (2.f + i); },    \
                 1e-5f);                                                       \
  }                                                                            \
                                                                               \
  TEST(suite, linearspace3) {                                                  \
    typedef LinearSpace3<Vec3<V>> L3;                                          \
    const L3 id(one);                                                          \
    /* a diagonal matrix has an easy determinant and inverse */                \
    const L3 d(Vec3<V>(ramp<V>(2.f), V(0.f), V(0.f)),                          \
               Vec3<V>(V(0.f), ramp<V>(3.f), V(0.f)),                          \
               Vec3<V>(V(0.f), V(0.f), ramp<V>(4.f)));                         \
    expect_lanes(                                                              \
        d.det(),                                                               \
        [](int i) { return (2.f + i) * (3.f + i) * (4.f + i); }, 1e-2f);       \
    expect_lanes((d * d.inverse()).vx.x, [](int) { return 1.f; }, 1e-4f);      \
    expect_lanes((d * d.inverse()).vy.y, [](int) { return 1.f; }, 1e-4f);      \
    expect_lanes((d * d.inverse()).vz.x, [](int) { return 0.f; }, 1e-4f);      \
    /* transposed is an involution and swaps rows with columns */              \
    const L3 a(Vec3<V>(ramp<V>(1.f), ramp<V>(2.f), ramp<V>(3.f)),              \
               Vec3<V>(ramp<V>(4.f), ramp<V>(5.f), ramp<V>(6.f)),              \
               Vec3<V>(ramp<V>(7.f), ramp<V>(8.f), ramp<V>(10.f)));            \
    expect_lanes(a.transposed().vx.y, [](int i) { return 4.f + i; }, 0.f);     \
    expect_lanes(a.row2().z, [](int i) { return 10.f + i; }, 0.f);             \
    expect_lanes(a.transposed().transposed().vy.z, [](int i) { return 6.f + i; }, \
                 0.f);                                                         \
    /* matrix * vector picks out a column */                                   \
    const Vec3<V> v = a * Vec3<V>(V(0.f), V(1.f), V(0.f));                     \
    expect_lanes(v.z, [](int i) { return 6.f + i; }, 1e-4f);                   \
    expect_lanes(xfmVector(a, Vec3<V>(V(0.f), V(1.f), V(0.f))).z,              \
                 [](int i) { return 6.f + i; }, 1e-4f);                        \
    expect_lanes((a * id).vx.x, [](int i) { return 1.f + i; }, 1e-4f);         \
    expect_lanes((-a).vz.z, [](int i) { return -(10.f + i); }, 0.f);           \
    expect_lanes((a + a).vx.x, [](int i) { return 2.f * (1.f + i); }, 0.f);    \
    expect_lanes((V(2.f) * a).vy.y, [](int i) { return 2.f * (5.f + i); },     \
                 0.f);                                                         \
  }                                                                            \
                                                                               \
  TEST(suite, frame_and_select) {                                              \
    typedef LinearSpace3<Vec3<V>> L3;                                          \
    /* frame(N) is orthonormal with N as its third column, per lane */         \
    const Vec3<V> n = normalize(Vec3<V>(ramp<V>(1.f), V(2.f), V(3.f)));        \
    const L3 f = frame(n);                                                     \
    for (int i = 0; i < V::size; ++i) {                                        \
      EXPECT_NEAR(f.vz.x[i], n.x[i], 1e-5f) << "at lane " << i;                \
      EXPECT_NEAR(dot(f.vx, f.vy)[i], 0.f, 1e-4f) << "at lane " << i;          \
      EXPECT_NEAR(dot(f.vx, f.vz)[i], 0.f, 1e-4f) << "at lane " << i;          \
      EXPECT_NEAR(length(f.vx)[i], 1.f, 1e-4f) << "at lane " << i;             \
    }                                                                          \
    /* select over a lane mask, on the Scalar's Bool */                        \
    float sel[V::size];                                                        \
    for (int i = 0; i < V::size; ++i)                                          \
      sel[i] = (i % 2 == 0) ? 0.f : 1.f;                                       \
    const typename V::Bool m = V::loadu(sel) < V(0.5f);                        \
    const L3 t(zero), g(one);                                                  \
    const L3 s = select(m, t, g);                                              \
    for (int i = 0; i < V::size; ++i)                                          \
      EXPECT_EQ(s.vx.x[i], (i % 2 == 0) ? 0.f : 1.f) << "at lane " << i;       \
  }

ESIMD_LINEARSPACE_TESTS(LinearSpace_vfloat4, vfloat4)
#if defined(__AVX__)
ESIMD_LINEARSPACE_TESTS(LinearSpace_vfloat8, vfloat8)
#endif
#if defined(__AVX512F__)
ESIMD_LINEARSPACE_TESTS(LinearSpace_vfloat16, vfloat16)
#endif
