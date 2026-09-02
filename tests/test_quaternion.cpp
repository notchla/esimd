// Copyright 2026 notchla liso.lorenzo@gmail.com
// SPDX-License-Identifier: Apache-2.0
//
// Correctness tests for QuaternionT<T> and the fastapprox transcendentals it
// rests on, both pulled into esimd as dependencies of LinearSpace3 (whose
// construction-from-quaternion constructor needs them).

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

} // namespace

////////////////////////////////////////////////////////////////////////////////
// Quaternion3f
////////////////////////////////////////////////////////////////////////////////

TEST(Quaternion3f, construction) {
  const Quaternion3f a(1.f, 2.f, 3.f, 4.f);
  EXPECT_EQ(a.r, 1.f);
  EXPECT_EQ(a.i, 2.f);
  EXPECT_EQ(a.j, 3.f);
  EXPECT_EQ(a.k, 4.f);
  // a bare scalar is a real quaternion
  const Quaternion3f s(5.f);
  EXPECT_EQ(s.r, 5.f);
  EXPECT_EQ(s.i, 0.f);
  // a Vec3 is a pure (imaginary) quaternion
  const Quaternion3f v(Vec3f(1.f, 2.f, 3.f));
  EXPECT_EQ(v.r, 0.f);
  EXPECT_EQ(v.k, 3.f);
  EXPECT_TRUE(v.v() == Vec3f(1.f, 2.f, 3.f));
  // a Vec4 maps (x,y,z,w) onto (r,i,j,k)
  const Quaternion3f w{Vec4f(1.f, 2.f, 3.f, 4.f)};
  EXPECT_EQ(w.r, 1.f);
  EXPECT_EQ(w.k, 4.f);
  // scalar + vector
  const Quaternion3f rv(1.f, Vec3f(2.f, 3.f, 4.f));
  EXPECT_TRUE(rv == a);
  // constants
  EXPECT_EQ(Quaternion3f(zero).r, 0.f);
  EXPECT_EQ(Quaternion3f(one).r, 1.f);
  EXPECT_EQ(Quaternion3f(one).i, 0.f);
  EXPECT_TRUE(a != s);
}

TEST(Quaternion3f, rotation) {
  // a quarter turn about +z takes +x to +y
  const Quaternion3f q =
      Quaternion3f::rotate(Vec3f(0.f, 0.f, 1.f), float(M_PI) * 0.5f);
  const Vec3f v = q * Vec3f(1.f, 0.f, 0.f);
  EXPECT_NEAR(v.x, 0.f, 1e-5f);
  EXPECT_NEAR(v.y, 1.f, 1e-5f);
  EXPECT_NEAR(v.z, 0.f, 1e-5f);
  // xfmPoint / xfmVector / xfmNormal are all the same rotation
  EXPECT_NEAR(xfmPoint(q, Vec3f(1.f, 0.f, 0.f)).y, 1.f, 1e-5f);
  EXPECT_NEAR(xfmVector(q, Vec3f(1.f, 0.f, 0.f)).y, 1.f, 1e-5f);
  EXPECT_NEAR(xfmNormal(q, Vec3f(1.f, 0.f, 0.f)).y, 1.f, 1e-5f);
  // a unit quaternion has unit norm, and the axis is unchanged
  EXPECT_NEAR(dot(q, q), 1.f, 1e-5f);
  EXPECT_NEAR((q * Vec3f(0.f, 0.f, 1.f)).z, 1.f, 1e-5f);
  // two quarter turns make a half turn
  const Vec3f h = (q * q) * Vec3f(1.f, 0.f, 0.f);
  EXPECT_NEAR(h.x, -1.f, 1e-5f);
  // the axis need not be normalized
  const Quaternion3f q2 =
      Quaternion3f::rotate(Vec3f(0.f, 0.f, 7.f), float(M_PI) * 0.5f);
  EXPECT_NEAR((q2 * Vec3f(1.f, 0.f, 0.f)).y, 1.f, 1e-5f);
}

TEST(Quaternion3f, from_basis_and_euler) {
  // building from an orthonormal basis round-trips through LinearSpace3
  const LinearSpace3f r =
      LinearSpace3f::rotate(Vec3f(0.f, 0.f, 1.f), float(M_PI) * 0.5f);
  const Quaternion3f q(r.vx, r.vy, r.vz);
  const Vec3f v = q * Vec3f(1.f, 0.f, 0.f);
  EXPECT_NEAR(v.x, 0.f, 1e-4f);
  EXPECT_NEAR(v.y, 1.f, 1e-4f);
  // yaw/pitch/roll, all zero, is the identity rotation
  const Quaternion3f e(0.f, 0.f, 0.f);
  EXPECT_NEAR(e.r, 1.f, 1e-5f);
  EXPECT_NEAR(e.i, 0.f, 1e-5f);
  const Vec3f id = e * Vec3f(1.f, 2.f, 3.f);
  EXPECT_NEAR(id.x, 1.f, 1e-5f);
  EXPECT_NEAR(id.z, 3.f, 1e-5f);
}

// Pins the yaw/pitch/roll convention
TEST(Quaternion3f, euler_convention) {
  const float Y = 0.3f, P = 0.5f, R = 0.7f;

  // Each angle drives one axis: yaw -> Y, pitch -> X, roll -> Z.
  const Quaternion3f y(Y, 0.f, 0.f), p(0.f, P, 0.f), r(0.f, 0.f, R);
  EXPECT_NEAR(y.r, std::cos(Y * 0.5f), 1e-5f);
  EXPECT_NEAR(y.j, std::sin(Y * 0.5f), 1e-5f); // j == Y
  EXPECT_NEAR(y.i, 0.f, 1e-6f);
  EXPECT_NEAR(y.k, 0.f, 1e-6f);
  EXPECT_NEAR(p.i, std::sin(P * 0.5f), 1e-5f); // i == X
  EXPECT_NEAR(p.j, 0.f, 1e-6f);
  EXPECT_NEAR(p.k, 0.f, 1e-6f);
  EXPECT_NEAR(r.k, std::sin(R * 0.5f), 1e-5f); // k == Z
  EXPECT_NEAR(r.i, 0.f, 1e-6f);
  EXPECT_NEAR(r.j, 0.f, 1e-6f);

  // Right-handed active rotation about each of those axes.
  const Vec3f ry = y * Vec3f(1.f, 0.f, 0.f); // about +Y: x -> (cos, 0, -sin)
  EXPECT_NEAR(ry.x, std::cos(Y), 1e-5f);
  EXPECT_NEAR(ry.y, 0.f, 1e-5f);
  EXPECT_NEAR(ry.z, -std::sin(Y), 1e-5f);
  const Vec3f rp = p * Vec3f(0.f, 1.f, 0.f); // about +X: y -> (0, cos, sin)
  EXPECT_NEAR(rp.y, std::cos(P), 1e-5f);
  EXPECT_NEAR(rp.z, std::sin(P), 1e-5f);
  const Vec3f rr = r * Vec3f(1.f, 0.f, 0.f); // about +Z: x -> (cos, sin, 0)
  EXPECT_NEAR(rr.x, std::cos(R), 1e-5f);
  EXPECT_NEAR(rr.y, std::sin(R), 1e-5f);

  // Composition is Qy(yaw) * Qx(pitch) * Qz(roll) -- note that the argument
  // order is yaw, pitch, roll but the rightmost factor acts first, so the
  // fixed-frame order is roll, then pitch, then yaw.
  const Quaternion3f Qx = Quaternion3f::rotate(Vec3f(1.f, 0.f, 0.f), P);
  const Quaternion3f Qy = Quaternion3f::rotate(Vec3f(0.f, 1.f, 0.f), Y);
  const Quaternion3f Qz = Quaternion3f::rotate(Vec3f(0.f, 0.f, 1.f), R);
  const Quaternion3f e(Y, P, R);
  const Quaternion3f composed = Qy * Qx * Qz;
  EXPECT_NEAR(e.r, composed.r, 1e-5f);
  EXPECT_NEAR(e.i, composed.i, 1e-5f);
  EXPECT_NEAR(e.j, composed.j, 1e-5f);
  EXPECT_NEAR(e.k, composed.k, 1e-5f);
  // the order matters: the reverse grouping is a different rotation
  EXPECT_GT(std::abs((Qz * Qy * Qx).i - e.i), 1e-3f);

  // and applying it agrees with applying roll, then pitch, then yaw by hand
  const Vec3f v(1.f, 2.f, 3.f);
  const Vec3f byHand = Qy * (Qx * (Qz * v));
  const Vec3f byEuler = e * v;
  EXPECT_NEAR(byEuler.x, byHand.x, 1e-4f);
  EXPECT_NEAR(byEuler.y, byHand.y, 1e-4f);
  EXPECT_NEAR(byEuler.z, byHand.z, 1e-4f);
}

TEST(Quaternion3f, algebra) {
  const Quaternion3f a(1.f, 2.f, 3.f, 4.f), b(5.f, 6.f, 7.f, 8.f);
  EXPECT_EQ((a + b).r, 6.f);
  EXPECT_EQ((a - b).k, -4.f);
  EXPECT_EQ((-a).i, -2.f);
  EXPECT_EQ((2.f * a).j, 6.f);
  EXPECT_EQ((a * 2.f).j, 6.f);
  EXPECT_EQ((a - 1.f).r, 0.f);
  EXPECT_EQ((1.f - a).i, -2.f);
  // conjugate flips the imaginary part
  EXPECT_EQ(conj(a).r, 1.f);
  EXPECT_EQ(conj(a).i, -2.f);
  // dot is the squared norm when both are the same
  EXPECT_NEAR(dot(a, a), 1.f + 4.f + 9.f + 16.f, 1e-4f);
  EXPECT_NEAR(abs(a), std::sqrt(30.f), 1e-4f);
  // q * q^-1 == 1
  const Quaternion3f inv = rcp(a);
  const Quaternion3f id = a * inv;
  EXPECT_NEAR(id.r, 1.f, 1e-4f);
  EXPECT_NEAR(id.i, 0.f, 1e-4f);
  EXPECT_NEAR(id.k, 0.f, 1e-4f);
  EXPECT_NEAR(dot(normalize(a), normalize(a)), 1.f, 1e-4f);
  // multiplication is associative but not commutative
  EXPECT_NEAR(((a * b) * a).r, (a * (b * a)).r, 1e-3f);
  EXPECT_NE((a * b).i, (b * a).i);

  Quaternion3f m(1.f, 2.f, 3.f, 4.f);
  m += Quaternion3f(1.f, 1.f, 1.f, 1.f);
  EXPECT_EQ(m.k, 5.f);
  m -= Quaternion3f(1.f, 1.f, 1.f, 1.f);
  EXPECT_EQ(m.k, 4.f);
  m *= 2.f;
  EXPECT_EQ(m.k, 8.f);
  /* /= by a scalar is a * rcp(b) upstream, so it is not bit-exact */
  m /= 2.f;
  EXPECT_NEAR(m.k, 4.f, 1e-5f);
}

TEST(Quaternion3f, interpolation) {
  const Quaternion3f q0(one);
  const Quaternion3f q1 =
      Quaternion3f::rotate(Vec3f(0.f, 0.f, 1.f), float(M_PI) * 0.5f);
  // lerp is componentwise
  EXPECT_NEAR(lerp(q0, q1, 0.f).r, q0.r, 1e-5f);
  EXPECT_NEAR(lerp(q0, q1, 1.f).k, q1.k, 1e-5f);
  EXPECT_NEAR(lerp(q0, q1, 0.5f).r, 0.5f * (q0.r + q1.r), 1e-5f);
  // slerp stays on the unit sphere and hits the endpoints
  EXPECT_NEAR(dot(slerp(q0, q1, 0.5f), slerp(q0, q1, 0.5f)), 1.f, 1e-3f);
  EXPECT_NEAR(slerp(q0, q1, 0.f).r, q0.r, 1e-3f);
  EXPECT_NEAR(slerp(q0, q1, 1.f).k, q1.k, 1e-3f);
  // halfway along a quarter turn is an eighth turn: +x lands at 45 degrees
  const Vec3f v = slerp(q0, q1, 0.5f) * Vec3f(1.f, 0.f, 0.f);
  EXPECT_NEAR(v.x, std::sqrt(0.5f), 1e-3f);
  EXPECT_NEAR(v.y, std::sqrt(0.5f), 1e-3f);
}

////////////////////////////////////////////////////////////////////////////////
// fastapprox -- the transcendentals slerp is built on
////////////////////////////////////////////////////////////////////////////////

TEST(fastapprox, scalar) {
  for (float x = -3.f; x <= 3.f; x += 0.25f) {
    EXPECT_NEAR(fastapprox::sin(x), std::sin(x), 1e-4f) << "at x=" << x;
    EXPECT_NEAR(fastapprox::cos(x), std::cos(x), 1e-4f) << "at x=" << x;
    EXPECT_NEAR(fastapprox::exp(x), std::exp(x), 1e-3f) << "at x=" << x;
  }
  // tan blows up near +-pi/2, so it is checked away from the poles
  for (float x = -1.3f; x <= 1.3f; x += 0.1f)
    EXPECT_NEAR(fastapprox::tan(x), std::tan(x), 1e-3f) << "at x=" << x;
  for (float x = -0.99f; x <= 0.99f; x += 0.1f) {
    EXPECT_NEAR(fastapprox::asin(x), std::asin(x), 1e-3f) << "at x=" << x;
    EXPECT_NEAR(fastapprox::acos(x), std::acos(x), 1e-3f) << "at x=" << x;
  }
  for (float x = 0.1f; x <= 10.f; x += 0.3f)
    EXPECT_NEAR(fastapprox::log(x), std::log(x), 1e-3f) << "at x=" << x;
  float s, c;
  fastapprox::sincos(0.7f, s, c);
  EXPECT_NEAR(s, std::sin(0.7f), 1e-4f);
  EXPECT_NEAR(c, std::cos(0.7f), 1e-4f);
  // This covers every entry point fastapprox still declares. Upstream's atan,
  // atan2 and pow are gone: they could not be instantiated at any T.
}

////////////////////////////////////////////////////////////////////////////////
// QuaternionT<V> -- one instantiation per vector width
////////////////////////////////////////////////////////////////////////////////

#define ESIMD_QUATERNION_TESTS(suite, V)                                       \
                                                                               \
  TEST(suite, construction_and_algebra) {                                      \
    const QuaternionT<V> a(ramp<V>(1.f), ramp<V>(2.f), ramp<V>(3.f),           \
                           ramp<V>(4.f));                                      \
    expect_lanes(a.r, [](int i) { return 1.f + i; }, 0.f);                     \
    expect_lanes(a.k, [](int i) { return 4.f + i; }, 0.f);                     \
    expect_lanes(a.v().z, [](int i) { return 4.f + i; }, 0.f);                 \
    /* pure and real forms */                                                  \
    const QuaternionT<V> p(Vec3<V>(V(1.f), V(2.f), V(3.f)));                   \
    expect_lanes(p.r, [](int) { return 0.f; }, 0.f);                           \
    expect_lanes(p.j, [](int) { return 2.f; }, 0.f);                           \
    expect_lanes(QuaternionT<V>(one).r, [](int) { return 1.f; }, 0.f);         \
    expect_lanes(QuaternionT<V>(zero).r, [](int) { return 0.f; }, 0.f);        \
    /* algebra, per lane */                                                    \
    expect_lanes((a + a).k, [](int i) { return 2.f * (4.f + i); }, 0.f);       \
    expect_lanes((a - a).r, [](int) { return 0.f; }, 0.f);                     \
    expect_lanes((-a).i, [](int i) { return -(2.f + i); }, 0.f);               \
    expect_lanes((V(2.f) * a).j, [](int i) { return 2.f * (3.f + i); }, 0.f);  \
    expect_lanes(conj(a).i, [](int i) { return -(2.f + i); }, 0.f);            \
    expect_lanes(                                                              \
        dot(a, a),                                                             \
        [](int i) {                                                            \
          const float r = 1.f + i, x = 2.f + i, y = 3.f + i, z = 4.f + i;      \
          return r * r + x * x + y * y + z * z;                                \
        },                                                                     \
        1e-2f);                                                                \
    /* q * q^-1 == 1 */                                                        \
    const QuaternionT<V> id = a * rcp(a);                                      \
    expect_lanes(id.r, [](int) { return 1.f; }, 1e-4f);                        \
    expect_lanes(id.k, [](int) { return 0.f; }, 1e-4f);                        \
    expect_lanes(dot(normalize(a), normalize(a)), [](int) { return 1.f; },     \
                 1e-4f);                                                       \
  }                                                                            \
                                                                               \
  TEST(suite, rotation_and_interpolation) {                                    \
    /* a quarter turn about +z, built by hand since rotate() needs sin/cos */  \
    const float h = float(M_PI) * 0.25f;                                       \
    const QuaternionT<V> q0(one);                                              \
    const QuaternionT<V> q1(V(std::cos(h)), V(0.f), V(0.f), V(std::sin(h)));   \
    const Vec3<V> v = q1 * Vec3<V>(V(1.f), V(0.f), V(0.f));                    \
    expect_lanes(v.x, [](int) { return 0.f; }, 1e-5f);                         \
    expect_lanes(v.y, [](int) { return 1.f; }, 1e-5f);                         \
    expect_lanes(xfmVector(q1, Vec3<V>(V(1.f), V(0.f), V(0.f))).y,             \
                 [](int) { return 1.f; }, 1e-5f);                              \
    /* lerp is componentwise */                                                \
    expect_lanes(lerp(q0, q1, V(0.f)).r, [](int) { return 1.f; }, 1e-5f);      \
    expect_lanes(lerp(q0, q1, V(1.f)).k, [h](int) { return std::sin(h); },     \
                 1e-5f);                                                       \
    /* slerp stays on the unit sphere and hits both endpoints */               \
    const QuaternionT<V> s = slerp(q0, q1, V(0.5f));                           \
    expect_lanes(dot(s, s), [](int) { return 1.f; }, 1e-3f);                   \
    expect_lanes(slerp(q0, q1, V(0.f)).r, [](int) { return 1.f; }, 1e-3f);     \
    expect_lanes(slerp(q0, q1, V(1.f)).k, [h](int) { return std::sin(h); },    \
                 1e-3f);                                                       \
    /* halfway along the quarter turn puts +x at 45 degrees */                 \
    const Vec3<V> m = s * Vec3<V>(V(1.f), V(0.f), V(0.f));                     \
    expect_lanes(m.x, [](int) { return std::sqrt(0.5f); }, 1e-3f);             \
    expect_lanes(m.y, [](int) { return std::sqrt(0.5f); }, 1e-3f);             \
  }                                                                            \
                                                                               \
  TEST(suite, fastapprox_lanes) {                                              \
    /* the same approximations slerp uses, checked lane-by-lane */             \
    const V x = ramp<V>(0.f) * V(0.1f);                                        \
    expect_lanes(fastapprox::sin(x), [](int i) { return std::sin(0.1f * i); }, \
                 1e-4f);                                                       \
    expect_lanes(fastapprox::cos(x), [](int i) { return std::cos(0.1f * i); }, \
                 1e-4f);                                                       \
    /* acos needs |x| <= 1, which the 16-wide ramp above would exceed */       \
    const V u = ramp<V>(0.f) * V(0.05f);                                       \
    expect_lanes(fastapprox::acos(u),                                          \
                 [](int i) { return std::acos(0.05f * i); }, 1e-3f);           \
    V s, c;                                                                    \
    fastapprox::sincos(x, s, c);                                               \
    expect_lanes(s, [](int i) { return std::sin(0.1f * i); }, 1e-4f);          \
    expect_lanes(c, [](int i) { return std::cos(0.1f * i); }, 1e-4f);          \
    expect_lanes(fastapprox::exp(x), [](int i) { return std::exp(0.1f * i); }, \
                 1e-3f);                                                       \
  }

ESIMD_QUATERNION_TESTS(Quaternion_vfloat4, vfloat4)
#if defined(__AVX__)
ESIMD_QUATERNION_TESTS(Quaternion_vfloat8, vfloat8)
#endif
#if defined(__AVX512F__)
ESIMD_QUATERNION_TESTS(Quaternion_vfloat16, vfloat16)
#endif
