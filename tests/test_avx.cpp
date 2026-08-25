// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Correctness tests for the AVX (256-bit) esimd types: the 8-wide vboolf8,
// vint8, vuint8, vfloat8 and the 4-wide double vboold4, vdouble4. Each op is
// checked element-by-element against a scalar reference. Compiled with plain
// AVX flags (see tests/CMakeLists.txt), so the two-SSE-halves integer paths
// (vint8_avx.h / vuint8_avx.h) are exercised here; AVX2 native paths are
// covered separately in Phase 3.

#include <esimd/avx.h>
#include "test_helpers.h"

using namespace esimd;
using namespace esimd_test;

////////////////////////////////////////////////////////////////////////////////
// vboolf8
////////////////////////////////////////////////////////////////////////////////

TEST(vboolf8, constructors_and_constants) {
  expect_mask(vboolf8(true),  {true, true, true, true, true, true, true, true});
  expect_mask(vboolf8(false), {false, false, false, false, false, false, false, false});
  expect_mask(vboolf8(true, false, true, false, false, true, false, true),
              {true, false, true, false, false, true, false, true});
  expect_mask(vboolf8(esimd::True),  {true, true, true, true, true, true, true, true});
  expect_mask(vboolf8(esimd::False), {false, false, false, false, false, false, false, false});
  expect_mask(vboolf8(0x5A /*0b01011010*/),
              {false, true, false, true, true, false, true, false});
}

TEST(vboolf8, logical_ops) {
  const vboolf8 a(true, true, false, false, true, true, false, false);
  const vboolf8 b(true, false, true, false, true, false, true, false);
  expect_mask(a & b, {true, false, false, false, true, false, false, false});
  expect_mask(a | b, {true, true, true, false, true, true, true, false});
  expect_mask(a ^ b, {false, true, true, false, false, true, true, false});
  expect_mask(!a,    {false, false, true, true, false, false, true, true});
  expect_mask(andn(a, b), {false, true, false, false, false, true, false, false}); // a & !b
}

TEST(vboolf8, compare_and_select) {
  const vboolf8 a(true, true, false, false, true, true, false, false);
  const vboolf8 b(true, false, true, false, true, false, true, false);
  expect_mask(a == b, {true, false, false, true, true, false, false, true});
  expect_mask(a != b, {false, true, true, false, false, true, true, false});
  expect_mask(select(b, a, vboolf8(false)),
              {true, false, false, false, true, false, false, false});
}

TEST(vboolf8, reductions_and_bits) {
  EXPECT_TRUE (all(vboolf8(true)));
  EXPECT_FALSE(all(vboolf8(true, true, true, true, true, true, true, false)));
  EXPECT_TRUE (any(vboolf8(false, false, false, false, false, false, true, false)));
  EXPECT_FALSE(any(vboolf8(false)));
  EXPECT_TRUE (none(vboolf8(false)));
  EXPECT_EQ(movemask(vboolf8(true, false, true, false, false, true, false, true)), size_t(0xA5));
  EXPECT_EQ(popcnt(vboolf8(true, false, true, true, false, false, true, false)), size_t(4));
  EXPECT_TRUE (reduce_and(vboolf8(true)));
  EXPECT_FALSE(reduce_and(vboolf8(true, true, true, true, true, true, true, false)));
  EXPECT_TRUE (reduce_or(vboolf8(false, false, false, false, false, false, false, true)));
}

////////////////////////////////////////////////////////////////////////////////
// vint8
////////////////////////////////////////////////////////////////////////////////

TEST(vint8, constructors_and_load_store) {
  expect_eq(vint8(7), {7, 7, 7, 7, 7, 7, 7, 7});
  expect_eq(vint8(1, 2, 3, 4, 5, 6, 7, 8), {1, 2, 3, 4, 5, 6, 7, 8});
  expect_eq(vint8(esimd::zero), {0, 0, 0, 0, 0, 0, 0, 0});
  expect_eq(vint8(esimd::step), {0, 1, 2, 3, 4, 5, 6, 7});

  alignas(32) int mem[8] = {5, 6, 7, 8, 9, 10, 11, 12};
  expect_eq(vint8::load(mem),  {5, 6, 7, 8, 9, 10, 11, 12});
  expect_eq(vint8::loadu(mem), {5, 6, 7, 8, 9, 10, 11, 12});
  alignas(32) int out[8] = {};
  vint8::store(out, vint8(1, 2, 3, 4, 5, 6, 7, 8));
  EXPECT_EQ(out[0], 1); EXPECT_EQ(out[7], 8);
}

TEST(vint8, arithmetic_bitwise_shift) {
  const vint8 a(1, 2, 3, 4, 5, 6, 7, 8), b(10, 20, 30, 40, 50, 60, 70, 80);
  expect_eq(a + b, {11, 22, 33, 44, 55, 66, 77, 88});
  expect_eq(b - a, {9, 18, 27, 36, 45, 54, 63, 72});
  expect_eq(a * vint8(2), {2, 4, 6, 8, 10, 12, 14, 16});
  expect_eq(-a, {-1, -2, -3, -4, -5, -6, -7, -8});
  expect_eq(abs(vint8(-1, 2, -3, 4, -5, 6, -7, 8)), {1, 2, 3, 4, 5, 6, 7, 8});
  expect_eq(a & vint8(0x3), {1, 2, 3, 0, 1, 2, 3, 0});
  expect_eq(a | vint8(0x10), {17, 18, 19, 20, 21, 22, 23, 24});
  expect_eq(a ^ a, {0, 0, 0, 0, 0, 0, 0, 0});
  expect_eq(vint8(1) << 3, {8, 8, 8, 8, 8, 8, 8, 8});
  expect_eq(vint8(64) >> 2, {16, 16, 16, 16, 16, 16, 16, 16});
  expect_eq(srl(vint8(-1), 30), {3, 3, 3, 3, 3, 3, 3, 3}); // logical shift
}

TEST(vint8, compare_min_max_select) {
  const vint8 a(1, 2, 3, 4, 5, 6, 7, 8), b(8, 7, 6, 5, 4, 3, 2, 1);
  expect_mask(a == vint8(1, 9, 3, 9, 5, 9, 7, 9),
              {true, false, true, false, true, false, true, false});
  expect_mask(a < b, {true, true, true, true, false, false, false, false});
  expect_mask(a >= b, {false, false, false, false, true, true, true, true});
  expect_eq(min(a, b), {1, 2, 3, 4, 4, 3, 2, 1});
  expect_eq(max(a, b), {8, 7, 6, 5, 5, 6, 7, 8});
  expect_eq(select(vboolf8(true, false, true, false, true, false, true, false), a, b),
            {1, 7, 3, 5, 5, 3, 7, 1});
}

TEST(vint8, movement_and_reductions) {
  expect_eq(unpacklo(vint8(0, 1, 2, 3, 4, 5, 6, 7), vint8(10, 11, 12, 13, 14, 15, 16, 17)),
            {0, 10, 1, 11, 4, 14, 5, 15});
  expect_eq(unpackhi(vint8(0, 1, 2, 3, 4, 5, 6, 7), vint8(10, 11, 12, 13, 14, 15, 16, 17)),
            {2, 12, 3, 13, 6, 16, 7, 17});
  expect_eq(shuffle<1, 0, 3, 2>(vint8(0, 1, 2, 3, 4, 5, 6, 7)), {1, 0, 3, 2, 5, 4, 7, 6});
  const vint8 v(3, 1, 4, 1, 5, 9, 2, 6);
  EXPECT_EQ(reduce_add(v), 31);
  EXPECT_EQ(reduce_min(v), 1);
  EXPECT_EQ(reduce_max(v), 9);
  EXPECT_EQ(toScalar(vint8(42, 0, 0, 0, 0, 0, 0, 0)), 42);
}

////////////////////////////////////////////////////////////////////////////////
// vuint8  (plain-AVX backend: no ordered compares / reduce_min/max)
////////////////////////////////////////////////////////////////////////////////

TEST(vuint8, arithmetic_bitwise_shift) {
  const vuint8 a(1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u), b(10u, 20u, 30u, 40u, 50u, 60u, 70u, 80u);
  expect_eq(a + b, {11u, 22u, 33u, 44u, 55u, 66u, 77u, 88u});
  expect_eq(b - a, {9u, 18u, 27u, 36u, 45u, 54u, 63u, 72u});
  expect_eq(a & vuint8(0x3u), {1u, 2u, 3u, 0u, 1u, 2u, 3u, 0u});
  expect_eq(vuint8(1u) << 3u, {8u, 8u, 8u, 8u, 8u, 8u, 8u, 8u});
  expect_eq(vuint8(64u) >> 2u, {16u, 16u, 16u, 16u, 16u, 16u, 16u, 16u});
}

TEST(vuint8, compare_minmax_select_reduce) {
  const vuint8 a(1u, 5u, 3u, 8u, 2u, 7u, 4u, 6u), b(4u, 2u, 6u, 7u, 2u, 9u, 1u, 6u);
  expect_mask(a == b, {false, false, false, false, true, false, false, true});
  expect_mask(a != b, {true, true, true, true, false, true, true, false});
  expect_eq(min(a, b), {1u, 2u, 3u, 7u, 2u, 7u, 1u, 6u});
  expect_eq(max(a, b), {4u, 5u, 6u, 8u, 2u, 9u, 4u, 6u});
  expect_eq(select(vboolf8(true, false, true, false, true, false, true, false), a, b),
            {1u, 2u, 3u, 7u, 2u, 9u, 4u, 6u});
  EXPECT_EQ(reduce_add(vuint8(1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u)), 36);
  EXPECT_EQ(toScalar(vuint8(42u, 0u, 0u, 0u, 0u, 0u, 0u, 0u)), 42u);
}

////////////////////////////////////////////////////////////////////////////////
// vfloat8
////////////////////////////////////////////////////////////////////////////////

TEST(vfloat8, constructors_and_constants) {
  expect_eq(vfloat8(2.5f), {2.5f, 2.5f, 2.5f, 2.5f, 2.5f, 2.5f, 2.5f, 2.5f});
  expect_eq(vfloat8(1, 2, 3, 4, 5, 6, 7, 8), {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
  expect_eq(vfloat8(esimd::zero), {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f});
  expect_eq(vfloat8(esimd::one),  {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f});
  expect_eq(vfloat8(esimd::step), {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f});
  expect_eq(vfloat8(vint8(1, 2, 3, 4, 5, 6, 7, 8)), {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
}

TEST(vfloat8, load_store) {
  alignas(32) float mem[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  expect_eq(vfloat8::load(mem),  {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
  expect_eq(vfloat8::loadu(mem), {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
  alignas(32) float out[8] = {};
  vfloat8::store(out, vfloat8(8, 7, 6, 5, 4, 3, 2, 1));
  EXPECT_FLOAT_EQ(out[0], 8.f); EXPECT_FLOAT_EQ(out[7], 1.f);
}

TEST(vfloat8, unary) {
  expect_eq(-vfloat8(1, -2, 3, -4, 5, -6, 7, -8), {-1.f, 2.f, -3.f, 4.f, -5.f, 6.f, -7.f, 8.f});
  expect_eq(abs(vfloat8(-1, 2, -3, 4, -5, 6, -7, 8)), {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
  expect_eq(sign(vfloat8(-2, 3, -4, 5, -6, 7, -8, 9)), {-1.f, 1.f, -1.f, 1.f, -1.f, 1.f, -1.f, 1.f});
  expect_eq(sqr(vfloat8(1, 2, 3, 4, 5, 6, 7, 8)), {1.f, 4.f, 9.f, 16.f, 25.f, 36.f, 49.f, 64.f});
  expect_eq(sqrt(vfloat8(1, 4, 9, 16, 25, 36, 49, 64)), {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
  expect_near(rcp(vfloat8(1, 2, 4, 8, 16, 32, 64, 128)),
              {1.f, 0.5f, 0.25f, 0.125f, 0.0625f, 0.03125f, 0.015625f, 0.0078125f}, 1e-4f);
  expect_near(rsqrt(vfloat8(1, 4, 16, 64, 256, 1024, 4096, 16384)),
              {1.f, 0.5f, 0.25f, 0.125f, 0.0625f, 0.03125f, 0.015625f, 0.0078125f}, 1e-3f);
}

TEST(vfloat8, binary) {
  const vfloat8 a(1, 2, 3, 4, 5, 6, 7, 8), b(8, 7, 6, 5, 4, 3, 2, 1);
  expect_eq(a + b, {9.f, 9.f, 9.f, 9.f, 9.f, 9.f, 9.f, 9.f});
  expect_eq(a - b, {-7.f, -5.f, -3.f, -1.f, 1.f, 3.f, 5.f, 7.f});
  expect_eq(a * b, {8.f, 14.f, 18.f, 20.f, 20.f, 18.f, 14.f, 8.f});
  expect_near(a / vfloat8(2), {0.5f, 1.f, 1.5f, 2.f, 2.5f, 3.f, 3.5f, 4.f}, 1e-5f);
  expect_eq(min(a, b), {1.f, 2.f, 3.f, 4.f, 4.f, 3.f, 2.f, 1.f});
  expect_eq(max(a, b), {8.f, 7.f, 6.f, 5.f, 5.f, 6.f, 7.f, 8.f});
}

TEST(vfloat8, ternary_fma) {
  const vfloat8 a(1, 2, 3, 4, 5, 6, 7, 8), b(2), c(1);
  expect_eq(madd (a, b, c), {3.f, 5.f, 7.f, 9.f, 11.f, 13.f, 15.f, 17.f});    //  a*b+c
  expect_eq(msub (a, b, c), {1.f, 3.f, 5.f, 7.f, 9.f, 11.f, 13.f, 15.f});     //  a*b-c
  expect_eq(nmadd(a, b, c), {-1.f, -3.f, -5.f, -7.f, -9.f, -11.f, -13.f, -15.f}); // -a*b+c
  expect_eq(nmsub(a, b, c), {-3.f, -5.f, -7.f, -9.f, -11.f, -13.f, -15.f, -17.f}); // -a*b-c
}

TEST(vfloat8, compare_and_select) {
  const vfloat8 a(1, 2, 3, 4, 5, 6, 7, 8), b(8, 2, 6, 4, 2, 6, 1, 8);
  expect_mask(a == b, {false, true, false, true, false, true, false, true});
  expect_mask(a <  b, {true, false, true, false, false, false, false, false});
  expect_mask(a >= b, {false, true, false, true, true, true, true, true});
  expect_eq(select(vboolf8(true, false, true, false, true, false, true, false), a, b),
            {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
}

TEST(vfloat8, rounding) {
  const vfloat8 v(1.4f, 1.6f, -1.4f, -1.6f, 2.5f, -2.5f, 0.4f, -0.6f);
  expect_eq(floor(v), {1.f, 1.f, -2.f, -2.f, 2.f, -3.f, 0.f, -1.f});
  expect_eq(ceil(v),  {2.f, 2.f, -1.f, -1.f, 3.f, -2.f, 1.f, 0.f});
#if !defined(ESIMD_ARM64)
  // NEON2X defines only floor/ceil for vfloat8; upstream has no trunc/round there.
  expect_eq(trunc(v), {1.f, 1.f, -1.f, -1.f, 2.f, -2.f, 0.f, 0.f});
#endif
}

TEST(vfloat8, movement_and_reductions) {
  expect_eq(unpacklo(vfloat8(0, 1, 2, 3, 4, 5, 6, 7), vfloat8(10, 11, 12, 13, 14, 15, 16, 17)),
            {0.f, 10.f, 1.f, 11.f, 4.f, 14.f, 5.f, 15.f});
  expect_eq(shuffle<1, 0, 3, 2>(vfloat8(0, 1, 2, 3, 4, 5, 6, 7)),
            {1.f, 0.f, 3.f, 2.f, 5.f, 4.f, 7.f, 6.f});
  const vfloat8 v(3, 1, 4, 1, 5, 9, 2, 6);
  EXPECT_FLOAT_EQ(reduce_add(v), 31.f);
  EXPECT_FLOAT_EQ(reduce_min(v), 1.f);
  EXPECT_FLOAT_EQ(reduce_max(v), 9.f);
  EXPECT_FLOAT_EQ(toScalar(vfloat8(9, 0, 0, 0, 0, 0, 0, 0)), 9.f);
}

////////////////////////////////////////////////////////////////////////////////
// vboold4 / vdouble4  (256-bit, 4-wide double)
////////////////////////////////////////////////////////////////////////////////

TEST(vboold4, construct_logical_reduce) {
  expect_mask(vboold4(esimd::True),  {true, true, true, true});
  expect_mask(vboold4(esimd::False), {false, false, false, false});
  expect_mask(vboold4(0x5 /*0b0101*/), {true, false, true, false});
  const vboold4 a(0xC /*1100*/), b(0xA /*1010*/);
  expect_mask(a & b, {false, false, false, true});
  expect_mask(a | b, {false, true, true, true});
  expect_mask(a ^ b, {false, true, true, false});
  expect_mask(!a,    {true, true, false, false});
  EXPECT_TRUE (all(vboold4(esimd::True)));
  EXPECT_TRUE (any(vboold4(0x2)));
  EXPECT_TRUE (none(vboold4(esimd::False)));
}

// vdouble4 is gated on __X86_64__ in avx.h, so it does not exist under NEON2X.
// (vboold4 above is unconditional and is exercised on ARM too.)
#if !defined(ESIMD_ARM64)

TEST(vdouble4, constructors_and_arithmetic) {
  expect_eq(vdouble4(2.5), {2.5, 2.5, 2.5, 2.5});
  expect_eq(vdouble4(1.0, 2.0, 3.0, 4.0), {1.0, 2.0, 3.0, 4.0});
  expect_eq(vdouble4(esimd::zero), {0.0, 0.0, 0.0, 0.0});
  expect_eq(vdouble4(esimd::step), {0.0, 1.0, 2.0, 3.0});
  const vdouble4 a(1.0, 2.0, 3.0, 4.0), b(4.0, 3.0, 2.0, 1.0);
  expect_eq(a + b, {5.0, 5.0, 5.0, 5.0});
  expect_eq(a - b, {-3.0, -1.0, 1.0, 3.0});
  expect_eq(a * b, {4.0, 6.0, 6.0, 4.0});
  expect_eq(min(a, b), {1.0, 2.0, 2.0, 1.0});
  expect_eq(max(a, b), {4.0, 3.0, 3.0, 4.0});
}

TEST(vdouble4, fma_compare_select_reduce) {
  const vdouble4 a(1.0, 2.0, 3.0, 4.0), b(2.0, 2.0, 2.0, 2.0), c(1.0, 1.0, 1.0, 1.0);
  expect_eq(madd (a, b, c), {3.0, 5.0, 7.0, 9.0});
  expect_eq(msub (a, b, c), {1.0, 3.0, 5.0, 7.0});
  expect_eq(nmadd(a, b, c), {-1.0, -3.0, -5.0, -7.0});
  const vdouble4 x(1.0, 2.0, 3.0, 4.0), y(4.0, 2.0, 2.0, 1.0);
  expect_mask(x == y, {false, true, false, false});
  expect_mask(x <  y, {true, false, false, false});
  expect_mask(x >= y, {false, true, true, true});
  expect_eq(select(vboold4(0x5), x, y), {1.0, 2.0, 3.0, 1.0});
  const vdouble4 v(3.0, 1.0, 4.0, 2.0);
  EXPECT_DOUBLE_EQ(reduce_add(v), 10.0);
  EXPECT_DOUBLE_EQ(reduce_min(v), 1.0);
  EXPECT_DOUBLE_EQ(reduce_max(v), 4.0);
  EXPECT_DOUBLE_EQ(toScalar(vdouble4(9.0, 0.0, 0.0, 0.0)), 9.0);
}

#endif // !ESIMD_ARM64
