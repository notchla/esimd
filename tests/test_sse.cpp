// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Correctness tests for the SSE (128-bit, 4-wide) esimd types: vboolf4, vint4,
// vuint4, vfloat4. Each operation is checked element-by-element against a plain
// scalar reference. Compiled with SSE4.2 flags (see tests/CMakeLists.txt).

#include <esimd/sse.h>
#include "test_helpers.h"

using namespace esimd;
using namespace esimd_test;

////////////////////////////////////////////////////////////////////////////////
// vboolf4
////////////////////////////////////////////////////////////////////////////////

TEST(vboolf4, constructors_and_constants) {
  expect_mask(vboolf4(true),  {true, true, true, true});
  expect_mask(vboolf4(false), {false, false, false, false});
  expect_mask(vboolf4(true, false, true, false), {true, false, true, false});
  expect_mask(vboolf4(esimd::True),  {true, true, true, true});
  expect_mask(vboolf4(esimd::False), {false, false, false, false});
  expect_mask(vboolf4(0x5 /*0b0101*/), {true, false, true, false});
}

TEST(vboolf4, logical_ops) {
  const vboolf4 a(true, true, false, false);
  const vboolf4 b(true, false, true, false);
  expect_mask(a & b, {true, false, false, false});
  expect_mask(a | b, {true, true, true, false});
  expect_mask(a ^ b, {false, true, true, false});
  expect_mask(!a,    {false, false, true, true});
  expect_mask(andn(a, b), {false, true, false, false}); // a & !b
}

TEST(vboolf4, compare_and_select) {
  const vboolf4 a(true, true, false, false);
  const vboolf4 b(true, false, true, false);
  expect_mask(a == b, {true, false, false, true});
  expect_mask(a != b, {false, true, true, false});
  expect_mask(select(vboolf4(true,false,true,false), a, b), {true, false, false, false});
}

TEST(vboolf4, reductions) {
  EXPECT_TRUE (all(vboolf4(true)));
  EXPECT_FALSE(all(vboolf4(true, true, true, false)));
  EXPECT_TRUE (any(vboolf4(false, false, true, false)));
  EXPECT_FALSE(any(vboolf4(false)));
  EXPECT_TRUE (none(vboolf4(false)));
  EXPECT_EQ(movemask(vboolf4(true, false, true, false)), size_t(0x5));
  EXPECT_EQ(popcnt(vboolf4(true, false, true, true)), size_t(3));
  EXPECT_TRUE (reduce_and(vboolf4(true)));
  EXPECT_FALSE(reduce_and(vboolf4(true, true, false, true)));
  EXPECT_TRUE (reduce_or(vboolf4(false, false, true, false)));
}

TEST(vboolf4, get_set_clear) {
  vboolf4 a(false);
  set(a, 1);
  expect_mask(a, {false, true, false, false});
  clear(a, 1);
  expect_mask(a, {false, false, false, false});
  EXPECT_FALSE(get(a, 0));
}

////////////////////////////////////////////////////////////////////////////////
// vint4
////////////////////////////////////////////////////////////////////////////////

TEST(vint4, constructors_and_load_store) {
  expect_eq(vint4(7), {7, 7, 7, 7});
  expect_eq(vint4(1, 2, 3, 4), {1, 2, 3, 4});
  expect_eq(vint4(esimd::zero), {0, 0, 0, 0});
  expect_eq(vint4(esimd::step), {0, 1, 2, 3});

  alignas(16) int mem[4] = {5, 6, 7, 8};
  expect_eq(vint4::load(mem),  {5, 6, 7, 8});
  expect_eq(vint4::loadu(mem), {5, 6, 7, 8});
  alignas(16) int out[4] = {0, 0, 0, 0};
  vint4::store(out, vint4(1, 2, 3, 4));
  EXPECT_EQ(out[0], 1); EXPECT_EQ(out[3], 4);
}

TEST(vint4, arithmetic) {
  const vint4 a(1, 2, 3, 4), b(10, 20, 30, 40);
  expect_eq(a + b, {11, 22, 33, 44});
  expect_eq(b - a, {9, 18, 27, 36});
  expect_eq(a * vint4(2), {2, 4, 6, 8});
  expect_eq(-a, {-1, -2, -3, -4});
  expect_eq(abs(vint4(-1, 2, -3, 4)), {1, 2, 3, 4});
  expect_eq(a + 100, {101, 102, 103, 104});
}

TEST(vint4, bitwise_and_shift) {
  const vint4 a(0b1100, 0b1010, 0xFF, 0x0F);
  expect_eq(a & vint4(0b1010), {0b1000, 0b1010, 0x0A, 0x0A});
  expect_eq(a | vint4(0b0001), {0b1101, 0b1011, 0xFF, 0x0F});
  expect_eq(a ^ a, {0, 0, 0, 0});
  expect_eq(vint4(1, 2, 3, 4) << 2, {4, 8, 12, 16});
  expect_eq(vint4(16, 8, 4, 2) >> 1, {8, 4, 2, 1});
  expect_eq(sll(vint4(1), 3), {8, 8, 8, 8});
  expect_eq(srl(vint4(-1), 30), {3, 3, 3, 3}); // logical shift
}

TEST(vint4, compare) {
  const vint4 a(1, 2, 3, 4), b(4, 3, 2, 1);
  expect_mask(a == vint4(1, 9, 3, 9), {true, false, true, false});
  expect_mask(a != b, {true, true, true, true});
  expect_mask(a < b,  {true, true, false, false});
  expect_mask(a <= vint4(1, 2, 2, 2), {true, true, false, false});
  expect_mask(a > b,  {false, false, true, true});
  expect_mask(a >= vint4(2, 2, 2, 2), {false, true, true, true});
}

TEST(vint4, min_max_select) {
  const vint4 a(1, 5, 3, 8), b(4, 2, 6, 7);
  expect_eq(min(a, b), {1, 2, 3, 7});
  expect_eq(max(a, b), {4, 5, 6, 8});
  expect_eq(select(vboolf4(true, false, true, false), a, b), {1, 2, 3, 7});
}

TEST(vint4, movement_and_reductions) {
  expect_eq(unpacklo(vint4(1, 2, 3, 4), vint4(5, 6, 7, 8)), {1, 5, 2, 6});
  expect_eq(unpackhi(vint4(1, 2, 3, 4), vint4(5, 6, 7, 8)), {3, 7, 4, 8});
  expect_eq(shuffle<1, 0, 3, 2>(vint4(1, 2, 3, 4)), {2, 1, 4, 3});
  const vint4 v(3, 1, 4, 2);
  EXPECT_EQ(reduce_add(v), 10);
  EXPECT_EQ(reduce_min(v), 1);
  EXPECT_EQ(reduce_max(v), 4);
  EXPECT_EQ(toScalar(vint4(9, 0, 0, 0)), 9);
}

////////////////////////////////////////////////////////////////////////////////
// vuint4
////////////////////////////////////////////////////////////////////////////////

TEST(vuint4, constructors_and_load_store) {
  expect_eq(vuint4(7u), {7u, 7u, 7u, 7u});
  expect_eq(vuint4(1u, 2u, 3u, 4u), {1u, 2u, 3u, 4u});
  expect_eq(vuint4(esimd::step), {0u, 1u, 2u, 3u});
  alignas(16) unsigned int mem[4] = {5, 6, 7, 8};
  expect_eq(vuint4::load(mem), {5u, 6u, 7u, 8u});
}

TEST(vuint4, arithmetic_bitwise_shift) {
  const vuint4 a(1u, 2u, 3u, 4u), b(10u, 20u, 30u, 40u);
  expect_eq(a + b, {11u, 22u, 33u, 44u});
  expect_eq(b - a, {9u, 18u, 27u, 36u});
  expect_eq(a & vuint4(1u), {1u, 0u, 1u, 0u});
  expect_eq(a | vuint4(8u), {9u, 10u, 11u, 12u});
  expect_eq(vuint4(1u, 2u, 3u, 4u) << 2u, {4u, 8u, 12u, 16u});
  expect_eq(vuint4(16u, 8u, 4u, 2u) >> 1u, {8u, 4u, 2u, 1u});
}

TEST(vuint4, compare_select_movement) {
  // Note: vuint4 intentionally omits *, ordered compares, min/max and reductions
  // in the SSE backend (only == / != are provided).
  const vuint4 a(1u, 5u, 3u, 8u), b(4u, 2u, 6u, 7u);
  expect_mask(a == vuint4(1u, 9u, 3u, 9u), {true, false, true, false});
  expect_mask(a != b, {true, true, true, true});
  expect_eq(select(vboolf4(true, false, true, false), a, b), {1u, 2u, 3u, 7u});
  expect_eq(unpacklo(vuint4(1u, 2u, 3u, 4u), vuint4(5u, 6u, 7u, 8u)), {1u, 5u, 2u, 6u});
  expect_eq(shuffle<1, 0, 3, 2>(vuint4(1u, 2u, 3u, 4u)), {2u, 1u, 4u, 3u});
  EXPECT_EQ(toScalar(vuint4(9u, 0u, 0u, 0u)), 9u);
}

////////////////////////////////////////////////////////////////////////////////
// vfloat4
////////////////////////////////////////////////////////////////////////////////

TEST(vfloat4, constructors_and_constants) {
  expect_eq(vfloat4(2.5f), {2.5f, 2.5f, 2.5f, 2.5f});
  expect_eq(vfloat4(1.f, 2.f, 3.f, 4.f), {1.f, 2.f, 3.f, 4.f});
  expect_eq(vfloat4(esimd::zero), {0.f, 0.f, 0.f, 0.f});
  expect_eq(vfloat4(esimd::one),  {1.f, 1.f, 1.f, 1.f});
  expect_eq(vfloat4(esimd::step), {0.f, 1.f, 2.f, 3.f});
  expect_eq(vfloat4(vint4(1, 2, 3, 4)), {1.f, 2.f, 3.f, 4.f}); // int -> float cast
}

TEST(vfloat4, load_store) {
  alignas(16) float mem[4] = {5.f, 6.f, 7.f, 8.f};
  expect_eq(vfloat4::load(mem),  {5.f, 6.f, 7.f, 8.f});
  expect_eq(vfloat4::loadu(mem), {5.f, 6.f, 7.f, 8.f});
  alignas(16) float out[4] = {};
  vfloat4::store(out, vfloat4(1.f, 2.f, 3.f, 4.f));
  EXPECT_FLOAT_EQ(out[2], 3.f);
}

TEST(vfloat4, unary) {
  expect_eq(-vfloat4(1.f, -2.f, 3.f, -4.f), {-1.f, 2.f, -3.f, 4.f});
  expect_eq(abs(vfloat4(-1.f, 2.f, -3.f, 4.f)), {1.f, 2.f, 3.f, 4.f});
  expect_eq(sign(vfloat4(-2.f, 3.f, -4.f, 5.f)), {-1.f, 1.f, -1.f, 1.f});
  expect_eq(sqr(vfloat4(1.f, 2.f, 3.f, 4.f)), {1.f, 4.f, 9.f, 16.f});
  expect_eq(sqrt(vfloat4(1.f, 4.f, 9.f, 16.f)), {1.f, 2.f, 3.f, 4.f});
  expect_near(rcp(vfloat4(1.f, 2.f, 4.f, 8.f)),   {1.f, 0.5f, 0.25f, 0.125f}, 1e-4f);
  expect_near(rsqrt(vfloat4(1.f, 4.f, 16.f, 64.f)), {1.f, 0.5f, 0.25f, 0.125f}, 1e-3f);
}

TEST(vfloat4, binary) {
  const vfloat4 a(1.f, 2.f, 3.f, 4.f), b(4.f, 3.f, 2.f, 1.f);
  expect_eq(a + b, {5.f, 5.f, 5.f, 5.f});
  expect_eq(a - b, {-3.f, -1.f, 1.f, 3.f});
  expect_eq(a * b, {4.f, 6.f, 6.f, 4.f});
  expect_near(a / b, {0.25f, 2.f/3.f, 1.5f, 4.f}, 1e-5f);
  expect_eq(min(a, b), {1.f, 2.f, 2.f, 1.f});
  expect_eq(max(a, b), {4.f, 3.f, 3.f, 4.f});
}

TEST(vfloat4, ternary_fma) {
  const vfloat4 a(1.f, 2.f, 3.f, 4.f), b(2.f, 2.f, 2.f, 2.f), c(1.f, 1.f, 1.f, 1.f);
  expect_eq(madd (a, b, c), {3.f, 5.f, 7.f, 9.f});   //  a*b+c
  expect_eq(msub (a, b, c), {1.f, 3.f, 5.f, 7.f});   //  a*b-c
  expect_eq(nmadd(a, b, c), {-1.f, -3.f, -5.f, -7.f}); // -a*b+c
  expect_eq(nmsub(a, b, c), {-3.f, -5.f, -7.f, -9.f}); // -a*b-c
}

TEST(vfloat4, compare_and_select) {
  const vfloat4 a(1.f, 2.f, 3.f, 4.f), b(4.f, 2.f, 2.f, 1.f);
  expect_mask(a == b, {false, true, false, false});
  expect_mask(a <  b, {true, false, false, false});
  expect_mask(a >= b, {false, true, true, true});
  expect_eq(select(vboolf4(true, false, true, false), a, b), {1.f, 2.f, 3.f, 1.f});
}

TEST(vfloat4, rounding) {
  const vfloat4 v(1.4f, 1.6f, -1.4f, -1.6f);
  expect_eq(floor(v), {1.f, 1.f, -2.f, -2.f});
  expect_eq(ceil(v),  {2.f, 2.f, -1.f, -1.f});
  expect_eq(trunc(v), {1.f, 1.f, -1.f, -1.f});
  expect_eq(round(v), {1.f, 2.f, -1.f, -2.f});
}

TEST(vfloat4, movement_and_reductions) {
  expect_eq(unpacklo(vfloat4(1, 2, 3, 4), vfloat4(5, 6, 7, 8)), {1.f, 5.f, 2.f, 6.f});
  expect_eq(shuffle<1, 0, 3, 2>(vfloat4(1, 2, 3, 4)), {2.f, 1.f, 4.f, 3.f});
  const vfloat4 v(3.f, 1.f, 4.f, 2.f);
  EXPECT_FLOAT_EQ(reduce_add(v), 10.f);
  EXPECT_FLOAT_EQ(reduce_min(v), 1.f);
  EXPECT_FLOAT_EQ(reduce_max(v), 4.f);
  EXPECT_FLOAT_EQ(toScalar(vfloat4(9.f, 0, 0, 0)), 9.f);
}

TEST(vfloat4, isnan) {
  expect_mask(isnan(vfloat4(1.f, NAN, 3.f, NAN)), {false, true, false, true});
}
