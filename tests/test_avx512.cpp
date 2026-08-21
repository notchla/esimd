// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Correctness tests for the AVX512 esimd types: the 16-wide vboolf16, vint16,
// vuint16, vfloat16 and the 8-wide vboold8, vdouble8, vllong8. Compiled with
// AVX512 flags (see tests/CMakeLists.txt). The entry header is <esimd/avx.h>:
// under __AVX512F__ it pulls in avx512.h, and under __AVX512VL__ the SSE/AVX
// small-width bool types resolve to their __mmask mask-register overrides, so
// this suite also exercises those mask paths.

#include <esimd/avx.h>
#include "test_helpers.h"

using namespace esimd;
using namespace esimd_test;

////////////////////////////////////////////////////////////////////////////////
// vboolf16 (__mmask16)
////////////////////////////////////////////////////////////////////////////////

TEST(vboolf16, masks) {
  expect_mask(vboolf16(esimd::True),
              {true, true, true, true, true, true, true, true,
               true, true, true, true, true, true, true, true});
  expect_mask(vboolf16(esimd::False),
              {false, false, false, false, false, false, false, false,
               false, false, false, false, false, false, false, false});
  // bit i of the mask -> lane i
  expect_mask(vboolf16(0x000F),
              {true, true, true, true, false, false, false, false,
               false, false, false, false, false, false, false, false});
  EXPECT_EQ(movemask(vboolf16(0xA5A5)), size_t(0xA5A5));
  EXPECT_EQ(popcnt(vboolf16(0xA5A5)), size_t(8));
  EXPECT_TRUE (all(vboolf16(0xFFFF)));
  EXPECT_FALSE(all(vboolf16(0xFFFE)));
  EXPECT_TRUE (any(vboolf16(0x0010)));
  EXPECT_FALSE(any(vboolf16(0x0000)));
  EXPECT_TRUE (none(vboolf16(0x0000)));
}

TEST(vboolf16, logical_set_get_clear) {
  const vboolf16 a(0xFF00), b(0x0FF0);
  EXPECT_EQ(movemask(a & b), size_t(0x0F00));
  EXPECT_EQ(movemask(a | b), size_t(0xFFF0));
  EXPECT_EQ(movemask(a ^ b), size_t(0xF0F0));
  EXPECT_EQ(movemask(andn(a, b)), size_t(0xF000)); // a & !b
  vboolf16 m(0x0000);
  set(m, 3); set(m, 10);
  EXPECT_EQ(movemask(m), size_t((1u << 3) | (1u << 10)));
  EXPECT_TRUE(get(m, 3));
  clear(m, 3);
  EXPECT_FALSE(get(m, 3));
}

////////////////////////////////////////////////////////////////////////////////
// vint16
////////////////////////////////////////////////////////////////////////////////

static const vint16 iota16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

TEST(vint16, constructors_and_load_store) {
  expect_eq(vint16(7), {7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7});
  expect_eq(vint16(esimd::step), {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
  expect_eq(iota16, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});

  alignas(64) int mem[16];
  for (int i = 0; i < 16; ++i) mem[i] = i + 1;
  expect_eq(vint16::load(mem),  {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  expect_eq(vint16::loadu(mem), {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  alignas(64) int out[16] = {};
  vint16::store(out, iota16);
  EXPECT_EQ(out[0], 0); EXPECT_EQ(out[15], 15);
}

TEST(vint16, arithmetic_native_mul) {
  const vint16 a = iota16 + 1;         // 1..16
  expect_eq(a + vint16(10), {11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26});
  expect_eq(a * vint16(2),  {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32}); // native mullo_epi32
  expect_eq(-a, {-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16});
  expect_eq(a & vint16(0x3), {1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0});
}

TEST(vint16, compare_min_max_select_reduce) {
  const vint16 a = iota16;                   // 0..15
  const vint16 b = vint16(15) - iota16;      // 15..0
  expect_mask(a < b, {true, true, true, true, true, true, true, true,
                      false, false, false, false, false, false, false, false});
  expect_eq(min(a, b), {0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0});
  expect_eq(max(a, b), {15, 14, 13, 12, 11, 10, 9, 8, 8, 9, 10, 11, 12, 13, 14, 15});
  expect_eq(select(vboolf16(0x00FF), a, b),
            {0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0});
  EXPECT_EQ(reduce_add(iota16), 120); // 0+..+15
  EXPECT_EQ(reduce_min(iota16), 0);
  EXPECT_EQ(reduce_max(iota16), 15);
  EXPECT_EQ(toScalar(vint16(42) + iota16), 42);
}

TEST(vint16, movement) {
  expect_eq(shuffle<1, 0, 3, 2>(iota16),
            {1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14}); // per-128-lane
}

////////////////////////////////////////////////////////////////////////////////
// vuint16
////////////////////////////////////////////////////////////////////////////////

TEST(vuint16, arithmetic_minmax_select_reduce) {
  const vuint16 a(1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u, 16u);
  expect_eq(a + vuint16(10u),
            {11u, 12u, 13u, 14u, 15u, 16u, 17u, 18u, 19u, 20u, 21u, 22u, 23u, 24u, 25u, 26u});
  expect_eq(a & vuint16(0x3u), {1u, 2u, 3u, 0u, 1u, 2u, 3u, 0u, 1u, 2u, 3u, 0u, 1u, 2u, 3u, 0u});
  const vuint16 b = vuint16(17u) - a;  // 16..1
  expect_eq(min(a, b), {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 8u, 7u, 6u, 5u, 4u, 3u, 2u, 1u});
  expect_eq(max(a, b), {16u, 15u, 14u, 13u, 12u, 11u, 10u, 9u, 9u, 10u, 11u, 12u, 13u, 14u, 15u, 16u});
  expect_eq(select(vboolf16(0x00FF), a, b),
            {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 8u, 7u, 6u, 5u, 4u, 3u, 2u, 1u});
  EXPECT_EQ(reduce_add(a), 136u); // 1+..+16
}

////////////////////////////////////////////////////////////////////////////////
// vfloat16
////////////////////////////////////////////////////////////////////////////////

TEST(vfloat16, constructors_and_unary) {
  expect_eq(vfloat16(2.5f), {2.5f, 2.5f, 2.5f, 2.5f, 2.5f, 2.5f, 2.5f, 2.5f,
                             2.5f, 2.5f, 2.5f, 2.5f, 2.5f, 2.5f, 2.5f, 2.5f});
  expect_eq(vfloat16(esimd::step), {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f,
                                    8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f});
  const vfloat16 s(esimd::step);
  expect_eq(abs(s - vfloat16(8.f)),
            {8.f, 7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f, 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f});
  expect_eq(sqrt(vfloat16(4.f)), {2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f,
                                  2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f});
  expect_near(rcp(vfloat16(4.f)), {0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f,
                                   0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f}, 1e-4f);
}

TEST(vfloat16, binary_fma_and_reduce) {
  const vfloat16 a(esimd::step);       // 0..15
  const vfloat16 b(2.f), c(1.f);
  expect_eq(a + b, {2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f,
                    10.f, 11.f, 12.f, 13.f, 14.f, 15.f, 16.f, 17.f});
  expect_eq(a * b, {0.f, 2.f, 4.f, 6.f, 8.f, 10.f, 12.f, 14.f,
                    16.f, 18.f, 20.f, 22.f, 24.f, 26.f, 28.f, 30.f});
  expect_eq(madd(a, b, c), {1.f, 3.f, 5.f, 7.f, 9.f, 11.f, 13.f, 15.f,
                            17.f, 19.f, 21.f, 23.f, 25.f, 27.f, 29.f, 31.f});
  expect_eq(min(a, vfloat16(8.f)), {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f,
                                    8.f, 8.f, 8.f, 8.f, 8.f, 8.f, 8.f, 8.f});
  EXPECT_FLOAT_EQ(reduce_add(a), 120.f);
  EXPECT_FLOAT_EQ(reduce_min(a), 0.f);
  EXPECT_FLOAT_EQ(reduce_max(a), 15.f);
  EXPECT_FLOAT_EQ(toScalar(a + vfloat16(9.f)), 9.f);
}

TEST(vfloat16, compare_select_round) {
  const vfloat16 a(esimd::step);
  expect_mask(a < vfloat16(4.f),
              {true, true, true, true, false, false, false, false,
               false, false, false, false, false, false, false, false});
  expect_eq(select(vboolf16(0x000F), a, vfloat16(-1.f)),
            {0.f, 1.f, 2.f, 3.f, -1.f, -1.f, -1.f, -1.f,
             -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f});
  const vfloat16 f(1.4f);
  expect_eq(floor(f), {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
                       1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f});
  expect_eq(ceil(f), {2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f,
                      2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f});
}

////////////////////////////////////////////////////////////////////////////////
// vboold8 / vdouble8 / vllong8  (8-wide)
////////////////////////////////////////////////////////////////////////////////

TEST(vboold8, masks_and_logical) {
  expect_mask(vboold8(esimd::True),  {true, true, true, true, true, true, true, true});
  expect_mask(vboold8(esimd::False), {false, false, false, false, false, false, false, false});
  expect_mask(vboold8(0x0F), {true, true, true, true, false, false, false, false});
  const vboold8 a(0xCC), b(0xAA);
  EXPECT_EQ(movemask(a & b), size_t(0x88));
  EXPECT_EQ(movemask(a | b), size_t(0xEE));
  EXPECT_EQ(movemask(a ^ b), size_t(0x66));
  EXPECT_TRUE(all(vboold8(0xFF)));
  EXPECT_TRUE(any(vboold8(0x10)));
  EXPECT_TRUE(none(vboold8(0x00)));
}

TEST(vdouble8, arithmetic_fma_compare_reduce) {
  const vdouble8 a(esimd::step);       // 0..7
  const vdouble8 b(2.0), c(1.0);
  expect_eq(a + b, {2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
  expect_eq(a * b, {0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0});
  expect_eq(madd(a, b, c), {1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0});
  expect_eq(min(a, vdouble8(4.0)), {0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0});
  expect_mask(a < vdouble8(4.0), {true, true, true, true, false, false, false, false});
  expect_eq(select(vboold8(0x0F), a, vdouble8(-1.0)),
            {0.0, 1.0, 2.0, 3.0, -1.0, -1.0, -1.0, -1.0});
  EXPECT_DOUBLE_EQ(reduce_add(a), 28.0);
  EXPECT_DOUBLE_EQ(reduce_min(a), 0.0);
  EXPECT_DOUBLE_EQ(reduce_max(a), 7.0);
  EXPECT_DOUBLE_EQ(toScalar(a + vdouble8(3.0)), 3.0);
}

TEST(vllong8, arithmetic_native_mul_compare_reduce) {
  const vllong8 a(esimd::step);        // 0..7
  const vllong8 one(1ll);
  const vllong8 b = a + one;           // 1..8
  expect_eq(b, {1ll, 2ll, 3ll, 4ll, 5ll, 6ll, 7ll, 8ll});
  expect_eq(b * vllong8(2ll), {2ll, 4ll, 6ll, 8ll, 10ll, 12ll, 14ll, 16ll}); // native mullo_epi64
  expect_eq(b + vllong8(10ll), {11ll, 12ll, 13ll, 14ll, 15ll, 16ll, 17ll, 18ll});
  expect_mask(a < vllong8(4ll), {true, true, true, true, false, false, false, false});
  expect_eq(min(a, vllong8(4ll)), {0ll, 1ll, 2ll, 3ll, 4ll, 4ll, 4ll, 4ll});
  expect_eq(select(vboold8(0x0F), a, vllong8(-1ll)),
            {0ll, 1ll, 2ll, 3ll, -1ll, -1ll, -1ll, -1ll});
  EXPECT_EQ(reduce_add(a), 28ll);
  EXPECT_EQ(toScalar(a + vllong8(5ll)), 5ll);
}
