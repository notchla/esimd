// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Correctness tests for the AVX2 native-integer esimd paths: vint8_avx2.h,
// vuint8_avx2.h and the 4-wide 64-bit vllong4_avx2.h. Compiled with AVX2 flags
// (see tests/CMakeLists.txt), so avx.h selects these native paths over the
// two-SSE-halves plain-AVX integer headers. Where an operation has the same
// semantics as the plain-AVX backend (Phase 2), the expected values match; the
// point here is that the AVX2 codegen produces identical results.

#include <esimd/avx.h>
#include "test_helpers.h"

using namespace esimd;
using namespace esimd_test;

////////////////////////////////////////////////////////////////////////////////
// vint8 (native AVX2: mullo_epi32, min/max_epi32, permute)
////////////////////////////////////////////////////////////////////////////////

TEST(vint8_avx2, arithmetic_native_mul) {
  const vint8 a(1, 2, 3, 4, 5, 6, 7, 8), b(2, 3, 4, 5, 6, 7, 8, 9);
  expect_eq(a + b, {3, 5, 7, 9, 11, 13, 15, 17});
  expect_eq(a * b, {2, 6, 12, 20, 30, 42, 56, 72});   // native _mm256_mullo_epi32
  expect_eq(a * vint8(3), {3, 6, 9, 12, 15, 18, 21, 24});
  expect_eq(-a, {-1, -2, -3, -4, -5, -6, -7, -8});
  expect_eq(abs(vint8(-1, 2, -3, 4, -5, 6, -7, 8)), {1, 2, 3, 4, 5, 6, 7, 8});
}

TEST(vint8_avx2, compare_min_max_select_reduce) {
  const vint8 a(1, 2, 3, 4, 5, 6, 7, 8), b(8, 7, 6, 5, 4, 3, 2, 1);
  expect_mask(a < b, {true, true, true, true, false, false, false, false});
  expect_mask(a >= b, {false, false, false, false, true, true, true, true});
  expect_eq(min(a, b), {1, 2, 3, 4, 4, 3, 2, 1});
  expect_eq(max(a, b), {8, 7, 6, 5, 5, 6, 7, 8});
  expect_eq(select(vboolf8(true, false, true, false, true, false, true, false), a, b),
            {1, 7, 3, 5, 5, 3, 7, 1});
  const vint8 v(3, 1, 4, 1, 5, 9, 2, 6);
  EXPECT_EQ(reduce_add(v), 31);
  EXPECT_EQ(reduce_min(v), 1);
  EXPECT_EQ(reduce_max(v), 9);
}

TEST(vint8_avx2, permute_and_shuffle) {
  const vint8 v(10, 11, 12, 13, 14, 15, 16, 17);
  const __m256i rev = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
  expect_eq(permute(v, rev), {17, 16, 15, 14, 13, 12, 11, 10}); // cross-lane permute
  expect_eq(shuffle<1, 0, 3, 2>(v), {11, 10, 13, 12, 15, 14, 17, 16}); // per-128-lane
  expect_eq(unpacklo(vint8(0, 1, 2, 3, 4, 5, 6, 7), vint8(10, 11, 12, 13, 14, 15, 16, 17)),
            {0, 10, 1, 11, 4, 14, 5, 15});
}

////////////////////////////////////////////////////////////////////////////////
// vuint8 (native AVX2: min/max_epu32, blend select). Ordered compares remain
// AVX512VL-gated, so only == / != are exercised here.
////////////////////////////////////////////////////////////////////////////////

TEST(vuint8_avx2, arithmetic_bitwise_shift) {
  const vuint8 a(1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u), b(10u, 20u, 30u, 40u, 50u, 60u, 70u, 80u);
  expect_eq(a + b, {11u, 22u, 33u, 44u, 55u, 66u, 77u, 88u});
  expect_eq(b - a, {9u, 18u, 27u, 36u, 45u, 54u, 63u, 72u});
  expect_eq(a & vuint8(0x3u), {1u, 2u, 3u, 0u, 1u, 2u, 3u, 0u});
  expect_eq(a | vuint8(0x10u), {17u, 18u, 19u, 20u, 21u, 22u, 23u, 24u});
  expect_eq(vuint8(1u) << 3u, {8u, 8u, 8u, 8u, 8u, 8u, 8u, 8u});
  expect_eq(vuint8(64u) >> 2u, {16u, 16u, 16u, 16u, 16u, 16u, 16u, 16u});
}

TEST(vuint8_avx2, compare_minmax_select_reduce) {
  const vuint8 a(1u, 5u, 3u, 8u, 2u, 7u, 4u, 6u), b(4u, 2u, 6u, 7u, 2u, 9u, 1u, 6u);
  expect_mask(a == b, {false, false, false, false, true, false, false, true});
  expect_mask(a != b, {true, true, true, true, false, true, true, false});
  expect_eq(min(a, b), {1u, 2u, 3u, 7u, 2u, 7u, 1u, 6u}); // native _mm256_min_epu32
  expect_eq(max(a, b), {4u, 5u, 6u, 8u, 2u, 9u, 4u, 6u});
  expect_eq(select(vboolf8(true, false, true, false, true, false, true, false), a, b),
            {1u, 2u, 3u, 7u, 2u, 9u, 4u, 6u});
  EXPECT_EQ(reduce_add(vuint8(1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u)), 36);
}

////////////////////////////////////////////////////////////////////////////////
// vllong4 (256-bit, 4-wide signed 64-bit)
////////////////////////////////////////////////////////////////////////////////

TEST(vllong4, constructors_and_load_store) {
  expect_eq(vllong4(7ll), {7ll, 7ll, 7ll, 7ll});
  expect_eq(vllong4(1ll, 2ll, 3ll, 4ll), {1ll, 2ll, 3ll, 4ll});
  expect_eq(vllong4(esimd::zero), {0ll, 0ll, 0ll, 0ll});
  expect_eq(vllong4(esimd::step), {0ll, 1ll, 2ll, 3ll});

  alignas(32) long long mem[4] = {5, 6, 7, 8};
  expect_eq(vllong4::load(mem),  {5ll, 6ll, 7ll, 8ll});
  expect_eq(vllong4::loadu(mem), {5ll, 6ll, 7ll, 8ll});
  alignas(32) long long out[4] = {};
  vllong4::store(out, vllong4(1ll, 2ll, 3ll, 4ll));
  EXPECT_EQ(out[0], 1ll); EXPECT_EQ(out[3], 4ll);
}

TEST(vllong4, arithmetic_bitwise) {
  const vllong4 a(1ll, 2ll, 3ll, 4ll), b(10ll, 20ll, 30ll, 40ll);
  expect_eq(a + b, {11ll, 22ll, 33ll, 44ll});
  expect_eq(b - a, {9ll, 18ll, 27ll, 36ll});
  expect_eq(-a, {-1ll, -2ll, -3ll, -4ll});
  // operator* is _mm256_mul_epi32: low 32 bits of each 64-bit lane, result 64-bit.
  expect_eq(a * b, {10ll, 40ll, 90ll, 160ll});
  expect_eq(a & vllong4(0x3ll), {1ll, 2ll, 3ll, 0ll});
  expect_eq(a | vllong4(0x10ll), {17ll, 18ll, 19ll, 20ll});
  expect_eq(a ^ a, {0ll, 0ll, 0ll, 0ll});
  expect_eq(vllong4(1ll) << 4ll, {16ll, 16ll, 16ll, 16ll});
}

TEST(vllong4, compare_select_reduce) {
  const vllong4 a(1ll, 2ll, 3ll, 4ll), b(4ll, 2ll, 2ll, 1ll);
  expect_mask(a == b, {false, true, false, false});
  expect_mask(a != b, {true, false, true, true});
  expect_mask(a <  b, {true, false, false, false}); // signed _mm256_cmpgt_epi64
  expect_mask(a >  b, {false, false, true, true});
  expect_mask(a >= b, {false, true, true, true});
  expect_eq(select(vboold4(0x5), a, b), {1ll, 2ll, 3ll, 1ll});
  const vllong4 v(3ll, 1ll, 4ll, 2ll);
  EXPECT_EQ(reduce_add(v), 10ll);
  EXPECT_EQ(toScalar(vllong4(42ll, 0ll, 0ll, 0ll)), 42ll);
}
