// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Integration tests for the top-level <esimd/esimd.h>: the ISA #include
// dispatch plus the cross-width helper templates (isfinite, foreach_unique,
// next_unique / next_unique_index, foreach2) and the default-width aliases
// (vfloatx / vintx / vboolx / VSIZEX). Compiled once per ISA, so these run at
// width 4 (SSE) and width 8 (AVX / AVX2 / AVX512) from the same source.

#include <esimd/esimd.h>
#include "test_helpers.h"

#include <set>
#include <vector>
#include <limits>

using namespace esimd;
using namespace esimd_test;

////////////////////////////////////////////////////////////////////////////////
// isfinite (cross-width)
////////////////////////////////////////////////////////////////////////////////

TEST(esimd_integration, isfinite) {
  alignas(64) float buf[VSIZEX];
  for (int i = 0; i < VSIZEX; ++i) buf[i] = float(i) - 2.f; // all finite
  const vfloatx a = vfloatx::loadu(buf);
  // isfinite's parameter is the alias vfloat<N> (a non-deduced context), so the
  // width is supplied explicitly.
  const vboolx fa = isfinite<VSIZEX>(a);
  for (int i = 0; i < VSIZEX; ++i) EXPECT_TRUE(fa[i]) << "lane " << i;

  buf[0] = std::numeric_limits<float>::infinity();
  buf[VSIZEX - 1] = std::numeric_limits<float>::quiet_NaN();
  const vboolx fb = isfinite<VSIZEX>(vfloatx::loadu(buf));
  EXPECT_FALSE(fb[0]);
  EXPECT_FALSE(fb[VSIZEX - 1]);
  for (int i = 1; i < VSIZEX - 1; ++i) EXPECT_TRUE(fb[i]) << "lane " << i;
}

////////////////////////////////////////////////////////////////////////////////
// foreach_unique (cross-width)
////////////////////////////////////////////////////////////////////////////////

TEST(esimd_integration, foreach_unique_partitions_lanes) {
  // Build a vintx whose lanes repeat a small set of values.
  alignas(64) int buf[VSIZEX];
  for (int i = 0; i < VSIZEX; ++i) buf[i] = (i % 3) * 10; // values in {0,10,20}
  const vintx vi = vintx::loadu(buf);

  std::set<int> seen_values;
  int lanes_covered = 0;
  int visits = 0;
  foreach_unique(vboolx(True), vi, [&](const vboolx& m, int value) {
    ++visits;
    EXPECT_TRUE(seen_values.insert(value).second) << "value visited twice: " << value;
    // every lane in this mask must actually hold `value`
    for (int i = 0; i < VSIZEX; ++i)
      if (m[i]) { EXPECT_EQ(buf[i], value); ++lanes_covered; }
  });

  EXPECT_EQ(lanes_covered, VSIZEX);           // masks partition all lanes
  const std::set<int> expected = {0, 10, 20};
  EXPECT_EQ(seen_values, expected);
  EXPECT_EQ(visits, int(expected.size()));
}

////////////////////////////////////////////////////////////////////////////////
// next_unique / next_unique_index (cross-width)
////////////////////////////////////////////////////////////////////////////////

TEST(esimd_integration, next_unique_drains_mask) {
  alignas(64) int buf[VSIZEX];
  for (int i = 0; i < VSIZEX; ++i) buf[i] = (i & 1) ? 7 : 3; // {3,7} interleaved
  const vintx vi = vintx::loadu(buf);

  vboolx valid(True);
  std::set<int> values;
  int total = 0;
  while (any(valid)) {
    vboolx vmask;
    const int v = next_unique(valid, vi, vmask);
    values.insert(v);
    for (int i = 0; i < VSIZEX; ++i) if (vmask[i]) { EXPECT_EQ(buf[i], v); ++total; }
  }
  EXPECT_EQ(total, VSIZEX);
  const std::set<int> expected = (VSIZEX == 1) ? std::set<int>{3} : std::set<int>{3, 7};
  EXPECT_EQ(values, expected);
}

TEST(esimd_integration, next_unique_index_returns_lane) {
  alignas(64) int buf[VSIZEX];
  for (int i = 0; i < VSIZEX; ++i) buf[i] = i * 100; // all distinct
  const vintx vi = vintx::loadu(buf);

  vboolx valid(True);
  int iterations = 0;
  while (any(valid)) {
    vboolx vmask;
    const int j = next_unique_index(valid, vi, vmask);
    ASSERT_GE(j, 0); ASSERT_LT(j, VSIZEX);
    // returned index j identifies a lane whose value the mask selects
    EXPECT_TRUE(vmask[j]);
    ++iterations;
  }
  EXPECT_EQ(iterations, VSIZEX); // all-distinct -> one iteration per lane
}

////////////////////////////////////////////////////////////////////////////////
// foreach2 (uses VSIZEX / vintx / vboolx / step internally)
////////////////////////////////////////////////////////////////////////////////

TEST(esimd_integration, foreach2_covers_grid_once) {
  const int x0 = 0, x1 = 5, y0 = 0, y1 = 3; // 15 cells, not a multiple of VSIZEX
  std::set<std::pair<int, int>> visited;
  int count = 0;
  foreach2(x0, x1, y0, y1, [&](const vboolx& valid, const vintx& X, const vintx& Y) {
    for (int i = 0; i < VSIZEX; ++i) {
      if (valid[i]) {
        ++count;
        const auto p = std::make_pair(int(X[i]), int(Y[i]));
        EXPECT_TRUE(visited.insert(p).second) << "cell visited twice: (" << p.first << "," << p.second << ")";
        EXPECT_GE(p.first, x0);  EXPECT_LT(p.first, x1);
        EXPECT_GE(p.second, y0); EXPECT_LT(p.second, y1);
      }
    }
  });
  EXPECT_EQ(count, (x1 - x0) * (y1 - y0)); // every cell exactly once
  EXPECT_EQ(int(visited.size()), (x1 - x0) * (y1 - y0));
}

////////////////////////////////////////////////////////////////////////////////
// default-width aliases resolve to the ISA's native width
////////////////////////////////////////////////////////////////////////////////

TEST(esimd_integration, default_width_aliases) {
#if defined(__AVX__)
  EXPECT_EQ(VSIZEX, 8);
#else
  EXPECT_EQ(VSIZEX, 4);
#endif
  // vfloatx/vintx/vboolx are the VSIZEX-wide types; a round-trip sanity check.
  const vintx a = vintx(step);
  EXPECT_EQ(reduce_add(a), (VSIZEX * (VSIZEX - 1)) / 2); // 0+1+..+(N-1)
  const vfloatx f = vfloatx(one);
  EXPECT_FLOAT_EQ(reduce_add(f), float(VSIZEX));
  EXPECT_TRUE(all(vboolx(True)));
}
