// Copyright 2026 notchla liso.lorenzo@gmail.com
// SPDX-License-Identifier: Apache-2.0
//
// Representative micro-benchmarks for the AVX2 native-integer paths (vint8,
// vuint8, vllong4). These track relative regressions, not absolute performance.

#include <esimd/avx.h>
#include <benchmark/benchmark.h>
#include <vector>

using namespace esimd;

static std::vector<vint8> make_int_data(size_t n) {
  std::vector<vint8> v(n);
  for (size_t i = 0; i < n; ++i)
    v[i] = vint8(int(i), int(i + 1), int(i + 2), int(i + 3),
                 int(i + 4), int(i + 5), int(i + 6), int(i + 7));
  return v;
}

static std::vector<vllong4> make_llong_data(size_t n) {
  std::vector<vllong4> v(n);
  for (size_t i = 0; i < n; ++i)
    v[i] = vllong4((long long)i, (long long)(i + 1), (long long)(i + 2), (long long)(i + 3));
  return v;
}

static void BM_vint8_mul(benchmark::State& state) {
  auto data = make_int_data(1024);
  const vint8 k(3);
  for (auto _ : state) {
    vint8 acc(zero);
    for (const auto& x : data) acc = acc + x * k; // native _mm256_mullo_epi32
    benchmark::DoNotOptimize(acc);
  }
}
BENCHMARK(BM_vint8_mul);

static void BM_vint8_min_max(benchmark::State& state) {
  auto data = make_int_data(1024);
  for (auto _ : state) {
    vint8 lo(vint8(1 << 30)), hi(vint8(-(1 << 30)));
    for (const auto& x : data) { lo = min(lo, x); hi = max(hi, x); }
    benchmark::DoNotOptimize(lo);
    benchmark::DoNotOptimize(hi);
  }
}
BENCHMARK(BM_vint8_min_max);

static void BM_vint8_reduce_add(benchmark::State& state) {
  auto data = make_int_data(1024);
  for (auto _ : state) {
    int acc = 0;
    for (const auto& x : data) acc += reduce_add(x);
    benchmark::DoNotOptimize(acc);
  }
}
BENCHMARK(BM_vint8_reduce_add);

static void BM_vllong4_add(benchmark::State& state) {
  auto data = make_llong_data(1024);
  for (auto _ : state) {
    vllong4 acc(zero);
    for (const auto& x : data) acc = acc + x; // native _mm256_add_epi64
    benchmark::DoNotOptimize(acc);
  }
}
BENCHMARK(BM_vllong4_add);

BENCHMARK_MAIN();
