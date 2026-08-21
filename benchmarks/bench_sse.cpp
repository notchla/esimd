// Copyright 2026 notchla liso.lorenzo@gmail.com
// SPDX-License-Identifier: Apache-2.0
//
// Representative micro-benchmarks for the SSE (4-wide) types. These track
// relative regressions, not absolute performance.

#include <esimd/sse.h>
#include <benchmark/benchmark.h>
#include <vector>

using namespace esimd;

static std::vector<vfloat4> make_data(size_t n) {
  std::vector<vfloat4> v(n);
  for (size_t i = 0; i < n; ++i)
    v[i] = vfloat4(float(i), float(i + 1), float(i + 2), float(i + 3));
  return v;
}

static void BM_vfloat4_add(benchmark::State& state) {
  auto data = make_data(1024);
  for (auto _ : state) {
    vfloat4 acc(zero);
    for (const auto& x : data) acc = acc + x;
    benchmark::DoNotOptimize(acc);
  }
}
BENCHMARK(BM_vfloat4_add);

static void BM_vfloat4_madd(benchmark::State& state) {
  auto data = make_data(1024);
  const vfloat4 k(1.5f);
  for (auto _ : state) {
    vfloat4 acc(zero);
    for (const auto& x : data) acc = madd(x, k, acc);
    benchmark::DoNotOptimize(acc);
  }
}
BENCHMARK(BM_vfloat4_madd);

static void BM_vfloat4_sqrt_rcp(benchmark::State& state) {
  auto data = make_data(1024);
  for (auto _ : state) {
    vfloat4 acc(zero);
    for (const auto& x : data) acc = acc + rcp(sqrt(x + vfloat4(one)));
    benchmark::DoNotOptimize(acc);
  }
}
BENCHMARK(BM_vfloat4_sqrt_rcp);

static void BM_vfloat4_reduce_add(benchmark::State& state) {
  auto data = make_data(1024);
  for (auto _ : state) {
    float acc = 0.f;
    for (const auto& x : data) acc += reduce_add(x);
    benchmark::DoNotOptimize(acc);
  }
}
BENCHMARK(BM_vfloat4_reduce_add);

BENCHMARK_MAIN();
