// Copyright 2026 notchla liso.lorenzo@gmail.com
// SPDX-License-Identifier: Apache-2.0
//
// Representative micro-benchmarks for the AVX512 types (vfloat16, vint16,
// vdouble8). These track relative regressions, not absolute performance.
// (AVX512 downclocking can make these noisy; use for regression tracking only.)

#include <esimd/avx.h>
#include <benchmark/benchmark.h>
#include <vector>

using namespace esimd;

static std::vector<vfloat16> make_float_data(size_t n) {
  std::vector<vfloat16> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = vfloat16(float(i)) + vfloat16(step);
  return v;
}

static std::vector<vdouble8> make_double_data(size_t n) {
  std::vector<vdouble8> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = vdouble8(double(i)) + vdouble8(step);
  return v;
}

static void BM_vfloat16_madd(benchmark::State& state) {
  auto data = make_float_data(1024);
  const vfloat16 k(1.5f);
  for (auto _ : state) {
    vfloat16 acc(zero);
    for (const auto& x : data) acc = madd(x, k, acc);
    benchmark::DoNotOptimize(acc);
  }
}
BENCHMARK(BM_vfloat16_madd);

static void BM_vfloat16_sqrt_rcp(benchmark::State& state) {
  auto data = make_float_data(1024);
  for (auto _ : state) {
    vfloat16 acc(zero);
    for (const auto& x : data) acc = acc + rcp(sqrt(x + vfloat16(one)));
    benchmark::DoNotOptimize(acc);
  }
}
BENCHMARK(BM_vfloat16_sqrt_rcp);

static void BM_vfloat16_reduce_add(benchmark::State& state) {
  auto data = make_float_data(1024);
  for (auto _ : state) {
    float acc = 0.f;
    for (const auto& x : data) acc += reduce_add(x);
    benchmark::DoNotOptimize(acc);
  }
}
BENCHMARK(BM_vfloat16_reduce_add);

static void BM_vdouble8_madd(benchmark::State& state) {
  auto data = make_double_data(1024);
  const vdouble8 k(1.5);
  for (auto _ : state) {
    vdouble8 acc(zero);
    for (const auto& x : data) acc = madd(x, k, acc);
    benchmark::DoNotOptimize(acc);
  }
}
BENCHMARK(BM_vdouble8_madd);

BENCHMARK_MAIN();
