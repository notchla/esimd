# Copyright 2009-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# One benchmark executable per ISA (unlabeled -> built but not run by ctest).

esimd_add_isa_target(esimd_bench_sse SSE42
  SOURCES bench_sse.cpp
  LINKS   benchmark::benchmark)

esimd_add_isa_target(esimd_bench_avx AVX
  SOURCES bench_avx.cpp
  LINKS   benchmark::benchmark)
