# Copyright 2026 notchla liso.lorenzo@gmail.com
# SPDX-License-Identifier: Apache-2.0

# One benchmark executable per ISA (unlabeled -> built but not run by ctest).

esimd_add_isa_target(esimd_bench_sse SSE42
  SOURCES bench_sse.cpp
  LINKS   benchmark::benchmark)

esimd_add_isa_target(esimd_bench_avx AVX
  SOURCES bench_avx.cpp
  LINKS   benchmark::benchmark)

esimd_add_isa_target(esimd_bench_avx2 AVX2
  SOURCES bench_avx2.cpp
  LINKS   benchmark::benchmark)
