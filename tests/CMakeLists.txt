# Copyright 2009-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# One test executable per ISA. Each compiles the same assertions with that ISA's
# flags, so every backend is validated. Host-unsupported ISAs are skipped by
# esimd_add_isa_target.

esimd_add_isa_target(esimd_test_sse42 SSE42
  SOURCES test_sse.cpp
  LINKS   GTest::gtest_main
  LABELS  sse)

esimd_add_isa_target(esimd_test_avx AVX
  SOURCES test_avx.cpp
  LINKS   GTest::gtest_main
  LABELS  avx)
