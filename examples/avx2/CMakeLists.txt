# Copyright 2026 notchla liso.lorenzo@gmail.com
# SPDX-License-Identifier: Apache-2.0
#
# Builds the AVX2 showcase with the AVX2 flag/define set (ESIMD_FLAGS_AVX2):
#   -mavx2 -mfma -mf16c -mbmi -mbmi2 -mlzcnt
#   -D__AVX2__ -D__AVX__ -D__SSE4_2__ -D__SSE4_1__ -D__LZCNT__ -D__BMI__

esimd_add_isa_target(esimd_example_avx2 AVX2
  SOURCES avx2_example.cpp)
