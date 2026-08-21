# Copyright 2026 notchla liso.lorenzo@gmail.com
# SPDX-License-Identifier: Apache-2.0
#
# Builds the AVX512 showcase with the AVX512 flag/define set (ESIMD_FLAGS_AVX512):
#   -march=skylake-avx512 -mavx2 -mfma -mf16c -mbmi -mbmi2 -mlzcnt
#   -D__AVX512F__ -D__AVX512VL__ -D__AVX512DQ__ -D__AVX512BW__
#   -D__AVX2__ -D__AVX__ -D__SSE4_2__ -D__SSE4_1__ -D__LZCNT__ -D__BMI__

esimd_add_isa_target(esimd_example_avx512 AVX512
  SOURCES avx512_example.cpp)
