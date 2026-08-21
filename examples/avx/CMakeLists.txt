# Copyright 2026 notchla liso.lorenzo@gmail.com
# SPDX-License-Identifier: Apache-2.0
#
# Builds the AVX showcase with the AVX flag/define set (ESIMD_FLAGS_AVX):
#   -mavx -mbmi -D__AVX__ -D__SSE4_2__ -D__SSE4_1__ -D__BMI__

esimd_add_isa_target(esimd_example_avx AVX
  SOURCES avx_example.cpp)
