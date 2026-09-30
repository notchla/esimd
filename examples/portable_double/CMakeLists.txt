# Copyright 2026 notchla liso.lorenzo@gmail.com
# SPDX-License-Identifier: Apache-2.0
#
# Same source built per ISA, like portable/, but only where vdoublexd has a SIMD
# backend: SSE and NEON have no vdouble class, and NEON2X lacks vdouble4.

esimd_add_isa_target(esimd_example_portable_double_avx    AVX    SOURCES portable_double_example.cpp)
esimd_add_isa_target(esimd_example_portable_double_avx2   AVX2   SOURCES portable_double_example.cpp)
esimd_add_isa_target(esimd_example_portable_double_avx512 AVX512 SOURCES portable_double_example.cpp)
