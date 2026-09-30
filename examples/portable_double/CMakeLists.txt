# Copyright 2026 notchla liso.lorenzo@gmail.com
# SPDX-License-Identifier: Apache-2.0
#
# Same source built per ISA, like portable/.

esimd_add_isa_target(esimd_example_portable_double_sse    SSE42  SOURCES portable_double_example.cpp)
esimd_add_isa_target(esimd_example_portable_double_avx    AVX    SOURCES portable_double_example.cpp)
esimd_add_isa_target(esimd_example_portable_double_avx2   AVX2   SOURCES portable_double_example.cpp)
esimd_add_isa_target(esimd_example_portable_double_avx512 AVX512 SOURCES portable_double_example.cpp)
esimd_add_isa_target(esimd_example_portable_double_neon   NEON   SOURCES portable_double_example.cpp)
esimd_add_isa_target(esimd_example_portable_double_neon2x NEON2X SOURCES portable_double_example.cpp)
