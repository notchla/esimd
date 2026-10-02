# Copyright 2026 notchla liso.lorenzo@gmail.com
# SPDX-License-Identifier: Apache-2.0
#
# The source names no fixed width (vfloatx / VSIZEX), so each per-ISA build
# processes a different lane count from the same file.

esimd_add_isa_target(esimd_example_portable_sse    SSE42  SOURCES portable_example.cpp)
esimd_add_isa_target(esimd_example_portable_avx    AVX    SOURCES portable_example.cpp)
esimd_add_isa_target(esimd_example_portable_avx2   AVX2   SOURCES portable_example.cpp)
esimd_add_isa_target(esimd_example_portable_avx512 AVX512 SOURCES portable_example.cpp)
esimd_add_isa_target(esimd_example_portable_neon   NEON   SOURCES portable_example.cpp)
esimd_add_isa_target(esimd_example_portable_neon2x NEON2X SOURCES portable_example.cpp)
