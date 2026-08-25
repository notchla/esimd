# Copyright 2026 notchla liso.lorenzo@gmail.com
# SPDX-License-Identifier: Apache-2.0
#
# The portable example names no fixed width (vfloatx / VSIZEX), so the SAME source
# is built once per ISA. Each executable processes a different lane count (4 under
# SSE, 8 under AVX/AVX2/AVX512) purely from its compile flags. Only the ISAs the
# host can execute are built (host-gated by esimd_add_isa_target).

esimd_add_isa_target(esimd_example_portable_sse    SSE42  SOURCES portable_example.cpp)
esimd_add_isa_target(esimd_example_portable_avx    AVX    SOURCES portable_example.cpp)
esimd_add_isa_target(esimd_example_portable_avx2   AVX2   SOURCES portable_example.cpp)
esimd_add_isa_target(esimd_example_portable_avx512 AVX512 SOURCES portable_example.cpp)
esimd_add_isa_target(esimd_example_portable_neon   NEON   SOURCES portable_example.cpp)
esimd_add_isa_target(esimd_example_portable_neon2x NEON2X SOURCES portable_example.cpp)
