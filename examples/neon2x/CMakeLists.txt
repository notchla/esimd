# Copyright 2026 notchla liso.lorenzo@gmail.com
# SPDX-License-Identifier: Apache-2.0
#
# Only configured when the target is AArch64 (see esimd_add_isa_target).

esimd_add_isa_target(esimd_example_neon2x NEON2X
  SOURCES ../avx2/avx2_example.cpp)
