# Copyright 2026 notchla liso.lorenzo@gmail.com
# SPDX-License-Identifier: Apache-2.0
#
# One showcase executable per instruction set. Each lives in its own subfolder
# and is compiled with that ISA's flags via esimd_add_isa_target (defined in the
# top-level CMakeLists), so only the ISAs the host can execute are built.

add_subdirectory(sse)
add_subdirectory(avx)
add_subdirectory(avx2)
add_subdirectory(avx512)

# ARM backends (configured only when targeting AArch64).
add_subdirectory(neon)
add_subdirectory(neon2x)

# Width-agnostic showcase (vfloatx / VSIZEX): the same source built for every ISA.
add_subdirectory(portable)
