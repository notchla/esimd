# Copyright 2026 notchla liso.lorenzo@gmail.com
# SPDX-License-Identifier: Apache-2.0
#
# Builds the SSE showcase with the SSE4.2 flag/define set. esimd_add_isa_target
# (top-level CMakeLists) applies ESIMD_FLAGS_SSE42 and links the header-only
# `esimd` interface target — no other dependency.
#
# Standalone equivalent (without this repo's CMake helper):
#   add_executable(esimd_example_sse sse_example.cpp)
#   target_include_directories(esimd_example_sse PRIVATE ${ESIMD}/include)
#   target_compile_options(esimd_example_sse PRIVATE
#     -msse4.2 -D__SSE__ -D__SSE2__ -D__SSE4_1__ -D__SSE4_2__)

esimd_add_isa_target(esimd_example_sse SSE42
  SOURCES sse_example.cpp)
