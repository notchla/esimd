# Copyright 2026 notchla liso.lorenzo@gmail.com
# SPDX-License-Identifier: Apache-2.0
#
# Standalone equivalent (without this repo's CMake helper):
#   add_executable(esimd_example_sse sse_example.cpp)
#   target_include_directories(esimd_example_sse PRIVATE ${ESIMD}/include)
#   target_compile_options(esimd_example_sse PRIVATE
#     -msse4.2 -D__SSE__ -D__SSE2__ -D__SSE4_1__ -D__SSE4_2__)

esimd_add_isa_target(esimd_example_sse SSE42
  SOURCES sse_example.cpp)
