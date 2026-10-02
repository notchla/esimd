# Copyright 2009-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# One test executable per ISA, each compiling the same assertions with that
# ISA's flags.

esimd_add_isa_target(esimd_test_sse42 SSE42
  SOURCES test_sse.cpp
  LINKS   GTest::gtest_main
  LABELS  sse)

# The SSE headers pick AVX / AVX512VL instructions when those are enabled, so
# the same suite is rebuilt at AVX2 and AVX512 to cover those branches.
esimd_add_isa_target(esimd_test_sse_avx2 AVX2
  SOURCES test_sse.cpp
  LINKS   GTest::gtest_main
  LABELS  avx2)

esimd_add_isa_target(esimd_test_sse_avx512 AVX512
  SOURCES test_sse.cpp
  LINKS   GTest::gtest_main
  LABELS  avx512)

esimd_add_isa_target(esimd_test_avx AVX
  SOURCES test_avx.cpp
  LINKS   GTest::gtest_main
  LABELS  avx)

esimd_add_isa_target(esimd_test_avx2 AVX2
  SOURCES test_avx2.cpp
  LINKS   GTest::gtest_main
  LABELS  avx2)

esimd_add_isa_target(esimd_test_avx512 AVX512
  SOURCES test_avx512.cpp
  LINKS   GTest::gtest_main
  LABELS  avx512)

# Integration test: <esimd/esimd.h> only, built once per ISA so the cross-width
# helpers and default-width aliases are validated at each native width.
esimd_add_isa_target(esimd_test_integration_sse42 SSE42
  SOURCES test_esimd.cpp
  LINKS   GTest::gtest_main
  LABELS  sse integration)

esimd_add_isa_target(esimd_test_integration_avx AVX
  SOURCES test_esimd.cpp
  LINKS   GTest::gtest_main
  LABELS  avx integration)

esimd_add_isa_target(esimd_test_integration_avx2 AVX2
  SOURCES test_esimd.cpp
  LINKS   GTest::gtest_main
  LABELS  avx2 integration)

esimd_add_isa_target(esimd_test_integration_avx512 AVX512
  SOURCES test_esimd.cpp
  LINKS   GTest::gtest_main
  LABELS  avx512 integration)

# ARM: NEON is 4-wide, so it runs the SSE suite; NEON2X is double-pumped
# 8-wide, so it runs the AVX and AVX2 suites.
esimd_add_isa_target(esimd_test_neon NEON
  SOURCES test_sse.cpp
  LINKS   GTest::gtest_main
  LABELS  neon)

esimd_add_isa_target(esimd_test_neon2x_avx NEON2X
  SOURCES test_avx.cpp
  LINKS   GTest::gtest_main
  LABELS  neon2x)

esimd_add_isa_target(esimd_test_neon2x NEON2X
  SOURCES test_avx2.cpp
  LINKS   GTest::gtest_main
  LABELS  neon2x)

esimd_add_isa_target(esimd_test_integration_neon NEON
  SOURCES test_esimd.cpp
  LINKS   GTest::gtest_main
  LABELS  neon integration)

esimd_add_isa_target(esimd_test_integration_neon2x NEON2X
  SOURCES test_esimd.cpp
  LINKS   GTest::gtest_main
  LABELS  neon2x integration)

# Optional trigonometry layer (<esimd/trig.h>), on every ISA: AVX2/AVX512/NEON/
# NEON2X exercise a SLEEF backend, SSE42/AVX the per-lane libm fallback.
esimd_add_isa_target(esimd_test_trig_sse42 SSE42
  SOURCES test_trig.cpp
  LINKS   GTest::gtest_main
  LABELS  sse trig)

esimd_add_isa_target(esimd_test_trig_avx AVX
  SOURCES test_trig.cpp
  LINKS   GTest::gtest_main
  LABELS  avx trig)

esimd_add_isa_target(esimd_test_trig_avx2 AVX2
  SOURCES test_trig.cpp
  LINKS   GTest::gtest_main
  LABELS  avx2 trig)

esimd_add_isa_target(esimd_test_trig_avx512 AVX512
  SOURCES test_trig.cpp
  LINKS   GTest::gtest_main
  LABELS  avx512 trig)

esimd_add_isa_target(esimd_test_trig_neon NEON
  SOURCES test_trig.cpp
  LINKS   GTest::gtest_main
  LABELS  neon trig)

esimd_add_isa_target(esimd_test_trig_neon2x NEON2X
  SOURCES test_trig.cpp
  LINKS   GTest::gtest_main
  LABELS  neon2x trig)

# Required by <esimd/trig.h>: without it SLEEF's error compensation breaks and
# the ULP assertions are void. esimd::esimd does not force it on consumers.
# esimd_add_isa_target skips ISAs the host cannot execute, hence if(TARGET).
# cl's /fp:precise has not contracted since VS 2022 17.0.
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  set(esimd_no_contract /fp:precise)
elseif(MSVC) # clang-cl
  set(esimd_no_contract /clang:-ffp-contract=off)
else()
  set(esimd_no_contract -ffp-contract=off)
endif()
foreach(t esimd_test_trig_sse42 esimd_test_trig_avx
          esimd_test_trig_avx2  esimd_test_trig_avx512
          esimd_test_trig_neon  esimd_test_trig_neon2x)
  if(TARGET ${t})
    target_compile_options(${t} PRIVATE ${esimd_no_contract})
  endif()
endforeach()

# Optional data-types layer (<esimd/types.h>), on every ISA.
esimd_add_isa_target(esimd_test_vec2_sse42 SSE42
  SOURCES test_vec2.cpp
  LINKS   GTest::gtest_main
  LABELS  sse vec2)

esimd_add_isa_target(esimd_test_vec2_avx AVX
  SOURCES test_vec2.cpp
  LINKS   GTest::gtest_main
  LABELS  avx vec2)

esimd_add_isa_target(esimd_test_vec2_avx2 AVX2
  SOURCES test_vec2.cpp
  LINKS   GTest::gtest_main
  LABELS  avx2 vec2)

esimd_add_isa_target(esimd_test_vec2_avx512 AVX512
  SOURCES test_vec2.cpp
  LINKS   GTest::gtest_main
  LABELS  avx512 vec2)

esimd_add_isa_target(esimd_test_vec2_neon NEON
  SOURCES test_vec2.cpp
  LINKS   GTest::gtest_main
  LABELS  neon vec2)

esimd_add_isa_target(esimd_test_vec2_neon2x NEON2X
  SOURCES test_vec2.cpp
  LINKS   GTest::gtest_main
  LABELS  neon2x vec2)

esimd_add_isa_target(esimd_test_vec3_sse42 SSE42
  SOURCES test_vec3.cpp
  LINKS   GTest::gtest_main
  LABELS  sse vec3)

esimd_add_isa_target(esimd_test_vec3_avx AVX
  SOURCES test_vec3.cpp
  LINKS   GTest::gtest_main
  LABELS  avx vec3)

esimd_add_isa_target(esimd_test_vec3_avx2 AVX2
  SOURCES test_vec3.cpp
  LINKS   GTest::gtest_main
  LABELS  avx2 vec3)

esimd_add_isa_target(esimd_test_vec3_avx512 AVX512
  SOURCES test_vec3.cpp
  LINKS   GTest::gtest_main
  LABELS  avx512 vec3)

esimd_add_isa_target(esimd_test_vec3_neon NEON
  SOURCES test_vec3.cpp
  LINKS   GTest::gtest_main
  LABELS  neon vec3)

esimd_add_isa_target(esimd_test_vec3_neon2x NEON2X
  SOURCES test_vec3.cpp
  LINKS   GTest::gtest_main
  LABELS  neon2x vec3)

esimd_add_isa_target(esimd_test_vec4_sse42 SSE42
  SOURCES test_vec4.cpp
  LINKS   GTest::gtest_main
  LABELS  sse vec4)

esimd_add_isa_target(esimd_test_vec4_avx AVX
  SOURCES test_vec4.cpp
  LINKS   GTest::gtest_main
  LABELS  avx vec4)

esimd_add_isa_target(esimd_test_vec4_avx2 AVX2
  SOURCES test_vec4.cpp
  LINKS   GTest::gtest_main
  LABELS  avx2 vec4)

esimd_add_isa_target(esimd_test_vec4_avx512 AVX512
  SOURCES test_vec4.cpp
  LINKS   GTest::gtest_main
  LABELS  avx512 vec4)

esimd_add_isa_target(esimd_test_vec4_neon NEON
  SOURCES test_vec4.cpp
  LINKS   GTest::gtest_main
  LABELS  neon vec4)

esimd_add_isa_target(esimd_test_vec4_neon2x NEON2X
  SOURCES test_vec4.cpp
  LINKS   GTest::gtest_main
  LABELS  neon2x vec4)

esimd_add_isa_target(esimd_test_linearspace_sse42 SSE42
  SOURCES test_linearspace.cpp
  LINKS   GTest::gtest_main
  LABELS  sse linearspace)

esimd_add_isa_target(esimd_test_linearspace_avx AVX
  SOURCES test_linearspace.cpp
  LINKS   GTest::gtest_main
  LABELS  avx linearspace)

esimd_add_isa_target(esimd_test_linearspace_avx2 AVX2
  SOURCES test_linearspace.cpp
  LINKS   GTest::gtest_main
  LABELS  avx2 linearspace)

esimd_add_isa_target(esimd_test_linearspace_avx512 AVX512
  SOURCES test_linearspace.cpp
  LINKS   GTest::gtest_main
  LABELS  avx512 linearspace)

esimd_add_isa_target(esimd_test_linearspace_neon NEON
  SOURCES test_linearspace.cpp
  LINKS   GTest::gtest_main
  LABELS  neon linearspace)

esimd_add_isa_target(esimd_test_linearspace_neon2x NEON2X
  SOURCES test_linearspace.cpp
  LINKS   GTest::gtest_main
  LABELS  neon2x linearspace)

esimd_add_isa_target(esimd_test_quaternion_sse42 SSE42
  SOURCES test_quaternion.cpp
  LINKS   GTest::gtest_main
  LABELS  sse quaternion)

esimd_add_isa_target(esimd_test_quaternion_avx AVX
  SOURCES test_quaternion.cpp
  LINKS   GTest::gtest_main
  LABELS  avx quaternion)

esimd_add_isa_target(esimd_test_quaternion_avx2 AVX2
  SOURCES test_quaternion.cpp
  LINKS   GTest::gtest_main
  LABELS  avx2 quaternion)

esimd_add_isa_target(esimd_test_quaternion_avx512 AVX512
  SOURCES test_quaternion.cpp
  LINKS   GTest::gtest_main
  LABELS  avx512 quaternion)

esimd_add_isa_target(esimd_test_quaternion_neon NEON
  SOURCES test_quaternion.cpp
  LINKS   GTest::gtest_main
  LABELS  neon quaternion)

esimd_add_isa_target(esimd_test_quaternion_neon2x NEON2X
  SOURCES test_quaternion.cpp
  LINKS   GTest::gtest_main
  LABELS  neon2x quaternion)
