# Copyright 2009-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# One test executable per ISA. Each compiles the same assertions with that ISA's
# flags, so every backend is validated. Host-unsupported ISAs are skipped by
# esimd_add_isa_target.

esimd_add_isa_target(esimd_test_sse42 SSE42
  SOURCES test_sse.cpp
  LINKS   GTest::gtest_main
  LABELS  sse)

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

# ARM: the same assertions, compiled against the NEON backends. NEON is 4-wide
# (so it runs the SSE suite) and NEON2X is double-pumped 8-wide (so it runs the
# AVX2 suite). Skipped automatically when the target is not AArch64.
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
if(ESIMD_ENABLE_TRIG)
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
  foreach(t esimd_test_trig_sse42 esimd_test_trig_avx
            esimd_test_trig_avx2  esimd_test_trig_avx512
            esimd_test_trig_neon  esimd_test_trig_neon2x)
    if(TARGET ${t})
      target_compile_options(${t} PRIVATE -ffp-contract=off)
    endif()
  endforeach()
endif()

# Optional data-types layer (<esimd/types.h>), on every ISA: Vec2<T> is checked
# at T = scalar and at every vector width the backend defines, plus Vec2fa.
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

# Same, for Vec3<T> and the three SSE companions vec3.h brings with it
# (Vec3ba / Vec3ia / Vec3fa+Vec3fx).
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
