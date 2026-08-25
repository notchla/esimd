# Copyright 2026 notchla liso.lorenzo@gmail.com
# SPDX-License-Identifier: Apache-2.0
#
# esimdISA — reusable per-ISA build helper for the header-only esimd library.
#
# This module is self-contained and safe to include() from anywhere: the esimd
# project's own build includes it, and downstream consumers get it too (the
# installed package config include()s it). It provides:
#
#   ESIMD_FLAGS_<ISA>            cached flag/define set for each ISA
#                                (<ISA> = SSE42 | AVX | AVX2 | AVX512 | NEON | NEON2X)
#   esimd_host_supports(<ISA> out)   sets `out` to whether the build host can
#                                     execute that ISA (cached probe)
#   esimd_add_isa_target(name <ISA> SOURCES ... [LINKS ...] [LABELS ...])
#                                adds an executable compiled with that ISA's
#                                flags, linked to esimd::esimd. Host-gated unless
#                                ESIMD_BUILD_ALL_ISA is ON. A ctest entry is
#                                registered only when LABELS are given.
#
# It expects the imported/alias target `esimd::esimd` to exist (the esimd build
# defines it; find_package(esimd) imports it).

include_guard(GLOBAL)
include(CheckCXXSourceRuns)

# ----------------------------------------------------------------------------
# Per-ISA flag/define sets. The -m flags authorize codegen; the matching -D
# macros tell the esimd headers which backend to expose. They must agree.
# Cached (INTERNAL) so they are visible in every scope that uses the helper,
# including a consumer project that only include()s this module.
# ----------------------------------------------------------------------------
set(ESIMD_FLAGS_SSE42
    -msse4.2 -D__SSE__ -D__SSE2__ -D__SSE4_1__ -D__SSE4_2__
    CACHE INTERNAL "esimd SSE4.2 flag/define set")
set(ESIMD_FLAGS_AVX
    -mavx -mbmi -D__AVX__ -D__SSE4_2__ -D__SSE4_1__ -D__BMI__
    CACHE INTERNAL "esimd AVX flag/define set")
set(ESIMD_FLAGS_AVX2
    -mavx2 -mfma -mf16c -mbmi -mbmi2 -mlzcnt
    -D__AVX2__ -D__AVX__ -D__SSE4_2__ -D__SSE4_1__ -D__LZCNT__ -D__BMI__
    CACHE INTERNAL "esimd AVX2 flag/define set")
set(ESIMD_FLAGS_AVX512
    -march=skylake-avx512 -mavx2 -mfma -mf16c -mbmi -mbmi2 -mlzcnt
    -D__AVX512F__ -D__AVX512VL__ -D__AVX512DQ__ -D__AVX512BW__
    -D__AVX2__ -D__AVX__ -D__SSE4_2__ -D__SSE4_1__ -D__LZCNT__ -D__BMI__
    CACHE INTERNAL "esimd AVX512 flag/define set")

# ARM: NEON is baseline on AArch64, so there are no -m codegen flags here -- the
# -D defines alone pick the backend, and the x86 intrinsics they name are
# supplied by the vendored sse2neon.h / avx2neon.h emulation headers.
#   NEON   -> 128-bit, 4-wide only (sse2neon)
#   NEON2X -> 256-bit, "double pumped": __m256 is a pair of __m128 (avx2neon)
# -flax-vector-conversions: avx2neon.h assigns freely between same-width NEON
# vector types (int32x4_t <-> __m128i etc.).
set(ESIMD_FLAGS_NEON
    -flax-vector-conversions
    -D__SSE4_2__ -D__SSE4_1__
    CACHE INTERNAL "esimd NEON flag/define set")
set(ESIMD_FLAGS_NEON2X
    -flax-vector-conversions
    -D__AVX2__ -D__AVX__ -D__SSE4_2__ -D__SSE4_1__ -D__BMI__ -D__BMI2__ -D__LZCNT__
    CACHE INTERNAL "esimd NEON2X (double pumped) flag/define set")

# Which ISA family the *target* belongs to. Used to skip ISAs that cannot exist
# on the target at all, before the (more expensive) host-execution probe.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
  set(ESIMD_TARGET_ARM ON)
else()
  set(ESIMD_TARGET_ARM OFF)
endif()
set(ESIMD_ISAS_X86 SSE42 AVX AVX2 AVX512)
set(ESIMD_ISAS_ARM NEON NEON2X)

# ----------------------------------------------------------------------------
# Host-ISA execution detection: does this machine execute <ISA> without SIGILL?
# ----------------------------------------------------------------------------
function(esimd_host_supports isa outvar)
  # CMAKE_REQUIRED_FLAGS must be a single space-separated string, not a list,
  # or check_cxx_source_runs mis-forwards the -D defines into try_run's cmake args.
  string(REPLACE ";" " " CMAKE_REQUIRED_FLAGS "${ESIMD_FLAGS_${isa}}")
  # Probe a representative intrinsic per ISA.
  if(isa STREQUAL "SSE42")
    set(probe "#include <immintrin.h>\nint main(){__m128i a=_mm_set1_epi32(1);return _mm_extract_epi32(a,0)-1;}")
  elseif(isa STREQUAL "AVX")
    set(probe "#include <immintrin.h>\nint main(){__m256 a=_mm256_set1_ps(1);return (int)_mm256_cvtss_f32(a)-1;}")
  elseif(isa STREQUAL "AVX2")
    set(probe "#include <immintrin.h>\nint main(){__m256i a=_mm256_set1_epi32(1);a=_mm256_add_epi32(a,a);return _mm256_extract_epi32(a,0)-2;}")
  elseif(isa STREQUAL "AVX512")
    set(probe "#include <immintrin.h>\nint main(){__m512i a=_mm512_set1_epi32(1);a=_mm512_add_epi32(a,a);return (int)_mm512_reduce_add_epi32(a)/16-2;}")
  elseif(isa STREQUAL "NEON" OR isa STREQUAL "NEON2X")
    set(probe "#include <arm_neon.h>\nint main(){int32x4_t a=vdupq_n_s32(1);return vgetq_lane_s32(vaddq_s32(a,a),0)-2;}")
  else()
    message(FATAL_ERROR "esimd_host_supports: unknown ISA '${isa}' (use SSE42|AVX|AVX2|AVX512|NEON|NEON2X)")
  endif()
  check_cxx_source_runs("${probe}" ESIMD_HOST_RUNS_${isa})
  set(${outvar} ${ESIMD_HOST_RUNS_${isa}} PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------
# Add a per-ISA executable compiled with that ISA's flags, linked to esimd.
# Registers a ctest entry gated on host support (only for labeled targets).
# ----------------------------------------------------------------------------
function(esimd_add_isa_target name isa)
  if(NOT DEFINED ESIMD_FLAGS_${isa})
    message(FATAL_ERROR "esimd_add_isa_target: unknown ISA '${isa}' (use SSE42|AVX|AVX2|AVX512|NEON|NEON2X)")
  endif()
  if(NOT TARGET esimd::esimd)
    message(FATAL_ERROR "esimd_add_isa_target: target esimd::esimd not found; "
                        "call find_package(esimd) or add_subdirectory(esimd) first")
  endif()
  cmake_parse_arguments(A "" "" "SOURCES;LINKS;LABELS" ${ARGN})
  if(ESIMD_TARGET_ARM)
    if(${isa} IN_LIST ESIMD_ISAS_X86)
      return()
    endif()
  elseif(${isa} IN_LIST ESIMD_ISAS_ARM)
    return()
  endif()
  esimd_host_supports(${isa} host_ok)
  if(NOT host_ok AND NOT ESIMD_BUILD_ALL_ISA)
    message(STATUS "esimd: host cannot execute ${isa}; skipping target ${name}")
    return()
  endif()
  add_executable(${name} ${A_SOURCES})
  target_link_libraries(${name} PRIVATE esimd::esimd ${A_LINKS})
  target_compile_options(${name} PRIVATE ${ESIMD_FLAGS_${isa}} -Wall)
  # Register a ctest entry only for labeled targets (tests). Unlabeled targets
  # (benchmarks, examples) build but are run manually.
  if(host_ok AND A_LABELS)
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES LABELS "${A_LABELS}")
  endif()
endfunction()
