# Copyright 2026 notchla liso.lorenzo@gmail.com
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.16)
project(esimd LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ----------------------------------------------------------------------------
# Header-only library target
# ----------------------------------------------------------------------------
add_library(esimd INTERFACE)
target_include_directories(esimd INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(esimd INTERFACE cxx_std_17)

option(ESIMD_BUILD_TESTS "Build correctness tests (GoogleTest)" ON)
option(ESIMD_BUILD_BENCHMARKS "Build benchmarks (Google Benchmark)" ON)
option(ESIMD_BUILD_EXAMPLES "Build the per-ISA showcase examples" ON)
option(ESIMD_BUILD_ALL_ISA "Build every ISA variant even if the host cannot run it" OFF)

# ----------------------------------------------------------------------------
# Per-ISA flag/define sets.
# ----------------------------------------------------------------------------
set(ESIMD_FLAGS_SSE42  -msse4.2 -D__SSE__ -D__SSE2__ -D__SSE4_1__ -D__SSE4_2__)
set(ESIMD_FLAGS_AVX    -mavx -mbmi -D__AVX__ -D__SSE4_2__ -D__SSE4_1__ -D__BMI__)
set(ESIMD_FLAGS_AVX2   -mavx2 -mfma -mf16c -mbmi -mbmi2 -mlzcnt
                       -D__AVX2__ -D__AVX__ -D__SSE4_2__ -D__SSE4_1__ -D__LZCNT__ -D__BMI__)
set(ESIMD_FLAGS_AVX512 -march=skylake-avx512 -mavx2 -mfma -mf16c -mbmi -mbmi2 -mlzcnt
                       -D__AVX512F__ -D__AVX512VL__ -D__AVX512DQ__ -D__AVX512BW__
                       -D__AVX2__ -D__AVX__ -D__SSE4_2__ -D__SSE4_1__ -D__LZCNT__ -D__BMI__)

# ----------------------------------------------------------------------------
# Host-ISA execution detection: only run (and, unless forced, build) the ISA
# variants this machine can actually execute, so an AVX512 binary never SIGILLs.
# ----------------------------------------------------------------------------
include(CheckCXXSourceRuns)
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
  else() # AVX512
    set(probe "#include <immintrin.h>\nint main(){__m512i a=_mm512_set1_epi32(1);a=_mm512_add_epi32(a,a);return (int)_mm512_reduce_add_epi32(a)/16-2;}")
  endif()
  check_cxx_source_runs("${probe}" ESIMD_HOST_RUNS_${isa})
  set(${outvar} ${ESIMD_HOST_RUNS_${isa}} PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------
# Helper: add a per-ISA executable (test or benchmark) with the right flags.
# Registers a ctest entry gated on host support (unless ESIMD_BUILD_ALL_ISA).
# ----------------------------------------------------------------------------
function(esimd_add_isa_target name isa)
  # remaining args = source files; links against provided libs via TARGET_LINKS var
  cmake_parse_arguments(A "" "" "SOURCES;LINKS;LABELS" ${ARGN})
  esimd_host_supports(${isa} host_ok)
  if(NOT host_ok AND NOT ESIMD_BUILD_ALL_ISA)
    message(STATUS "esimd: host cannot execute ${isa}; skipping target ${name}")
    return()
  endif()
  add_executable(${name} ${A_SOURCES})
  target_link_libraries(${name} PRIVATE esimd ${A_LINKS})
  target_compile_options(${name} PRIVATE ${ESIMD_FLAGS_${isa}} -Wall)
  # Register a ctest entry only for labeled targets (tests). Benchmarks are left
  # unlabeled so they build but are run manually rather than during `ctest`.
  if(host_ok AND A_LABELS)
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES LABELS "${A_LABELS}")
  endif()
endfunction()

# ----------------------------------------------------------------------------
# Third-party (submodules)
# ----------------------------------------------------------------------------
if(ESIMD_BUILD_TESTS)
  enable_testing()
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  add_subdirectory(extern/googletest EXCLUDE_FROM_ALL)
endif()

if(ESIMD_BUILD_BENCHMARKS)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  add_subdirectory(extern/benchmark EXCLUDE_FROM_ALL)
endif()

if(ESIMD_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
  add_subdirectory(tests)
endif()
if(ESIMD_BUILD_BENCHMARKS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/CMakeLists.txt")
  add_subdirectory(benchmarks)
endif()
if(ESIMD_BUILD_EXAMPLES AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/CMakeLists.txt")
  add_subdirectory(examples)
endif()
