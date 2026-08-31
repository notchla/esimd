# Copyright 2026 notchla liso.lorenzo@gmail.com
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.16)
project(esimd LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ----------------------------------------------------------------------------
# Header-only library target (+ esimd::esimd alias so the same name works both
# in-tree and for find_package(esimd) consumers).
# ----------------------------------------------------------------------------
add_library(esimd INTERFACE)
add_library(esimd::esimd ALIAS esimd)
target_include_directories(esimd INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(esimd INTERFACE cxx_std_17)

option(ESIMD_BUILD_TESTS "Build correctness tests (GoogleTest)" ON)
option(ESIMD_BUILD_BENCHMARKS "Build benchmarks (Google Benchmark)" ON)
option(ESIMD_BUILD_EXAMPLES "Build the per-ISA showcase examples" ON)
option(ESIMD_BUILD_ALL_ISA "Build every ISA variant even if the host cannot run it" OFF)
option(ESIMD_INSTALL "Generate install/export rules for esimd" ON)
option(ESIMD_ENABLE_TRIG "Enable the SLEEF-backed trigonometry layer (<esimd/trig.h>)" ON)

# Negative define, so the enabled default also holds for consumers who put
# include/ on their path without CMake. On the interface target, so
# find_package(esimd) consumers inherit this build's choice. Note that TUs
# including <esimd/trig.h> must also be compiled with -ffp-contract=off (see the
# README); that is left to the consumer rather than forced onto the interface
# target, which would disable FMA contraction in every TU linking esimd::esimd.
if(NOT ESIMD_ENABLE_TRIG)
  target_compile_definitions(esimd INTERFACE ESIMD_DISABLE_TRIG)
endif()

# ----------------------------------------------------------------------------
# Per-ISA flag sets + esimd_host_supports() + esimd_add_isa_target(). Kept in a
# standalone, installable module so downstream consumers get the same helper.
# ----------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(esimdISA)

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

# ----------------------------------------------------------------------------
# Install / export: headers + the esimd::esimd interface target + a
# find_package(esimd) config that also ships the esimdISA helper module.
# ----------------------------------------------------------------------------
if(ESIMD_INSTALL)
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)

  set(ESIMD_CMAKE_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/esimd")

  # Headers (the whole header-only library).
  if(ESIMD_ENABLE_TRIG)
    install(DIRECTORY include/esimd DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  else()
    install(DIRECTORY include/esimd DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
            PATTERN "detail/sleef" EXCLUDE)
  endif()

  # The interface target -> esimdTargets.cmake (namespaced esimd::).
  install(TARGETS esimd EXPORT esimdTargets)
  install(EXPORT esimdTargets
          NAMESPACE esimd::
          FILE esimdTargets.cmake
          DESTINATION ${ESIMD_CMAKE_DIR})

  # The reusable ISA helper module, shipped alongside the config.
  install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/cmake/esimdISA.cmake"
          DESTINATION ${ESIMD_CMAKE_DIR})

  # Generate and install esimdConfig.cmake + version file.
  configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/esimdConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/esimdConfig.cmake"
    INSTALL_DESTINATION ${ESIMD_CMAKE_DIR})
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/esimdConfigVersion.cmake"
    VERSION 1.0.0
    COMPATIBILITY SameMajorVersion
    ARCH_INDEPENDENT)
  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/esimdConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/esimdConfigVersion.cmake"
    DESTINATION ${ESIMD_CMAKE_DIR})
endif()
