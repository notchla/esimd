# Copyright 2026 notchla liso.lorenzo@gmail.com
# SPDX-License-Identifier: Apache-2.0
#
# Cross-compile esimd for AArch64 and run the results under qemu user-mode
# emulation. Requires g++-aarch64-linux-gnu and qemu-user:
#
#   sudo apt install g++-aarch64-linux-gnu qemu-user
#   cmake -B build-arm -S . -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-aarch64-linux-gnu.cmake
#
# CMAKE_CROSSCOMPILING_EMULATOR is honoured by both try_run (so the NEON ISA
# probes in esimdISA actually execute) and add_test (so ctest runs the suites).

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER   aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

set(CMAKE_CROSSCOMPILING_EMULATOR /usr/bin/qemu-aarch64 -L /usr/aarch64-linux-gnu)
