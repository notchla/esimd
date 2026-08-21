# esimd (easy-simd)

A **header-only** C++17 SIMD abstraction layer, ported from
[Embree](https://github.com/RenderKit/embree)'s `common/simd`. It provides uniform
vector types — `vfloat<N>`, `vint<N>`, `vuint<N>`, `vbool<N>`, `vdouble<N>`,
`vllong<N>` — that wrap x86 SSE / AVX / AVX2 / AVX512 intrinsics behind one API, with
scalar fallbacks and width-generic aliases (`vfloat4/8/16`, `vfloatx`, `VSIZEX`).

Original code © Intel Corporation, licensed Apache-2.0 (see `LICENSE`). This is an
extraction/repackaging into a standalone header-only library.

> **Scope:** x86 only. ARM/NEON and WebAssembly backends from upstream are currently
> excluded.

## Usage

Add `include/` to your include path and include the master header (it selects the
widest ISA enabled by your compiler flags):

```cpp
#include <esimd/esimd.h>
using namespace esimd;

vfloat4 a(1,2,3,4), b(4,3,2,1);
vfloat4 c = a + b;        // {5,5,5,5}
```

Because the headers dispatch on the compiler's predefined ISA macros, compile your
translation unit with matching flags (the `-m` codegen flag and the matching `-D`
define must agree).

## Using esimd from your own CMake project

esimd exports an interface target `esimd::esimd` (just the include path) and ships a
reusable helper module, `esimdISA`, that provides the per-ISA flag sets and the
`esimd_add_isa_target()` function used by this repo's own tests/examples.

Consume it either way:

```cmake
# (a) installed package
find_package(esimd REQUIRED)          # imports esimd::esimd and includes esimdISA

# (b) vendored in-tree
# add_subdirectory(extern/esimd)

# link only (you supply the ISA flags yourself):
add_executable(app main.cpp)
target_link_libraries(app PRIVATE esimd::esimd)
target_compile_options(app PRIVATE ${ESIMD_FLAGS_AVX2})   # or -mavx2 -D__AVX2__ ...

# ...or let the helper build a host-gated per-ISA executable for you:
esimd_add_isa_target(app_sse    SSE42  SOURCES main.cpp)
esimd_add_isa_target(app_avx512 AVX512 SOURCES main.cpp)
```

`esimd_add_isa_target(name <ISA> SOURCES ... [LINKS ...] [LABELS ...])` compiles the
sources with `ESIMD_FLAGS_<ISA>`, links `esimd::esimd`, and — unless
`ESIMD_BUILD_ALL_ISA=ON` — only builds the ISAs the configure host can actually
execute. `<ISA>` is one of `SSE42 | AVX | AVX2 | AVX512`. Pass `LABELS` to also
register a `ctest` entry.

## Layout

```
include/esimd/            public headers
  esimd.h                 master include (ISA dispatch + cross-width helpers)
  varying.h               scalar fallback types, vtypes<N>, width aliases
  sse.h avx.h avx512.h    per-ISA entry points
  v*4_sse2.h              SSE / 128-bit 4-wide types
  v*8_avx*.h v*4_avx2.h   AVX / AVX2 / 256-bit 8-wide types
  v*16_avx512.h v*8_avx512.h  AVX512 / 512-bit types
  detail/                 support headers (platform, intrinsics, constants, emath,
                          header-only alloc, simd_wrapper_types)
extern/                   googletest, benchmark (git submodules)
tests/                    GoogleTest correctness suites (one exe per ISA)
benchmarks/               Google Benchmark perf suites (one exe per ISA)
examples/                 runnable showcases, one subfolder per instruction set
  sse/ avx/ avx2/ avx512/ each: a demo .cpp + CMakeLists showing how to
                          include the library and build for that ISA
  portable/               width-agnostic demo (vfloatx / VSIZEX): one source
                          built for every ISA
cmake/                    esimdISA.cmake helper module + find_package config
```

The library is 100% headers. Consumers need only
`include/` on their include path.

## Building the tests & benchmarks

```sh
git submodule update --init --recursive
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Test/benchmark executables are compiled **once per ISA** (`esimd_test_sse42`,
`esimd_test_avx`, `esimd_test_avx2`, `esimd_test_avx512`, and matching `esimd_bench_*`).
By default only the ISA variants the build host can *execute* are built and run
(detected at configure time), so an AVX512 binary never faults on an older CPU. Pass
`-DESIMD_BUILD_ALL_ISA=ON` to force-build every variant for compile-only inspection.

Correctness tests validate each vector operation element-by-element against a plain
scalar reference (exact for integer/logic/compare; ULP/epsilon tolerance for
`rcp`/`rsqrt`/rounding), and the same assertions run against every ISA backend.
