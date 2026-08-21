# esimd

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
translation unit with matching flags

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
