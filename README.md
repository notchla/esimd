# esimd (easy-simd)

A **header-only** C++17 SIMD abstraction layer, ported from
[Embree](https://github.com/RenderKit/embree)'s `common/simd`. It provides uniform
vector types — `vfloat<N>`, `vint<N>`, `vuint<N>`, `vbool<N>`, `vdouble<N>`,
`vllong<N>` — that wrap x86 SSE / AVX / AVX2 / AVX512 and ARM NEON intrinsics behind
one API, with scalar fallbacks and width-generic aliases (`vfloat4/8/16`, `vfloatx`,
`VSIZEX`).

Original code © Intel Corporation, licensed Apache-2.0 (see `LICENSE`). This is an
extraction/repackaging into a standalone header-only library. The vendored
`include/esimd/detail/arm/sse2neon.h` is © SSE2NEON Contributors and licensed MIT
(see the header's own license block).

> **Scope:** x86 and AArch64. The WebAssembly backend from upstream is excluded.

## Supported ISAs

| ISA      | Arch    | Width      | Backend                                        |
|----------|---------|------------|------------------------------------------------|
| `SSE42`  | x86     | 4          | native SSE4.2                                  |
| `AVX`    | x86     | 4, 8       | native AVX                                     |
| `AVX2`   | x86     | 4, 8       | native AVX2                                    |
| `AVX512` | x86     | 4, 8, 16   | native AVX-512                                 |
| `NEON`   | AArch64 | 4          | `sse2neon.h` — real NEON behind `__m128`       |
| `NEON2X` | AArch64 | 4, 8       | `avx2neon.h` — `__m256` as a pair of `__m128`  |

On ARM there are **no `-m` codegen flags**: NEON is baseline on AArch64, so the `-D`
defines alone select the backend, and the x86 intrinsics they name are supplied by
the vendored emulation headers in `include/esimd/detail/arm/`. `NEON2X` is upstream's
"double pumped" mode — it defines `__AVX__`/`__AVX2__`, so `VSIZEX` is 8 and the
8-wide types are available, each 256-bit register being emulated as two NEON
registers.

### ARM API gaps

A few operations have no ARM implementation upstream and are therefore absent under
`NEON`/`NEON2X`:

- `vfloat8`: `trunc`, `round`, `permute`, the
  `shuffle<0,0,2,2>` / `<1,1,3,3>` / `<0,1,0,1>` specializations, and the
  `vreduce_*2`/`*4` staged reductions
- `vint8` / `vuint8`: `permute`, `align_shift_right`
- `vllong4`, `vdouble4`: absent entirely

> **GCC note:** the ARM flag sets include `-flax-vector-conversions`. `avx2neon.h`
> assigns freely between same-width NEON vector types; Clang permits that by default, GCC needs the flag.

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
execute. `<ISA>` is one of `SSE42 | AVX | AVX2 | AVX512 | NEON | NEON2X`. ISAs from
the wrong architecture family are skipped silently, so a single `CMakeLists.txt` can
list both the x86 and the ARM variants. Pass `LABELS` to also register a `ctest` entry.

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
    arm/                  vendored NEON emulation: sse2neon.h (MIT), avx2neon.h,
                          emulation.h — pulled in automatically on AArch64
extern/                   googletest, benchmark (git submodules)
tests/                    GoogleTest correctness suites (one exe per ISA)
benchmarks/               Google Benchmark perf suites (one exe per ISA)
examples/                 runnable showcases, one subfolder per instruction set
  sse/ avx/ avx2/ avx512/ each: a demo .cpp + CMakeLists showing how to
                          include the library and build for that ISA
  neon/ neon2x/           the ARM backends, reusing the sse/ and avx2/ sources
                          unchanged — same code, different flag set
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

### Building and testing the ARM backends

On an AArch64 machine the NEON targets are picked up automatically. To build and test
them from an x86-64 host, cross-compile and run under qemu user-mode emulation:

```sh
sudo apt install g++-aarch64-linux-gnu qemu-user

cmake -B build-arm -S . -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-aarch64-linux-gnu.cmake
cmake --build build-arm
ctest --test-dir build-arm --output-on-failure
```

This builds `esimd_test_neon` (the SSE suite at
4-wide), `esimd_test_neon2x_avx` and `esimd_test_neon2x` (the AVX and AVX2 suites at
8-wide), plus the integration suites.

Correctness tests validate each vector operation element-by-element against a plain
scalar reference (exact for integer/logic/compare; ULP/epsilon tolerance for
`rcp`/`rsqrt`/rounding), and the same assertions run against every ISA backend.
