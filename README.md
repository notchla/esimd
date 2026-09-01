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
(see the header's own license block). The vendored `include/esimd/detail/sleef/`
headers are © Naoki Shibata and contributors, licensed under the Boost Software
License 1.0, and are used only by the optional `<esimd/trig.h>` layer.

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
define must agree). Translation units that include `<esimd/trig.h>` additionally need
`-ffp-contract=off` — see [Trigonometry](#trigonometry-optional).

## Trigonometry (optional)

`<esimd/trig.h>` adds SLEEF-backed trigonometry to the existing vector types. It is
enabled by default (`ESIMD_ENABLE_TRIG`) but never implicit: `<esimd/esimd.h>` does
not pull it in.

```cpp
#include <esimd/trig.h>
using namespace esimd;

vfloat8 x(0.5f);
vfloat8 s = sin(x);              // <=1   ULP  (SLEEF u10)
vfloat8 f = sin_u35(x);          // <=3.5 ULP, faster
vfloat8 q = fast_sin(x);         // <=3500 ULP, float only, |x| < 125000

vfloat8 sn, cs; sincos(x, sn, cs);   // one shared range reduction
```

`sin cos tan sincos asin acos atan atan2`, each in three flavours: the plain name
(SLEEF `u10`), a `_u35` suffix (SLEEF `u35`), and `fast_sin` / `fast_cos` (SLEEF
`u3500`, single precision only — SLEEF has no `fasttan` and no double `u3500`).
Overloads exist for every `vfloat`/`vdouble` width the active ISA defines.

> **Every translation unit including `<esimd/trig.h>` must be compiled with
> `-ffp-contract=off`.** SLEEF's double-double sequences reconstruct the rounding
> error of an unfused `a*b`, which a contracted FMA discards; without the flag the ULP
> bounds do not hold. The headers ask for it with `#pragma STDC FP_CONTRACT OFF`, which
> Clang honours and GCC ignores. esimd does not add it to `esimd::esimd`, since that
> would disable contraction in every TU linking the target, trig or not:
>
> ```cmake
> target_link_libraries(app PRIVATE esimd::esimd)
> target_compile_options(app PRIVATE -ffp-contract=off)
> ```

## Trigonometry backends

Each width dispatches to the SLEEF implementation built for that same instruction set
— no cross-ISA emulation. SLEEF only generates inline headers for FMA-capable targets,
so SSE4.2 and plain AVX fall back to a per-lane libm loop.

| ISA      | vfloat4    | vfloat8              | vfloat16   | vdouble4 | vdouble8   |
|----------|------------|----------------------|------------|----------|------------|
| `SSE42`  | *scalar*   | —                    | —          | —        | —          |
| `AVX`    | *scalar*   | *scalar*             | —          | *scalar* | —          |
| `AVX2`   | `avx2128`  | `avx2`               | —          | `avx2`   | —          |
| `AVX512` | `avx2128`  | `avx2`               | `avx512f`  | `avx2`   | `avx512f`  |
| `NEON`   | `advsimd`  | —                    | —          | —        | —          |
| `NEON2X` | `advsimd`  | `advsimd` ×2 (lo/hi) | —          | —        | —          |

`—` means the ISA has no such type at all, not that the function is missing.

### The SSE4.2 / AVX scalar fallback

`<esimd/trig.h>` defines `ESIMD_TRIG_SCALAR_FALLBACK` when this path is active. It is
**not vectorised** — one libm call per lane, several times slower than the SLEEF path
— and all three accuracy flavours collapse to the same call, so `sin_u35` and
`fast_sin` buy nothing there. Accuracy is your libm's rather than SLEEF's; glibc stays
inside the `u10` bound, and the test suite holds the fallback to it.

### Switching it off

`-DESIMD_ENABLE_TRIG=OFF` makes `<esimd/trig.h>` inert (it still compiles, but declares
nothing and leaves `ESIMD_HAS_TRIG` undefined), skips the `esimd_test_trig_*` targets,
and drops the vendored SLEEF headers from the install. It travels on the
`esimd::esimd` interface target as `ESIMD_DISABLE_TRIG`, so `find_package(esimd)`
consumers inherit the choice; the define is negative, so putting `include/` on your
path without CMake still gets the enabled default.

`ESIMD_HAS_TRIG` says the API is present. It is defined on every ISA unless the layer
is switched off, so guard on it only to stay compatible with an `OFF` build:

```cpp
#include <esimd/trig.h>
#ifdef ESIMD_HAS_TRIG
  vfloatx s = sin(v);
#endif
```

### Regenerating the SLEEF headers

The vendored headers in `include/esimd/detail/sleef/` are generated artifacts, checked
in so consumers need no SLEEF build. They are self-contained: nothing links `libsleef`.
To refresh them from a SLEEF checkout:

```sh
# x86 -> sleefinline_avx2128.h, sleefinline_avx2.h, sleefinline_avx512f.h
cmake -B build -S . -DSLEEF_BUILD_INLINE_HEADERS=ON -DCMAKE_INSTALL_PREFIX=$PWD/install
cmake --build build

# aarch64 -> sleefinline_advsimd.h (reuses the native build for the host tools)
cmake -B build-arm64 -S . -DSLEEF_BUILD_INLINE_HEADERS=ON \
      -DCMAKE_INSTALL_PREFIX=$PWD/install-arm64 -DNATIVE_BUILD_DIR=$PWD/build \
      -DCMAKE_TOOLCHAIN_FILE=<esimd>/cmake/toolchain-aarch64-linux-gnu.cmake
cmake --build build-arm64
```

Copy the resulting `build*/include/sleefinline_*.h` into
`include/esimd/detail/sleef/`.

## Data types

`<esimd/types.h>` adds the geometric types embree builds on top of the simd
registers. It is opt-in by include — `<esimd/esimd.h>` does not pull it in — and, unlike
`<esimd/trig.h>`, needs no CMake option, no vendored third-party headers and no
special compile flags. Just include it.

```cpp
#include <esimd/types.h>
using namespace esimd;

Vec2f  p(3.0f, 4.0f);
float  d = length(p);          // 5.0f

Vec2<vfloat8> q(vfloat8(3.0f), vfloat8(4.0f));
vfloat8 l = length(q);         // 8 lanes at once -- same code, T = vfloat8

Vec2fa a(1.0f, 2.0f);          // SSE-backed, x/y in one __m128
```

So far the layer provides:

| type | what it is |
|------|------------|
| `Vec2<T>` | generic 2D vector. `T` is any scalar (`Vec2b`/`Vec2i`/`Vec2f`) or any esimd vector type (`Vec2<vfloat4>`, `Vec2<vfloat8>`, `Vec2<vfloat16>`, …) |
| `Vec2fa` | 2 floats held in a `__m128`, 16-byte aligned (`Vec2fa_t` is an alias) |

`Vec2<T>` is component-wise plumbing over `T`, so every operation it offers is one
`T` already has: arithmetic, `min`/`max`, the `madd`/`msub`/`nmadd`/`nmsub` family,
reductions, `dot`/`cross`/`length`/`normalize`/`normalize_safe`/`distance`/`det`,
`select` (on a `bool`, a `Vec2<bool>`, or a `T::Bool` mask), `lerp`, `maxDim` and
`shift_right_1`. Some operation are defined only for some vector width: `frac` exists for `Vec2<vfloat4>`/`Vec2<vfloat8>` but not `Vec2<vfloat16>`,
and `maxDim` — which branches on a scalar comparison — is meaningful only for a
scalar `T`.

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

Build options: `ESIMD_BUILD_TESTS`, `ESIMD_BUILD_BENCHMARKS`, `ESIMD_BUILD_EXAMPLES`,
`ESIMD_INSTALL` and `ESIMD_ENABLE_TRIG` default to `ON`; `ESIMD_BUILD_ALL_ISA`
defaults to `OFF`.

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
  trig.h                  optional SLEEF-backed trigonometry (opt-in include)
  types.h                 data types
  varying.h               scalar fallback types, vtypes<N>, width aliases
  sse.h avx.h avx512.h    per-ISA entry points
  v*4_sse2.h              SSE / 128-bit 4-wide types
  v*8_avx*.h v*4_avx2.h   AVX / AVX2 / 256-bit 8-wide types
  v*16_avx512.h v*8_avx512.h  AVX512 / 512-bit types
  detail/                 support headers (platform, intrinsics, constants, emath,
                          header-only alloc, simd_wrapper_types)
    arm/                  vendored NEON emulation: sse2neon.h (MIT), avx2neon.h,
                          emulation.h — pulled in automatically on AArch64
    sleef/                vendored SLEEF 4.0.0 inline headers (BSL-1.0), one per
                          ISA — used only by trig.h
  types/                  the optional data types themselves: vec2.h (Vec2<T>)
                          and vec2fa.h (Vec2fa) — reached via types.h
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
`esimd_test_avx`, `esimd_test_avx2`, `esimd_test_avx512`, and matching `esimd_bench_*`),
plus `esimd_test_trig_*` for every ISA — the SLEEF backends and the scalar fallback
are held to the same ULP bounds (`ctest -L trig` runs just those) — and
`esimd_test_vec2_*`, which checks `Vec2<T>` at `T` = scalar and at every vector width
the backend defines, plus `Vec2fa` (`ctest -L vec2`).
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
