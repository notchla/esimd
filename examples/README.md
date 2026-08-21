# esimd examples

One runnable showcase per instruction set. Every example includes the **same**
master header — `#include <esimd/esimd.h>` — and the *only* thing that selects the
vectorization width is the set of compiler flags you build it with. That is the
whole point of these folders: same source, different `-m`/`-D` flags → different ISA.

| folder     | native types shown                          | build flags |
|------------|---------------------------------------------|-------------|
| `sse/`     | `vfloat4`, `vint4`, `vboolf4` (128-bit)     | `-msse4.2 -D__SSE__ -D__SSE2__ -D__SSE4_1__ -D__SSE4_2__` |
| `avx/`     | `vfloat8`, `vdouble4` (256-bit)             | `-mavx -mbmi -D__AVX__ -D__SSE4_2__ -D__SSE4_1__ -D__BMI__` |
| `avx2/`    | `vint8`, `vuint8`, `vllong4` (native int)   | `-mavx2 -mfma -mf16c -mbmi -mbmi2 -mlzcnt -D__AVX2__ -D__AVX__ -D__SSE4_2__ -D__SSE4_1__ -D__LZCNT__ -D__BMI__` |
| `avx512/`  | `vfloat16`, `vdouble8`, `vboolf16` masks    | `-march=skylake-avx512 -mavx2 -mfma -mf16c -mbmi -mbmi2 -mlzcnt -D__AVX512F__ -D__AVX512VL__ -D__AVX512DQ__ -D__AVX512BW__ -D__AVX2__ -D__AVX__ -D__SSE4_2__ -D__SSE4_1__ -D__LZCNT__ -D__BMI__` |

The `-m` flags tell the compiler which instructions it may emit; the matching `-D`
macros tell the esimd headers which backend to expose. They must agree — esimd's
CMake pairs them for you in `ESIMD_FLAGS_<ISA>`.

## Build & run — with CMake (part of the main build)

```sh
cmake -B build -S .
cmake --build build
./build/examples/sse/esimd_example_sse
./build/examples/avx/esimd_example_avx
./build/examples/avx2/esimd_example_avx2
./build/examples/avx512/esimd_example_avx512
```

Only the ISAs your CPU can actually execute are built (host detection at configure
time). Pass `-DESIMD_BUILD_EXAMPLES=OFF` to skip them, or `-DESIMD_BUILD_ALL_ISA=ON`
to force-build every variant for compile-only inspection.

## Build & run — standalone (one file, one compiler command)

esimd is header-only, so a consumer needs nothing but `include/` on the path plus the
ISA flags. Each example's header comment carries its exact command; e.g. for SSE:

```sh
g++ -std=c++17 -I../../include \
    -msse4.2 -D__SSE__ -D__SSE2__ -D__SSE4_1__ -D__SSE4_2__ \
    sse/sse_example.cpp -o sse_example
./sse_example
```
