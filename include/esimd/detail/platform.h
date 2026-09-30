// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cassert>
#include <stdint.h>

////////////////////////////////////////////////////////////////////////////////
/// detect platform
////////////////////////////////////////////////////////////////////////////////

/* detect 32 or 64 Intel platform */
#if defined(__x86_64__) || defined(__ia64__) || defined(_M_X64)
#define __X86_64__
#define __X86_ASM__
#elif defined(__i386__) || defined(_M_IX86)
#define __X86_ASM__
#endif

/* detect 64 bit ARM platform */
#if defined(__aarch64__) || defined(_M_ARM64)
#define ESIMD_ARM64
#endif

#if defined(__X86_64__) || defined(ESIMD_ARM64)
#define __64BIT__
#endif

/* detect Windows platform */
#if (defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)) && !defined(__CYGWIN__)
#  if !defined(__WIN32__)
#     define __WIN32__
#  endif
#endif

/* MSVC predefines only __AVX__, __AVX2__ and __AVX512*__, never the SSE family
   or __FMA__, so derive what /arch guarantees. /arch:SSE4.2 sets no macro at
   all; pass /D__SSE4_1__ /D__SSE4_2__ for it (esimdISA.cmake does). __BMI__ and
   __LZCNT__ are deliberately not derived: intrinsics.h maps them to GCC builtins. */
#if defined(_MSC_VER) && !defined(__clang__)
#  if defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#    if !defined(__SSE__)
#      define __SSE__
#    endif
#    if !defined(__SSE2__)
#      define __SSE2__
#    endif
#  endif
#  if defined(__AVX__)
#    if !defined(__SSE3__)
#      define __SSE3__
#    endif
#    if !defined(__SSSE3__)
#      define __SSSE3__
#    endif
#    if !defined(__SSE4_1__)
#      define __SSE4_1__
#    endif
#    if !defined(__SSE4_2__)
#      define __SSE4_2__
#    endif
#  endif
#  if defined(__AVX2__) && !defined(__FMA__)
#    define __FMA__
#  endif
#endif

////////////////////////////////////////////////////////////////////////////////
/// Macros
////////////////////////////////////////////////////////////////////////////////

#if defined(__WIN32__) && !defined(__MINGW32__)
#define __restrict__           //__restrict // causes issues with MSVC
#if !defined(__aligned)
#define __aligned(...)           __declspec(align(__VA_ARGS__))
#endif

#else
#if !defined(__forceinline)
#define __forceinline          inline __attribute__((always_inline))
#endif
#if !defined(__aligned)
#define __aligned(...)           __attribute__((aligned(__VA_ARGS__)))
#endif
#endif

#if defined(__clang__) || defined(__GNUC__)
  #define MAYBE_UNUSED __attribute__((unused))
#else
  #define MAYBE_UNUSED
#endif

#if !defined(likely)
#if defined(_MSC_VER)
#define   likely(expr) (expr)
#define unlikely(expr) (expr)
#else
#define   likely(expr) __builtin_expect((bool)(expr),true )
#define unlikely(expr) __builtin_expect((bool)(expr),false)
#endif
#endif

////////////////////////////////////////////////////////////////////////////////
/// Disable some compiler warnings
////////////////////////////////////////////////////////////////////////////////

#if defined(_MSC_VER)
#pragma warning(disable:4800) // forcing value to bool 'true' or 'false' (performance warning)
#pragma warning(disable:4244) // conversion, possible loss of data
#pragma warning(disable:4267) // conversion from 'size_t' to 'const int', possible loss of data
#pragma warning(disable:4503) // decorated name length exceeded, name was truncated
#pragma warning(disable:4180) // qualifier applied to function type has no meaning; ignored
#pragma warning(disable:4258) // definition from the for loop is ignored; the definition from the enclosing scope is used
#endif

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#pragma GCC diagnostic ignored "-Wattributes"
#pragma GCC diagnostic ignored "-Wmisleading-indentation"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wparentheses"
#endif
