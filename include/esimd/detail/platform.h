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

#if defined(__X86_64__)
#define __64BIT__
#endif

/* detect Windows platform */
#if (defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)) && !defined(__CYGWIN__)
#  if !defined(__WIN32__)
#     define __WIN32__
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
