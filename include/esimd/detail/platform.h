// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <cstddef>
#include <cassert>
#include <cstdlib>
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

/* detect Linux platform */
#if defined(linux) || defined(__linux__) || defined(__LINUX__)
#  if !defined(__LINUX__)
#     define __LINUX__
#  endif
#  if !defined(__UNIX__)
#     define __UNIX__
#  endif
#endif

/* detect Windows 95/98/NT/2000/XP/Vista/7/8/10 platform */
#if (defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)) && !defined(__CYGWIN__)
#  if !defined(__WIN32__)
#     define __WIN32__
#  endif
#endif

/* detect Cygwin platform */
#if defined(__CYGWIN__)
#  if !defined(__UNIX__)
#     define __UNIX__
#  endif
#endif

/* try to detect other Unix systems */
#if defined(__unix__) || defined (unix) || defined(__unix) || defined(_unix)
#  if !defined(__UNIX__)
#     define __UNIX__
#  endif
#endif

////////////////////////////////////////////////////////////////////////////////
/// Macros
////////////////////////////////////////////////////////////////////////////////

#ifdef __WIN32__
#  if defined(EMBREE_STATIC_LIB)
#    define dll_export
#    define dll_import
#  else
#    define dll_export __declspec(dllexport)
#    define dll_import __declspec(dllimport)
#  endif
#else
#  define dll_export __attribute__ ((visibility ("default")))
#  define dll_import
#endif

#if defined(__WIN32__) && !defined(__MINGW32__)
#if !defined(__noinline)
#define __noinline             __declspec(noinline)
#endif
//#define __forceinline        __forceinline
//#define __restrict           __restrict
#define __restrict__           //__restrict // causes issues with MSVC
#if !defined(__thread)
#define __thread               __declspec(thread)
#endif
#if !defined(__aligned)
#define __aligned(...)           __declspec(align(__VA_ARGS__))
#endif
//#define __FUNCTION__           __FUNCTION__
#define debugbreak()           __debugbreak()

#else
#if !defined(__noinline)
#define __noinline             __attribute__((noinline))
#endif
#if !defined(__forceinline)
#define __forceinline          inline __attribute__((always_inline))
#endif
//#define __restrict             __restrict
//#define __thread               __thread
#if !defined(__aligned)
#define __aligned(...)           __attribute__((aligned(__VA_ARGS__)))
#endif
#if !defined(__FUNCTION__)
#define __FUNCTION__           __PRETTY_FUNCTION__
#endif
#define debugbreak()           asm ("int $3")
#endif

#if defined(__clang__) || defined(__GNUC__)
  #define MAYBE_UNUSED __attribute__((unused))
#else
  #define MAYBE_UNUSED
#endif

#if !defined(_unused)
#define _unused(x) ((void)(x))
#endif

#define DELETED  = delete

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
/// Basic types
////////////////////////////////////////////////////////////////////////////////

/* default floating-point type */
namespace esimd {
  typedef float real;
}

/* windows does not have ssize_t */
#if defined(__WIN32__)
#if defined(__64BIT__)
typedef int64_t ssize_t;
#else
typedef int32_t ssize_t;
#endif
#endif

////////////////////////////////////////////////////////////////////////////////
/// Disable some compiler warnings
////////////////////////////////////////////////////////////////////////////////

#if defined(_MSC_VER)
#pragma warning(disable:4800) // forcing value to bool 'true' or 'false' (performance warning)
#pragma warning(disable:4244) // 'argument' : conversion from 'ssize_t' to 'unsigned int', possible loss of data
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
