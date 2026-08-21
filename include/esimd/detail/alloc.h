// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Header-only replacement for embree's common/sys/alloc.h. The upstream header
// declared alignedMalloc/alignedFree (defined in alloc.cpp) and defined the
// ALIGNED_STRUCT_/ALIGNED_CLASS_ macros. Here alignedMalloc/alignedFree are
// provided as inline wrappers over _mm_malloc/_mm_free so the SIMD library needs
// no linked translation unit. Only the pieces the simd headers actually use are
// kept (the aligned_allocator/os_allocator/IDPool templates are dropped).

#pragma once

#include "platform.h"
#include <immintrin.h> // _mm_malloc / _mm_free

namespace esimd
{
  __forceinline void* alignedMalloc(size_t size, size_t align)
  {
    if (size == 0) return nullptr;
    assert((align & (align-1)) == 0);
    return _mm_malloc(size, align);
  }

  __forceinline void alignedFree(void* ptr)
  {
    if (ptr) _mm_free(ptr);
  }

#define ALIGNED_STRUCT_(align)                                            \
  void* operator new(size_t size) { return alignedMalloc(size,align); }   \
  void operator delete(void* ptr) { alignedFree(ptr); }                   \
  void* operator new[](size_t size) { return alignedMalloc(size,align); } \
  void operator delete[](void* ptr) { alignedFree(ptr); }

#define ALIGNED_CLASS_(align)                                         \
 public:                                                              \
    ALIGNED_STRUCT_(align)                                            \
 private:
}
