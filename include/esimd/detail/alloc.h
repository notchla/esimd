// Copyright 2009-2021 Intel Corporation
// Copyright 2026 notchla liso.lorenzo@gmail.com
// SPDX-License-Identifier: Apache-2.0
//
// alignedMalloc/alignedFree are inline over _mm_malloc/_mm_free so the library
// needs no linked translation unit.

#pragma once

#include "platform.h"
#include "intrinsics.h" // _mm_malloc / _mm_free (sse2neon supplies both on ARM)

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
