// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sse.h"

#if defined(__AVX512VL__)
#include "vboolf8_avx512.h"
#include "vboold4_avx512.h"
#else
#include "vboolf8_avx.h"
#include "vboold4_avx.h"
#endif

#if defined(__AVX2__)
#include "vint8_avx2.h"
#include "vuint8_avx2.h"
#else
#include "vint8_avx.h"
#include "vuint8_avx.h"
#endif
#if ESIMD_HAS_VLLONG4
#include "vllong4_avx2.h"
#endif
#include "vfloat8_avx.h"
#if ESIMD_HAS_VDOUBLE4
#include "vdouble4_avx.h"
#endif

#if ESIMD_ISA_AVX512
#include "avx512.h"
#endif
