#ifndef ASSEMBLY_HELPER_H
#define ASSEMBLY_HELPER_H

#include <iostream>
#include <cstdlib>
#include <memory>
#include <string>

#include <immintrin.h>
#include <emmintrin.h>

#include "Fastor.h"

#include "_KinematicMeasures_.h"
#include "_GeometricStiffness_.h"
#include "_MassIntegrand_.h"


#ifndef LL_TYPES
#define LL_TYPES
using Real = double;
using Integer = std::int64_t;
using UInteger = std::uint64_t;
#endif

using V = Fastor::SIMDVector<Real>;


// Helper functions
/*---------------------------------------------------------------------------------------------*/
#ifndef CUSTOM_ALLOCATION_
#define CUSTOM_ALLOCATION_
template<typename T>
FASTOR_INLINE T *allocate(Integer size) {
#if defined(__AVX__)
    T *out = (T*)_mm_malloc(sizeof(T)*size,32);
#elif defined(__SSE__)
    T *out = (T*)_mm_malloc(sizeof(T)*size,16);
#else
    T *out = (T*)malloc(sizeof(T)*size);
#endif
    return out;
}

template<typename T>
FASTOR_INLINE void deallocate(T *a) {
#if defined(__SSE__)
    _mm_free(a);
#else
    free(a);
#endif
}
#endif


// For checks only
//-------------------
template<typename T>
inline T sum(const T *arr, int size) {
    T val = 0;
    for (auto i=0; i<size; ++i)
        val += arr[i];
    return val;
}

template<typename T>
inline Real norm(const T *arr, int size) {
    Real val = 0;
    for (auto i=0; i<size; ++i)
        val += arr[i]*arr[i];
    return std::sqrt(val);
}
//-------------------


#endif // ASSEMBLY_HELPER_H