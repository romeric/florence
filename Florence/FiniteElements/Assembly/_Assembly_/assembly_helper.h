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



using Integer = long long;
using UInteger = unsigned long long;
using Real = double;

using V = Fastor::SIMDVector<Real>;


// Helper functions
/*---------------------------------------------------------------------------------------------*/
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


/*-------------------
template<typename T>
FASTOR_INLINE void fill_(T *__restrict__ a, Integer size, T num) {
    V _vec(num);
    size_t i=0;
    for (; i<ROUND_DOWN(size,V::Size); i+=V::Size) {
        _vec.store(a+i,true);
    }
    for (; i<size; ++i) {
        a[i] = num;
    }
}



template<typename T>
FASTOR_INLINE void iadd_(T *__restrict__ a, const T *__restrict__ b, Integer size) {
    size_t i=0;
    for (; i<ROUND_DOWN(size,V::Size); i+=V::Size) {
        (V(a+i)+V(b+i)).store(a+i);
    }
    for (; i<size; ++i) {
        a[i] += b[i];
    }
}
-------------------*/
/*---------------------------------------------------------------------------------------------*/


















// IJV Filler
/*---------------------------------------------------------------------------------------------*/
FASTOR_INLINE
void fill_triplet(  const Integer *i,
                    const Integer *j,
                    const Real *coeff,
                    int *I,
                    int *J,
                    Real *V,
                    Integer elem,
                    Integer nvar,
                    Integer nodeperelem,
                    const UInteger *elements,
                    Integer i_shape,
                    Integer j_shape
                    ) {

    Integer *current_row_column = allocate<Integer>(nvar*nodeperelem);
    Integer *full_current_row = allocate<Integer>(i_shape);
    Integer *full_current_column = allocate<Integer>(j_shape);

    Integer ndof = nvar*nodeperelem;

    Integer const_elem_retriever;
    for (Integer counter=0; counter<nodeperelem; ++counter) {
        const_elem_retriever = nvar*elements[elem*nodeperelem+counter];
        for (Integer ncounter=0; ncounter<nvar; ++ncounter) {
            current_row_column[nvar*counter+ncounter] = const_elem_retriever+ncounter;
        }
    }

    // memcpy(full_current_row,i,i_shape*sizeof(Integer));
    // memcpy(full_current_column,j,j_shape*sizeof(Integer));

    Integer const_I_retriever;
    for (Integer counter=0; counter<ndof; ++counter) {
        const_I_retriever = current_row_column[counter];
        for (Integer iterator=0; iterator<ndof; ++iterator) {
            full_current_row[counter*ndof+iterator]    = const_I_retriever;
            full_current_column[counter*ndof+iterator] = current_row_column[iterator];
        }
    }


    Integer low, high;
    low = ndof*ndof*elem;
    high = ndof*ndof*(elem+1);

    Integer incrementer = 0;
    for (Integer counter = low; counter < high; ++counter) {
        I[counter] = full_current_row[incrementer];
        J[counter] = full_current_column[incrementer];
        V[counter] = coeff[incrementer];

        incrementer += 1;
    }

    deallocate(full_current_row);
    deallocate(full_current_column);
    deallocate(current_row_column);
}
/*---------------------------------------------------------------------------------------------*/




#endif // ASSEMBLY_HELPER_H