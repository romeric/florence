#include <cstdint>

#include "Fastor/Fastor.h"

using Fastor::_mm_loadul3_ps;
using Fastor::_mm256_loadul3_pd;


#ifndef LL_TYPES
#define LL_TYPES
using Real = double;
using Integer = std::int64_t;
using UInteger = std::uint64_t;
#endif


#ifndef CUSTOM_ALLOCATION_
#define CUSTOM_ALLOCATION_
template<typename T>
FASTOR_INLINE T *allocate(Integer size) {
#if defined(__AVX512F__)
    T *out = (T*)_mm_malloc(sizeof(T)*size,64);
#elif defined(__AVX__)
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
#if defined(__SSE__) || defined(__AVX__) || defined(__AVX512F__)
    _mm_free(a);
#else
    free(a);
#endif
}
#endif



FASTOR_INLINE void _SIMD_BDB_Integrator_DF_2D_(
    Real *HBT,
    Real *BDB,
    const Real* SpatialGradient,
    const Real* H_Voigt,
    int noderpelem)
{

    constexpr int ndim = 2;
    constexpr int nvar = 2;
    constexpr int H_VoigtSize = 3;
    int ndof = nvar*noderpelem;


    using VEC = Fastor::SIMDVector<Real,simd_abi::avx>;
    constexpr int Size = VEC::Size;
    int ROUND_AVX = ROUND_DOWN(noderpelem,Size);

    // Real *HBT = allocate<Real>(ndof*H_VoigtSize);

    VEC a1, a2, out1, out2;

    VEC b1(_mm256_loadul3_pd(&H_Voigt[0]));
    VEC b2(_mm256_loadul3_pd(&H_Voigt[3]));
    VEC b3(_mm256_loadul3_pd(&H_Voigt[6]));

    for (int i=0; i<noderpelem; ++i) {

        a1.set(SpatialGradient[i*ndim]);
        a2.set(SpatialGradient[i*ndim+1]);

        out1 = a1*b1 + a2*b3;
        out2 = a2*b2 + a1*b3;

        out1.store(&HBT[2*i*H_VoigtSize],false);
        out2.store(&HBT[(2*i+1)*H_VoigtSize],false);

    }

    VEC a3;
    int jj = 0;
    for (int i=0; i<ndof; ++i) {

        a1.set(HBT[i*H_VoigtSize]);
        a2.set(HBT[i*H_VoigtSize+1]);
        a3.set(HBT[i*H_VoigtSize+2]);

        int j = 0;
        for ( ; j<ROUND_AVX; j+=Size) {
            b1.set(SpatialGradient[(j+3)*ndim],SpatialGradient[(j+2)*ndim],SpatialGradient[(j+1)*ndim],SpatialGradient[(j+0)*ndim]);
            b2.set(SpatialGradient[(j+3)*ndim+1],SpatialGradient[(j+2)*ndim+1],SpatialGradient[(j+1)*ndim+1],SpatialGradient[(j+0)*ndim+1]);

            out1 = a1*b1+a3*b2;
            out2 = a2*b2+a3*b1;

            Fastor::data_setter(BDB, out1, i*ndof+j*nvar,   nvar);
            Fastor::data_setter(BDB, out2, i*ndof+j*nvar+1, nvar);
        }
        jj = j;
    }

    for (int i=0; i<ndof; ++i) {

        Real _a1 = HBT[i*H_VoigtSize];
        Real _a2 = HBT[i*H_VoigtSize+1];
        Real _a3 = HBT[i*H_VoigtSize+2];

        int j = jj;
        for ( ; j<noderpelem; ++j) {

            Real _b1 = SpatialGradient[(j+0)*ndim];
            Real _b2 = SpatialGradient[(j+0)*ndim+1];

            Real _out1 = _a1*_b1+_a3*_b2;
            Real _out2 = _a2*_b2+_a3*_b1;

            BDB[i*ndof+j*nvar] = _out1;
            BDB[i*ndof+j*nvar+1] = _out2;
        }
    }

    // deallocate(HBT);
}