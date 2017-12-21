#include <Fastor.h>

using Fastor::SIMDVector;

#ifdef __AVX__

// (2xk) x (kx2) matrices
FASTOR_INLINE
void _matmul_2k2(size_t K, const double * FASTOR_RESTRICT a, const double * FASTOR_RESTRICT b, double * FASTOR_RESTRICT out) {

    __m128d out_row0 = ZEROPD;
    __m128d out_row1 = ZEROPD;

    for (size_t i=0; i<K; ++i) {
        __m128d brow = _mm_loadu_pd(&b[i*2]);
#ifndef __FMA__
        // row 0
        __m128d a_vec0 = _mm_set1_pd(a[i]);
        out_row0 = _mm_add_pd(out_row0,_mm_mul_pd(a_vec0,brow));
        // row 1
        __m128d a_vec1 = _mm_set1_pd(a[K+i]);
        out_row1 = _mm_add_pd(out_row1,_mm_mul_pd(a_vec1,brow));
#else
        // row 0
        __m128d a_vec0 = _mm_set1_pd(a[i]);
        out_row0 = _mm_fmadd_pd(a_vec0,brow,out_row0);
        // row 1
        __m128d a_vec1 = _mm_set1_pd(a[K+i]);
        out_row1 = _mm_fmadd_pd(a_vec1,brow,out_row1);
#endif
    }
    _mm_store_pd(out,out_row0);
    _mm_storeu_pd(out+2,out_row1);
}


// (2xk) x (kx2) matrices
FASTOR_INLINE
void _matmul_2k2(size_t K, const float * FASTOR_RESTRICT a, const float * FASTOR_RESTRICT b, float * FASTOR_RESTRICT out) {

    __m128 out_row0 = ZEROPS;
    __m128 out_row1 = ZEROPS;

    for (size_t i=0; i<K; i++) {
        __m128 brow = _mm_loadu_ps(&b[i*2]);
#ifndef __FMA__
        // row 0
        __m128 a_vec0 = _mm_set1_ps(a[i]);
        out_row0 = _mm_add_ps(out_row0,_mm_mul_ps(a_vec0,brow));
        // row 1
        __m128 a_vec1 = _mm_set1_ps(a[K+i]);
        out_row1 = _mm_add_ps(out_row1,_mm_mul_ps(a_vec1,brow));
#else
        // row 0
        __m128 a_vec0 = _mm_set1_ps(a[i]);
        out_row0 = _mm_fmadd_ps(a_vec0,brow,out_row0);
        // row 1
        __m128 a_vec1 = _mm_set1_ps(a[K+i]);
        out_row1 = _mm_fmadd_ps(a_vec1,brow,out_row1);
#endif
    }
    _mm_store_ps(out,_mm_shuffle_ps(out_row0,out_row1,_MM_SHUFFLE(1,0,1,0)));
}


// (3xk) x (kx3) matrices
FASTOR_INLINE
void _matmul_3k3(size_t K, const double * FASTOR_RESTRICT a, const double * FASTOR_RESTRICT b, double * FASTOR_RESTRICT out) {

    __m256d out_row0 = VZEROPD;
    __m256d out_row1 = VZEROPD;
    __m256d out_row2 = VZEROPD;

    for (size_t i=0; i<K; ++i) {
        __m256d brow = _mm256_loadul3_pd(&b[i*3]);
#ifndef __FMA__
        // row 0
        __m256d a_vec0 = _mm256_set1_pd(a[i]);
        out_row0 = _mm256_add_pd(out_row0,_mm256_mul_pd(a_vec0,brow));
        // row 1
        __m256d a_vec1 = _mm256_set1_pd(a[K+i]);
        out_row1 = _mm256_add_pd(out_row1,_mm256_mul_pd(a_vec1,brow));
        // row 2
        __m256d a_vec2 = _mm256_set1_pd(a[2*K+i]);
        out_row2 = _mm256_add_pd(out_row2,_mm256_mul_pd(a_vec2,brow));
#else
        // row 0
        __m256d a_vec0 = _mm256_set1_pd(a[i]);
        out_row0 = _mm256_fmadd_pd(a_vec0,brow,out_row0);
        // row 1
        __m256d a_vec1 = _mm256_set1_pd(a[K+i]);
        out_row1 = _mm256_fmadd_pd(a_vec1,brow,out_row1);
        // row 2
        __m256d a_vec2 = _mm256_set1_pd(a[2*K+i]);
        out_row2 = _mm256_fmadd_pd(a_vec2,brow,out_row2);
#endif
    }
    _mm256_store_pd(out,out_row0);
    _mm256_storeu_pd(out+3,out_row1);
    _mm256_storeu_pd(out+6,out_row2);
}


// (3xk) x (kx3) matrices
FASTOR_INLINE
void _matmul_3k3(size_t K, const float * FASTOR_RESTRICT a, const float * FASTOR_RESTRICT b, float * FASTOR_RESTRICT out) {

    __m128 out_row0 = ZEROPS;
    __m128 out_row1 = ZEROPS;
    __m128 out_row2 = ZEROPS;

    for (size_t i=0; i<K; ++i) {
        __m128 brow = _mm_loadul3_ps(&b[i*3]);
#ifndef __FMA__
        // row 0
        __m128 a_vec0 = _mm_set1_ps(a[i]);
        out_row0 = _mm_add_ps(out_row0,_mm_mul_ps(a_vec0,brow));
        // row 1
        __m128 a_vec1 = _mm_set1_ps(a[K+i]);
        out_row1 = _mm_add_ps(out_row1,_mm_mul_ps(a_vec1,brow));
        // row 2
        __m128 a_vec2 = _mm_set1_ps(a[2*K+i]);
        out_row2 = _mm_add_ps(out_row2,_mm_mul_ps(a_vec2,brow));
#else
        // row 0
        __m128 a_vec0 = _mm_set1_ps(a[i]);
        out_row0 = _mm_fmadd_ps(a_vec0,brow,out_row0);
        // row 1
        __m128 a_vec1 = _mm_set1_ps(a[K+i]);
        out_row1 = _mm_fmadd_ps(a_vec1,brow,out_row1);
        // row 2
        __m128 a_vec2 = _mm_set1_ps(a[2*K+i]);
        out_row2 = _mm_fmadd_ps(a_vec2,brow,out_row2);
#endif

    }
    _mm_store_ps(out,out_row0);
    _mm_storeu_ps(out+3,out_row1);
    _mm_storeu_ps(out+6,out_row2);
}



// (2x2) x (2xn) matrices
template<typename T>
FASTOR_INLINE
void _matmul_22k(size_t N, const T* FASTOR_RESTRICT a, const T* FASTOR_RESTRICT b, T* FASTOR_RESTRICT out) {


    using V256 = SIMDVector<T,256>;
    using V128 = SIMDVector<T,128>;
    constexpr size_t M = 2;

    constexpr size_t SIZE_AVX = V256::Size;
    constexpr size_t SIZE_SSE = V128::Size;
    const size_t ROUND_AVX = ROUND_DOWN(N,SIZE_AVX);
    const size_t ROUND_SSE = ROUND_DOWN(N,SIZE_SSE);

    size_t k=0;
    for (; k<ROUND_AVX; k+=SIZE_AVX) {

        V256 out_row0, out_row1, vec_a0, vec_a1;
        for (size_t i=0; i<2; ++i) {
            V256 brow; brow.load(&b[i*N+k],false);
            vec_a0.set(a[i]);
            vec_a1.set(a[i+M]);
#ifndef __FMA__
            out_row0 += vec_a0*brow;
            out_row1 += vec_a1*brow;
#else
            out_row0 = fmadd(vec_a0,brow,out_row0);
            out_row1 = fmadd(vec_a1,brow,out_row1);
#endif
        }
        out_row0.store(out+k,false);
        out_row1.store(out+N+k,false);
    }

    for (; k<ROUND_SSE; k+=SIZE_SSE) {
        V128 out_row0, out_row1, vec_a0, vec_a1, brow;
        for (size_t i=0; i<2; ++i) {
            V128 brow; brow.load(&b[i*N+k],false);
            vec_a0.set(a[i]);
            vec_a1.set(a[i+M]);
#ifndef __FMA__
            out_row0 += vec_a0*brow;
            out_row1 += vec_a1*brow;
#else
            out_row0 = fmadd(vec_a0,brow,out_row0);
            out_row1 = fmadd(vec_a1,brow,out_row1);
#endif
        }
        out_row0.store(out+k,false);
        out_row1.store(out+N+k,false);
    }

    for (; k<N; k++) {
        T out_row0=0., out_row1=0.;
        for (size_t i=0; i<2; ++i) {
            T brow = b[i*N+k];
            out_row0 += a[i]*brow;
            out_row1 += a[i+M]*brow;
        }
        out[k] = out_row0;
        out[N+k] = out_row1;
    }
}


// (3x3) x (3xn) matrices
template<typename T>
FASTOR_INLINE
void _matmul_33k(size_t N, const T* FASTOR_RESTRICT a, const T* FASTOR_RESTRICT b, T* FASTOR_RESTRICT out) {


    using V256 = SIMDVector<T,256>;
    using V128 = SIMDVector<T,128>;
    constexpr size_t M = 3;

    constexpr size_t SIZE_AVX = V256::Size;
    constexpr size_t SIZE_SSE = V128::Size;
    const size_t ROUND_AVX = ROUND_DOWN(N,SIZE_AVX);
    const size_t ROUND_SSE = ROUND_DOWN(N,SIZE_SSE);

    size_t k=0;
    for (; k<ROUND_AVX; k+=SIZE_AVX) {

        V256 out_row0, out_row1, out_row2, vec_a0, vec_a1, vec_a2;
        for (size_t i=0; i<3; ++i) {
            V256 brow; brow.load(&b[i*N+k],false);
            vec_a0.set(a[i]);
            vec_a1.set(a[i+M]);
            vec_a2.set(a[i+2*M]);
#ifndef __FMA__
            out_row0 += vec_a0*brow;
            out_row1 += vec_a1*brow;
            out_row2 += vec_a2*brow;
#else
            out_row0 = fmadd(vec_a0,brow,out_row0);
            out_row1 = fmadd(vec_a1,brow,out_row1);
            out_row2 = fmadd(vec_a2,brow,out_row2);
#endif
        }
        out_row0.store(out+k,false);
        out_row1.store(out+N+k,false);
        out_row2.store(out+2*N+k,false);
    }

    for (; k<ROUND_SSE; k+=SIZE_SSE) {
        V128 out_row0, out_row1, out_row2, vec_a0, vec_a1, vec_a2;
        for (size_t i=0; i<3; ++i) {
            V128 brow; brow.load(&b[i*N+k],false);
            vec_a0.set(a[i]);
            vec_a1.set(a[i+M]);
            vec_a2.set(a[i+2*M]);
#ifndef __FMA__
            out_row0 += vec_a0*brow;
            out_row1 += vec_a1*brow;
            out_row2 += vec_a2*brow;
#else
            out_row0 = fmadd(vec_a0,brow,out_row0);
            out_row1 = fmadd(vec_a1,brow,out_row1);
            out_row2 = fmadd(vec_a2,brow,out_row2);
#endif
        }
        out_row0.store(out+k,false);
        out_row1.store(out+N+k,false);
        out_row2.store(out+2*N+k,false);
    }

    for (; k<N; k++) {
        T out_row0=0., out_row1=0., out_row2=0.;
        for (size_t i=0; i<3; ++i) {
            T brow = b[i*N+k];
            out_row0 += a[i]*brow;
            out_row1 += a[i+M]*brow;
            out_row2 += a[i+2*M]*brow;
        }
        out[k] = out_row0;
        out[N+k] = out_row1;
        out[2*N+k] = out_row2;
    }
}


#endif


// Non-sqaure matrices
template<typename T>
FASTOR_INLINE
void _matmul_(size_t M, size_t N, size_t K, const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {

#ifdef __AVX__
    if (M==3 && N==3 && N!=K) {
        _matmul_3k3(K,a,b,out);
        return;
    }
    else if (M==2 && N==2 && N!=K) {
        _matmul_2k2(K,a,b,out);
        return;
    }
    // The following specialisations don't make much of a difference (at least for SP)
    // Need thorough performance checks
    if (M==3 && K==3 && N!=K) {
        _matmul_33k(N,a,b,out);
        return;
    }
    else if (M==2 && K==2 && N!=K) {
        _matmul_22k(N,a,b,out);
        return;
    }
#endif

    using V256 = SIMDVector<T,256>;
    using V128 = SIMDVector<T,128>;

    constexpr size_t SIZE_AVX = V256::Size;
    constexpr size_t SIZE_SSE = V128::Size;
    const size_t ROUND_AVX = ROUND_DOWN(N,SIZE_AVX);
    const size_t ROUND_SSE = ROUND_DOWN(N,SIZE_SSE);

    for (size_t j=0; j<M; ++j) {
        size_t k=0;
        for (; k<ROUND_AVX; k+=SIZE_AVX) {
            V256 out_row, vec_a;
            for (size_t i=0; i<K; ++i) {
                V256 brow; brow.load(&b[i*N+k],false);
                vec_a.set(a[j*K+i]);
#ifndef __FMA__
                out_row += vec_a*brow;
#else
                out_row = fmadd(vec_a,brow,out_row);
#endif
            }
            out_row.store(out+k+N*j,false);
        }

        for (; k<ROUND_SSE; k+=SIZE_SSE) {
            V128 out_row, vec_a;
            for (size_t i=0; i<K; ++i) {
                V128 brow; brow.load(&b[i*N+k],false);
                vec_a.set(a[j*K+i]);
#ifndef __FMA__
                out_row += vec_a*brow;
#else
                out_row = fmadd(vec_a,brow,out_row);
#endif
            }
            out_row.store(out+k+N*j,false);
        }

        for (; k<N; k++) {
            T out_row = 0.;
            for (size_t i=0; i<K; ++i) {
                out_row += a[j*K+i]*b[i*N+k];
            }
            out[N*j+k] = out_row;
        }
    }
}

