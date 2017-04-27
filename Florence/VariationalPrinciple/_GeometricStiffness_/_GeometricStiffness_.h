#ifdef __SSE4_2__
#include <emmintrin.h>
#endif

typedef double Real;

// The cost of numerical integration of geometric stiffness matrix
// is "ngauss x [(55) x noderperelem**2]" for 2D completely unrolled version
// and "ngauss x [(48) x noderperelem**2]" for 2D unrolled + SSE version.
// For 3D problems the cost is "ngauss x [(98) x noderperelem**2]" for completely
// unrolled version. These costs are for both displacement and displacement potential
// formulations

// Note the following strategy assumes that the Cauchy stress is symmetric



#define ABI 1
#ifdef __SSE4_2__
#define Aligned
#endif

#ifdef __SSE4_2__
static inline __m128d h_add_pd(__m128d a) {
    // 4 cycles
    return _mm_add_pd(a,_mm_shuffle_pd(a,a,0x1));
}
#endif



#if ABI==0

static inline void _GeometricStiffnessFiller_(Real *geometric_stiffness, 
    const Real *SpatialGradient, const Real *CauchyStressTensor, Real *detJ, 
    int ndim, int nvar, int nodeperelem, int nguass) {

    for (int g=0; g<nguass; ++g) {

        for (int a=0; a<nodeperelem; ++a) {
            for (int b=0; b<nodeperelem; ++b) {

                Real dum=0.;
                for (int i=0; i<ndim; ++i) {
                    for (int j=0; j<ndim; ++j) {
                        dum += SpatialGradient[g*nodeperelem*ndim+a*ndim+i]*CauchyStressTensor[g*ndim*ndim+i*ndim+j]*\
                            SpatialGradient[g*nodeperelem*ndim+b*ndim+j];
                    }
                }

                for (int i=0; i<ndim; ++i) {
                    geometric_stiffness[(a*nvar+i)*nodeperelem*nvar+(b*nvar+i)] += dum*detJ[g];
                }
            }
        }
    }
} 

#elif ABI==1

static inline void _GeometricStiffnessFiller_(Real *geometric_stiffness, 
    const Real *SpatialGradient, const Real *CauchyStressTensor, const Real *detJ, 
    const int ndim, const int nvar, const int nodeperelem, const int nguass) {

    if (ndim==3) {

        for (int g=0; g<nguass; ++g) {

            for (int a=0; a<nodeperelem; ++a) {
                for (int b=0; b<nodeperelem; ++b) {

                    // If SSE fails due numpy array alignment - use this
                    Real a0 = SpatialGradient[g*nodeperelem*3+a*3];
                    Real a1 = SpatialGradient[g*nodeperelem*3+a*3+1];
                    Real a2 = SpatialGradient[g*nodeperelem*3+a*3+2];

                    Real b0 = SpatialGradient[g*nodeperelem*3+b*3];
                    Real b1 = SpatialGradient[g*nodeperelem*3+b*3+1];
                    Real b2 = SpatialGradient[g*nodeperelem*3+b*3+2];

                    Real s00 = CauchyStressTensor[g*9];
                    Real s01 = CauchyStressTensor[g*9+1];
                    Real s02 = CauchyStressTensor[g*9+2];
                    Real s11 = CauchyStressTensor[g*9+4];
                    Real s12 = CauchyStressTensor[g*9+5];
                    Real s22 = CauchyStressTensor[g*9+8];                    
                    
                    Real dum0 = a0*(s00*b0+s01*b1+s02*b2);
                    Real dum1 = a1*(s01*b0+s11*b1+s12*b2);
                    Real dum2 = a2*(s02*b0+s12*b1+s22*b2); 
                    Real dum = dum0 + dum1 + dum2;

                    geometric_stiffness[(a*nvar)*nodeperelem*nvar+(b*nvar)] += dum*detJ[g];
                    geometric_stiffness[(a*nvar+1)*nodeperelem*nvar+(b*nvar+1)] += dum*detJ[g];
                    geometric_stiffness[(a*nvar+2)*nodeperelem*nvar+(b*nvar+2)] += dum*detJ[g];
                }
            }
        }
    }

    else if (ndim==2) {

        for (int g=0; g<nguass; ++g) {

            for (int a=0; a<nodeperelem; ++a) {
                for (int b=0; b<nodeperelem; ++b) {

#ifdef Aligned
                    // Assuming numpy array is aligned
                    __m128d as = _mm_load_pd(&SpatialGradient[g*nodeperelem*2+a*2]);
                    __m128d bs = _mm_load_pd(&SpatialGradient[g*nodeperelem*2+b*2]);
                    __m128d s_row0 = _mm_load_pd(&CauchyStressTensor[g*4]);
                    __m128d s_row1 = _mm_load_pd(&CauchyStressTensor[g*4+2]);

                    __m128d mul0 = _mm_mul_pd(s_row0,bs);  mul0 = h_add_pd(mul0);
                    __m128d mul1 = _mm_mul_pd(s_row1,bs);  mul1 = h_add_pd(mul1);
                    __m128d mull = _mm_add_pd(_mm_mul_pd(as,mul0),_mm_mul_pd(_mm_shuffle_pd(as,as,0x1),mul1));

                    Real dum = _mm_cvtsd_f64(mull);
#else
                    // If SSE fails due numpy array alignment - use this
                    Real a0 = SpatialGradient[g*nodeperelem*2+a*2];
                    Real a1 = SpatialGradient[g*nodeperelem*2+a*2+1];

                    Real b0 = SpatialGradient[g*nodeperelem*2+b*2];
                    Real b1 = SpatialGradient[g*nodeperelem*2+b*2+1];

                    Real s00 = CauchyStressTensor[g*4];
                    Real s01 = CauchyStressTensor[g*4+1];
                    // Real s10 = CauchyStressTensor[g*4+2];
                    Real s11 = CauchyStressTensor[g*4+3];
                    
                    Real dum0 = a0*(s00*b0+s01*b1);
                    Real dum1 = a1*(s01*b0+s11*b1); 
                    Real dum = dum0 + dum1;
#endif

                    geometric_stiffness[(a*nvar)*nodeperelem*nvar+(b*nvar)] += dum*detJ[g];
                    geometric_stiffness[(a*nvar+1)*nodeperelem*nvar+(b*nvar+1)] += dum*detJ[g];
                }
            }
        }
    }
} 

#endif