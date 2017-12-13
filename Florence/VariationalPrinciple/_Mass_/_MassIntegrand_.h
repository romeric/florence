#include <algorithm>
#include <numeric>

#ifdef HAS_MKL
#include <mkl.h>
#else
#include <cblas.h>
#endif

#ifdef __SSE4_2__
#include <emmintrin.h>
#endif

typedef double Real;


inline void _MassIntegrand_Filler_(Real *mass,
    const Real* bases,
    const Real* detJ,
    int ngauss,
    int noderpelem,
    int ndim,
    int nvar,
    Real rho) {


    int local_size = nvar*noderpelem;

#ifdef __SSE4_2__
    Real *N = (Real*)_mm_malloc(nvar*local_size*sizeof(Real),32);
    Real *rhoNN = (Real*)_mm_malloc(local_size*local_size*sizeof(Real),32);
#else
    Real *N = (Real*)malloc(nvar*local_size*sizeof(Real));
    Real *rhoNN = (Real*)malloc(local_size*local_size*sizeof(Real));
#endif

    std::fill(N,N+nvar*local_size,0.);

    for (int igauss = 0; igauss < ngauss; ++igauss) {

        // Fill mass integrand
        for (int j=0; j<noderpelem; ++j) {
            const Real bases_j = bases[j*ngauss+igauss];
            for (int ivar=0; ivar<ndim; ++ivar) {
                N[j*nvar*nvar+ivar*nvar+ivar] = bases_j;
            }
        }

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
            local_size, local_size, nvar, rho, N, nvar, N, nvar, 0.0, rhoNN, local_size);

        // Multiply mass with detJ
        const Real detJ_igauss = detJ[igauss];
        for (int i=0; i<local_size*local_size; ++i) {
            mass[i] += rhoNN[i]*detJ_igauss;
        }

    }

#ifdef __SSE4_2__
    _mm_free(N);
    _mm_free(rhoNN);
#else
    free(N);
    free(rhoNN);
#endif
}




inline void _ConstantMassIntegrand_Filler_(Real *mass,
    const Real* constant_mass_integrand,
    const Real* detJ,
    int ngauss,
    int local_capacity) {

    for (int igauss = 0; igauss < ngauss; ++igauss) {
        // Multiply mass with detJ
        const Real detJ_igauss = detJ[igauss];
        for (int i=0; i<local_capacity; ++i) {
            mass[i] += constant_mass_integrand[igauss*local_capacity+i]*detJ_igauss;
        }
    }
}