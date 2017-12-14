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


inline void GetTotalTraction_(Real *TotalTraction,
    const Real *ElectricDisplacementx, int ndim) {
    // if (ndim==3) {
    //     TotalTraction[0] = ElectricDisplacementx[0];
    //     TotalTraction[1] = ElectricDisplacementx[1];
    //     TotalTraction[2] = ElectricDisplacementx[2];
    // }
    // else if (ndim == 2) {
    //     TotalTraction[0] = ElectricDisplacementx[0];
    //     TotalTraction[1] = ElectricDisplacementx[1];
    // }
    std::copy(ElectricDisplacementx,ElectricDisplacementx+ndim,TotalTraction);
}


inline void FillConstitutiveB_(Real *B, const Real* SpatialGradient,
                     int ndim, int rows, int cols) {
    int i = 0;

    if (ndim == 3) {

        for (; i<rows; ++i) {
            // Store in registers
            const Real a0 = SpatialGradient[i*ndim];
            const Real a1 = SpatialGradient[i*ndim+1];
            const Real a2 = SpatialGradient[i*ndim+2];

            // ELECTROSTATIC TERMS
            B[i*cols+0]     = a0;
            B[i*cols+1]     = a1;
            B[i*cols+2]     = a2;
        }
    }

    else if (ndim == 2) {

        for (; i<rows; ++i) {
            // Store in registers
            const Real a0 = SpatialGradient[i*ndim];
            const Real a1 = SpatialGradient[i*ndim+1];

            // ELECTROSTATIC TERMS
            B[i*cols+0]     = a0;
            B[i*cols+1]     = a1;
        }
    }
}
#include <iostream>
inline void _ConstitutiveStiffnessLaplacian_Filler_(Real *stiffness, Real *traction,
    const Real* SpatialGradient,
    const Real* ElectricDisplacementx,
    const Real* H_Voigt,
    const Real* detJ,
    int ngauss,
    int noderpelem,
    int ndim,
    int nvar,
    int H_VoigtSize,
    int requires_geometry_update) {


    int local_size = nvar*noderpelem;

    Real *t;

#ifdef __SSE4_2__
    t = (Real*)_mm_malloc(ndim*sizeof(Real),32);
    Real *B = (Real*)_mm_malloc(H_VoigtSize*local_size*sizeof(Real),32);
    Real *HBT = (Real*)_mm_malloc(H_VoigtSize*local_size*sizeof(Real),32);
    Real *BDB_1 = (Real*)_mm_malloc(local_size*local_size*sizeof(Real),32);
#else
    t = (Real*)malloc(ndim*sizeof(Real));
    Real *B = (Real*)malloc(H_VoigtSize*local_size*sizeof(Real));
    Real *HBT = (Real*)malloc(H_VoigtSize*local_size*sizeof(Real));
    Real *BDB_1 = (Real*)malloc(local_size*local_size*sizeof(Real));
#endif

    std::fill(B,B+H_VoigtSize*local_size,0.);

    for (int igauss = 0; igauss < ngauss; ++igauss) {

        FillConstitutiveB_(B,&SpatialGradient[igauss*ndim*noderpelem],ndim,noderpelem,H_VoigtSize);

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
            H_VoigtSize, local_size, H_VoigtSize, 1.0, &H_Voigt[igauss*H_VoigtSize*H_VoigtSize], H_VoigtSize, B, H_VoigtSize, 0.0, HBT, local_size);

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            local_size, local_size, H_VoigtSize, 1.0, B, H_VoigtSize, HBT, local_size, 0.0, BDB_1, local_size);

        // Multiply stiffness with detJ
        const Real detJ_igauss = detJ[igauss];
        for (int i=0; i<local_size*local_size; ++i) {
            stiffness[i] += BDB_1[i]*detJ_igauss;
        }

        // Compute tractions
        GetTotalTraction_(t,
            &ElectricDisplacementx[igauss*ndim], ndim);

        // Multiply B with traction - for loop is okay
        for (int i=0; i<local_size; ++i) {
            Real tmp = 0;
            for (int j=0; j<H_VoigtSize; ++j) {
                tmp += B[i*H_VoigtSize+j]*t[j];
            }
            traction[i] += tmp*detJ_igauss;
        }
    }

#ifdef __SSE4_2__
    _mm_free(t);
    _mm_free(B);
    _mm_free(HBT);
    _mm_free(BDB_1);
#else
    free(t);
    free(B);
    free(HBT);
    free(BDB_1);
#endif
}