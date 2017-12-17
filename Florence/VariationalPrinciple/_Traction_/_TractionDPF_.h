#include <algorithm>
#include <numeric>

#ifdef __SSE4_2__
#include <emmintrin.h>
#endif

typedef double Real;


inline void GetTotalTraction_DPF_(Real *TotalTraction, const Real *CauchyStressTensor,
    const Real *ElectricDisplacementx, int ndim) {
    if (ndim==3) {
        TotalTraction[0] = CauchyStressTensor[0];
        TotalTraction[1] = CauchyStressTensor[4];
        TotalTraction[2] = CauchyStressTensor[8];
        TotalTraction[3] = CauchyStressTensor[1];
        TotalTraction[4] = CauchyStressTensor[2];
        TotalTraction[5] = CauchyStressTensor[5];

        TotalTraction[6] = ElectricDisplacementx[0];
        TotalTraction[7] = ElectricDisplacementx[1];
        TotalTraction[8] = ElectricDisplacementx[2];
    }
    else if (ndim == 2) {
        TotalTraction[0] = CauchyStressTensor[0];
        TotalTraction[1] = CauchyStressTensor[3];
        TotalTraction[2] = CauchyStressTensor[1];

        TotalTraction[3] = ElectricDisplacementx[0];
        TotalTraction[4] = ElectricDisplacementx[1];
    }
}


inline void FillConstitutiveB_DPF_(Real *B, const Real* SpatialGradient,
                     int ndim, int nvar, int rows, int cols) {
    int i = 0;

    if (ndim == 3) {

        for (; i<rows; ++i) {

            // Store in registers
            const Real a0 = SpatialGradient[i*ndim];
            const Real a1 = SpatialGradient[i*ndim+1];
            const Real a2 = SpatialGradient[i*ndim+2];

            // MECHANICAL TERMS
            B[i*cols*nvar]              = a0;
            B[i*cols*nvar+cols+1]       = a1;
            B[i*cols*nvar+2*(cols+1)]   = a2;

            B[i*cols*nvar+cols+5]       = a2;
            B[i*cols*nvar+2*cols+5]     = a1;

            B[i*cols*nvar+4]            = a2;
            B[i*cols*nvar+2*cols+4]     = a0;

            B[i*cols*nvar+3]            = a1;
            B[i*cols*nvar+cols+3]       = a0;

            // ELECTROSTATIC TERMS
            B[i*cols*nvar+3*cols+6]     = a0;
            B[i*cols*nvar+3*cols+7]     = a1;
            B[i*cols*nvar+3*cols+8]     = a2;
        }
    }

    else if (ndim == 2) {

        for (; i<rows; ++i) {

            // Store in registers
            const Real a0 = SpatialGradient[i*ndim];
            const Real a1 = SpatialGradient[i*ndim+1];

            // MECHANICAL TERMS
            B[i*cols*nvar]              = a0;
            B[i*cols*nvar+cols+1]       = a1;

            B[i*cols*nvar+2]            = a1;
            B[i*cols*nvar+cols+2]       = a0;

            // ELECTROSTATIC TERMS
            B[i*cols*nvar+2*cols+3]     = a0;
            B[i*cols*nvar+2*cols+4]     = a1;
        }
    }
}
inline void _TractionDPF_Filler_(Real *traction,
    const Real* SpatialGradient,
    const Real* ElectricDisplacementx,
    const Real* CauchyStressTensor,
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
    if (ndim==3) {
        t = (Real*)_mm_malloc(9*sizeof(Real),32);
    }
    else if (ndim==2) {
        t = (Real*)_mm_malloc(5*sizeof(Real),32);
    }

    Real *B = (Real*)_mm_malloc(H_VoigtSize*local_size*sizeof(Real),32);
#else
    if (ndim==3) {
        t = (Real*)malloc(9*sizeof(Real));
    }
    else if (ndim==2) {
        t = (Real*)malloc(5*sizeof(Real));
    }

    Real *B = (Real*)malloc(H_VoigtSize*local_size*sizeof(Real));
#endif

    std::fill(B,B+H_VoigtSize*local_size,0.);

    for (int igauss = 0; igauss < ngauss; ++igauss) {

        FillConstitutiveB_DPF_(B,&SpatialGradient[igauss*ndim*noderpelem],ndim,nvar,noderpelem,H_VoigtSize);

        if (requires_geometry_update==1) {
            // Compute tractions
            GetTotalTraction_DPF_(t,
                &CauchyStressTensor[igauss*ndim*ndim],
                &ElectricDisplacementx[igauss*ndim], ndim);

            // Multiply B with traction - for loop is okay
            const Real detJ_igauss = detJ[igauss];
            for (int i=0; i<local_size; ++i) {
                Real tmp = 0;
                for (int j=0; j<H_VoigtSize; ++j) {
                    tmp += B[i*H_VoigtSize+j]*t[j];
                }
                // local_traction[i] = tmp;
                traction[i] += tmp*detJ_igauss;
            }
        }
    }

#ifdef __SSE4_2__
    _mm_free(t);
    _mm_free(B);
#else
    free(t);
    free(B);
#endif
}