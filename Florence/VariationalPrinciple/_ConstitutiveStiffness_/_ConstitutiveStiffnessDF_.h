#ifndef CONSTITUTIVE_DF_H
#define CONSTITUTIVE_DF_H

#include <algorithm>
#include <numeric>
#include "SIMD_BDB_Integrator.h"

#ifdef HAS_MKL
#include <mkl.h>
#else
#include <cblas.h>
#endif


inline void GetTotalTraction_(Real *TotalTraction, const Real *CauchyStressTensor, int ndim) {
    if (ndim==3) {
        TotalTraction[0] = CauchyStressTensor[0];
        TotalTraction[1] = CauchyStressTensor[4];
        TotalTraction[2] = CauchyStressTensor[8];
        TotalTraction[3] = CauchyStressTensor[1];
        TotalTraction[4] = CauchyStressTensor[2];
        TotalTraction[5] = CauchyStressTensor[5];
    }
    else if (ndim == 2) {
        TotalTraction[0] = CauchyStressTensor[0];
        TotalTraction[1] = CauchyStressTensor[3];
        TotalTraction[2] = CauchyStressTensor[1];
    }
}


inline void FillConstitutiveB_(Real *B, const Real* SpatialGradient,
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
        }
    }
}





inline void _ConstitutiveStiffnessIntegrandDF_Filler_(
    Real *stiffness,
    Real *traction,
    const Real* SpatialGradient,
    const Real* CauchyStressTensor,
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
    if (ndim==3) { t = allocate<Real>(6);}
    else if (ndim==2) { t =  allocate<Real>(3);}

    Real *B = allocate<Real>(H_VoigtSize*local_size);
    Real *HBT = allocate<Real>(H_VoigtSize*local_size);
    Real *BDB_1 = allocate<Real>(local_size*local_size);

    std::fill(B,B+H_VoigtSize*local_size,0.);

    for (int igauss = 0; igauss < ngauss; ++igauss) {

        FillConstitutiveB_(B,&SpatialGradient[igauss*ndim*noderpelem],ndim,nvar,noderpelem,H_VoigtSize);
        // if (ndim == 3) {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                H_VoigtSize, local_size, H_VoigtSize, 1.0, &H_Voigt[igauss*H_VoigtSize*H_VoigtSize], H_VoigtSize, B, H_VoigtSize, 0.0, HBT, local_size);

            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                local_size, local_size, H_VoigtSize, 1.0, B, H_VoigtSize, HBT, local_size, 0.0, BDB_1, local_size);
        // }
        // else
        // {
        //     _SIMD_BDB_Integrator_DF_2D_(
        //         HBT,
        //         BDB_1,
        //         &SpatialGradient[igauss*ndim*noderpelem],
        //         &H_Voigt[igauss*H_VoigtSize*H_VoigtSize],
        //         noderpelem
        //         );
        // }

        // Multiply stiffness with detJ
        const Real detJ_igauss = detJ[igauss];
        for (int i=0; i<local_size*local_size; ++i) {
            stiffness[i] += BDB_1[i]*detJ_igauss;
        }

        if (requires_geometry_update==1) {
            // Compute tractions
            GetTotalTraction_(t, &CauchyStressTensor[igauss*ndim*ndim], ndim);

            // Multiply B with traction - for loop is okay
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

    deallocate(t);
    deallocate(B);
    deallocate(HBT);
    deallocate(BDB_1);
}

#endif