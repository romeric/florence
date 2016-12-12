#include "_det_inv_.h"
#include <math.h>


void KinematicMeasures(Real *SpatialGradient_, Real *F_, Real *detJ, const Real *Jm_,
    const Real *AllGauss_, const Real *LagrangeElemCoords_, const Real *EulerElemCoords_,
    int ngauss, int ndim, int nodeperelem, int update)  {

    Real *ParentGradientX     = (Real*)calloc(sizeof(Real),ngauss*ndim*ndim);
    Real *invParentGradientX  = (Real*)calloc(sizeof(Real),ngauss*ndim*ndim);
    Real *ParentGradientx     = (Real*)calloc(sizeof(Real),ngauss*ndim*ndim);
    Real *invParentGradientx  = (Real*)calloc(sizeof(Real),ngauss*ndim*ndim);
    Real *MaterialGradient    = (Real*)calloc(sizeof(Real),ngauss*ndim*nodeperelem);


    for (size_t i=0; i<ndim; ++i) {
        for (size_t j=0; j<nodeperelem; ++j) {
            for (size_t k=0; k<ngauss; ++k) {
                for (size_t l=0; l<ndim; ++l) {
                    ParentGradientX[k*ndim*ndim+i*ndim+l] += Jm_[i*ngauss*nodeperelem+j*ngauss+k]*LagrangeElemCoords_[j*ndim+l];
                    ParentGradientx[k*ndim*ndim+i*ndim+l] += Jm_[i*ngauss*nodeperelem+j*ngauss+k]*EulerElemCoords_[j*ndim+l];
                }
            }
        }
    }

    // Find inverse of the isoparametric mapping (Jacobian) 
    if (ndim==3) {
        for (size_t g=0; g<ngauss; ++g) {
            inv3x3(ParentGradientX+g*9, invParentGradientX+g*9);
            inv3x3(ParentGradientx+g*9, invParentGradientx+g*9);
        }
    } 
    else if (ndim==2) {
        for (size_t g=0; g<ngauss; ++g) {
            inv2x2(ParentGradientX+g*4, invParentGradientX+g*4);
            inv2x2(ParentGradientx+g*4, invParentGradientx+g*4);
        }
    }

    // Find material and spatial gradients 
    for (size_t i=0; i<ngauss; ++i) {
        for (size_t j=0; j<ndim; ++j) {
            for (size_t k=0; k<ndim; ++k) {
                for (size_t l=0; l<nodeperelem; ++l) {
                    MaterialGradient[i*ndim*nodeperelem+j*nodeperelem+l] += invParentGradientX[i*ndim*ndim+j*ndim+k]*\
                        Jm_[k*ngauss*nodeperelem+l*ngauss+i];
                    SpatialGradient_[i*ndim*nodeperelem+l*ndim+j] += invParentGradientx[i*ndim*ndim+j*ndim+k]*\
                        Jm_[k*ngauss*nodeperelem+l*ngauss+i];
                }
            }
        }
    }


    // Compute deformation gradient F  
    for (size_t i=0; i<nodeperelem; ++i) {
        for (size_t j=0; j<ndim; ++j) {
            for (size_t k=0; k<ngauss; ++k) {
                for (size_t l=0; l<ndim; ++l) {
                    F_[k*ndim*ndim+j*ndim+l] += EulerElemCoords_[i*ndim+j]*\
                        MaterialGradient[k*ndim*nodeperelem+l*nodeperelem+i];
                }
            }
        }
    }


    if (update==1) {
        if (ndim==3) {
            for (size_t i=0; i<ngauss; ++i) {
                // detJ[i] = AllGauss_[i]*fabs(det3x3(F_+i*9))*fabs(det3x3(ParentGradientX+i*9));
                detJ[i] = AllGauss_[i]*fabs(det3x3(ParentGradientx+i*9));
            }
        }
        else if (ndim==2) {
            for (size_t i=0; i<ngauss; ++i) {
                // detJ[i] = AllGauss_[i]*fabs(det2x2(F_+i*4))*fabs(det2x2(ParentGradientX+i*4));
                detJ[i] = AllGauss_[i]*fabs(det2x2(ParentGradientx+i*4));
            }
        }
    }
    else if (update==0) {
        if (ndim==3) {
            for (size_t i=0; i<ngauss; ++i) {
                detJ[i] = AllGauss_[i]*fabs(det3x3(ParentGradientX+i*9));
            }
        }
        else if (ndim==2) {
            for (size_t i=0; i<ngauss; ++i) {
                detJ[i] = AllGauss_[i]*fabs(det2x2(ParentGradientX+i*4));
            }
        }
    }



    // for (size_t i=0; i<ndim*ndim*ngauss; ++i) {
    //     printf("%9.9f, ",ParentGradientx[i]);
    //     // printf("%9.9f, ", invParentGradientX[i]);
    //     // printf("%9.9f, ", F_[i]);
    // }

    // printf("\n\n");
    // for (size_t i=0; i<ndim*nodeperelem*ngauss; ++i) {
    //     printf("%9.9f, ", MaterialGradient[i]);
    // }
    // printf("\n\n");


    free(ParentGradientX);
    free(invParentGradientX);
    free(ParentGradientx);
    free(invParentGradientx);
    free(MaterialGradient);

}
