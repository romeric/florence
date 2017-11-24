#include "_det_inv_.h"
#include "_matmul_.h"
#include <math.h>

#ifdef USE_AVX_VERSION

template<int ndim, typename std::enable_if<ndim==2,bool>::type = 0>
FASTOR_INLINE void KinematicMeasures_(Real *SpatialGradient_, Real *F_, Real *detJ, const Real *Jm_,
    const Real *AllGauss_, const Real *LagrangeElemCoords_, const Real *EulerElemCoords_,
    int ngauss, int nodeperelem, int update)  {

    Real ParentGradientX[ndim*ndim];
    Real invParentGradientX[ndim*ndim];
    Real ParentGradientx[ndim*ndim];
    Real invParentGradientx[ndim*ndim];
    Real current_F[ndim*ndim];
    Real current_Ft[ndim*ndim];

    Real *MaterialGradient    = (Real*)malloc(sizeof(Real)*ndim*nodeperelem);
    Real *current_sp          = (Real*)malloc(sizeof(Real)*ndim*nodeperelem);
    Real *current_Jm          = (Real*)malloc(sizeof(Real)*nodeperelem*ndim);

    for (int g=0; g<ngauss; ++g) {

        for (int j=0; j<nodeperelem; ++j) {
            current_Jm[j] = Jm_[j*ngauss+g];
            current_Jm[nodeperelem+j] = Jm_[ngauss*nodeperelem+j*ngauss+g];
        }

        _matmul_(ndim,ndim,nodeperelem,current_Jm,LagrangeElemCoords_,ParentGradientX);
        _matmul_(ndim,ndim,nodeperelem,current_Jm,EulerElemCoords_,ParentGradientx);

        const Real detX = invdet2x2(ParentGradientX,invParentGradientX);
        const Real detx = invdet2x2(ParentGradientx,invParentGradientx);

        if (update==1) {
            detJ[g] = AllGauss_[g]*fabs(detx);
        }
        else {
            detJ[g] = AllGauss_[g]*fabs(detX);
        }

        _matmul_(ndim,nodeperelem,ndim,invParentGradientX,current_Jm,MaterialGradient);
        _matmul_(ndim,nodeperelem,ndim,invParentGradientx,current_Jm,current_sp);

        for (int i=0; i<nodeperelem; ++i) {
            SpatialGradient_[g*ndim*nodeperelem+i*ndim] = current_sp[i];
            SpatialGradient_[g*ndim*nodeperelem+i*ndim+1] = current_sp[nodeperelem+i];
        }

        // Compute deformation gradient F
        _matmul_(ndim,ndim,nodeperelem,MaterialGradient,EulerElemCoords_,current_F);
        Fastor::_transpose<Real,ndim,ndim>(current_F,current_Ft);
        std::copy(current_Ft,current_Ft+ndim*ndim,&F_[g*ndim*ndim]);
    }

    free(MaterialGradient);
    free(current_Jm);
    free(current_sp);
}



template<int ndim, typename std::enable_if<ndim==3,bool>::type = 0>
FASTOR_INLINE void KinematicMeasures_(Real *SpatialGradient_, Real *F_, Real *detJ, const Real *Jm_,
    const Real *AllGauss_, const Real *LagrangeElemCoords_, const Real *EulerElemCoords_,
    int ngauss, int nodeperelem, int update)  {

    Real ParentGradientX[ndim*ndim];
    Real invParentGradientX[ndim*ndim];
    Real ParentGradientx[ndim*ndim];
    Real invParentGradientx[ndim*ndim];
    Real current_F[ndim*ndim];
    Real current_Ft[ndim*ndim];

    Real *MaterialGradient    = (Real*)malloc(sizeof(Real)*ndim*nodeperelem);
    Real *current_sp          = (Real*)malloc(sizeof(Real)*ndim*nodeperelem);
    Real *current_Jm          = (Real*)malloc(sizeof(Real)*nodeperelem*ndim);

    for (int g=0; g<ngauss; ++g) {

        for (int j=0; j<nodeperelem; ++j) {
            current_Jm[j] = Jm_[j*ngauss+g];
            current_Jm[nodeperelem+j] = Jm_[ngauss*nodeperelem+j*ngauss+g];
            current_Jm[2*nodeperelem+j] = Jm_[2*ngauss*nodeperelem+j*ngauss+g];
        }

        _matmul_(ndim,ndim,nodeperelem,current_Jm,LagrangeElemCoords_,ParentGradientX);
        _matmul_(ndim,ndim,nodeperelem,current_Jm,EulerElemCoords_,ParentGradientx);

        const Real detX = invdet3x3(ParentGradientX,invParentGradientX);
        const Real detx = invdet3x3(ParentGradientx,invParentGradientx);

        if (update==1) {
            detJ[g] = AllGauss_[g]*fabs(detx);
        }
        else {
            detJ[g] = AllGauss_[g]*fabs(detX);
        }

        _matmul_(ndim,nodeperelem,ndim,invParentGradientX,current_Jm,MaterialGradient);
        _matmul_(ndim,nodeperelem,ndim,invParentGradientx,current_Jm,current_sp);

        for (int i=0; i<nodeperelem; ++i) {
            SpatialGradient_[g*ndim*nodeperelem+i*ndim] = current_sp[i];
            SpatialGradient_[g*ndim*nodeperelem+i*ndim+1] = current_sp[nodeperelem+i];
            SpatialGradient_[g*ndim*nodeperelem+i*ndim+2] = current_sp[2*nodeperelem+i];
        }

        // Compute deformation gradient F
        _matmul_(ndim,ndim,nodeperelem,MaterialGradient,EulerElemCoords_,current_F);
        Fastor::_transpose<Real,ndim,ndim>(current_F,current_Ft);
        std::copy(current_Ft,current_Ft+ndim*ndim,&F_[g*ndim*ndim]);
    }

    free(MaterialGradient);
    free(current_Jm);
    free(current_sp);
}



inline
void KinematicMeasures(Real *SpatialGradient_, Real *F_, Real *detJ, const Real *Jm_,
    const Real *AllGauss_, const Real *LagrangeElemCoords_, const Real *EulerElemCoords_,
    int ngauss, int ndim, int nodeperelem, int update)  {

    if (ndim == 3) {
        KinematicMeasures_<3>(SpatialGradient_, F_, detJ, Jm_,
            AllGauss_, LagrangeElemCoords_, EulerElemCoords_,
            ngauss, nodeperelem, update);
    }
    else {
        KinematicMeasures_<2>(SpatialGradient_, F_, detJ, Jm_,
            AllGauss_, LagrangeElemCoords_, EulerElemCoords_,
            ngauss, nodeperelem, update);
    }
}



#else


void KinematicMeasures(Real *SpatialGradient_, Real *F_, Real *detJ, const Real *Jm_,
    const Real *AllGauss_, const Real *LagrangeElemCoords_, const Real *EulerElemCoords_,
    int ngauss, int ndim, int nodeperelem, int update)  {

    Real *ParentGradientX     = (Real*)calloc(sizeof(Real),ngauss*ndim*ndim);
    Real *invParentGradientX  = (Real*)calloc(sizeof(Real),ngauss*ndim*ndim);
    Real *ParentGradientx     = (Real*)calloc(sizeof(Real),ngauss*ndim*ndim);
    Real *invParentGradientx  = (Real*)calloc(sizeof(Real),ngauss*ndim*ndim);
    Real *MaterialGradient    = (Real*)calloc(sizeof(Real),ngauss*ndim*nodeperelem);


    for (int i=0; i<ndim; ++i) {
        for (int j=0; j<nodeperelem; ++j) {
            for (int k=0; k<ngauss; ++k) {
                for (int l=0; l<ndim; ++l) {
                    ParentGradientX[k*ndim*ndim+i*ndim+l] += Jm_[i*ngauss*nodeperelem+j*ngauss+k]*LagrangeElemCoords_[j*ndim+l];
                    ParentGradientx[k*ndim*ndim+i*ndim+l] += Jm_[i*ngauss*nodeperelem+j*ngauss+k]*EulerElemCoords_[j*ndim+l];
                }
            }
        }
    }

    // Find inverse of the isoparametric mapping (Jacobian)
    if (ndim==3) {
        for (int g=0; g<ngauss; ++g) {
            inv3x3(ParentGradientX+g*9, invParentGradientX+g*9);
            inv3x3(ParentGradientx+g*9, invParentGradientx+g*9);
        }
    }
    else if (ndim==2) {
        for (int g=0; g<ngauss; ++g) {
            inv2x2(ParentGradientX+g*4, invParentGradientX+g*4);
            inv2x2(ParentGradientx+g*4, invParentGradientx+g*4);
        }
    }

    // Find material and spatial gradients
    for (int i=0; i<ngauss; ++i) {
        for (int j=0; j<ndim; ++j) {
            for (int k=0; k<ndim; ++k) {
                for (int l=0; l<nodeperelem; ++l) {
                    MaterialGradient[i*ndim*nodeperelem+j*nodeperelem+l] += invParentGradientX[i*ndim*ndim+j*ndim+k]*\
                        Jm_[k*ngauss*nodeperelem+l*ngauss+i];
                    SpatialGradient_[i*ndim*nodeperelem+l*ndim+j] += invParentGradientx[i*ndim*ndim+j*ndim+k]*\
                        Jm_[k*ngauss*nodeperelem+l*ngauss+i];
                }
            }
        }
    }


    // Compute deformation gradient F
    for (int i=0; i<nodeperelem; ++i) {
        for (int j=0; j<ndim; ++j) {
            for (int k=0; k<ngauss; ++k) {
                for (int l=0; l<ndim; ++l) {
                    F_[k*ndim*ndim+j*ndim+l] += EulerElemCoords_[i*ndim+j]*\
                        MaterialGradient[k*ndim*nodeperelem+l*nodeperelem+i];
                }
            }
        }
    }


    if (update==1) {
        if (ndim==3) {
            for (int i=0; i<ngauss; ++i) {
                // detJ[i] = AllGauss_[i]*fabs(det3x3(F_+i*9))*fabs(det3x3(ParentGradientX+i*9));
                detJ[i] = AllGauss_[i]*fabs(det3x3(ParentGradientx+i*9));
            }
        }
        else if (ndim==2) {
            for (int i=0; i<ngauss; ++i) {
                // detJ[i] = AllGauss_[i]*fabs(det2x2(F_+i*4))*fabs(det2x2(ParentGradientX+i*4));
                detJ[i] = AllGauss_[i]*fabs(det2x2(ParentGradientx+i*4));
            }
        }
    }
    else if (update==0) {
        if (ndim==3) {
            for (int i=0; i<ngauss; ++i) {
                detJ[i] = AllGauss_[i]*fabs(det3x3(ParentGradientX+i*9));
            }
        }
        else if (ndim==2) {
            for (int i=0; i<ngauss; ++i) {
                detJ[i] = AllGauss_[i]*fabs(det2x2(ParentGradientX+i*4));
            }
        }
    }

    free(ParentGradientX);
    free(invParentGradientX);
    free(ParentGradientx);
    free(invParentGradientx);
    free(MaterialGradient);

}


#endif
