#ifndef _MASSINTEGRAND__H
#define _MASSINTEGRAND__H

#include <vector>
#include <algorithm>
#include <numeric>
#include <functional>
#include <cstdint>

#ifdef HAS_MKL
#include <mkl.h>
#else
#include <cblas.h>
#endif

#ifdef __SSE4_2__
#include <emmintrin.h>
#include <mm_malloc.h>
#endif

#include "_det_inv_.h"
#include "_matmul_.h"

using Real = double;
using Integer = std::int64_t;
using UInteger = std::uint64_t;


#ifndef CUSTOM_ALLOCATION_
#define CUSTOM_ALLOCATION_
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
#endif



inline void _MassIntegrand_Filler_(Real *mass,
    const Real* bases,
    const Real* detJ,
    int ngauss,
    int noderpelem,
    int ndim,
    int nvar,
    Real rho) {


    int local_size = nvar*noderpelem;

    Real *N     = allocate<Real>(nvar*local_size);
    Real *rhoNN = allocate<Real>(local_size*local_size);

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

    deallocate(N);
    deallocate(rhoNN);
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





inline void _ExplicitConstantMassIntegrand_(
    const UInteger* elements,
    const Real* points,
    const Real* Jm,
    const Real* AllGauss,
    const Real* constant_mass_integrand,
    Real *mass,
    Integer nelem,
    Integer ndim,
    Integer nvar,
    Integer ngauss,
    Integer nodeperelem,
    Integer local_capacity) {


    Integer ndof                    = nodeperelem*nvar;
    Real *LagrangeElemCoords        = allocate<Real>(nodeperelem*ndim);
    Real *MaterialGradient          = allocate<Real>(ndim*nodeperelem);
    Real *ParentGradientX           = allocate<Real>(ndim*ndim);
    Real detJ                       = 0.;
    Real *massel                    = allocate<Real>(local_capacity);
    Real *massel_lumped             = allocate<Real>(ndof);


    // PRE-COMPUTE ISOPARAMETRIC GRADIENTS
    std::vector<std::vector<Real>> current_Jms(ngauss);
    for (int g=0; g<ngauss; ++g) {
        std::vector<Real> current_Jm(ndim*nodeperelem);
        for (int j=0; j<nodeperelem; ++j) {
            for (int k=0; k<ndim; ++k) {
                current_Jm[k*nodeperelem+j] = Jm[k*ngauss*nodeperelem+j*ngauss+g];
            }
        }
        current_Jms[g] = current_Jm;
    }

    // LOOP OVER ELEMETNS
    for (Integer elem=0; elem < nelem; ++elem) {

        // GET THE FIELDS AT THE ELEMENT LEVEL
        for (Integer i=0; i<nodeperelem; ++i) {
            const Integer inode = elements[elem*nodeperelem+i];
            for (Integer j=0; j<ndim; ++j) {
                LagrangeElemCoords[i*ndim+j] = points[inode*ndim+j];
            }
        }

        std::fill_n(massel,local_capacity,0.);

        for (Integer igauss = 0; igauss < ngauss; ++igauss) {

            {
                // USING A STL BASED FILLER REMOVES THE ANNOYING BUG
                std::fill_n(ParentGradientX,ndim*ndim,0.);
                _matmul_(ndim,ndim,nodeperelem,current_Jms[igauss].data(),LagrangeElemCoords,ParentGradientX);
                const Real detX = det3x3(ParentGradientX);
                detJ = AllGauss[igauss]*std::abs(detX);
            }

            // Multiply constant part of mass with detJ
#ifdef __AVX__
            using V2 = Fastor::SIMDVector<Real>;
            V2 _va, _vb, _vout;
            _vb.set(detJ);
            int Vsize = V2::Size;
            int ROUND_ = ROUND_DOWN(local_capacity,Vsize);
            int i=0;
            for (; i<ROUND_; i+=Vsize) {
                _va.load(&constant_mass_integrand[igauss*local_capacity+i]);
                _vout.load(&massel[i]);
#ifdef __FMA__
                _vout = fmadd(_va,_vb,_vout);
#else
                _vout += _va*_vb;
#endif
                _vout.store(&massel[i],false);
            }
            for ( ; i<local_capacity; ++i) {
                massel[i] += constant_mass_integrand[igauss*local_capacity+i]*detJ;
            }
#else
            for (int i=0; i<local_capacity; ++i) {
                massel[i] += constant_mass_integrand[igauss*local_capacity+i]*detJ;
            }
#endif
        }

        // LUMP MASS
        for (Integer i=0; i<ndof; ++i) {
            massel_lumped[i] = std::accumulate(&massel[i*ndof], &massel[(i+1)*ndof],0.);
        }

        // ASSEMBLE LUMPED MASS
        {
            for (Integer i = 0; i<nodeperelem; ++i) {
                UInteger T_idx = elements[elem*nodeperelem+i]*nvar;
                for (Integer iterator = 0; iterator < nvar; ++iterator) {
                    mass[T_idx+iterator] += massel_lumped[i*nvar+iterator];
                }
            }
        }
    }

    deallocate(ParentGradientX);
    deallocate(MaterialGradient);
    deallocate(LagrangeElemCoords);
    deallocate(massel);
    deallocate(massel_lumped);
}

#endif