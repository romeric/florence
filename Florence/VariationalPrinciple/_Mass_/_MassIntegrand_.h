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

#include "SparseAssemblyNative.h"
#include "_det_inv_.h"
#include "_matmul_.h"

#ifndef LL_TYPES
#define LL_TYPES
using Real = double;
using Integer = std::int64_t;
using UInteger = std::uint64_t;
#endif




/*---------------------------------------------------------------------------------------------*/
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
/*---------------------------------------------------------------------------------------------*/




/*---------------------------------------------------------------------------------------------*/
#ifndef SPARSE_TRIPLET_FILLER
#define SPARSE_TRIPLET_FILLER
// IJV Filler
FASTOR_INLINE
void fill_triplet(  const Integer *i,
                    const Integer *j,
                    const Real *coeff,
                    int *I,
                    int *J,
                    Real *V,
                    Integer elem,
                    Integer nvar,
                    Integer nodeperelem,
                    const UInteger *elements,
                    Integer i_shape,
                    Integer j_shape
                    ) {

    Integer ndof = nvar*nodeperelem;
    Integer *current_row_column = allocate<Integer>(nvar*nodeperelem);

    Integer const_elem_retriever;
    for (Integer counter=0; counter<nodeperelem; ++counter) {
        const_elem_retriever = nvar*elements[elem*nodeperelem+counter];
        for (Integer ncounter=0; ncounter<nvar; ++ncounter) {
            current_row_column[nvar*counter+ncounter] = const_elem_retriever+ncounter;
        }
    }

    Integer icounter = 0;
    Integer ncounter = ndof*ndof*elem;

    Integer const_I_retriever;
    for (Integer counter=0; counter<ndof; ++counter) {
        const_I_retriever = current_row_column[counter];
        for (Integer iterator=0; iterator<ndof; ++iterator) {
            I[ncounter] = const_I_retriever;
            J[ncounter] = current_row_column[iterator];
            V[ncounter] = coeff[icounter];
            ncounter++;
            icounter++;
        }
    }

    deallocate(current_row_column);
}



FASTOR_INLINE
void fill_global_data(
                    const Integer *i,
                    const Integer *j,
                    const Real *coeff,
                    int *I,
                    int *J,
                    Real *V,
                    Integer elem,
                    Integer nvar,
                    Integer nodeperelem,
                    const UInteger *elements,
                    Integer i_shape,
                    Integer j_shape,
                    int recompute_sparsity_pattern,
                    int squeeze_sparsity_pattern,
                    const int *data_local_indices,
                    const int *data_global_indices,
                    const unsigned long *sorted_elements,
                    const long *sorter
    )
{

    if (recompute_sparsity_pattern) {
        fill_triplet(i,j,coeff,I,J,V,elem,nvar,nodeperelem,elements,i_shape,j_shape);
    }
    else {
        if (squeeze_sparsity_pattern) {
            SparseAssemblyNativeCSR_RecomputeDataIndex_(
                coeff,
                J,
                I,
                V,
                elem,
                nvar,
                nodeperelem,
                sorted_elements,
                sorter);
        }
        else {
            const int ndof = nvar*nodeperelem;
            const int local_capacity = ndof*ndof;
            SparseAssemblyNativeCSR_(
                coeff,
                data_local_indices,
                data_global_indices,
                elem,
                local_capacity,
                V
                );
        }
    }
}

#endif
/*---------------------------------------------------------------------------------------------*/










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










inline void _GenericConstantMassIntegrand_(
    const UInteger* elements,
    const Real* points,
    const Real* Jm,
    const Real* AllGauss,
    const Real* constant_mass_integrand,
    Integer nelem,
    Integer ndim,
    Integer nvar,
    Integer ngauss,
    Integer nodeperelem,
    Integer local_capacity,
    Integer mass_type,
    const Integer* local_rows_mass,
    const Integer* local_cols_mass,
    int *I_mass,
    int *J_mass,
    Real *V_mass,
    Real *mass,
    int recompute_sparsity_pattern,
    int squeeze_sparsity_pattern,
    const int *data_local_indices,
    const int *data_global_indices,
    const unsigned long *sorted_elements,
    const long *sorter
    ) {


    Integer ndof                    = nodeperelem*nvar;
    Real *LagrangeElemCoords        = allocate<Real>(nodeperelem*ndim);
    Real *MaterialGradient          = allocate<Real>(ndim*nodeperelem);
    // +1 TO AVOID OVERSTEPPONG IN MATMUL UNALIGNED STORE - EXTREMELY DANGEROUSE BUG
    int plus1                       = ndim == 2 ? 0 : 1;
    Real *ParentGradientX           = allocate<Real>(ndim*ndim+plus1);
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
                std::fill_n(ParentGradientX,ndim*ndim+plus1,0.);
                _matmul_(ndim,ndim,nodeperelem,current_Jms[igauss].data(),LagrangeElemCoords,ParentGradientX);
                const Real detX = _det_(ndim, ParentGradientX);
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
                _va.load(&constant_mass_integrand[igauss*local_capacity+i],false);
                _vout.load(&massel[i],false);
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

        // FOR LUMP MASS MATRIX
        if (mass_type == 0) {
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
        else {
            fill_global_data(   local_rows_mass,
                                local_cols_mass,
                                massel,
                                I_mass,
                                J_mass,
                                V_mass,
                                elem,
                                nvar,
                                nodeperelem,
                                elements,
                                local_capacity,
                                local_capacity,
                                recompute_sparsity_pattern,
                                squeeze_sparsity_pattern,
                                data_local_indices,
                                data_global_indices,
                                sorted_elements,
                                sorter);
        }
    }

    deallocate(ParentGradientX);
    deallocate(MaterialGradient);
    deallocate(LagrangeElemCoords);
    deallocate(massel);
    deallocate(massel_lumped);
}







inline void _SymmetricConstantMassIntegrand_(
    const UInteger* elements,
    const Real* points,
    const Real* Jm,
    const Real* AllGauss,
    const Real* constant_mass_integrand,
    Integer nelem,
    Integer ndim,
    Integer nvar,
    Integer ngauss,
    Integer nodeperelem,
    Integer local_capacity,
    Integer mass_type,
    const Integer* local_rows_mass,
    const Integer* local_cols_mass,
    int *I_mass,
    int *J_mass,
    Real *V_mass,
    Real *mass,
    int recompute_sparsity_pattern,
    int squeeze_sparsity_pattern,
    const int *data_local_indices,
    const int *data_global_indices,
    const unsigned long *sorted_elements,
    const long *sorter
    ) {


    const Integer ndof                      = nodeperelem*nvar;
    const Integer symmetric_local_capacity  = ndof*(ndof+1)/2;
    Real *LagrangeElemCoords                = allocate<Real>(nodeperelem*ndim);
    Real *MaterialGradient                  = allocate<Real>(ndim*nodeperelem);
    // +1 TO AVOID OVERSTEPPONG IN MATMUL UNALIGNED STORE - EXTREMELY DANGEROUSE BUG
    int plus1                               = ndim == 2 ? 0 : 1;
    Real *ParentGradientX                   = allocate<Real>(ndim*ndim+plus1);
    Real detJ                               = 0.;
    Real *massel                            = allocate<Real>(local_capacity);
    Real *symmetric_massel                  = allocate<Real>(symmetric_local_capacity);
    Real *massel_lumped                     = allocate<Real>(ndof);

    // PRE-COMPUTE SYMMETRIC PART OF CONSTANT MASS WITHOUT ZEROS
    Real *symmetric_constant_mass_integrand = allocate<Real>(ngauss*symmetric_local_capacity);
    Integer counter = 0;
    for (Integer igauss = 0; igauss < ngauss; ++igauss) {
        for (Integer i=0; i<ndof; ++i) {
            for (Integer j=i; j<ndof; ++j) {
                const Real value = constant_mass_integrand[igauss*ndof*ndof+i*ndof+j];
                symmetric_constant_mass_integrand[counter] = value;
                counter++;
            }
        }
    }

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
        std::fill_n(symmetric_massel,symmetric_local_capacity,0.);

        for (Integer igauss = 0; igauss < ngauss; ++igauss) {

            {
                // USING A STL BASED FILLER REMOVES THE ANNOYING BUG
                std::fill_n(ParentGradientX,ndim*ndim+plus1,0.);
                _matmul_(ndim,ndim,nodeperelem,current_Jms[igauss].data(),LagrangeElemCoords,ParentGradientX);
                const Real detX = _det_(ndim, ParentGradientX);
                detJ = AllGauss[igauss]*std::abs(detX);
            }

            // Multiply constant part of mass with detJ
#ifdef __AVX__
            using V2 = Fastor::SIMDVector<Real>;
            V2 _va, _vb, _vout;
            _vb.set(detJ);
            int Vsize = V2::Size;
            int ROUND_ = ROUND_DOWN(symmetric_local_capacity,Vsize);
            int i=0;
            for (; i<ROUND_; i+=Vsize) {
                _va.load(&symmetric_constant_mass_integrand[igauss*symmetric_local_capacity+i],false);
                _vout.load(&symmetric_massel[i],false);
#ifdef __FMA__
                _vout = fmadd(_va,_vb,_vout);
#else
                _vout += _va*_vb;
#endif
                _vout.store(&symmetric_massel[i],false);
            }
            for ( ; i<symmetric_local_capacity; ++i) {
                symmetric_massel[i] += symmetric_constant_mass_integrand[igauss*symmetric_local_capacity+i]*detJ;
            }
#else
            for (int i=0; i<symmetric_local_capacity; ++i) {
                symmetric_massel[i] += symmetric_constant_mass_integrand[igauss*symmetric_local_capacity+i]*detJ;
            }
#endif
        }

        // BUILD TOTAL LOCAL MASS MATRIX
        Integer counter = 0;
        for (Integer i=0; i<ndof; ++i) {
            for (Integer j=i; j<ndof; ++j) {
                const Real value = symmetric_massel[counter];
                massel[i*ndof+j] = value;
                massel[j*ndof+i] = value;
                counter++;
            }
        }

        // FOR LUMP MASS MATRIX
        if (mass_type == 0) {
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
        else {
            fill_global_data(   local_rows_mass,
                                local_cols_mass,
                                massel,
                                I_mass,
                                J_mass,
                                V_mass,
                                elem,
                                nvar,
                                nodeperelem,
                                elements,
                                local_capacity,
                                local_capacity,
                                recompute_sparsity_pattern,
                                squeeze_sparsity_pattern,
                                data_local_indices,
                                data_global_indices,
                                sorted_elements,
                                sorter);
        }
    }

    deallocate(ParentGradientX);
    deallocate(MaterialGradient);
    deallocate(LagrangeElemCoords);
    deallocate(massel);
    deallocate(symmetric_constant_mass_integrand);
    deallocate(symmetric_massel);
    deallocate(massel_lumped);
}






inline void _SymmetricNonZeroConstantMassIntegrand_(
    const UInteger* elements,
    const Real* points,
    const Real* Jm,
    const Real* AllGauss,
    const Real* constant_mass_integrand,
    Integer nelem,
    Integer ndim,
    Integer nvar,
    Integer ngauss,
    Integer nodeperelem,
    Integer local_capacity,
    Integer mass_type,
    const Integer* local_rows_mass,
    const Integer* local_cols_mass,
    int *I_mass,
    int *J_mass,
    Real *V_mass,
    Real *mass,
    int recompute_sparsity_pattern,
    int squeeze_sparsity_pattern,
    const int *data_local_indices,
    const int *data_global_indices,
    const unsigned long *sorted_elements,
    const long *sorter
    ) {


    const Integer ndof                      = nodeperelem*nvar;
    Real *LagrangeElemCoords                = allocate<Real>(nodeperelem*ndim);
    Real *MaterialGradient                  = allocate<Real>(ndim*nodeperelem);
    // +1 TO AVOID OVERSTEPPONG IN MATMUL UNALIGNED STORE - EXTREMELY DANGEROUSE BUG
    int plus1                               = ndim == 2 ? 0 : 1;
    Real *ParentGradientX                   = allocate<Real>(ndim*ndim+plus1);
    Real detJ                               = 0.;
    Real *massel                            = allocate<Real>(local_capacity);
    Real *massel_lumped                     = allocate<Real>(ndof);

    // PRE-COMPUTE SPARSITY PATTERN OF THE LOCAL MATRIX -
    // LOCAL MASS MATRICES TYPICALLY HAVE LOTS OF NONZEROS
    constexpr Real tolerance = 1e-12;
    std::vector<Integer> nonzero_i_indices, nonzero_j_indices;
    Integer counter = 0;
    for (Integer i=0; i<ndof; ++i) {
        for (Integer j=i; j<ndof; ++j) {
            if (std::abs(constant_mass_integrand[i*ndof+j]) > tolerance) {
                nonzero_i_indices.push_back(i);
                nonzero_j_indices.push_back(j);
            }
        }
    }
    // PRE-COMPUTE SYMMETRIC PART OF CONSTANT MASS WITHOUT ZEROS
    const Integer symmetric_local_capacity = nonzero_i_indices.size();
    Real *symmetric_massel                  = allocate<Real>(symmetric_local_capacity);
    Real *symmetric_constant_mass_integrand = allocate<Real>(ngauss*symmetric_local_capacity);
    counter = 0;
    for (Integer igauss = 0; igauss < ngauss; ++igauss) {
        for (Integer i=0; i<symmetric_local_capacity; ++i) {
            const Real value = constant_mass_integrand[igauss*ndof*ndof+nonzero_i_indices[i]*ndof+nonzero_j_indices[i]];
            symmetric_constant_mass_integrand[counter] = value;
            counter++;
        }
    }

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
        std::fill_n(symmetric_massel,symmetric_local_capacity,0.);

        for (Integer igauss = 0; igauss < ngauss; ++igauss) {

            {
                // USING A STL BASED FILLER REMOVES THE ANNOYING BUG
                std::fill_n(ParentGradientX,ndim*ndim+plus1,0.);
                _matmul_(ndim,ndim,nodeperelem,current_Jms[igauss].data(),LagrangeElemCoords,ParentGradientX);
                const Real detX = _det_(ndim, ParentGradientX);
                detJ = AllGauss[igauss]*std::abs(detX);
            }

            // Multiply constant part of mass with detJ
#ifdef __AVX__
            using V2 = Fastor::SIMDVector<Real>;
            V2 _va, _vb, _vout;
            _vb.set(detJ);
            int Vsize = V2::Size;
            int ROUND_ = ROUND_DOWN(symmetric_local_capacity,Vsize);
            int i=0;
            for (; i<ROUND_; i+=Vsize) {
                _va.load(&symmetric_constant_mass_integrand[igauss*symmetric_local_capacity+i],false);
                _vout.load(&symmetric_massel[i],false);
#ifdef __FMA__
                _vout = fmadd(_va,_vb,_vout);
#else
                _vout += _va*_vb;
#endif
                _vout.store(&symmetric_massel[i],false);
            }
            for ( ; i<symmetric_local_capacity; ++i) {
                symmetric_massel[i] += symmetric_constant_mass_integrand[igauss*symmetric_local_capacity+i]*detJ;
            }
#else
            for (int i=0; i<symmetric_local_capacity; ++i) {
                symmetric_massel[i] += symmetric_constant_mass_integrand[igauss*symmetric_local_capacity+i]*detJ;
            }
#endif
        }

        // BUILD TOTAL LOCAL MASS MATRIX
        for (Integer i=0; i<symmetric_local_capacity; ++i) {
            const Real value = symmetric_massel[i];
            massel[nonzero_i_indices[i]*ndof+nonzero_j_indices[i]] = value;

        }

        for (Integer i=0; i<ndof; ++i) {
            for (Integer j=i; j<ndof; ++j) {
                massel[j*ndof+i] = massel[i*ndof+j];

            }
        }

        // FOR LUMP MASS MATRIX
        if (mass_type == 0) {
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
        else {
            fill_global_data(   local_rows_mass,
                                local_cols_mass,
                                massel,
                                I_mass,
                                J_mass,
                                V_mass,
                                elem,
                                nvar,
                                nodeperelem,
                                elements,
                                local_capacity,
                                local_capacity,
                                recompute_sparsity_pattern,
                                squeeze_sparsity_pattern,
                                data_local_indices,
                                data_global_indices,
                                sorted_elements,
                                sorter);
        }
    }

    deallocate(ParentGradientX);
    deallocate(MaterialGradient);
    deallocate(LagrangeElemCoords);
    deallocate(massel);
    deallocate(symmetric_constant_mass_integrand);
    deallocate(symmetric_massel);
    deallocate(massel_lumped);
}



#endif