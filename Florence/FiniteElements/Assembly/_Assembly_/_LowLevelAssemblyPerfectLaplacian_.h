#ifndef _LOWLEVELASSEMBLYDPF__H
#define _LOWLEVELASSEMBLYDPF__H

#include "assembly_helper.h"
#include "_MassIntegrand_.h"




template<Integer ndim>
void _GlobalAssemblyPerfectLaplacian__(const Real *points,
                        const UInteger* elements,
                        const Real* Eulerp,
                        const Real* bases,
                        const Real* Jm,
                        const Real* AllGauss,
                        Integer ngauss,
                        Integer nelem,
                        Integer nodeperelem,
                        Integer nnode,
                        int *I_stiff,
                        int *J_stiff,
                        Real *V_stiff,
                        Real *e_tensor,
                        Integer is_hessian_symmetric,
                        int recompute_sparsity_pattern,
                        int squeeze_sparsity_pattern,
                        const int *data_local_indices,
                        const int *data_global_indices,
                        const UInteger *sorted_elements,
                        const Integer *sorter
                        );



#ifdef __AVX__


// Kinematics
template<int ndim, typename std::enable_if<ndim==2,bool>::type = 0>
FASTOR_INLINE void KinematicMeasures__(Real *MaterialGradient, Real &detJ,
    const Real *current_Jm, Real AllGauss_, const Real *LagrangeElemCoords_,
    int nodeperelem)  {

    FASTOR_ALIGN Real ParentGradientX[ndim*ndim];
    FASTOR_ALIGN Real invParentGradientX[ndim*ndim];

    std::fill_n(ParentGradientX,ndim*ndim,0.);
    std::fill_n(invParentGradientX,ndim*ndim,0.);
    std::fill_n(MaterialGradient,nodeperelem*ndim,0.);

    _matmul_2k2(nodeperelem,current_Jm,LagrangeElemCoords_,ParentGradientX);
    const Real detX = invdet2x2(ParentGradientX,invParentGradientX);
    detJ = AllGauss_*fabs(detX);
    _matmul_22k(nodeperelem,invParentGradientX,current_Jm,MaterialGradient);
}

template<int ndim, typename std::enable_if<ndim==3,bool>::type = 0>
FASTOR_INLINE void KinematicMeasures__(Real *MaterialGradient, Real &detJ,
    const Real *current_Jm, Real AllGauss_, const Real *LagrangeElemCoords_,
    int nodeperelem)  {

    FASTOR_ALIGN Real ParentGradientX[ndim*ndim];
    FASTOR_ALIGN Real invParentGradientX[ndim*ndim];

    std::fill_n(ParentGradientX,ndim*ndim,0.);
    std::fill_n(invParentGradientX,ndim*ndim,0.);
    std::fill_n(MaterialGradient,nodeperelem*ndim,0.);

    _matmul_3k3(nodeperelem,current_Jm,LagrangeElemCoords_,ParentGradientX);
    const Real detX = invdet3x3(ParentGradientX,invParentGradientX);
    detJ = AllGauss_*fabs(detX);
    _matmul_33k(nodeperelem,invParentGradientX,current_Jm,MaterialGradient);
}
//




template<>
void _GlobalAssemblyPerfectLaplacian__<2>(const Real *points,
                        const UInteger* elements,
                        const Real* Eulerp,
                        const Real* bases,
                        const Real* Jm,
                        const Real* AllGauss,
                        Integer ngauss,
                        Integer nelem,
                        Integer nodeperelem,
                        Integer nnode,
                        int *I_stiff,
                        int *J_stiff,
                        Real *V_stiff,
                        Real *e_tensor,
                        Integer is_hessian_symmetric,
                        int recompute_sparsity_pattern,
                        int squeeze_sparsity_pattern,
                        const int *data_local_indices,
                        const int *data_global_indices,
                        const UInteger *sorted_elements,
                        const Integer *sorter
                        ) {

    constexpr Integer ndim = 2;
    Integer ndof = nodeperelem;
    Integer local_capacity = ndof*ndof;

    Real *LagrangeElemCoords        = allocate<Real>(nodeperelem*ndim);
    Real *ElectricPotentialElem     = allocate<Real>(nodeperelem);

    Real *current_Jm                = allocate<Real>(nodeperelem*ndim);
    Real *MaterialGradient          = allocate<Real>(ndim*nodeperelem);
    Real *eM                        = allocate<Real>(ndim*nodeperelem);
    Real detJ                       = 0;

    Real *BDB                       = allocate<Real>(local_capacity);
    Real *stiffness                 = allocate<Real>(local_capacity);

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

            LagrangeElemCoords[i*2+0] = points[inode*2+0];
            LagrangeElemCoords[i*2+1] = points[inode*2+1];

            ElectricPotentialElem[i] = Eulerp[inode];
        }

        std::fill(stiffness,stiffness+local_capacity,0.);
        std::fill(BDB,BDB+local_capacity,0.);

        for (int g=0; g<ngauss; ++g) {

            // COMPUTE KINEMATIC MEASURES
            KinematicMeasures__<2>(  MaterialGradient,
                                    detJ,
                                    current_Jms[g].data(),
                                    AllGauss[g],
                                    LagrangeElemCoords,
                                    nodeperelem
                                    );

            // Extremely consise way
            // _transpose_(ndim,nodeperelem,MaterialGradient,MaterialGradientT);
            // _matmul_(ndof,ndim,ndim,MaterialGradientT,e_tensor,eM);
            // _matmul_(ndof,ndof,ndim,eM,MaterialGradient,BDB);
            // for (int i=0; i<local_capacity; ++i) {
            //     stiffness[i] += BDB[i]*detJ;
            // }

            _matmul_22k(nodeperelem,e_tensor,MaterialGradient,eM);

            if (is_hessian_symmetric) {
                for (int i=0; i<nodeperelem; ++i) {
                    const Real a0 = MaterialGradient[i];
                    const Real a1 = MaterialGradient[i+nodeperelem];

                    for (int j=i; j<nodeperelem; ++j) {
                        const Real b0 = eM[j];
                        const Real b1 = eM[j+nodeperelem];

                        BDB[i*nodeperelem+j] += (a0*b0 + a1*b1)*detJ;
                    }
                }
            }
            else {
                for (int i=0; i<nodeperelem; ++i) {
                    const Real a0 = MaterialGradient[i];
                    const Real a1 = MaterialGradient[i+nodeperelem];

                    for (int j=0; j<nodeperelem; ++j) {
                        const Real b0 = eM[j];
                        const Real b1 = eM[j+nodeperelem];

                        BDB[i*nodeperelem+j] = a0*b0 + a1*b1;
                    }
                }

                for (int i=0; i<local_capacity; ++i) {
                    stiffness[i] += BDB[i]*detJ;
                }
            }
        }

        if (is_hessian_symmetric) {
            // Fill the lower half now
            std::copy(BDB,BDB+local_capacity,stiffness);
            for (int i=0; i<nodeperelem; ++i) {
                for (int j=i; j<nodeperelem; ++j) {
                    stiffness[j*nodeperelem+i] = BDB[i*nodeperelem+j];
                }
            }
        }

        // ASSEMBLE CONSTITUTIVE STIFFNESS
        fill_global_data(
                nullptr,
                nullptr,
                stiffness,
                I_stiff,
                J_stiff,
                V_stiff,
                elem,
                1,
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

    deallocate(LagrangeElemCoords);
    deallocate(ElectricPotentialElem);
    deallocate(current_Jm);
    deallocate(MaterialGradient);
    deallocate(eM);

    deallocate(BDB);
    deallocate(stiffness);
}



template<>
void _GlobalAssemblyPerfectLaplacian__<3>(const Real *points,
                        const UInteger* elements,
                        const Real* Eulerp,
                        const Real* bases,
                        const Real* Jm,
                        const Real* AllGauss,
                        Integer ngauss,
                        Integer nelem,
                        Integer nodeperelem,
                        Integer nnode,
                        int *I_stiff,
                        int *J_stiff,
                        Real *V_stiff,
                        Real *e_tensor,
                        Integer is_hessian_symmetric,
                        int recompute_sparsity_pattern,
                        int squeeze_sparsity_pattern,
                        const int *data_local_indices,
                        const int *data_global_indices,
                        const UInteger *sorted_elements,
                        const Integer *sorter
                        ) {

    constexpr Integer ndim = 3;
    Integer ndof = nodeperelem;
    Integer local_capacity = ndof*ndof;

    Real *LagrangeElemCoords        = allocate<Real>(nodeperelem*ndim);
    Real *ElectricPotentialElem     = allocate<Real>(nodeperelem);

    Real *current_Jm                = allocate<Real>(nodeperelem*ndim);
    Real *MaterialGradient          = allocate<Real>(ndim*nodeperelem);
    Real *eM                        = allocate<Real>(ndim*nodeperelem);
    Real detJ                       = 0;

    Real *BDB                       = allocate<Real>(local_capacity);
    Real *stiffness                 = allocate<Real>(local_capacity);

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

            LagrangeElemCoords[i*3+0] = points[inode*3+0];
            LagrangeElemCoords[i*3+1] = points[inode*3+1];
            LagrangeElemCoords[i*3+2] = points[inode*3+2];

            ElectricPotentialElem[i] = Eulerp[inode];
        }

        std::fill(stiffness,stiffness+local_capacity,0.);
        std::fill(BDB,BDB+local_capacity,0.);

        for (int g=0; g<ngauss; ++g) {

            // COMPUTE KINEMATIC MEASURES
            std::fill(MaterialGradient,MaterialGradient+nodeperelem*ndim,0.);

            KinematicMeasures__<3>(  MaterialGradient,
                                    detJ,
                                    current_Jms[g].data(),
                                    AllGauss[g],
                                    LagrangeElemCoords,
                                    nodeperelem
                                    );

            // Extremely consise way - Gives almost identical timing
            // _transpose_(ndim,nodeperelem,MaterialGradient,MaterialGradientT);
            // _matmul_(ndof,ndim,ndim,MaterialGradientT,e_tensor,eM);
            // _matmul_(ndof,ndof,ndim,eM,MaterialGradient,BDB);
            // for (int i=0; i<local_capacity; ++i) {
            //     stiffness[i] += BDB[i]*detJ;
            // }

            _matmul_33k(nodeperelem,e_tensor,MaterialGradient,eM);

            if (is_hessian_symmetric) {
                for (int i=0; i<nodeperelem; ++i) {
                    const Real a0 = MaterialGradient[i];
                    const Real a1 = MaterialGradient[i+nodeperelem];
                    const Real a2 = MaterialGradient[i+2*nodeperelem];

                    for (int j=i; j<nodeperelem; ++j) {
                        const Real b0 = eM[j];
                        const Real b1 = eM[j+nodeperelem];
                        const Real b2 = eM[j+2*nodeperelem];

                        BDB[i*nodeperelem+j] += (a0*b0 + a1*b1 + a2*b2)*detJ;
                    }
                }
            }
            else {
                for (int i=0; i<nodeperelem; ++i) {
                    const Real a0 = MaterialGradient[i];
                    const Real a1 = MaterialGradient[i+nodeperelem];
                    const Real a2 = MaterialGradient[i+2*nodeperelem];

                    for (int j=0; j<nodeperelem; ++j) {
                        const Real b0 = eM[j];
                        const Real b1 = eM[j+nodeperelem];
                        const Real b2 = eM[j+2*nodeperelem];

                        BDB[i*nodeperelem+j] = a0*b0 + a1*b1 + a2*b2;
                    }
                }

                for (int i=0; i<local_capacity; ++i) {
                    stiffness[i] += BDB[i]*detJ;
                }
            }
        }

        if (is_hessian_symmetric) {
            // Fill the lower half now
            std::copy(BDB,BDB+local_capacity,stiffness);
            for (int i=0; i<nodeperelem; ++i) {
                for (int j=i; j<nodeperelem; ++j) {
                    stiffness[j*nodeperelem+i] = BDB[i*nodeperelem+j];
                }
            }
        }


        // ASSEMBLE CONSTITUTIVE STIFFNESS
        fill_global_data(
                nullptr,
                nullptr,
                stiffness,
                I_stiff,
                J_stiff,
                V_stiff,
                elem,
                1,
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

    deallocate(current_Jm);
    deallocate(LagrangeElemCoords);
    deallocate(ElectricPotentialElem);
    deallocate(MaterialGradient);
    deallocate(eM);

    deallocate(BDB);
    deallocate(stiffness);
}












#else
















// Kinematics
template<int ndim, typename std::enable_if<ndim==2,bool>::type = 0>
FASTOR_INLINE void KinematicMeasures__(Real *MaterialGradient, Real &detJ,
    const Real *current_Jm, Real AllGauss_, const Real *LagrangeElemCoords_,
    int nodeperelem)  {

    FASTOR_ALIGN Real ParentGradientX[ndim*ndim];
    FASTOR_ALIGN Real invParentGradientX[ndim*ndim];

    _matmul_(ndim,ndim,nodeperelem,current_Jm,LagrangeElemCoords_,ParentGradientX);
    const Real detX = invdet2x2(ParentGradientX,invParentGradientX);
    detJ = AllGauss_*fabs(detX);
    _matmul_(ndim,nodeperelem,ndim,invParentGradientX,current_Jm,MaterialGradient);
}

template<int ndim, typename std::enable_if<ndim==3,bool>::type = 0>
FASTOR_INLINE void KinematicMeasures__(Real *MaterialGradient, Real &detJ,
    const Real *current_Jm, Real AllGauss_, const Real *LagrangeElemCoords_,
    int nodeperelem)  {

    FASTOR_ALIGN Real ParentGradientX[ndim*ndim];
    FASTOR_ALIGN Real invParentGradientX[ndim*ndim];

    _matmul_(ndim,ndim,nodeperelem,current_Jm,LagrangeElemCoords_,ParentGradientX);
    const Real detX = invdet3x3(ParentGradientX,invParentGradientX);
    detJ = AllGauss_*fabs(detX);
    _matmul_(ndim,nodeperelem,ndim,invParentGradientX,current_Jm,MaterialGradient);
}
//




template<>
void _GlobalAssemblyPerfectLaplacian__<2>(const Real *points,
                        const UInteger* elements,
                        const Real* Eulerp,
                        const Real* bases,
                        const Real* Jm,
                        const Real* AllGauss,
                        Integer ngauss,
                        Integer nelem,
                        Integer nodeperelem,
                        Integer nnode,
                        int *I_stiff,
                        int *J_stiff,
                        Real *V_stiff,
                        Real *e_tensor,
                        Integer is_hessian_symmetric,
                        int recompute_sparsity_pattern,
                        int squeeze_sparsity_pattern,
                        const int *data_local_indices,
                        const int *data_global_indices,
                        const UInteger *sorted_elements,
                        const Integer *sorter
                        ) {

    constexpr Integer ndim = 2;
    Integer ndof = nodeperelem;
    Integer local_capacity = ndof*ndof;

    Real *LagrangeElemCoords        = allocate<Real>(nodeperelem*ndim);
    Real *ElectricPotentialElem     = allocate<Real>(nodeperelem);

    Real *current_Jm                = allocate<Real>(nodeperelem*ndim);
    Real *MaterialGradient          = allocate<Real>(ndim*nodeperelem);
    Real *eM                        = allocate<Real>(ndim*nodeperelem);
    Real detJ                       = 0;

    Real *BDB                       = allocate<Real>(local_capacity);
    Real *stiffness                 = allocate<Real>(local_capacity);

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

            LagrangeElemCoords[i*2+0] = points[inode*2+0];
            LagrangeElemCoords[i*2+1] = points[inode*2+1];

            ElectricPotentialElem[i] = Eulerp[inode];
        }

        std::fill(stiffness,stiffness+local_capacity,0.);
        std::fill(BDB,BDB+local_capacity,0.);

        for (int g=0; g<ngauss; ++g) {

            // COMPUTE KINEMATIC MEASURES
            std::fill(MaterialGradient,MaterialGradient+nodeperelem*ndim,0.);

            KinematicMeasures__<2>(  MaterialGradient,
                                    detJ,
                                    current_Jms[g].data(),
                                    AllGauss[g],
                                    LagrangeElemCoords,
                                    nodeperelem
                                    );

            // Extremely consise way
            // _transpose_(ndim,nodeperelem,MaterialGradient,MaterialGradientT);
            // _matmul_(ndof,ndim,ndim,MaterialGradientT,e_tensor,eM);
            // _matmul_(ndof,ndof,ndim,eM,MaterialGradient,BDB);
            // for (int i=0; i<local_capacity; ++i) {
            //     stiffness[i] += BDB[i]*detJ;
            // }

            _matmul_22k(nodeperelem,e_tensor,MaterialGradient,eM);

            if (is_hessian_symmetric) {
                for (int i=0; i<nodeperelem; ++i) {
                    const Real a0 = MaterialGradient[i];
                    const Real a1 = MaterialGradient[i+nodeperelem];

                    for (int j=i; j<nodeperelem; ++j) {
                        const Real b0 = eM[j];
                        const Real b1 = eM[j+nodeperelem];

                        BDB[i*nodeperelem+j] += (a0*b0 + a1*b1)*detJ;
                    }
                }
            }
            else {
                for (int i=0; i<nodeperelem; ++i) {
                    const Real a0 = MaterialGradient[i];
                    const Real a1 = MaterialGradient[i+nodeperelem];

                    for (int j=0; j<nodeperelem; ++j) {
                        const Real b0 = eM[j];
                        const Real b1 = eM[j+nodeperelem];

                        BDB[i*nodeperelem+j] = a0*b0 + a1*b1;
                    }
                }

                for (int i=0; i<local_capacity; ++i) {
                    stiffness[i] += BDB[i]*detJ;
                }
            }
        }

        if (is_hessian_symmetric) {
            // Fill the lower half now
            std::copy(BDB,BDB+local_capacity,stiffness);
            for (int i=0; i<nodeperelem; ++i) {
                for (int j=i; j<nodeperelem; ++j) {
                    stiffness[j*nodeperelem+i] = BDB[i*nodeperelem+j];
                }
            }
        }

        // ASSEMBLE CONSTITUTIVE STIFFNESS
        fill_global_data(
                nullptr,
                nullptr,
                stiffness,
                I_stiff,
                J_stiff,
                V_stiff,
                elem,
                1,
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

    deallocate(LagrangeElemCoords);
    deallocate(ElectricPotentialElem);
    deallocate(current_Jm);
    deallocate(MaterialGradient);
    deallocate(eM);

    deallocate(BDB);
    deallocate(stiffness);
}



template<>
void _GlobalAssemblyPerfectLaplacian__<3>(const Real *points,
                        const UInteger* elements,
                        const Real* Eulerp,
                        const Real* bases,
                        const Real* Jm,
                        const Real* AllGauss,
                        Integer ngauss,
                        Integer nelem,
                        Integer nodeperelem,
                        Integer nnode,
                        int *I_stiff,
                        int *J_stiff,
                        Real *V_stiff,
                        Real *e_tensor,
                        Integer is_hessian_symmetric,
                        int recompute_sparsity_pattern,
                        int squeeze_sparsity_pattern,
                        const int *data_local_indices,
                        const int *data_global_indices,
                        const UInteger *sorted_elements,
                        const Integer *sorter
                        ) {

    constexpr Integer ndim = 3;
    Integer ndof = nodeperelem;
    Integer local_capacity = ndof*ndof;

    Real *LagrangeElemCoords        = allocate<Real>(nodeperelem*ndim);
    Real *ElectricPotentialElem     = allocate<Real>(nodeperelem);

    Real *current_Jm                = allocate<Real>(nodeperelem*ndim);
    Real *MaterialGradient          = allocate<Real>(ndim*nodeperelem);
    Real *eM                        = allocate<Real>(ndim*nodeperelem);
    Real detJ                       = 0;

    Real *BDB                       = allocate<Real>(local_capacity);
    Real *stiffness                 = allocate<Real>(local_capacity);

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

            LagrangeElemCoords[i*3+0] = points[inode*3+0];
            LagrangeElemCoords[i*3+1] = points[inode*3+1];
            LagrangeElemCoords[i*3+2] = points[inode*3+2];

            ElectricPotentialElem[i] = Eulerp[inode];
        }

        std::fill(stiffness,stiffness+local_capacity,0.);
        std::fill(BDB,BDB+local_capacity,0.);

        for (int g=0; g<ngauss; ++g) {

            // COMPUTE KINEMATIC MEASURES
            std::fill(MaterialGradient,MaterialGradient+nodeperelem*ndim,0.);

            KinematicMeasures__<3>( MaterialGradient,
                                    detJ,
                                    current_Jms[g].data(),
                                    AllGauss[g],
                                    LagrangeElemCoords,
                                    nodeperelem
                                    );

            // Extremely consise way - Gives almost identical timing
            // _transpose_(ndim,nodeperelem,MaterialGradient,MaterialGradientT);
            // _matmul_(ndof,ndim,ndim,MaterialGradientT,e_tensor,eM);
            // _matmul_(ndof,ndof,ndim,eM,MaterialGradient,BDB);
            // for (int i=0; i<local_capacity; ++i) {
            //     stiffness[i] += BDB[i]*detJ;
            // }

            _matmul_33k(nodeperelem,e_tensor,MaterialGradient,eM);

            if (is_hessian_symmetric) {
                for (int i=0; i<nodeperelem; ++i) {
                    const Real a0 = MaterialGradient[i];
                    const Real a1 = MaterialGradient[i+nodeperelem];
                    const Real a2 = MaterialGradient[i+2*nodeperelem];

                    for (int j=i; j<nodeperelem; ++j) {
                        const Real b0 = eM[j];
                        const Real b1 = eM[j+nodeperelem];
                        const Real b2 = eM[j+2*nodeperelem];

                        BDB[i*nodeperelem+j] += (a0*b0 + a1*b1 + a2*b2)*detJ;
                    }
                }
            }
            else {
                for (int i=0; i<nodeperelem; ++i) {
                    const Real a0 = MaterialGradient[i];
                    const Real a1 = MaterialGradient[i+nodeperelem];
                    const Real a2 = MaterialGradient[i+2*nodeperelem];

                    for (int j=0; j<nodeperelem; ++j) {
                        const Real b0 = eM[j];
                        const Real b1 = eM[j+nodeperelem];
                        const Real b2 = eM[j+2*nodeperelem];

                        BDB[i*nodeperelem+j] = a0*b0 + a1*b1 + a2*b2;
                    }
                }

                for (int i=0; i<local_capacity; ++i) {
                    stiffness[i] += BDB[i]*detJ;
                }
            }
        }

        if (is_hessian_symmetric) {
            // Fill the lower half now
            std::copy(BDB,BDB+local_capacity,stiffness);
            for (int i=0; i<nodeperelem; ++i) {
                for (int j=i; j<nodeperelem; ++j) {
                    stiffness[j*nodeperelem+i] = BDB[i*nodeperelem+j];
                }
            }
        }


        // ASSEMBLE CONSTITUTIVE STIFFNESS
        fill_global_data(
                nullptr,
                nullptr,
                stiffness,
                I_stiff,
                J_stiff,
                V_stiff,
                elem,
                1,
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

    deallocate(current_Jm);
    deallocate(LagrangeElemCoords);
    deallocate(ElectricPotentialElem);
    deallocate(MaterialGradient);
    deallocate(eM);

    deallocate(BDB);
    deallocate(stiffness);
}






#endif









void _GlobalAssemblyPerfectLaplacian_(
                        const Real *points,
                        const UInteger* elements,
                        const Real* Eulerp,
                        const Real* bases,
                        const Real* Jm,
                        const Real* AllGauss,
                        Integer ndim,
                        Integer ngauss,
                        Integer nelem,
                        Integer nodeperelem,
                        Integer nnode,
                        int *I_stiff,
                        int *J_stiff,
                        Real *V_stiff,
                        Real *e_tensor,
                        Integer is_hessian_symmetric,
                        int recompute_sparsity_pattern,
                        int squeeze_sparsity_pattern,
                        const int *data_local_indices,
                        const int *data_global_indices,
                        const UInteger *sorted_elements,
                        const Integer *sorter
                        ) {

    if (ndim==3) {
        _GlobalAssemblyPerfectLaplacian__<3>(points,
                        elements,
                        Eulerp,
                        bases,
                        Jm,
                        AllGauss,
                        ngauss,
                        nelem,
                        nodeperelem,
                        nnode,
                        I_stiff,
                        J_stiff,
                        V_stiff,
                        e_tensor,
                        is_hessian_symmetric,
                        recompute_sparsity_pattern,
                        squeeze_sparsity_pattern,
                        data_local_indices,
                        data_global_indices,
                        sorted_elements,
                        sorter
                        );
    }
    else {
        _GlobalAssemblyPerfectLaplacian__<2>(points,
                        elements,
                        Eulerp,
                        bases,
                        Jm,
                        AllGauss,
                        ngauss,
                        nelem,
                        nodeperelem,
                        nnode,
                        I_stiff,
                        J_stiff,
                        V_stiff,
                        e_tensor,
                        is_hessian_symmetric,
                        recompute_sparsity_pattern,
                        squeeze_sparsity_pattern,
                        data_local_indices,
                        data_global_indices,
                        sorted_elements,
                        sorter
                        );

    }
}







#endif // _LOWLEVELASSEMBLYDPF__H