#ifndef _LOWLEVELASSEMBLYDPF__H
#define _LOWLEVELASSEMBLYDPF__H


#include "assembly_helper.h"

// FASTOR_INLINE void _transpose_(int M, int N, const Real * FASTOR_RESTRICT a, Real * FASTOR_RESTRICT out) {
//     for (int i=0; i< M; ++i)
//         for (int j=0; j<N; ++j)
//             out[j*M+i] = a[i*N+j];
// }


// Kinematics
template<int ndim, typename std::enable_if<ndim==2,bool>::type = 0>
FASTOR_INLINE void KinematicMeasures__(Real *MaterialGradient, Real &detJ,
    const Real *current_Jm, Real AllGauss_, const Real *LagrangeElemCoords_,
    int nodeperelem)  {

    Real FASTOR_ALIGN ParentGradientX[ndim*ndim];
    Real FASTOR_ALIGN invParentGradientX[ndim*ndim];

    _matmul_(ndim,ndim,nodeperelem,current_Jm,LagrangeElemCoords_,ParentGradientX);
    const Real detX = invdet2x2(ParentGradientX,invParentGradientX);
    detJ = AllGauss_*fabs(detX);
    _matmul_(ndim,nodeperelem,ndim,invParentGradientX,current_Jm,MaterialGradient);
}

template<int ndim, typename std::enable_if<ndim==3,bool>::type = 0>
FASTOR_INLINE void KinematicMeasures__(Real *MaterialGradient, Real &detJ,
    const Real *current_Jm, Real AllGauss_, const Real *LagrangeElemCoords_,
    int nodeperelem)  {

    Real FASTOR_ALIGN ParentGradientX[ndim*ndim];
    Real FASTOR_ALIGN invParentGradientX[ndim*ndim];

    _matmul_(ndim,ndim,nodeperelem,current_Jm,LagrangeElemCoords_,ParentGradientX);
    const Real detX = invdet3x3(ParentGradientX,invParentGradientX);
    detJ = AllGauss_*fabs(detX);
    _matmul_(ndim,nodeperelem,ndim,invParentGradientX,current_Jm,MaterialGradient);
}
//




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
                        Integer is_hessian_symmetric
                        );



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
                        Integer is_hessian_symmetric
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

    Integer *current_row_column     = allocate<Integer>(ndof);
    Integer *full_current_row       = allocate<Integer>(local_capacity);
    Integer *full_current_column    = allocate<Integer>(local_capacity);


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

            for (int j=0; j<nodeperelem; ++j) {
                current_Jm[j] = Jm[j*ngauss+g];
                current_Jm[nodeperelem+j] = Jm[ngauss*nodeperelem+j*ngauss+g];
            }

            // COMPUTE KINEMATIC MEASURES
            std::fill(MaterialGradient,MaterialGradient+nodeperelem*ndim,0.);

            KinematicMeasures__<2>(  MaterialGradient,
                                    detJ,
                                    current_Jm,
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

            if (is_hessian_symmetric) {
                _matmul_(ndim,nodeperelem,ndim,e_tensor,MaterialGradient,eM);
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
                _matmul_(ndim,nodeperelem,ndim,e_tensor,MaterialGradient,eM);
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
        {
            for (Integer counter=0; counter<nodeperelem; ++counter) {
                current_row_column[counter] = elements[elem*nodeperelem+counter];
            }

            Integer const_I_retriever;
            for (Integer counter=0; counter<ndof; ++counter) {
                const_I_retriever = current_row_column[counter];
                for (Integer iterator=0; iterator<ndof; ++iterator) {
                    full_current_row[counter*ndof+iterator]    = const_I_retriever;
                    full_current_column[counter*ndof+iterator] = current_row_column[iterator];
                }
            }

            Integer low, high;
            low = local_capacity*elem;
            high = local_capacity*(elem+1);

            Integer incrementer = 0;
            for (Integer counter = low; counter < high; ++counter) {
                I_stiff[counter] = full_current_row[incrementer];
                J_stiff[counter] = full_current_column[incrementer];
                V_stiff[counter] = stiffness[incrementer];

                incrementer += 1;
            }
        }

    }


    deallocate(LagrangeElemCoords);
    deallocate(ElectricPotentialElem);
    deallocate(current_Jm);
    deallocate(MaterialGradient);
    deallocate(eM);

    deallocate(BDB);
    deallocate(stiffness);

    deallocate(full_current_row);
    deallocate(full_current_column);
    deallocate(current_row_column);
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
                        Integer is_hessian_symmetric
                        ) {

    constexpr Integer ndim = 3;
    Integer ndof = nodeperelem;
    Integer local_capacity = ndof*ndof;

    Real *LagrangeElemCoords        = allocate<Real>(nodeperelem*ndim);
    Real *ElectricPotentialElem     = allocate<Real>(nodeperelem);

    Real *current_Jm                = allocate<Real>(nodeperelem*ndim);
    Real *MaterialGradient          = allocate<Real>(ndim*nodeperelem);
    // Real *MaterialGradientT         = allocate<Real>(ndim*nodeperelem);
    Real *eM                        = allocate<Real>(ndim*nodeperelem);
    Real detJ                       = 0;

    Real *BDB                       = allocate<Real>(local_capacity);
    Real *stiffness                 = allocate<Real>(local_capacity);

    Integer *current_row_column     = allocate<Integer>(ndof);
    Integer *full_current_row       = allocate<Integer>(local_capacity);
    Integer *full_current_column    = allocate<Integer>(local_capacity);


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

            for (int j=0; j<nodeperelem; ++j) {
                current_Jm[j] = Jm[j*ngauss+g];
                current_Jm[nodeperelem+j] = Jm[ngauss*nodeperelem+j*ngauss+g];
                current_Jm[2*nodeperelem+j] = Jm[2*ngauss*nodeperelem+j*ngauss+g];
            }

            // COMPUTE KINEMATIC MEASURES
            std::fill(MaterialGradient,MaterialGradient+nodeperelem*ndim,0.);

            KinematicMeasures__<3>(  MaterialGradient,
                                    detJ,
                                    current_Jm,
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

            if (is_hessian_symmetric) {
                _matmul_(ndim,nodeperelem,ndim,e_tensor,MaterialGradient,eM);
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
                _matmul_(ndim,nodeperelem,ndim,e_tensor,MaterialGradient,eM);
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
        {
            for (Integer counter=0; counter<nodeperelem; ++counter) {
                current_row_column[counter] = elements[elem*nodeperelem+counter];
            }

            Integer const_I_retriever;
            for (Integer counter=0; counter<ndof; ++counter) {
                const_I_retriever = current_row_column[counter];
                for (Integer iterator=0; iterator<ndof; ++iterator) {
                    full_current_row[counter*ndof+iterator]    = const_I_retriever;
                    full_current_column[counter*ndof+iterator] = current_row_column[iterator];
                }
            }

            Integer low, high;
            low = local_capacity*elem;
            high = local_capacity*(elem+1);

            Integer incrementer = 0;
            for (Integer counter = low; counter < high; ++counter) {
                I_stiff[counter] = full_current_row[incrementer];
                J_stiff[counter] = full_current_column[incrementer];
                V_stiff[counter] = stiffness[incrementer];

                incrementer += 1;
            }
        }


    }

    deallocate(current_Jm);
    deallocate(LagrangeElemCoords);
    deallocate(ElectricPotentialElem);
    deallocate(MaterialGradient);
    // deallocate(MaterialGradientT);
    deallocate(eM);

    deallocate(BDB);
    deallocate(stiffness);

    deallocate(full_current_row);
    deallocate(full_current_column);
    deallocate(current_row_column);
}










void _GlobalAssemblyPerfectLaplacian_(const Real *points,
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
                        Integer is_hessian_symmetric
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
                        is_hessian_symmetric
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
                        is_hessian_symmetric
                        );

    }
}


#endif // _LOWLEVELASSEMBLYDPF__H

















// #ifndef _LOWLEVELASSEMBLYDPF__H
// #define _LOWLEVELASSEMBLYDPF__H

// #include "assembly_helper.h"
// #include "_ConstitutiveStiffnessDPF_.h"
// #include "_IsotropicElectroMechanics_101_.h"

// void _GlobalAssemblyDPF_(const Real *points,
//                         const UInteger* elements,
//                         const Real* Eulerx,
//                         const Real* Eulerp,
//                         const Real* bases,
//                         const Real* Jm,
//                         const Real* AllGauss,
//                         Integer ndim,
//                         Integer ngauss,
//                         Integer nelem,
//                         Integer nodeperelem,
//                         Integer nnode,
//                         Integer H_VoigtSize,
//                         Integer requires_geometry_update,
//                         Integer* local_rows_stiffness,
//                         Integer* local_cols_stiffness,
//                         int *I_stiff,
//                         int *J_stiff,
//                         Real *V_stiff,
//                         Real *T,
//                         Real eps_1,
//                         Real eps_2,
//                         Real eps_3,
//                         Real eps_e,
//                         const Real *anisotropic_orientations
//                         ) {

//     Integer ndof = nvar*nodeperelem;
//     Integer local_capacity = ndof*ndof;


//     Real *LagrangeElemCoords        = allocate<Real>(nodeperelem*ndim);
//     Real *EulerElemCoords           = allocate<Real>(nodeperelem*ndim);
//     Real *ElectricPotentialElem     = allocate<Real>(nodeperelem);

//     Real *F                         = allocate<Real>(ngauss*ndim*ndim);
//     Real *SpatialGradient           = allocate<Real>(ngauss*nodeperelem*ndim);
//     Real *detJ                      = allocate<Real>(ngauss);

//     Real *ElectricFieldx            = allocate<Real>(ngauss*ndim);

//     Real *D                         = allocate<Real>(ngauss*ndim);
//     Real *stress                    = allocate<Real>(ngauss*ndim*ndim);
//     Real *hessian                   = allocate<Real>(ngauss*H_VoigtSize*H_VoigtSize);

//     Real *traction                  = allocate<Real>(ndof);
//     Real *stiffness                 = allocate<Real>(local_capacity);
//     Real *geometric_stiffness       = allocate<Real>(local_capacity);
//     Real *mass                      = allocate<Real>(local_capacity);

//     Integer *current_row_column     = allocate<Integer>(ndof);
//     Integer *full_current_row       = allocate<Integer>(local_capacity);
//     Integer *full_current_column    = allocate<Integer>(local_capacity);

//     auto mat_obj = _IsotropicElectroMechanics_101_<Real>(mu,lamb,eps_1);


//     // LOOP OVER ELEMETNS
//     for (Integer elem=0; elem < nelem; ++elem) {

//         // GET THE FIELDS AT THE ELEMENT LEVEL
//         for (Integer i=0; i<nodeperelem; ++i) {
//             const Integer inode = elements[elem*nodeperelem+i];
//             ElectricPotentialElem[i] = Eulerp[inode];
//         }

//         // COMPUTE KINEMATIC MEASURES
//         std::fill(F,F+ngauss*ndim*ndim,0.);
//         std::fill(SpatialGradient,SpatialGradient+ngauss*nodeperelem*ndim,0.);
//         std::fill(detJ,detJ+ngauss,0.);
//         KinematicMeasures(  SpatialGradient,
//                             F,
//                             detJ,
//                             Jm,
//                             AllGauss,
//                             LagrangeElemCoords,
//                             EulerElemCoords,
//                             ngauss,
//                             ndim,
//                             nodeperelem,
//                             requires_geometry_update
//                             );

//         // COMPUTE ELECTRIC FIELD
//         for (Integer i=0; i<ngauss; ++i) {
//             for (Integer k=0; k<ndim; ++k) {
//                 Real iE = 0;
//                 for (Integer j=0; j<nodeperelem; ++j) {
//                     iE += SpatialGradient[i*nodeperelem*ndim+j*ndim+k]*ElectricPotentialElem[j];
//                 }
//                 ElectricFieldx[i*ndim+k] = -iE;
//             }
//         }

//         // COMPUTE KINETIC MEASURES
//         mat_obj.KineticMeasures(D, stress, hessian, ndim, ngauss, F, ElectricFieldx);

//         // COMPUTE CONSTITUTIVE STIFFNESS AND TRACTION
//         std::fill(stiffness,stiffness+local_capacity,0.);
//         std::fill(traction,traction+ndof,0.);
//         _ConstitutiveStiffnessIntegrandDPF_Filler_(
//             stiffness,
//             traction,
//             SpatialGradient,
//             D,
//             stress,
//             hessian,
//             detJ,
//             ngauss,
//             nodeperelem,
//             ndim,
//             nvar,
//             H_VoigtSize,
//             requires_geometry_update);

//         // COMPUTE GEOMETRIC STIFFNESS
//         std::fill(geometric_stiffness,geometric_stiffness+local_capacity,0);
//         _GeometricStiffnessFiller_( geometric_stiffness,
//                                     SpatialGradient,
//                                     stress,
//                                     detJ,
//                                     ndim,
//                                     nvar,
//                                     nodeperelem,
//                                     ngauss);


//         for (Integer i=0; i<local_capacity; ++i) {
//             stiffness[i] += geometric_stiffness[i];
//         }

//         // ASSEMBLE CONSTITUTIVE STIFFNESS
//         {

//             Integer const_elem_retriever;
//             for (Integer counter=0; counter<nodeperelem; ++counter) {
//                 const_elem_retriever = nvar*elements[elem*nodeperelem+counter];
//                 for (Integer ncounter=0; ncounter<nvar; ++ncounter) {
//                     current_row_column[nvar*counter+ncounter] = const_elem_retriever+ncounter;
//                 }
//             }

//             Integer const_I_retriever;
//             for (Integer counter=0; counter<ndof; ++counter) {
//                 const_I_retriever = current_row_column[counter];
//                 for (Integer iterator=0; iterator<ndof; ++iterator) {
//                     full_current_row[counter*ndof+iterator]    = const_I_retriever;
//                     full_current_column[counter*ndof+iterator] = current_row_column[iterator];
//                 }
//             }


//             Integer low, high;
//             low = local_capacity*elem;
//             high = local_capacity*(elem+1);

//             Integer incrementer = 0;
//             for (Integer counter = low; counter < high; ++counter) {
//                 I_stiff[counter] = full_current_row[incrementer];
//                 J_stiff[counter] = full_current_column[incrementer];
//                 V_stiff[counter] = stiffness[incrementer];

//                 incrementer += 1;
//             }

//         }

//         // ASSEMBLE TRACTIONS
//         {
//             for (Integer i = 0; i<nodeperelem; ++i) {
//                 UInteger T_idx = elements[elem*nodeperelem+i]*nvar;
//                 for (Integer iterator = 0; iterator < nvar; ++iterator) {
//                     T[T_idx+iterator] += traction[i*nvar+iterator];
//                 }
//             }
//         }

//     }

//     // ASSEMBLE MASS
//     if (is_dynamic) {
//         // This is performed only once as mass integrand is Lagrangian
//         // hence not mixing this with stiffness and mass integrand is beneficial
//         for (Integer elem=0 ; elem<nelem; ++elem) {

//             // GET THE FIELDS AT THE ELEMENT LEVEL
//             for (Integer i=0; i<nodeperelem; ++i) {
//                 const Integer inode = elements[elem*nodeperelem+i];
//                 for (Integer j=0; j<ndim; ++j) {
//                     LagrangeElemCoords[i*ndim+j] = points[inode*ndim+j];
//                     EulerElemCoords[i*ndim+j] = Eulerx[inode*ndim+j];
//                 }
//             }

//             // COMPUTE KINEMATIC MEASURES
//             std::fill(F,F+ngauss*ndim*ndim,0.);
//             std::fill(SpatialGradient,SpatialGradient+ngauss*nodeperelem*ndim,0.);
//             std::fill(detJ,detJ+ngauss,0.);
//             KinematicMeasures(  SpatialGradient,
//                                 F,
//                                 detJ,
//                                 Jm,
//                                 AllGauss,
//                                 LagrangeElemCoords,
//                                 EulerElemCoords,
//                                 ngauss,
//                                 ndim,
//                                 nodeperelem,
//                                 0
//                                 );

//             std::fill(mass,mass+local_capacity,0);

//             // Call MassIntegrand
//             _MassIntegrand_Filler_( mass,
//                                     bases,
//                                     detJ,
//                                     ngauss,
//                                     nodeperelem,
//                                     ndim,
//                                     nvar,
//                                     rho);

//             // Fill IJV
//             fill_triplet(   local_rows_mass,
//                             local_cols_mass,
//                             mass,
//                             I_mass,
//                             J_mass,
//                             V_mass,
//                             elem,
//                             nvar,
//                             nodeperelem,
//                             elements,
//                             local_capacity,
//                             local_capacity);
//         }
//     }


//     deallocate(LagrangeElemCoords);
//     deallocate(EulerElemCoords);
//     deallocate(ElectricPotentialElem);

//     deallocate(F);
//     deallocate(SpatialGradient);
//     deallocate(detJ);
//     deallocate(ElectricFieldx);
//     deallocate(D);
//     deallocate(stress);
//     deallocate(hessian);
//     deallocate(traction);
//     deallocate(stiffness);
//     deallocate(geometric_stiffness);
//     deallocate(mass);

//     deallocate(full_current_row);
//     deallocate(full_current_column);
//     deallocate(current_row_column);

// }


// #endif // _LOWLEVELASSEMBLYDPF__H