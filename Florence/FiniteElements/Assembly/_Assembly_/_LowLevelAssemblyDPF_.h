#ifndef _LOWLEVELASSEMBLYDPF__H
#define _LOWLEVELASSEMBLYDPF__H

#include "assembly_helper.h"
#include "_ConstitutiveStiffnessDPF_.h"
#include "_IsotropicElectroMechanics_101_.h"

void _GlobalAssemblyDPF_(const Real *points,
                        const UInteger* elements,
                        const Real* Eulerx,
                        const Real* Eulerp,
                        const Real* bases,
                        const Real* Jm,
                        const Real* AllGauss,
                        Integer ndim,
                        Integer nvar,
                        Integer ngauss,
                        Integer nelem,
                        Integer nodeperelem,
                        Integer nnode,
                        Integer H_VoigtSize,
                        Integer requires_geometry_update,
                        Integer* local_rows_stiffness,
                        Integer* local_cols_stiffness,
                        int *I_stiff,
                        int *J_stiff,
                        Real *V_stiff,
                        Real *T,
                        int recompute_sparsity_pattern,
                        int squeeze_sparsity_pattern,
                        const int *data_local_indices,
                        const int *data_global_indices,
                        const UInteger *sorted_elements,
                        const Integer *sorter,
                        Real rho,
                        Real mu,
                        Real mu1,
                        Real mu2,
                        Real mu3,
                        Real mue,
                        Real lamb,
                        Real eps_1,
                        Real eps_2,
                        Real eps_3,
                        Real eps_e,
                        const Real *anisotropic_orientations
                        ) {

    Integer ndof = nvar*nodeperelem;
    Integer local_capacity = ndof*ndof;

    Real *LagrangeElemCoords        = allocate<Real>(nodeperelem*ndim);
    Real *EulerElemCoords           = allocate<Real>(nodeperelem*ndim);
    Real *ElectricPotentialElem     = allocate<Real>(nodeperelem);

    Real *F                         = allocate<Real>(ngauss*ndim*ndim);
    Real *SpatialGradient           = allocate<Real>(ngauss*nodeperelem*ndim);
    Real *detJ                      = allocate<Real>(ngauss);

    Real *ElectricFieldx            = allocate<Real>(ngauss*ndim);

    Real *D                         = allocate<Real>(ngauss*ndim);
    Real *stress                    = allocate<Real>(ngauss*ndim*ndim);
    Real *hessian                   = allocate<Real>(ngauss*H_VoigtSize*H_VoigtSize);

    Real *traction                  = allocate<Real>(ndof);
    Real *stiffness                 = allocate<Real>(local_capacity);
    Real *geometric_stiffness       = allocate<Real>(local_capacity);

    auto mat_obj = _IsotropicElectroMechanics_101_<Real>(mu,lamb,eps_1);

    // LOOP OVER ELEMETNS
    for (Integer elem=0; elem < nelem; ++elem) {

        // GET THE FIELDS AT THE ELEMENT LEVEL
        for (Integer i=0; i<nodeperelem; ++i) {
            const Integer inode = elements[elem*nodeperelem+i];
            for (Integer j=0; j<ndim; ++j) {
                LagrangeElemCoords[i*ndim+j] = points[inode*ndim+j];
                EulerElemCoords[i*ndim+j] = Eulerx[inode*ndim+j];
            }
            ElectricPotentialElem[i] = Eulerp[inode];
        }

        // COMPUTE KINEMATIC MEASURES
        std::fill(F,F+ngauss*ndim*ndim,0.);
        std::fill(SpatialGradient,SpatialGradient+ngauss*nodeperelem*ndim,0.);
        std::fill(detJ,detJ+ngauss,0.);
        KinematicMeasures(  SpatialGradient,
                            F,
                            detJ,
                            Jm,
                            AllGauss,
                            LagrangeElemCoords,
                            EulerElemCoords,
                            ngauss,
                            ndim,
                            nodeperelem,
                            requires_geometry_update
                            );

        // COMPUTE ELECTRIC FIELD
        for (Integer i=0; i<ngauss; ++i) {
            for (Integer k=0; k<ndim; ++k) {
                Real iE = 0;
                for (Integer j=0; j<nodeperelem; ++j) {
                    iE += SpatialGradient[i*nodeperelem*ndim+j*ndim+k]*ElectricPotentialElem[j];
                }
                ElectricFieldx[i*ndim+k] = -iE;
            }
        }

        // COMPUTE KINETIC MEASURES
        mat_obj.KineticMeasures(D, stress, hessian, ndim, ngauss, F, ElectricFieldx);

        // COMPUTE CONSTITUTIVE STIFFNESS AND TRACTION
        std::fill(stiffness,stiffness+local_capacity,0.);
        std::fill(traction,traction+ndof,0.);
        _ConstitutiveStiffnessIntegrandDPF_Filler_(
            stiffness,
            traction,
            SpatialGradient,
            D,
            stress,
            hessian,
            detJ,
            ngauss,
            nodeperelem,
            ndim,
            nvar,
            H_VoigtSize,
            requires_geometry_update);

        // COMPUTE GEOMETRIC STIFFNESS
        std::fill(geometric_stiffness,geometric_stiffness+local_capacity,0);
        _GeometricStiffnessFiller_( geometric_stiffness,
                                    SpatialGradient,
                                    stress,
                                    detJ,
                                    ndim,
                                    nvar,
                                    nodeperelem,
                                    ngauss);

        for (Integer i=0; i<local_capacity; ++i) {
            stiffness[i] += geometric_stiffness[i];
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

        // ASSEMBLE TRACTIONS
        {
            for (Integer i = 0; i<nodeperelem; ++i) {
                UInteger T_idx = elements[elem*nodeperelem+i]*nvar;
                for (Integer iterator = 0; iterator < nvar; ++iterator) {
                    T[T_idx+iterator] += traction[i*nvar+iterator];
                }
            }
        }

    }

    deallocate(LagrangeElemCoords);
    deallocate(EulerElemCoords);
    deallocate(ElectricPotentialElem);

    deallocate(F);
    deallocate(SpatialGradient);
    deallocate(detJ);
    deallocate(ElectricFieldx);
    deallocate(D);
    deallocate(stress);
    deallocate(hessian);
    deallocate(traction);
    deallocate(stiffness);
    deallocate(geometric_stiffness);
}


#endif // _LOWLEVELASSEMBLYDPF__H