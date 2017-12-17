#ifndef _LOWLEVELASSEMBLYDPF__H
#define _LOWLEVELASSEMBLYDPF__H

#include "assembly_helper.h"
#include "_TractionDF_.h"
#include "_TractionDPF_.h"

#include "_NeoHookean_2_.h"
#include "_MooneyRivlin_0_.h"
#include "_ExplicitMooneyRivlin_0_.h"
#include "_IsotropicElectroMechanics_101_.h"
#include "_IsotropicElectroMechanics_108_.h"

void _GlobalAssemblyExplicit_DF_DPF_(const Real *points,
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
                        Real *T,
                        Integer is_dynamic,
                        Integer* local_rows_mass,
                        Integer* local_cols_mass,
                        int *I_mass,
                        int *J_mass,
                        Real *V_mass,
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
                        const Real *anisotropic_orientations,
                        int material_number,
                        int formulation_number
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
    Real *mass                      = allocate<Real>(local_capacity);

    Integer *current_row_column     = allocate<Integer>(ndof);
    Integer *full_current_row       = allocate<Integer>(local_capacity);
    Integer *full_current_column    = allocate<Integer>(local_capacity);

    auto mat_obj0 = _NeoHookean_2_<Real>(mu,lamb);
    auto mat_obj1 = _MooneyRivlin_0_<Real>(mu1,mu2,lamb);
    auto mat_obj2 = _ExplicitMooneyRivlin_0_<Real>(mu1,mu2,lamb);
    auto mat_obj3 = _IsotropicElectroMechanics_101_<Real>(mu,lamb,eps_1);
    auto mat_obj4 = _IsotropicElectroMechanics_108_<Real>(mu1,mu2,lamb,eps_2);


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
        if (material_number==0) {
            mat_obj0.KineticMeasures(stress, hessian, ndim, ngauss, F);
        }
        else if (material_number==1) {
            mat_obj1.KineticMeasures(stress, hessian, ndim, ngauss, F);
        }
        else if (material_number==2) {
            mat_obj2.KineticMeasures(stress, ndim, ngauss, F);
        }
        else if (material_number==3) {
            mat_obj3.KineticMeasures(D, stress, hessian, ndim, ngauss, F, ElectricFieldx);
        }
        else if (material_number==4) {
            mat_obj4.KineticMeasures(D, stress, hessian, ndim, ngauss, F, ElectricFieldx);
        }

        // COMPUTE CONSTITUTIVE STIFFNESS AND TRACTION
        std::fill(traction,traction+ndof,0.);
        if (formulation_number==1) {
            _TractionDPF_Filler_(
                traction,
                SpatialGradient,
                D,
                stress,
                detJ,
                ngauss,
                nodeperelem,
                ndim,
                nvar,
                H_VoigtSize,
                requires_geometry_update);
        }
        else {
            _TractionDF_Filler_(
                traction,
                SpatialGradient,
                stress,
                detJ,
                ngauss,
                nodeperelem,
                ndim,
                nvar,
                H_VoigtSize,
                requires_geometry_update);
        }

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

    // ASSEMBLE MASS
    if (is_dynamic) {
        // This is performed only once as mass integrand is Lagrangian
        // hence not mixing this with stiffness and mass integrand is beneficial
        for (Integer elem=0 ; elem<nelem; ++elem) {

            // GET THE FIELDS AT THE ELEMENT LEVEL
            for (Integer i=0; i<nodeperelem; ++i) {
                const Integer inode = elements[elem*nodeperelem+i];
                for (Integer j=0; j<ndim; ++j) {
                    LagrangeElemCoords[i*ndim+j] = points[inode*ndim+j];
                    EulerElemCoords[i*ndim+j] = Eulerx[inode*ndim+j];
                }
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
                                0
                                );

            std::fill(mass,mass+local_capacity,0);

            // Call MassIntegrand
            _MassIntegrand_Filler_( mass,
                                    bases,
                                    detJ,
                                    ngauss,
                                    nodeperelem,
                                    ndim,
                                    nvar,
                                    rho);

            // Fill IJV
            fill_triplet(   local_rows_mass,
                            local_cols_mass,
                            mass,
                            I_mass,
                            J_mass,
                            V_mass,
                            elem,
                            nvar,
                            nodeperelem,
                            elements,
                            local_capacity,
                            local_capacity);
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
    deallocate(mass);

    deallocate(full_current_row);
    deallocate(full_current_column);
    deallocate(current_row_column);

}


#endif // _LOWLEVELASSEMBLYDPF__H