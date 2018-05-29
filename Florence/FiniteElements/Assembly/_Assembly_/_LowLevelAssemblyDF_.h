#ifndef _LOWLEVELASSEMBLYDF__H
#define _LOWLEVELASSEMBLYDF__H

#include "assembly_helper.h"
#include "_ConstitutiveStiffnessDF_.h"
#include "_MooneyRivlin_.h"

void _GlobalAssemblyDF_(const Real *points,
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

    Real *stress                    = allocate<Real>(ngauss*ndim*ndim);
    Real *hessian                   = allocate<Real>(ngauss*H_VoigtSize*H_VoigtSize);

    Real *traction                  = allocate<Real>(ndof);
    Real *stiffness                 = allocate<Real>(local_capacity);
    Real *geometric_stiffness       = allocate<Real>(local_capacity);
    Real *mass                      = allocate<Real>(local_capacity);

    Integer *current_row_column     = allocate<Integer>(ndof);
    Integer *full_current_row       = allocate<Integer>(local_capacity);
    Integer *full_current_column    = allocate<Integer>(local_capacity);

    auto mat_obj = _MooneyRivlin_<Real>(mu1,mu2,lam);


    // LOOP OVER ELEMETNS
    for (Integer elem=0; elem < nelem; ++elem) {

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
                            requires_geometry_update
                            );

        // COMPUTE KINETIC MEASURES
        mat_obj.KineticMeasures(stress, hessian, ndim, ngauss, F);

        // COMPUTE CONSTITUTIVE STIFFNESS AND TRACTION
        std::fill(stiffness,stiffness+local_capacity,0.);
        std::fill(traction,traction+ndof,0.);
        _ConstitutiveStiffnessIntegrandDF_Filler_(
            stiffness,
            traction,
            SpatialGradient,
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
        std::fill(geometric_stiffness,geometric_stiffness+local_capacity,0.);
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
        {

            Integer const_elem_retriever;
            for (Integer counter=0; counter<nodeperelem; ++counter) {
                const_elem_retriever = nvar*elements[elem*nodeperelem+counter];
                for (Integer ncounter=0; ncounter<nvar; ++ncounter) {
                    current_row_column[nvar*counter+ncounter] = const_elem_retriever+ncounter;
                }
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
                                1
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
    deallocate(stress);
    deallocate(hessian);
    deallocate(traction);
    deallocate(stiffness);
    deallocate(geometric_stiffness);
    deallocate(mass);

    deallocate(full_current_row);
    deallocate(full_current_column);
    deallocate(current_row_column);

}


#endif // _LOWLEVELASSEMBLYDF__H