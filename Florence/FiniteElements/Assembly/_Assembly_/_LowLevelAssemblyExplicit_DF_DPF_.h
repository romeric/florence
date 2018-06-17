#ifndef _LOWLEVELASSEMBLYDPF__H
#define _LOWLEVELASSEMBLYDPF__H



#include "assembly_helper.h"
#include "_TractionDF_.h"
#include "_TractionDPF_.h"

#include "_ExplicitMooneyRivlin_.h"
#include "_ExplicitIsotropicElectroMechanics_108_.h"
#include "_NeoHookean_.h"
#include "_MooneyRivlin_.h"
#include "_NearlyIncompressibleMooneyRivlin_.h"
#include "_IsotropicElectroMechanics_101_.h"
#include "_IsotropicElectroMechanics_105_.h"
#include "_IsotropicElectroMechanics_106_.h"
#include "_IsotropicElectroMechanics_107_.h"
#include "_IsotropicElectroMechanics_108_.h"
#include "_LinearElastic_.h"


#ifndef EXPLICIT_OLD

// Kinematics
//
template<int ndim, typename std::enable_if<ndim==2,bool>::type = 0>
FASTOR_INLINE void KinematicMeasures__(Real *MaterialGradient, Real *current_sp, Real *current_F, Real &detJ,
    const Real *current_Jm, Real AllGauss_, const Real *LagrangeElemCoords_, const Real *EulerElemCoords_,
    int ngauss, int nodeperelem, int update)  {

    Real FASTOR_ALIGN ParentGradientX[ndim*ndim];
    Real FASTOR_ALIGN invParentGradientX[ndim*ndim];
    Real FASTOR_ALIGN ParentGradientx[ndim*ndim];
    Real FASTOR_ALIGN invParentGradientx[ndim*ndim];
    Real FASTOR_ALIGN current_Ft[ndim*ndim];

    _matmul_(ndim,ndim,nodeperelem,current_Jm,LagrangeElemCoords_,ParentGradientX);
    _matmul_(ndim,ndim,nodeperelem,current_Jm,EulerElemCoords_,ParentGradientx);

    const Real detX = invdet2x2(ParentGradientX,invParentGradientX);
    const Real detx = invdet2x2(ParentGradientx,invParentGradientx);

    detJ = update==1 ? AllGauss_*fabs(detx) : AllGauss_*fabs(detX);

    _matmul_(ndim,nodeperelem,ndim,invParentGradientX,current_Jm,MaterialGradient);
    _matmul_(ndim,nodeperelem,ndim,invParentGradientx,current_Jm,current_sp);

    // Compute deformation gradient F
    _matmul_(ndim,ndim,nodeperelem,MaterialGradient,EulerElemCoords_,current_Ft);
    Fastor::_transpose<Real,ndim,ndim>(current_Ft,current_F);
}

template<int ndim, typename std::enable_if<ndim==3,bool>::type = 0>
FASTOR_INLINE void KinematicMeasures__(Real *MaterialGradient, Real *current_sp, Real *current_F, Real &detJ,
    const Real *current_Jm, Real AllGauss_, const Real *LagrangeElemCoords_, const Real *EulerElemCoords_,
    int ngauss, int nodeperelem, int update)  {

    Real FASTOR_ALIGN ParentGradientX[ndim*ndim];
    Real FASTOR_ALIGN invParentGradientX[ndim*ndim];
    Real FASTOR_ALIGN ParentGradientx[ndim*ndim];
    Real FASTOR_ALIGN invParentGradientx[ndim*ndim];
    Real FASTOR_ALIGN current_Ft[ndim*ndim];

    _matmul_(ndim,ndim,nodeperelem,current_Jm,LagrangeElemCoords_,ParentGradientX);
    _matmul_(ndim,ndim,nodeperelem,current_Jm,EulerElemCoords_,ParentGradientx);

    const Real detX = invdet3x3(ParentGradientX,invParentGradientX);
    const Real detx = invdet3x3(ParentGradientx,invParentGradientx);

    detJ = update==1 ? AllGauss_*fabs(detx) : AllGauss_*fabs(detX);

    _matmul_(ndim,nodeperelem,ndim,invParentGradientX,current_Jm,MaterialGradient);
    _matmul_(ndim,nodeperelem,ndim,invParentGradientx,current_Jm,current_sp);

    // Compute deformation gradient F
    _matmul_(ndim,ndim,nodeperelem,MaterialGradient,EulerElemCoords_,current_Ft);
    Fastor::_transpose<Real,ndim,ndim>(current_Ft,current_F);
}
// // //




template<Integer ndim>
void _GlobalAssemblyExplicit_DF_DPF_(const Real *points,
                        const UInteger* elements,
                        const Real* Eulerx,
                        const Real* Eulerp,
                        const Real* bases,
                        const Real* Jm,
                        const Real* AllGauss,
                        Integer nvar,
                        Integer ngauss,
                        Integer nelem,
                        Integer nodeperelem,
                        Integer nnode,
                        Integer H_VoigtSize,
                        Integer requires_geometry_update,
                        Real *T,
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
                        );



template<>
void _GlobalAssemblyExplicit_DF_DPF_<2>(const Real *points,
                        const UInteger* elements,
                        const Real* Eulerx,
                        const Real* Eulerp,
                        const Real* bases,
                        const Real* Jm,
                        const Real* AllGauss,
                        Integer nvar,
                        Integer ngauss,
                        Integer nelem,
                        Integer nodeperelem,
                        Integer nnode,
                        Integer H_VoigtSize,
                        Integer requires_geometry_update,
                        Real *T,
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

    constexpr Integer ndim = 2;
    Integer ndof = nvar*nodeperelem;

    Real *LagrangeElemCoords        = allocate<Real>(nodeperelem*ndim);
    Real *EulerElemCoords           = allocate<Real>(nodeperelem*ndim);
    Real *ElectricPotentialElem     = allocate<Real>(nodeperelem);

    Real *current_Jm                = allocate<Real>(nodeperelem*ndim);
    Real *MaterialGradient          = allocate<Real>(ndim*nodeperelem);
    Real *SpatialGradient           = allocate<Real>(nodeperelem*ndim);
    Real detJ                       = 0;

    Real FASTOR_ALIGN F[ndim*ndim];
    Real FASTOR_ALIGN ElectricFieldx[ndim];
    Tensor<Real,ndim> D;
    Tensor<Real,ndim,ndim> stress;

    Real *local_traction            = allocate<Real>(ndof);
    Real *traction                  = allocate<Real>(ndof);


    auto mat_obj0 = _ExplicitMooneyRivlin_<Real>(mu1,mu2,lamb);
    auto mat_obj1 = _NeoHookean_<Real>(mu,lamb);
    auto mat_obj2 = _MooneyRivlin_<Real>(mu1,mu2,lamb);
    auto mat_obj3 = _NearlyIncompressibleMooneyRivlin_<Real>(mu1,mu2,mu3);
    auto mat_obj4 = _IsotropicElectroMechanics_101_<Real>(mu,lamb,eps_1);
    auto mat_obj5 = _IsotropicElectroMechanics_105_<Real>(mu1,mu2,lamb,eps_1,eps_2);
    auto mat_obj6 = _IsotropicElectroMechanics_106_<Real>(mu1,mu2,lamb,eps_1,eps_2);
    auto mat_obj7 = _IsotropicElectroMechanics_107_<Real>(mu1,mu2,mue,lamb,eps_1,eps_2,eps_e);
    auto mat_obj8 = _IsotropicElectroMechanics_108_<Real>(mu1,mu2,lamb,eps_2);
    auto mat_obj9 = _ExplicitIsotropicElectroMechanics_108_<Real>(mu1,mu2,lamb,eps_2);
    auto mat_obj10 = _LinearElastic_<Real>(mu,lamb);


    // LOOP OVER ELEMETNS
    for (Integer elem=0; elem < nelem; ++elem) {

        // GET THE FIELDS AT THE ELEMENT LEVEL
        for (Integer i=0; i<nodeperelem; ++i) {
            const Integer inode = elements[elem*nodeperelem+i];

            LagrangeElemCoords[i*2+0] = points[inode*2+0];
            LagrangeElemCoords[i*2+1] = points[inode*2+1];

            EulerElemCoords[i*2+0] = Eulerx[inode*2+0];
            EulerElemCoords[i*2+1] = Eulerx[inode*2+1];

            ElectricPotentialElem[i] = Eulerp[inode];
        }

        std::fill(traction,traction+ndof,0.);

        for (int g=0; g<ngauss; ++g) {

            for (int j=0; j<nodeperelem; ++j) {
                current_Jm[j] = Jm[j*ngauss+g];
                current_Jm[nodeperelem+j] = Jm[ngauss*nodeperelem+j*ngauss+g];
            }


            // COMPUTE KINEMATIC MEASURES
            std::fill(F,F+ndim*ndim,0.);
            std::fill(MaterialGradient,MaterialGradient+nodeperelem*ndim,0.);
            std::fill(SpatialGradient,SpatialGradient+nodeperelem*ndim,0.);

            KinematicMeasures__<2>(  MaterialGradient,
                                    SpatialGradient,
                                    F,
                                    detJ,
                                    current_Jm,
                                    AllGauss[g],
                                    LagrangeElemCoords,
                                    EulerElemCoords,
                                    ngauss,
                                    nodeperelem,
                                    requires_geometry_update
                                    );


            // COMPUTE ELECTRIC FIELD
            Real iE0 = 0, iE1 = 0;
            for (Integer j=0; j<nodeperelem; ++j) {
                const Real potE = ElectricPotentialElem[j];
                iE0 += SpatialGradient[j]*potE;
                iE1 += SpatialGradient[nodeperelem+j]*potE;
            }
            ElectricFieldx[0] = -iE0;
            ElectricFieldx[1] = -iE1;


            // COMPUTE KINETIC MEASURES
            if (material_number==0) {
                stress = mat_obj0.template _KineticMeasures_<Real,ndim>(F);
            }
            else if (material_number==9) {
                std::tie(D,stress) = mat_obj9.template _KineticMeasures_<Real,ndim>(F,ElectricFieldx);
            }
            else if (material_number==1) {
                std::tie(stress,std::ignore) = mat_obj1.template _KineticMeasures_<Real,ndim>(F);
            }
            else if (material_number==2) {
                std::tie(stress,std::ignore) = mat_obj2.template _KineticMeasures_<Real,ndim>(F);
            }
            else if (material_number==3) {
                std::tie(stress,std::ignore) = mat_obj3.template _KineticMeasures_<Real,ndim>(F);
            }
            else if (material_number==4) {
                std::tie(D,stress,std::ignore) = mat_obj4.template _KineticMeasures_<Real,ndim>(F,ElectricFieldx);
            }
            else if (material_number==5) {
                std::tie(D,stress,std::ignore) = mat_obj5.template _KineticMeasures_<Real,ndim>(F,ElectricFieldx);
            }
            else if (material_number==6) {
                std::tie(D,stress,std::ignore) = mat_obj6.template _KineticMeasures_<Real,ndim>(F,ElectricFieldx);
            }
            else if (material_number==7) {
                std::tie(D,stress,std::ignore) = mat_obj7.template _KineticMeasures_<Real,ndim>(F,ElectricFieldx);
            }
            else if (material_number==8) {
                std::tie(D,stress,std::ignore) = mat_obj8.template _KineticMeasures_<Real,ndim>(F,ElectricFieldx);
            }
            else if (material_number==10) {
                std::tie(stress,std::ignore) = mat_obj10.template _KineticMeasures_<Real,ndim>(F);
            }


            // COMPUTE TRACTION
            if (formulation_number == 0) {

                const Real s11 = stress(0,0);
                const Real s12 = stress(0,1);
                const Real s22 = stress(1,1);

                for (int i=0; i<nodeperelem; ++i) {
                    const Real a0 = SpatialGradient[i];
                    const Real a1 = SpatialGradient[nodeperelem+i];
                    local_traction[2*i]   = a0*s11 + a1*s12;
                    local_traction[2*i+1] = a0*s12 + a1*s22;
                }
            }
            else if (formulation_number == 1) {

                const Real s11 = stress(0,0);
                const Real s12 = stress(0,1);
                const Real s22 = stress(1,1);

                const Real d1  = D(0);
                const Real d2  = D(1);

                for (int i=0; i<nodeperelem; ++i) {
                    const Real a0 = SpatialGradient[i];
                    const Real a1 = SpatialGradient[nodeperelem+i];
                    local_traction[3*i]   = a0*s11 + a1*s12;
                    local_traction[3*i+1] = a0*s12 + a1*s22;
                    local_traction[3*i+2] = a0*d1 + a1*d2;
                }
            }

            for (int i=0; i<ndof; ++i) {
                traction[i] += local_traction[i]*detJ;
            }
        }


        // ASSEMBLE TRACTIONS
        for (Integer i = 0; i<nodeperelem; ++i) {
            UInteger T_idx = elements[elem*nodeperelem+i]*nvar;
            for (Integer iterator = 0; iterator < nvar; ++iterator) {
                T[T_idx+iterator] += traction[i*nvar+iterator];
            }
        }

    }


    deallocate(LagrangeElemCoords);
    deallocate(EulerElemCoords);
    deallocate(ElectricPotentialElem);
    deallocate(current_Jm);
    deallocate(MaterialGradient);
    deallocate(SpatialGradient);
    deallocate(local_traction);
    deallocate(traction);
}



template<>
void _GlobalAssemblyExplicit_DF_DPF_<3>(const Real *points,
                        const UInteger* elements,
                        const Real* Eulerx,
                        const Real* Eulerp,
                        const Real* bases,
                        const Real* Jm,
                        const Real* AllGauss,
                        Integer nvar,
                        Integer ngauss,
                        Integer nelem,
                        Integer nodeperelem,
                        Integer nnode,
                        Integer H_VoigtSize,
                        Integer requires_geometry_update,
                        Real *T,
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

    constexpr Integer ndim = 3;
    Integer ndof = nvar*nodeperelem;

    Real *LagrangeElemCoords        = allocate<Real>(nodeperelem*ndim);
    Real *EulerElemCoords           = allocate<Real>(nodeperelem*ndim);
    Real *ElectricPotentialElem     = allocate<Real>(nodeperelem);

    Real *current_Jm                = allocate<Real>(nodeperelem*ndim);
    Real *MaterialGradient          = allocate<Real>(ndim*nodeperelem);
    Real *SpatialGradient           = allocate<Real>(nodeperelem*ndim);
    Real detJ                       = 0;

    Real FASTOR_ALIGN F[ndim*ndim];
    Real FASTOR_ALIGN ElectricFieldx[ndim];
    Tensor<Real,ndim> D;
    Tensor<Real,ndim,ndim> stress;

    Real *local_traction            = allocate<Real>(ndof);
    Real *traction                  = allocate<Real>(ndof);


    auto mat_obj0 = _ExplicitMooneyRivlin_<Real>(mu1,mu2,lamb);
    auto mat_obj1 = _NeoHookean_<Real>(mu,lamb);
    auto mat_obj2 = _MooneyRivlin_<Real>(mu1,mu2,lamb);
    auto mat_obj3 = _NearlyIncompressibleMooneyRivlin_<Real>(mu1,mu2,mu3);
    auto mat_obj4 = _IsotropicElectroMechanics_101_<Real>(mu,lamb,eps_1);
    auto mat_obj5 = _IsotropicElectroMechanics_105_<Real>(mu1,mu2,lamb,eps_1,eps_2);
    auto mat_obj6 = _IsotropicElectroMechanics_106_<Real>(mu1,mu2,lamb,eps_1,eps_2);
    auto mat_obj7 = _IsotropicElectroMechanics_107_<Real>(mu1,mu2,mue,lamb,eps_1,eps_2,eps_e);
    auto mat_obj8 = _IsotropicElectroMechanics_108_<Real>(mu1,mu2,lamb,eps_2);
    auto mat_obj9 = _ExplicitIsotropicElectroMechanics_108_<Real>(mu1,mu2,lamb,eps_2);
    auto mat_obj10 = _LinearElastic_<Real>(mu,lamb);


    // LOOP OVER ELEMETNS
    for (Integer elem=0; elem < nelem; ++elem) {

        // GET THE FIELDS AT THE ELEMENT LEVEL
        for (Integer i=0; i<nodeperelem; ++i) {
            const Integer inode = elements[elem*nodeperelem+i];

            LagrangeElemCoords[i*3+0] = points[inode*3+0];
            LagrangeElemCoords[i*3+1] = points[inode*3+1];
            LagrangeElemCoords[i*3+2] = points[inode*3+2];

            EulerElemCoords[i*3+0] = Eulerx[inode*3+0];
            EulerElemCoords[i*3+1] = Eulerx[inode*3+1];
            EulerElemCoords[i*3+2] = Eulerx[inode*3+2];

            ElectricPotentialElem[i] = Eulerp[inode];
        }

        std::fill(traction,traction+ndof,0.);

        for (int g=0; g<ngauss; ++g) {

            for (int j=0; j<nodeperelem; ++j) {
                current_Jm[j] = Jm[j*ngauss+g];
                current_Jm[nodeperelem+j] = Jm[ngauss*nodeperelem+j*ngauss+g];
                current_Jm[2*nodeperelem+j] = Jm[2*ngauss*nodeperelem+j*ngauss+g];
            }


            // COMPUTE KINEMATIC MEASURES
            std::fill(F,F+ndim*ndim,0.);
            std::fill(MaterialGradient,MaterialGradient+nodeperelem*ndim,0.);
            std::fill(SpatialGradient,SpatialGradient+nodeperelem*ndim,0.);

            KinematicMeasures__<3>(  MaterialGradient,
                                    SpatialGradient,
                                    F,
                                    detJ,
                                    current_Jm,
                                    AllGauss[g],
                                    LagrangeElemCoords,
                                    EulerElemCoords,
                                    ngauss,
                                    nodeperelem,
                                    requires_geometry_update
                                    );


            // COMPUTE ELECTRIC FIELD
            Real iE0 = 0, iE1 = 0, iE2 = 0;
            for (Integer j=0; j<nodeperelem; ++j) {
                const Real potE = ElectricPotentialElem[j];
                iE0 += SpatialGradient[j]*potE;
                iE1 += SpatialGradient[nodeperelem+j]*potE;
                iE2 += SpatialGradient[2*nodeperelem+j]*potE;
            }
            ElectricFieldx[0] = -iE0;
            ElectricFieldx[1] = -iE1;
            ElectricFieldx[2] = -iE2;


            // COMPUTE KINETIC MEASURES
            if (material_number==0) {
                stress = mat_obj0.template _KineticMeasures_<Real,ndim>(F);
            }
            else if (material_number==9) {
                std::tie(D,stress) = mat_obj9.template _KineticMeasures_<Real,ndim>(F,ElectricFieldx);
            }
            else if (material_number==1) {
                std::tie(stress,std::ignore) = mat_obj1.template _KineticMeasures_<Real,ndim>(F);
            }
            else if (material_number==2) {
                std::tie(stress,std::ignore) = mat_obj2.template _KineticMeasures_<Real,ndim>(F);
            }
            else if (material_number==3) {
                std::tie(stress,std::ignore) = mat_obj3.template _KineticMeasures_<Real,ndim>(F);
            }
            else if (material_number==4) {
                std::tie(D,stress,std::ignore) = mat_obj4.template _KineticMeasures_<Real,ndim>(F,ElectricFieldx);
            }
            else if (material_number==5) {
                std::tie(D,stress,std::ignore) = mat_obj5.template _KineticMeasures_<Real,ndim>(F,ElectricFieldx);
            }
            else if (material_number==6) {
                std::tie(D,stress,std::ignore) = mat_obj6.template _KineticMeasures_<Real,ndim>(F,ElectricFieldx);
            }
            else if (material_number==7) {
                std::tie(D,stress,std::ignore) = mat_obj7.template _KineticMeasures_<Real,ndim>(F,ElectricFieldx);
            }
            else if (material_number==8) {
                std::tie(D,stress,std::ignore) = mat_obj8.template _KineticMeasures_<Real,ndim>(F,ElectricFieldx);
            }
            else if (material_number==10) {
                std::tie(stress,std::ignore) = mat_obj10.template _KineticMeasures_<Real,ndim>(F);
            }


            if (formulation_number == 0) {

                const Real s11 = stress(0,0);
                const Real s12 = stress(0,1);
                const Real s13 = stress(0,2);
                const Real s22 = stress(1,1);
                const Real s23 = stress(1,2);
                const Real s33 = stress(2,2);

                for (int i=0; i<nodeperelem; ++i) {

                    const Real a0 = SpatialGradient[i];
                    const Real a1 = SpatialGradient[nodeperelem+i];
                    const Real a2 = SpatialGradient[2*nodeperelem+i];

                    local_traction[3*i]   = a0*s11 + a1*s12 + a2*s13;
                    local_traction[3*i+1] = a0*s12 + a1*s22 + a2*s23;
                    local_traction[3*i+2] = a0*s13 + a1*s23 + a2*s33;
                }
            }
            else if (formulation_number == 1) {

                const Real s11 = stress(0,0);
                const Real s12 = stress(0,1);
                const Real s13 = stress(0,2);
                const Real s22 = stress(1,1);
                const Real s23 = stress(1,2);
                const Real s33 = stress(2,2);

                const Real d1  = D(0);
                const Real d2  = D(1);
                const Real d3  = D(2);

                for (int i=0; i<nodeperelem; ++i) {

                    const Real a0 = SpatialGradient[i];
                    const Real a1 = SpatialGradient[nodeperelem+i];
                    const Real a2 = SpatialGradient[2*nodeperelem+i];

                    local_traction[4*i]   = a0*s11 + a1*s12 + a2*s13;
                    local_traction[4*i+1] = a0*s12 + a1*s22 + a2*s23;
                    local_traction[4*i+2] = a0*s13 + a1*s23 + a2*s33;
                    local_traction[4*i+3] = a0*d1 + a1*d2 + a2*d3;
                }
            }

            for (int i=0; i<ndof; ++i) {
                traction[i] += local_traction[i]*detJ;
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

    deallocate(current_Jm);
    deallocate(LagrangeElemCoords);
    deallocate(EulerElemCoords);
    deallocate(ElectricPotentialElem);
    deallocate(MaterialGradient);
    deallocate(SpatialGradient);
    deallocate(local_traction);
    deallocate(traction);
}










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

    if (ndim==3) {
        _GlobalAssemblyExplicit_DF_DPF_<3>(points,
                        elements,
                        Eulerx,
                        Eulerp,
                        bases,
                        Jm,
                        AllGauss,
                        nvar,
                        ngauss,
                        nelem,
                        nodeperelem,
                        nnode,
                        H_VoigtSize,
                        requires_geometry_update,
                        T,
                        mu,
                        mu1,
                        mu2,
                        mu3,
                        mue,
                        lamb,
                        eps_1,
                        eps_2,
                        eps_3,
                        eps_e,
                        anisotropic_orientations,
                        material_number,
                        formulation_number
                        );
    }
    else {

        _GlobalAssemblyExplicit_DF_DPF_<2>(points,
                        elements,
                        Eulerx,
                        Eulerp,
                        bases,
                        Jm,
                        AllGauss,
                        nvar,
                        ngauss,
                        nelem,
                        nodeperelem,
                        nnode,
                        H_VoigtSize,
                        requires_geometry_update,
                        T,
                        mu,
                        mu1,
                        mu2,
                        mu3,
                        mue,
                        lamb,
                        eps_1,
                        eps_2,
                        eps_3,
                        eps_e,
                        anisotropic_orientations,
                        material_number,
                        formulation_number
                        );

    }
}





#else





// COMPATIBLE TO ASSEMBLY FOR IMPLICIT ROUTINES - WELL TESTED
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


    auto mat_obj0 = _NeoHookean_<Real>(mu,lamb);
    auto mat_obj1 = _MooneyRivlin_<Real>(mu1,mu2,lamb);
    auto mat_obj2 = _ExplicitMooneyRivlin_<Real>(mu1,mu2,lamb);
    auto mat_obj3 = _IsotropicElectroMechanics_101_<Real>(mu,lamb,eps_1);
    auto mat_obj4 = _IsotropicElectroMechanics_108_<Real>(mu1,mu2,lamb,eps_2);


    // LOOP OVER ELEMETNS
    for (Integer elem=0; elem < nelem; ++elem) {

        // GET THE FIELDS AT THE ELEMENT LEVEL

        if (ndim==3) {
            for (Integer i=0; i<nodeperelem; ++i) {
                const Integer inode = elements[elem*nodeperelem+i];

                LagrangeElemCoords[i*3+0] = points[inode*3+0];
                LagrangeElemCoords[i*3+1] = points[inode*3+1];
                LagrangeElemCoords[i*3+2] = points[inode*3+2];

                EulerElemCoords[i*3+0] = Eulerx[inode*3+0];
                EulerElemCoords[i*3+1] = Eulerx[inode*3+1];
                EulerElemCoords[i*3+2] = Eulerx[inode*3+2];

                ElectricPotentialElem[i] = Eulerp[inode];
            }
        }
        else {
            for (Integer i=0; i<nodeperelem; ++i) {
                const Integer inode = elements[elem*nodeperelem+i];

                LagrangeElemCoords[i*2+0] = points[inode*2+0];
                LagrangeElemCoords[i*2+1] = points[inode*2+1];

                EulerElemCoords[i*2+0] = Eulerx[inode*2+0];
                EulerElemCoords[i*2+1] = Eulerx[inode*2+1];

                ElectricPotentialElem[i] = Eulerp[inode];
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

        // COMPUTE ELECTRIC FIELD
        if (ndim==3) {
            for (Integer i=0; i<ngauss; ++i) {
                Real iE0 = 0, iE1 = 0, iE2 = 0;
                for (Integer j=0; j<nodeperelem; ++j) {
                    const Real potE = ElectricPotentialElem[j];
                    iE0 += SpatialGradient[i*nodeperelem*3+j*3+0]*potE;
                    iE1 += SpatialGradient[i*nodeperelem*3+j*3+1]*potE;
                    iE2 += SpatialGradient[i*nodeperelem*3+j*3+2]*potE;
                }
                ElectricFieldx[i*3+0] = -iE0;
                ElectricFieldx[i*3+1] = -iE0;
                ElectricFieldx[i*3+2] = -iE0;
            }
        }
        else {
            for (Integer i=0; i<ngauss; ++i) {
                Real iE0 = 0, iE1 = 0;
                for (Integer j=0; j<nodeperelem; ++j) {
                    const Real potE = ElectricPotentialElem[j];
                    iE0 += SpatialGradient[i*nodeperelem*2+j*2+0]*potE;
                    iE1 += SpatialGradient[i*nodeperelem*2+j*2+1]*potE;
                }
                ElectricFieldx[i*2+0] = -iE0;
                ElectricFieldx[i*2+1] = -iE0;
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

// TURN THIS OFF FOR THE TIME BEING
// AS A MORE EFFICIENT PYTHON VERSION IS AVAILABLE
#if 0
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
#endif


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

}

#endif


#endif // _LOWLEVELASSEMBLYDPF__H








// TWO GENERATION OLD - ORIGINAL NON-UNROLLED VERSION


// #ifndef _LOWLEVELASSEMBLYDPF__H
// #define _LOWLEVELASSEMBLYDPF__H

// #include "assembly_helper.h"
// #include "_TractionDF_.h"
// #include "_TractionDPF_.h"

// #include "_NeoHookean_.h"
// #include "_MooneyRivlin_.h"
// #include "_ExplicitMooneyRivlin_.h"
// #include "_IsotropicElectroMechanics_101_.h"
// #include "_IsotropicElectroMechanics_108_.h"

// void _GlobalAssemblyExplicit_DF_DPF_(const Real *points,
//                         const UInteger* elements,
//                         const Real* Eulerx,
//                         const Real* Eulerp,
//                         const Real* bases,
//                         const Real* Jm,
//                         const Real* AllGauss,
//                         Integer ndim,
//                         Integer nvar,
//                         Integer ngauss,
//                         Integer nelem,
//                         Integer nodeperelem,
//                         Integer nnode,
//                         Integer H_VoigtSize,
//                         Integer requires_geometry_update,
//                         Real *T,
//                         Integer is_dynamic,
//                         Integer* local_rows_mass,
//                         Integer* local_cols_mass,
//                         int *I_mass,
//                         int *J_mass,
//                         Real *V_mass,
//                         Real rho,
//                         Real mu,
//                         Real mu1,
//                         Real mu2,
//                         Real mu3,
//                         Real mue,
//                         Real lamb,
//                         Real eps_1,
//                         Real eps_2,
//                         Real eps_3,
//                         Real eps_e,
//                         const Real *anisotropic_orientations,
//                         int material_number,
//                         int formulation_number
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
//     Real *mass                      = allocate<Real>(local_capacity);


//     auto mat_obj0 = _NeoHookean_<Real>(mu,lamb);
//     auto mat_obj1 = _MooneyRivlin_<Real>(mu1,mu2,lamb);
//     auto mat_obj2 = _ExplicitMooneyRivlin_<Real>(mu1,mu2,lamb);
//     auto mat_obj3 = _IsotropicElectroMechanics_101_<Real>(mu,lamb,eps_1);
//     auto mat_obj4 = _IsotropicElectroMechanics_108_<Real>(mu1,mu2,lamb,eps_2);


//     // LOOP OVER ELEMETNS
//     for (Integer elem=0; elem < nelem; ++elem) {

//         // GET THE FIELDS AT THE ELEMENT LEVEL
//         for (Integer i=0; i<nodeperelem; ++i) {
//             const Integer inode = elements[elem*nodeperelem+i];
//             for (Integer j=0; j<ndim; ++j) {
//                 LagrangeElemCoords[i*ndim+j] = points[inode*ndim+j];
//                 EulerElemCoords[i*ndim+j] = Eulerx[inode*ndim+j];
//             }
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
//         if (material_number==0) {
//             mat_obj0.KineticMeasures(stress, hessian, ndim, ngauss, F);
//         }
//         else if (material_number==1) {
//             mat_obj1.KineticMeasures(stress, hessian, ndim, ngauss, F);
//         }
//         else if (material_number==2) {
//             mat_obj2.KineticMeasures(stress, ndim, ngauss, F);
//         }
//         else if (material_number==3) {
//             mat_obj3.KineticMeasures(D, stress, hessian, ndim, ngauss, F, ElectricFieldx);
//         }
//         else if (material_number==4) {
//             mat_obj4.KineticMeasures(D, stress, hessian, ndim, ngauss, F, ElectricFieldx);
//         }

//         // COMPUTE CONSTITUTIVE STIFFNESS AND TRACTION
//         std::fill(traction,traction+ndof,0.);
//         if (formulation_number==1) {
//             _TractionDPF_Filler_(
//                 traction,
//                 SpatialGradient,
//                 D,
//                 stress,
//                 detJ,
//                 ngauss,
//                 nodeperelem,
//                 ndim,
//                 nvar,
//                 H_VoigtSize,
//                 requires_geometry_update);
//         }
//         else {
//             _TractionDF_Filler_(
//                 traction,
//                 SpatialGradient,
//                 stress,
//                 detJ,
//                 ngauss,
//                 nodeperelem,
//                 ndim,
//                 nvar,
//                 H_VoigtSize,
//                 requires_geometry_update);
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
//     deallocate(mass);

// }


// #endif // _LOWLEVELASSEMBLYDPF__H