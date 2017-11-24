#include "_MaterialBase_.h"
#include "_LegendreTransform_.h"

template<typename U>
class _Piezoelectric_100_ : public _MaterialBase_<U> {
public:
    U mu1;
    U mu2;
    U mu3;
    U lamb;
    U eps_1;
    U eps_2;
    U eps_3;

    _Piezoelectric_100_() = default;

    FASTOR_INLINE
    _Piezoelectric_100_(U mu1, U mu2, U mu3, U lamb, U eps_1, U eps_2, U eps_3) {
        this->mu1 = mu1;
        this->mu2 = mu2;
        this->mu3 = mu3;
        this->lamb = lamb;
        this->eps_1 = eps_1;
        this->eps_2 = eps_2;
        this->eps_3 = eps_3;
    }

    FASTOR_INLINE
    void SetParameters(U mu1, U mu2, U mu3, U lamb, U eps_1, U eps_2, U eps_3){
        this->mu1 = mu1;
        this->mu2 = mu2;
        this->mu3 = mu3;
        this->lamb = lamb;
        this->eps_1 = eps_1;
        this->eps_2 = eps_2;
        this->eps_3 = eps_3;
    }


    template<typename T=U, size_t ndim>
    FASTOR_INLINE
    std::tuple<Tensor<T,ndim>,Tensor<T,ndim,ndim>, typename ElectroMechanicsHessianType<T,ndim>::return_type>
    _KineticMeasures_(const T *Fnp, const T *Enp, const T *Nnp) {

        auto coeff = sqrt(mu3/eps_3);

        // CREATE FASTOR TENSORS
        Tensor<T,ndim,ndim> F;
        Tensor<T,ndim> E;
        Tensor<T,ndim> N;
        // COPY NUMPY ARRAY TO FASTOR TENSORS
        copy_numpy(F,Fnp);
        copy_numpy(E,Enp);
        copy_numpy(N,Nnp);

        // FIND THE KINEMATIC MEASURES
        Tensor<Real,ndim,ndim> I; I.eye2();
        auto J = determinant(F);
        auto H = cofactor(F);
        auto b = matmul(F,transpose(F));
        auto FN = matmul(F,N);
        auto HN = matmul(H,N);
        auto outerFN = outer(FN,FN);
        auto outerHN = outer(HN,HN);
        auto innerHN = inner(HN,HN);

        // COMPUTE ELECTRIC DISPLACEMENT
        auto inv = inverse(static_cast<decltype(b)>(J/eps_1*inverse(b) + 1./eps_2*I + 2.*J*coeff*I));
        auto D = matmul(inv, static_cast<decltype(E)>(E - 2.*coeff*FN + 2./J*coeff*HN) );

        auto innerDD = inner(D,D);
        auto outerDD = outer(D,D);
        auto DFN = outer(D,FN);
        auto FND = transpose(DFN);

        // COMPUTE CAUCHY STRESS TENSOR
        T trb = trace(b);
        if (ndim == 2) {
            trb += 1.;
        }

        Tensor<T,ndim,ndim> sigma_mech = 2.*mu1/J*b + \
            2.*mu2/J*(trb*b - matmul(b,b)) - \
            2.*(mu1+2*mu2+mu3)/J*I + \
            lamb*(J-1)*I +\
            2*mu3/J*outerFN +\
            2*mu3/J*innerHN*I - 2*mu3/J*outerHN;

        Tensor<T,ndim,ndim> sigma_electric = 1./eps_2*(outerDD - 0.5*innerDD*I) +\
            2.*J*coeff*outerDD + 2.*coeff*( DFN + FND );

        Tensor<T,ndim,ndim> sigma = sigma_mech + sigma_electric;

        // FIND ELASTICITY TENSOR
        auto II_ijkl = einsum<Index<i,j>,Index<k,l>>(I,I);
        auto II_ikjl = permutation<Index<i,k,j,l>>(II_ijkl);
        auto II_iljk = permutation<Index<i,l,j,k>>(II_ijkl);

        auto bb_ijkl = einsum<Index<i,j>,Index<k,l>>(b,b);
        auto bb_ikjl = permutation<Index<i,k,j,l>>(bb_ijkl);
        auto bb_iljk = permutation<Index<i,l,j,k>>(bb_ijkl);

        auto IHN_ijkl = einsum<Index<i,j>,Index<k,l>>(I,outerHN);
        auto HNI_ijkl = einsum<Index<i,j>,Index<k,l>>(outerHN,I);

        auto IHN_ikjl = permutation<Index<i,k,j,l>>(IHN_ijkl);
        auto IHN_iljk = permutation<Index<i,l,j,k>>(IHN_ijkl);
        auto IHN_jlik = permutation<Index<j,l,i,k>>(IHN_ijkl);
        auto IHN_jkil = permutation<Index<j,k,i,l>>(IHN_ijkl);

        auto IDD_ijkl = einsum<Index<i,j>,Index<k,l>>(I,outerDD);
        auto DDI_ijkl = einsum<Index<i,j>,Index<k,l>>(outerDD,I);

        Tensor<T,ndim,ndim,ndim,ndim> C_mech = 2.0*mu2/J*(2.0*bb_ijkl - bb_ikjl - bb_iljk) + \
            (2.*(mu1+2*mu2+mu3)/J - lamb*(J-1.) ) * (II_ikjl + II_iljk) + lamb*(2.*J-1.)*II_ijkl -\
            4.*mu3/J*( IHN_ijkl + HNI_ijkl ) + \
            2.*mu3/J*innerHN*(2.0*II_ijkl - II_ikjl - II_iljk) +\
            2.*mu3/J * ( IHN_ikjl + IHN_iljk + IHN_jlik + IHN_jkil );

        Tensor<T,ndim,ndim,ndim,ndim> C_elect = 1./eps_2*(0.5*innerDD*( II_ijkl + II_ikjl + II_iljk) - \
                    IDD_ijkl - DDI_ijkl );

        Tensor<T,ndim,ndim,ndim,ndim> elasticity = C_mech + C_elect;

        // FIND COUPLING TENSOR
        auto ID_ijk = einsum<Index<i,j>,Index<k>>(I,D);
        auto ID_ikj = permutation<Index<i,k,j>>(ID_ijk);
        auto ID_jki = permutation<Index<j,k,i>>(ID_ijk);

        auto IFN_ijk = einsum<Index<i,j>,Index<k>>(I,FN);
        auto IFN_ikj = permutation<Index<i,k,j>>(IFN_ijk);
        auto IFN_jki = permutation<Index<j,k,i>>(IFN_ijk);

        Tensor<T,ndim,ndim,ndim> coupling =  1./eps_2*(ID_ikj + ID_jki - ID_ijk) + \
            2.*J*coeff*(ID_ikj + ID_jki) + \
            2.*coeff*(IFN_ikj + IFN_jki);

        // FIND DIELELCTRIC TENSOR
        Tensor<T,ndim,ndim> dielectric = J/eps_1*inverse(b)  + 1./eps_2*I + 2.*J*coeff*I;
        // PERFORM LEGENDRE TRANSFORM
        auto legendre_transform = _LegendreTransform_<T>();
        auto hessian = legendre_transform.InternalEnergyToEnthalpy(elasticity,coupling,dielectric);

        auto kinetics = std::make_tuple(D,sigma,hessian);
        return kinetics;
    }



    template<typename T>
    void KineticMeasures(T* Dnp, T *Snp, T* Hnp, int ndim, int ngauss, const T *Fnp, const T *Enp, const T *Nnp);

};

template<> template<>
void _Piezoelectric_100_<Real>::KineticMeasures<Real>(Real *Dnp, Real *Snp, Real* Hnp,
    int ndim, int ngauss, const Real *Fnp, const Real *Enp, const Real *Nnp) {

    if (ndim==3) {
        Tensor<Real,3> D;
        Tensor<Real,3,3> stress;
        Tensor<Real,9,9> hessian;
        for (int g=0; g<ngauss; ++g) {
            std::tie(D,stress,hessian) =_KineticMeasures_<Real,3>(Fnp+9*g, Enp+3*g, Nnp);
            copy_fastor(Dnp,D,g*3);
            copy_fastor(Snp,stress,g*9);
            copy_fastor(Hnp,hessian,g*81);
        }
    }
    else if (ndim==2) {
        Tensor<Real,2> D;
        Tensor<Real,2,2> stress;
        Tensor<Real,5,5> hessian;
        for (int g=0; g<ngauss; ++g) {
            std::tie(D,stress,hessian) =_KineticMeasures_<Real,2>(Fnp+4*g, Enp+2*g, Nnp);
            copy_fastor(Dnp,D,g*2);
            copy_fastor(Snp,stress,g*4);
            copy_fastor(Hnp,hessian,g*25);
        }
    }
}
