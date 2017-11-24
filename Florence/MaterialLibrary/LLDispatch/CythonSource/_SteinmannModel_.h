#include "_MaterialBase_.h"
#include "_LegendreTransform_.h"

template<typename U>
class _SteinmannModel_ : public _MaterialBase_<U> {
public:
    U mu;
    U lamb;
    U c1;
    U c2;
    U eps_1;

    _SteinmannModel_() = default;

    FASTOR_INLINE
    _SteinmannModel_(U mu, U lamb, U c1, U c2, U eps_1) {
        this->mu = mu;
        this->lamb = lamb;
        this->c1 = c1;
        this->c2 = c2;
        this->eps_1 = eps_1;
    }

    FASTOR_INLINE
    void SetParameters(U mu, U lamb, U c1, U c2, U eps_1) {
        this->mu = mu;
        this->lamb = lamb;
        this->c1 = c1;
        this->c2 = c2;
        this->eps_1 = eps_1;
    }


    template<typename T=U, size_t ndim>
    FASTOR_INLINE
    std::tuple<Tensor<T,ndim>,Tensor<T,ndim,ndim>, typename ElectroMechanicsHessianType<T,ndim>::return_type>
    _KineticMeasures_(const T *Fnp, const T *Enp) {

        // CREATE FASTOR TENSORS
        Tensor<T,ndim,ndim> F;
        Tensor<T,ndim> E;
        // COPY NUMPY ARRAY TO FASTOR TENSORS
        copy_numpy(F,Fnp);
        copy_numpy(E,Enp);

        // FIND THE KINEMATIC MEASURES
        Tensor<Real,ndim,ndim> I; I.eye2();
        auto J = determinant(F);
        auto b = matmul(F,transpose(F));

        auto bE = matmul(b,E);
        // auto innerbE = inner(bE,bE); // unused check
        auto outerbE = outer(bE,bE);
        // auto EbE = inner(E,bE); // unused check
        auto bb = matmul(b,b);

        // COMPUTE ELECTRIC DISPLACEMENT
        Tensor<T,ndim> D = eps_1*E - 2.*c1/J*bE - 2.*c2/J*matmul(bb,E);

        auto innerEE = inner(E,E);
        auto outerEE = outer(E,E);

        auto varcoeff = std::log(J);

        Tensor<T,ndim,ndim> sigma = mu/J*(b-I)  + lamb/J*varcoeff*I + \
            eps_1*(outerEE-0.5*innerEE*I) + 2.0*c2/J*outerbE;


        // FIND ELASTICITY TENSOR
        auto II_ijkl = einsum<Index<i,j>,Index<k,l>>(I,I);
        auto II_ikjl = permutation<Index<i,k,j,l>>(II_ijkl);
        auto II_iljk = permutation<Index<i,l,j,k>>(II_ijkl);

        auto IEE_ijkl = einsum<Index<i,j>,Index<k,l>>(I,outerEE);
        auto EEI_ijkl = einsum<Index<i,j>,Index<k,l>>(outerEE,I);
        auto IEE_ikjl = permutation<Index<i,k,j,l>>(IEE_ijkl);
        auto IEE_iljk = permutation<Index<i,l,j,k>>(IEE_ijkl);
        auto EEI_ikjl = permutation<Index<i,k,j,l>>(EEI_ijkl);
        auto EEI_iljk = permutation<Index<i,l,j,k>>(EEI_ijkl);

        Tensor<T,ndim,ndim,ndim,ndim> elasticity = lamb/J*II_ijkl - (lamb*varcoeff - mu)/J*(II_ikjl+II_iljk) +\
            eps_1*(IEE_ijkl + EEI_ijkl - EEI_ikjl - EEI_iljk - IEE_ikjl - IEE_iljk ) +\
            eps_1*innerEE*(0.5*(II_ikjl + II_iljk - II_ijkl));

        // FIND COUPLING TENSOR
        auto IE_ijk = einsum<Index<i,j>,Index<k>>(I,E);
        auto IE_ikj = permutation<Index<i,k,j>>(IE_ijk);
        auto IE_jki = permutation<Index<j,k,i>>(IE_ijk);
        auto bE_ijk = einsum<Index<i,j>,Index<k>>(b,bE);
        auto bE_ikj = permutation<Index<i,k,j>>(bE_ijk);
        auto bE_jki = permutation<Index<j,k,i>>(bE_ijk);

        Tensor<T,ndim,ndim,ndim> coupling =  eps_1*(IE_ikj + IE_jki - IE_ijk) +\
            2.0*c2/J*(bE_ikj + bE_jki);

        // FIND DIELELCTRIC TENSOR
        Tensor<T,ndim,ndim> dielectric = - eps_1*I +  2.0*c1/J*b + 2.0*c2/J*bb;

        auto hessian = make_electromechanical_hessian(voigt(elasticity),voigt(coupling),dielectric);

        auto kinetics = std::make_tuple(D,sigma,hessian);
        return kinetics;
    }



    template<typename T>
    void KineticMeasures(T* Dnp, T *Snp, T* Hnp, int ndim, int ngauss, const T *Fnp, const T *Enp);

};

template<> template<>
void _SteinmannModel_<Real>::KineticMeasures<Real>(Real *Dnp, Real *Snp, Real* Hnp,
    int ndim, int ngauss, const Real *Fnp, const Real *Enp) {

    if (ndim==3) {
        Tensor<Real,3> D;
        Tensor<Real,3,3> stress;
        Tensor<Real,9,9> hessian;
        for (int g=0; g<ngauss; ++g) {
            std::tie(D,stress,hessian) =_KineticMeasures_<Real,3>(Fnp+9*g, Enp+3*g);
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
            std::tie(D,stress,hessian) =_KineticMeasures_<Real,2>(Fnp+4*g, Enp+2*g);
            copy_fastor(Dnp,D,g*2);
            copy_fastor(Snp,stress,g*4);
            copy_fastor(Hnp,hessian,g*25);
        }
    }
}
