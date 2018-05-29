#include "_MaterialBase_.h"

template<typename U>
class _MooneyRivlin_ : public _MaterialBase_<U> {
public:
    U mu1;
    U mu2;
    U lamb;

    _MooneyRivlin_() = default;

    FASTOR_INLINE
    _MooneyRivlin_(U mu1, U mu2, U lamb) {
        this->mu1 = mu1;
        this->mu2 = mu2;
        this->lamb = lamb;
    }

    FASTOR_INLINE
    void SetParameters(U mu1, U mu2, U lamb){
        this->mu1 = mu1;
        this->mu2 = mu2;
        this->lamb = lamb;
    }


    template<typename T=U, size_t ndim>
    FASTOR_INLINE
    std::tuple<Tensor<T,ndim,ndim>, typename MechanicsHessianType<T,ndim>::return_type>
    _KineticMeasures_(const T *Fnp) {

        // CREATE FASTOR TENSORS
        Tensor<T,ndim,ndim> F;
        // COPY NUMPY ARRAY TO FASTOR TENSORS
        copy_numpy(F,Fnp);

        // FIND THE KINEMATIC MEASURES
        Tensor<Real,ndim,ndim> I; I.eye2();
        auto J = determinant(F);
        // auto H = cofactor(F);
        auto b = matmul(F,transpose(F));

        // COMPUTE CAUCHY STRESS TENSOR
        T trb = trace(b);
        if (ndim == 2) {
            trb += 1.;
        }

        Tensor<T,ndim,ndim> stress = 2.*mu1/J*b + \
            2.*mu2/J*(trb*b - matmul(b,b)) - \
            2.*(mu1+2*mu2)/J*I + \
            lamb*(J-1)*I;

        // FIND ELASTICITY TENSOR
        auto II_ijkl = einsum<Index<i,j>,Index<k,l>>(I,I);
        auto II_ikjl = permutation<Index<i,k,j,l>>(II_ijkl);
        auto II_iljk = permutation<Index<i,l,j,k>>(II_ijkl);

        auto bb_ijkl = einsum<Index<i,j>,Index<k,l>>(b,b);
        auto bb_ikjl = permutation<Index<i,k,j,l>>(bb_ijkl);
        auto bb_iljk = permutation<Index<i,l,j,k>>(bb_ijkl);

        Tensor<T,ndim,ndim,ndim,ndim> elasticity = 2.0*mu2/J*(2.0*bb_ijkl - bb_ikjl - bb_iljk) + \
            (2.*(mu1+2*mu2)/J - lamb*(J-1.) ) * (II_ikjl + II_iljk) + lamb*(2.*J-1.)*II_ijkl;

        auto hessian = voigt(elasticity);

        auto kinetics = std::make_tuple(stress,hessian);
        return kinetics;
    }



    template<typename T>
    void KineticMeasures(T *Snp, T* Hnp, int ndim, int ngauss, const T *Fnp);

};

template<> template<>
void _MooneyRivlin_<Real>::KineticMeasures<Real>(Real *Snp, Real* Hnp,
    int ndim, int ngauss, const Real *Fnp) {

    if (ndim==3) {
        Tensor<Real,3> D;
        Tensor<Real,3,3> stress;
        Tensor<Real,6,6> hessian;
        for (int g=0; g<ngauss; ++g) {
            std::tie(stress,hessian) =_KineticMeasures_<Real,3>(Fnp+9*g);
            copy_fastor(Snp,stress,g*9);
            copy_fastor(Hnp,hessian,g*36);
        }
    }
    else if (ndim==2) {
        Tensor<Real,2> D;
        Tensor<Real,2,2> stress;
        Tensor<Real,3,3> hessian;
        for (int g=0; g<ngauss; ++g) {
            std::tie(stress,hessian) =_KineticMeasures_<Real,2>(Fnp+4*g);
            copy_fastor(Snp,stress,g*4);
            copy_fastor(Hnp,hessian,g*9);
        }
    }
}
