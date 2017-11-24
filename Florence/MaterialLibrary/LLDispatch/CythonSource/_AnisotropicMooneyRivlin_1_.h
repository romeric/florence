#include "_MaterialBase_.h"

template<typename U>
class _AnisotropicMooneyRivlin_1_ : public _MaterialBase_<U> {
public:
    U mu1;
    U mu2;
    U mu3;
    U lamb;

    FASTOR_INLINE _AnisotropicMooneyRivlin_1_() = default;

    FASTOR_INLINE
    _AnisotropicMooneyRivlin_1_(U mu1, U mu2, U mu3, U lamb) {
        this->mu1 = mu1;
        this->mu2 = mu2;
        this->mu3 = mu3;
        this->lamb = lamb;
    }

    FASTOR_INLINE
    void SetParameters(U mu1, U mu2, U mu3, U lamb){
        this->mu1 = mu1;
        this->mu2 = mu2;
        this->mu3 = mu3;
        this->lamb = lamb;
    }


    template<typename T=U, size_t ndim>
    FASTOR_INLINE
    std::tuple<Tensor<T,ndim,ndim>, typename MechanicsHessianType<T,ndim>::return_type>
    _KineticMeasures_(const T *Fnp, const T *Nnp) {

        // CREATE FASTOR TENSORS
        Tensor<T,ndim,ndim> F;
        Tensor<T,ndim> N;
        // COPY NUMPY ARRAY TO FASTOR TENSORS
        copy_numpy(F,Fnp);
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

        // COMPUTE CAUCHY STRESS TENSOR
        T trb = trace(b);
        if (ndim == 2) {
            trb += 1.;
        }

        Tensor<T,ndim,ndim> sigma = 2.*mu1/J*b + \
            2.*mu2/J*(trb*b - matmul(b,b)) - \
            2.*(mu1+2.*mu2+mu3)/J*I + \
            lamb*(J-1.)*I +\
            2.*mu3/J*outerFN +\
            2.*mu3/J*innerHN*I - 2.*mu3/J*outerHN;


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


        Tensor<T,ndim,ndim,ndim,ndim> elasticity = 2.0*mu2/J*(2.0*bb_ijkl - bb_ikjl - bb_iljk) + \
            (2.*(mu1+2*mu2+mu3)/J - lamb*(J-1.) ) * (II_ikjl + II_iljk) + lamb*(2.*J-1.)*II_ijkl -\
            4.*mu3/J*( IHN_ijkl + HNI_ijkl ) + \
            2.*mu3/J*innerHN*(2.0*II_ijkl - II_ikjl - II_iljk) +\
            2.*mu3/J * ( IHN_ikjl + IHN_iljk + IHN_jlik + IHN_jkil );

        auto hessian = voigt(elasticity);
        // print(hessian);
        auto kinetics = std::make_tuple(sigma,hessian);
        return kinetics;
    }



    template<typename T>
    void KineticMeasures(T *Snp, T* Hnp, int ndim, int ngauss, const T *Fnp, const T *Nnp);

};

template<> template<>
void _AnisotropicMooneyRivlin_1_<Real>::KineticMeasures<Real>(Real *Snp, Real* Hnp,
    int ndim, int ngauss, const Real *Fnp, const Real *Nnp) {

    if (ndim==3) {
        Tensor<Real,3,3> stress;
        Tensor<Real,6,6> hessian;
        for (int g=0; g<ngauss; ++g) {
            std::tie(stress,hessian) =_KineticMeasures_<Real,3>(Fnp+9*g, Nnp);
            copy_fastor(Snp,stress,g*9);
            copy_fastor(Hnp,hessian,g*36);
        }
    }
    else if (ndim==2) {
        Tensor<Real,2,2> stress;
        Tensor<Real,3,3> hessian;
        for (int g=0; g<ngauss; ++g) {
            std::tie(stress,hessian) =_KineticMeasures_<Real,2>(Fnp+4*g, Nnp);
            copy_fastor(Snp,stress,g*4);
            copy_fastor(Hnp,hessian,g*9);
        }
    }
}
