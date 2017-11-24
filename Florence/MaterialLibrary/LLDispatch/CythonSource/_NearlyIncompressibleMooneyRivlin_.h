#include "_MaterialBase_.h"

template<typename U>
class _NearlyIncompressibleMooneyRivlin_ : public _MaterialBase_<U> {
public:
    U alpha;
    U beta;
    U kappa;

    _NearlyIncompressibleMooneyRivlin_() = default;

    FASTOR_INLINE
    _NearlyIncompressibleMooneyRivlin_(U alpha, U beta, U kappa) {
        this->alpha = alpha;
        this->beta = beta;
        this->kappa = kappa;
    }

    FASTOR_INLINE
    void SetParameters(U alpha, U beta, U kappa){
        this->alpha = alpha;
        this->beta = beta;
        this->kappa = kappa;
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
        auto H = cofactor(F);
        auto b = matmul(F,transpose(F));
        auto g = matmul(H,transpose(H));

        // COMPUTE CAUCHY STRESS TENSOR
        T trb = trace(b);
        T trg = trace(g);
        if (ndim == 2) {
            trb += 1.;
            trg += J*J;
        }

        T coeff0 = std::pow(J,-5./3.);
        T coeff1 = std::sqrt(trg);
        // T coeff2 = std::pow(trg,-1./2.);
        T coeff2 = 1./coeff1;
        // T coeff3 = std::pow(trg,3./2.);
        T coeff3 = trg*coeff1;
        // T coeff4 = std::pow(J,-3.);
        T coeff4 = 1./(J*J*J);


        Tensor<T,ndim,ndim> stress = 2.*alpha*coeff0*b - 2./3.*alpha*coeff0*(trb)*I + \
                beta*coeff4*coeff3*I - 3*beta*coeff4*coeff1*g + \
                + kappa*(J-1.0)*I;

        // FIND ELASTICITY TENSOR
        auto II_ijkl = einsum<Index<i,j>,Index<k,l>>(I,I);
        auto II_ikjl = permutation<Index<i,k,j,l>>(II_ijkl);
        auto II_iljk = permutation<Index<i,l,j,k>>(II_ijkl);

        auto bI_ijkl = einsum<Index<i,j>,Index<k,l>>(b,I);
        auto Ib_ijkl = einsum<Index<i,j>,Index<k,l>>(I,b);

        auto gI_ijkl = einsum<Index<i,j>,Index<k,l>>(g,I);
        auto Ig_ijkl = einsum<Index<i,j>,Index<k,l>>(I,g);
        auto gI_ikjl = permutation<Index<i,k,j,l>>(gI_ijkl);
        auto gI_iljk = permutation<Index<i,l,j,k>>(gI_ijkl);
        auto Ig_ikjl = permutation<Index<i,k,j,l>>(Ig_ijkl);
        auto Ig_iljk = permutation<Index<i,l,j,k>>(Ig_ijkl);

        auto gg_ijkl = einsum<Index<i,j>,Index<k,l>>(g,g);

        Tensor<T,ndim,ndim,ndim,ndim> elasticity = -4/3.*alpha*coeff0*( bI_ijkl + Ib_ijkl ) + \
            4.*alpha/9.*coeff0*trb*II_ijkl + \
            2/3.*alpha*coeff0*trb*(II_ikjl + II_iljk) + \
            beta*coeff4*coeff3*( II_ijkl - II_ikjl - II_iljk ) - \
            3.*beta*coeff4*coeff1*( gI_ijkl + Ig_ijkl ) + \
            3.*beta*coeff4*coeff1*( gI_ikjl + gI_iljk + Ig_ikjl + Ig_iljk ) + \
            3.*beta*coeff4*coeff2*gg_ijkl  + \
            kappa*(2.0*J-1)*II_ijkl - kappa*(J-1)*(II_ikjl + II_iljk);

        auto hessian = voigt(elasticity);

        auto kinetics = std::make_tuple(stress,hessian);
        return kinetics;
    }



    template<typename T>
    void KineticMeasures(T *Snp, T* Hnp, int ndim, int ngauss, const T *Fnp);

};

template<> template<>
void _NearlyIncompressibleMooneyRivlin_<Real>::KineticMeasures<Real>(Real *Snp, Real* Hnp,
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
