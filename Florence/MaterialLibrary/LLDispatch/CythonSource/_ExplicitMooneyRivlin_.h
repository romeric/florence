#include "_MaterialBase_.h"

template<typename U>
class _ExplicitMooneyRivlin_ : public _MaterialBase_<U> {
public:
    U mu1;
    U mu2;
    U lamb;

    _ExplicitMooneyRivlin_() = default;

    FASTOR_INLINE
    _ExplicitMooneyRivlin_(U mu1, U mu2, U lamb) {
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
    Tensor<T,ndim,ndim>
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

        return stress;
    }



    template<typename T>
    void KineticMeasures(T *Snp, int ndim, int ngauss, const T *Fnp);

};

template<> template<>
void _ExplicitMooneyRivlin_<Real>::KineticMeasures<Real>(Real *Snp,
    int ndim, int ngauss, const Real *Fnp) {

    if (ndim==3) {
        Tensor<Real,3,3> stress;
        for (int g=0; g<ngauss; ++g) {
            stress =_KineticMeasures_<Real,3>(Fnp+9*g);
            copy_fastor(Snp,stress,g*9);
        }
    }
    else if (ndim==2) {
        Tensor<Real,2,2> stress;
        for (int g=0; g<ngauss; ++g) {
            stress =_KineticMeasures_<Real,2>(Fnp+4*g);
            copy_fastor(Snp,stress,g*4);
        }
    }
}
