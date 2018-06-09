#include "_MaterialBase_.h"
#include "_LegendreTransform_.h"

template<typename U>
class _ExplicitIsotropicElectroMechanics_108_: public _MaterialBase_<U> {
public:
    U mu1;
    U mu2;
    U lamb;
    U eps_2;

    _ExplicitIsotropicElectroMechanics_108_() = default;

    FASTOR_INLINE
    _ExplicitIsotropicElectroMechanics_108_(U mu1, U mu2, U lamb, U eps_2) {
        this->mu1 = mu1;
        this->mu2 = mu2;
        this->lamb = lamb;
        this->eps_2 = eps_2;
    }

    FASTOR_INLINE
    void SetParameters(U mu1, U mu2, U lamb, U eps_2){
        this->mu1 = mu1;
        this->mu2 = mu2;
        this->lamb = lamb;
        this->eps_2 = eps_2;
    }


    template<typename T=U, size_t ndim>
    FASTOR_INLINE
    std::tuple<Tensor<T,ndim>,Tensor<T,ndim,ndim> >
    _KineticMeasures_(const T *Fnp, const T *Enp) {

        // CREATE FASTOR TENSORS
        Tensor<T,ndim,ndim> F;
        Tensor<T,ndim> E;
        // COPY NUMPY ARRAY TO FASTOR TENSORS
        copy_numpy(F,Fnp);
        copy_numpy(E,Enp);

        // FIND THE KINEMATIC MEASURES
        Tensor<T,ndim,ndim> I; I.eye2();
        auto J = determinant(F);
        // auto H = cofactor(F);
        auto b = matmul(F,transpose(F));

        // COMPUTE ELECTRIC DISPLACEMENT
        Tensor<T,ndim> D = eps_2*E;

        auto innerDD = inner(D,D);
        auto outerDD = outer(D,D);

        // COMPUTE CAUCHY STRESS TENSOR
        T trb = trace(b);
        if (ndim == 2) {
            trb += 1.;
        }

        Tensor<T,ndim,ndim> sigma_mech = 2.*mu1/J*b + \
            2.*mu2/J*(trb*b - matmul(b,b)) - \
            2.*(mu1+2*mu2)/J*I + \
            lamb*(J-1)*I;

        Tensor<T,ndim,ndim> sigma_electric = 1./eps_2*(outerDD - 0.5*innerDD*I);
        Tensor<T,ndim,ndim> sigma = sigma_mech + sigma_electric;

        auto kinetics = std::make_tuple(D,sigma);
        return kinetics;
    }


    template<typename T>
    void KineticMeasures(T* Dnp, T *Snp, int ndim, int ngauss, const T *Fnp, const T *Enp);


};

template<> template<>
void _ExplicitIsotropicElectroMechanics_108_<Real>::KineticMeasures<Real>(Real *Dnp, Real *Snp,
    int ndim, int ngauss, const Real *Fnp, const Real *Enp) {

    if (ndim==3) {
        Tensor<Real,3> D;
        Tensor<Real,3,3> stress;
        for (int g=0; g<ngauss; ++g) {
            std::tie(D,stress) =_KineticMeasures_<Real,3>(Fnp+9*g, Enp+3*g);
            copy_fastor(Dnp,D,g*3);
            copy_fastor(Snp,stress,g*9);
        }
    }
    else if (ndim==2) {
        Tensor<Real,2> D;
        Tensor<Real,2,2> stress;
        for (int g=0; g<ngauss; ++g) {
            std::tie(D,stress) =_KineticMeasures_<Real,2>(Fnp+4*g, Enp+2*g);
            copy_fastor(Dnp,D,g*2);
            copy_fastor(Snp,stress,g*4);
        }
    }
}
