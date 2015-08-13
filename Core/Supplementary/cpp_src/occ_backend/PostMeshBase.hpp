#ifndef POSTMESHBASE_HPP
#define POSTMESHBASE_HPP

#ifdef EIGEN_INC_HPP
#include <EIGEN_INC.hpp>
#endif

#ifdef OCC_INC_HPP
#include <OCC_INC.hpp>
#endif

#ifndef CNP_FUNCS_HPP
#define CNP_FUNC_HPP
#include <CNPFuncs.hpp>
#endif

//#ifndef AUX_FUNCS_HPP
//#define AUX_FUNCS_HPP
#include <AuxFuncs.hpp>
//#endif


class PostMeshBase
{
public:
    PostMeshBase();
    PostMeshBase(std::string &element_type, const UInteger &dim) : mesh_element_type(element_type), ndim(dim) {
//        this->mesh_element_type = element_type;
//        this->ndim = dim;
        this->condition = 1.0e10;
        this->scale = 1.;
    }
    ~PostMeshBase();

    Real condition;
    Real scale;
    std::string mesh_element_type;
    UInteger ndim;
//    Eigen::EigenBase<Real> xx;

//    UInteger degree;
//    TopoDS_Shape imported_shape;
//    UInteger no_of_shapes;

//    virtual void ReadIGES(const char *filename);
//    void ReadIGES(const char *filename);
};

#endif // POSTMESHBASE

