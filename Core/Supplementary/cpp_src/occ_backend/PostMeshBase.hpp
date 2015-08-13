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
        this->condition = 1.0e10;
        this->scale = 1.;
    }
    ~PostMeshBase();


    std::string mesh_element_type;
    UInteger ndim;
    Real condition;
    Real scale;
    Eigen::MatrixUI mesh_elements;
    Eigen::MatrixR mesh_points;
    Eigen::MatrixUI mesh_edges;
    Eigen::MatrixUI mesh_faces;
    Eigen::MatrixUI projection_criteria;
//    Eigen::EigenBase<Real> xx;
//    Eigen::MatrixBase<Real> xx;

//    UInteger degree;
    TopoDS_Shape imported_shape;
    UInteger no_of_shapes;
    std::vector<gp_Pnt> geometry_points;
    std::vector<Handle_Geom_Curve> geometry_curves;
    std::vector<Handle_Geom_Surface> geometry_surfaces;
    std::vector<UInteger> geometry_curves_types;
    std::vector<UInteger> geometry_surfaces_types;


    virtual void Init(std::string &etype, const UInteger &dim);
    void SetScale(Real &scale);
    void SetCondition(Real &condition);
    void SetProjectionCriteria(UInteger *criteria, const Integer &rows, const Integer &cols);
    void SetMeshElements(UInteger *arr, const Integer &rows, const Integer &cols);
    void SetMeshPoints(Real *arr, const Integer &rows, const Integer &cols);
    void SetMeshEdges(UInteger *arr, const Integer &rows, const Integer &cols);
    void SetMeshFaces(UInteger *arr, const Integer &rows, const Integer &cols);
    void ScaleMesh();
    std::string GetMeshElementType();
    void ReadIGES(const char *filename);
    void ReadSTEP(const char *filename);


    static Eigen::MatrixI Read(std::string &filename);
    static Eigen::MatrixUI ReadI(std::string &filename, char delim);
    static Eigen::MatrixR ReadR(std::string &filename, char delim);
    void CheckMesh();

    void GetGeomVertices();
    void GetGeomEdges();
    void GetGeomFaces();

protected:


private:
    void SetDimension(const UInteger &dim);
    void SetMeshElementType(std::string &type);
};

#endif // POSTMESHBASE

