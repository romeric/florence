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

#include <PyInterface.hpp>


class PostMeshBase
{
public:
    //! PUBLIC MEMBER FUNCTIONS OF POSTMESH BASE CLASS. SHORTER FUNCTIONS ARE
    //! DECLARED AND DEFINED IN THE HEADER TO PROVIDE THEM WITH THE POSSIBILITY
    //! OF BEING INLINED BY THE COMPILER. ALTHOUGH DEEMED UN-NECESSARY, USER-
    //! DEFINED COPY/MOVE CONSTRUCTORS AND OPERATORS ARE IMPLEMENTED TO RESTRICT
    //! COPY OF DATA MEMBERS TO A SPECIFIC SET.

    inline PostMeshBase()
    {
        this->scale = 1.0;
        this->condition = 1.0e10;
        this->projection_precision = 1.0e-04;
    }

    inline PostMeshBase(std::string &element_type, const UInteger &dim) : mesh_element_type(element_type), ndim(dim) {
        this->condition = 1.0e10;
        this->scale = 1.;
        this->projection_precision = 1.0e-04;
    }

    PostMeshBase(const PostMeshBase& other);
    PostMeshBase& operator=(const PostMeshBase& other);
    PostMeshBase(PostMeshBase&& other);
    PostMeshBase& operator=(PostMeshBase&& other);

    inline ~PostMeshBase(){}

    inline void Init(std::string &etype, const UInteger &dim)
    {
        this->mesh_element_type = etype;
        this->ndim = dim;
        this->scale = 1.0;
        this->condition = 1.0e10;
        this->projection_precision = 1.0e-04;
    }

    inline void SetScale(const Real &scale)
    {
        this->scale = scale;
    }

    inline void SetCondition(const Real &condition)
    {
        this->condition = condition;
    }

    inline void SetProjectionPrecision(const Real &precision)
    {
        if (precision < 1e-01)
            this->projection_precision = precision;
        else
            std::cerr << "Prescribed precision " << precision << " too high. Decrease it." << std::endl;
    }

    inline void SetProjectionCriteria(UInteger *criteria, const Integer &rows, const Integer &cols)
    {
        this->projection_criteria = Eigen::Map<Eigen::MatrixUI>(criteria,rows,cols);
    }

    inline void SetMeshElements(UInteger *arr, const Integer &rows, const Integer &cols)
    {
        this->mesh_elements = Eigen::Map<Eigen::MatrixUI>(arr,rows,cols);
    }

    inline void SetMeshPoints(Real *arr, const Integer &rows, const Integer &cols)
    {
        this->mesh_points = Eigen::Map<Eigen::MatrixR>(arr,rows,cols);
    //    new (&this->mesh_points) Eigen::Map<Eigen::MatrixR> (arr,rows,cols);
    //    print (this->mesh_points.rows(),this->mesh_points.cols());
    }

    inline void SetMeshEdges(UInteger *arr, const Integer &rows, const Integer &cols)
    {
        this->mesh_edges = Eigen::Map<Eigen::MatrixUI>(arr,rows,cols);
    }

    inline void SetMeshFaces(UInteger *arr, const Integer &rows, const Integer &cols)
    {
        this->mesh_faces = Eigen::Map<Eigen::MatrixUI>(arr,rows,cols);
    }

    inline void ScaleMesh()
    {
        this->mesh_points *=this->scale;
    }

    inline std::string GetMeshElementType()
    {
        return this->mesh_element_type;
    }

    inline void SetFeketePoints(Real *arr, const Integer &rows, const Integer &cols)
    {
        this->fekete = Eigen::Map<Eigen::MatrixR>(arr,rows,cols);
    }

    void ReadIGES(const char *filename);
    void ReadSTEP(const char *filename);
    static Eigen::MatrixI Read(std::string &filename);
    static Eigen::MatrixUI ReadI(std::string &filename, char delim);
    static Eigen::MatrixR ReadR(std::string &filename, char delim);
    void CheckMesh();

    void GetGeomVertices();
    void GetGeomEdges();
    void GetGeomFaces();


    std::string mesh_element_type;
    UInteger ndim;
    Real scale;
    Real condition;
    Real projection_precision;
    Eigen::MatrixUI mesh_elements;
    Eigen::MatrixR mesh_points;
    Eigen::MatrixUI mesh_edges;
    Eigen::MatrixUI mesh_faces;
    Eigen::MatrixUI projection_criteria;

    UInteger degree;
    TopoDS_Shape imported_shape;
    UInteger no_of_shapes;
    std::vector<gp_Pnt> geometry_points;
    std::vector<Handle_Geom_Curve> geometry_curves;
    std::vector<Handle_Geom_Surface> geometry_surfaces;
    std::vector<UInteger> geometry_curves_types;
    std::vector<UInteger> geometry_surfaces_types;
    const char *projection_method;
    Eigen::MatrixR displacements_BC;
    Eigen::MatrixI index_nodes;
    Eigen::MatrixI nodes_dir;
    Eigen::MatrixR fekete;

protected:
    void ComputeProjectionCriteria();

private:
    void SetDimension(const UInteger &dim)
    {
        this->ndim=dim;
    }

    void SetMeshElementType(std::string &type)
    {
        this->mesh_element_type = type;
    }
};

#endif // POSTMESHBASE

