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

    inline PostMeshBase(const PostMeshBase& other) : \
        scale(other.scale), condition(other.condition), projection_precision(other.projection_precision)
    {
        // Copy constructor
        this->mesh_element_type = other.mesh_element_type;
        this->ndim = other.ndim;
        this->mesh_elements = other.mesh_elements;
        this->mesh_points = other.mesh_points;
        this->mesh_edges = other.mesh_edges;
        this->mesh_faces = other.mesh_faces;
        this->projection_criteria = other.projection_criteria;
        this->degree = degree;
        this->imported_shape = other.imported_shape;
        this->no_of_shapes = other.no_of_shapes;
        this->geometry_points = other.geometry_points;
        this->geometry_curves = other.geometry_curves;
        this->geometry_surfaces = other.geometry_surfaces;
        this->geometry_curves_types = other.geometry_curves_types;
        this->geometry_surfaces_types = other.geometry_surfaces_types;
        this->projection_method = other.projection_method;
        this->displacements_BC = other.displacements_BC;
        this->index_nodes = other.index_nodes;
        this->nodes_dir = other.nodes_dir;
        this->fekete = other.fekete;
    }

    inline PostMeshBase& operator=(const PostMeshBase& other)
    {
        // Copy assignment operator
        this->scale = other.scale;
        this->condition = other.condition;
        this->projection_precision = other.projection_precision;

        this->mesh_element_type = other.mesh_element_type;
        this->ndim = other.ndim;
        this->mesh_elements = other.mesh_elements;
        this->mesh_points = other.mesh_points;
        this->mesh_edges = other.mesh_edges;
        this->mesh_faces = other.mesh_faces;
        this->projection_criteria = other.projection_criteria;
        this->degree = degree;
        this->imported_shape = other.imported_shape;
        this->no_of_shapes = other.no_of_shapes;
        this->geometry_points = other.geometry_points;
        this->geometry_curves = other.geometry_curves;
        this->geometry_surfaces = other.geometry_surfaces;
        this->geometry_curves_types = other.geometry_curves_types;
        this->geometry_surfaces_types = other.geometry_surfaces_types;
        this->projection_method = other.projection_method;
        this->displacements_BC = other.displacements_BC;
        this->index_nodes = other.index_nodes;
        this->nodes_dir = other.nodes_dir;
        this->fekete = other.fekete;

        return *this;
    }

    inline PostMeshBase(PostMeshBase&& other) : \
        scale(other.scale), condition(other.condition), projection_precision(other.projection_precision)
    {
        //! Move constructor for PostMeshBase class
        this->mesh_element_type = other.mesh_element_type;
        this->ndim = other.ndim;
        this->mesh_elements = std::move(other.mesh_elements);
        this->mesh_points = std::move(other.mesh_points);
        this->mesh_edges = std::move(other.mesh_edges);
        this->mesh_faces = std::move(other.mesh_faces);
        this->projection_criteria = std::move(other.projection_criteria);
        this->degree = degree;
        this->imported_shape = std::move(other.imported_shape);
        this->no_of_shapes = other.no_of_shapes;
        this->geometry_points = std::move(other.geometry_points);
        this->geometry_curves = std::move(other.geometry_curves);
        this->geometry_surfaces = std::move(other.geometry_surfaces);
        this->geometry_curves_types = std::move(other.geometry_curves_types);
        this->geometry_surfaces_types = std::move(other.geometry_surfaces_types);
        this->projection_method = std::move(other.projection_method);
        this->displacements_BC = std::move(other.displacements_BC);
        this->index_nodes = std::move(other.index_nodes);
        this->nodes_dir = std::move(other.nodes_dir);
        this->fekete = std::move(other.fekete);

        //! NB: While STL containers implement move semantics, Eigen does not. So a proper
        //! "move" happens when std::vector is used while a moved Eigen::Matrix<T,Options> will
        //! still be in scope
    }

    inline PostMeshBase& operator=(PostMeshBase&& other)
    {
        // Move assignment operator
        this->scale = other.scale;
        this->condition = other.condition;
        this->projection_precision = other.projection_precision;

        this->mesh_element_type = other.mesh_element_type;
        this->ndim = other.ndim;
        this->mesh_elements = std::move(other.mesh_elements);
        this->mesh_points = std::move(other.mesh_points);
        this->mesh_edges = std::move(other.mesh_edges);
        this->mesh_faces = std::move(other.mesh_faces);
        this->projection_criteria = std::move(other.projection_criteria);
        this->degree = degree;
        this->imported_shape = std::move(other.imported_shape);
        this->no_of_shapes = other.no_of_shapes;
        this->geometry_points = std::move(other.geometry_points);
        this->geometry_curves = std::move(other.geometry_curves);
        this->geometry_surfaces = std::move(other.geometry_surfaces);
        this->geometry_curves_types = std::move(other.geometry_curves_types);
        this->geometry_surfaces_types = std::move(other.geometry_surfaces_types);
        this->projection_method = std::move(other.projection_method);
        this->displacements_BC = std::move(other.displacements_BC);
        this->index_nodes = std::move(other.index_nodes);
        this->nodes_dir = std::move(other.nodes_dir);
        this->fekete = std::move(other.fekete);

        //! NB: While STL containers implement move semantics, Eigen does not. So a proper
        //! "move" happens when std::vector is used while a moved Eigen::Matrix<T,Options> will
        //! still be in scope

        return *this;
    }

    virtual inline ~PostMeshBase(){}

    virtual inline void Init(std::string &etype, const UInteger &dim)
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

