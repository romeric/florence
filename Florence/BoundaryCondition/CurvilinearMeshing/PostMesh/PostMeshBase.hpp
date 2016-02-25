#ifndef POSTMESHBASE_HPP
#define POSTMESHBASE_HPP

#ifdef EIGEN_INC_HPP
#include <EIGEN_INC.hpp>
#endif

#include <OCC_INC.hpp>

#ifndef CNP_FUNCS_HPP
#define CNP_FUNC_HPP
#include <CNPFuncs.hpp>
#endif

#include <AuxFuncs.hpp>
#include <PyInterface.hpp>


class PostMeshBase
{
public:
    //! PUBLIC MEMBER FUNCTIONS OF POSTMESH BASE CLASS. SHORTER FUNCTIONS ARE
    //! DECLARED AND DEFINED IN THE HEADER TO PROVIDE THEM WITH THE POSSIBILITY
    //! OF BEING INLINED BY THE COMPILER. ALTHOUGH DEEMED UN-NECESSARY, USER-
    //! DEFINED COPY/MOVE CONSTRUCTORS AND OPERATORS ARE IMPLEMENTED TO RESTRICT
    //! COPY OF DATA MEMBERS TO A SPECIFIC SET.

    ALWAYS_INLINE PostMeshBase()
    {
        this->scale = 1.0;
        this->condition = 1.0e10;
        this->projection_precision = 1.0e-04;
    }

    ALWAYS_INLINE PostMeshBase(std::string &element_type, const UInteger &dim) \
        : mesh_element_type(element_type), ndim(dim) {
        this->condition = 1.0e10;
        this->scale = 1.;
        this->projection_precision = 1.0e-04;
    }

    PostMeshBase(const PostMeshBase& other) \
    noexcept(std::is_copy_constructible<PostMeshBase>::value);
    PostMeshBase& operator=(const PostMeshBase& other) \
    noexcept(std::is_copy_assignable<PostMeshBase>::value);
    PostMeshBase(PostMeshBase&& other) noexcept;
    PostMeshBase& operator=(PostMeshBase&& other) noexcept;

    ALWAYS_INLINE ~PostMeshBase() = default;

    ALWAYS_INLINE void Init(std::string &etype, const UInteger &dim)
    {
        this->mesh_element_type = etype;
        this->ndim = dim;
        this->scale = 1.0;
        this->condition = 1.0e10;
        this->projection_precision = 1.0e-04;
    }

    ALWAYS_INLINE void SetScale(const Real &scale)
    {
        this->scale = scale;
    }

    ALWAYS_INLINE void SetCondition(const Real &condition)
    {
        this->condition = condition;
    }

    ALWAYS_INLINE void SetProjectionPrecision(const Real &precision)
    {
        if (precision < 1e-01)
            this->projection_precision = precision;
        else
            std::cerr << "Prescribed precision " << precision << " too high. Decrease it." << std::endl;
    }

    ALWAYS_INLINE void SetProjectionCriteria(UInteger *criteria, const Integer &rows, const Integer &cols)
    {
        this->projection_criteria = Eigen::Map<Eigen::MatrixUI>(criteria,rows,cols);
    }

    ALWAYS_INLINE void SetMeshElements(UInteger *arr, const Integer &rows, const Integer &cols)
    {
    #if !defined(WRAP_DATA)
        this->mesh_elements = Eigen::Map<Eigen::MatrixUI>(arr,rows,cols);
    #else
        Eigen::WrapRawBuffer<UInteger> Wrapper;
        Wrapper.data = arr;    Wrapper.rows = rows;    Wrapper.cols = cols;
        this->mesh_elements = std::move(Wrapper.asPostMeshMatrix());
    #endif
    }

    ALWAYS_INLINE void SetMeshPoints(Real *arr, const Integer &rows, const Integer &cols)
    {
    #if !defined(WRAP_DATA)
        this->mesh_points = Eigen::Map<Eigen::MatrixR>(arr,rows,cols);
    #else
        Eigen::WrapRawBuffer<Real> Wrapper;
        Wrapper.data = arr;    Wrapper.rows = rows;    Wrapper.cols = cols;
        this->mesh_points = std::move(Wrapper.asPostMeshMatrix());
    #endif
    }

    ALWAYS_INLINE void SetMeshEdges(UInteger *arr, const Integer &rows, const Integer &cols)
    {
    #if !defined(WRAP_DATA)
        this->mesh_edges = Eigen::Map<Eigen::MatrixUI>(arr,rows,cols);
    #else
        Eigen::WrapRawBuffer<UInteger> Wrapper;
        Wrapper.data = arr;    Wrapper.rows = rows;    Wrapper.cols = cols;
        this->mesh_edges = std::move(Wrapper.asPostMeshMatrix());
    #endif
    }

    ALWAYS_INLINE void SetMeshFaces(UInteger *arr, const Integer &rows, const Integer &cols)
    {
    #if !defined(WRAP_DATA)
        this->mesh_faces = Eigen::Map<Eigen::MatrixUI>(arr,rows,cols);
    #else
        Eigen::WrapRawBuffer<UInteger> Wrapper;
        Wrapper.data = arr;    Wrapper.rows = rows;    Wrapper.cols = cols;
        this->mesh_faces = std::move(Wrapper.asPostMeshMatrix());
    #endif
    }

    ALWAYS_INLINE void ScaleMesh()
    {
        this->mesh_points *=this->scale;
    }

    ALWAYS_INLINE std::string GetMeshElementType()
    {
        return this->mesh_element_type;
    }

    ALWAYS_INLINE void SetNodalSpacing(Real *arr, const Integer &rows, const Integer &cols)
    {
    #if !defined(WRAP_DATA)
        this->fekete = Eigen::Map<Eigen::MatrixR>(arr,rows,cols);
    #else
        Eigen::WrapRawBuffer<Real> Wrapper;
        Wrapper.data = arr;    Wrapper.rows = rows;    Wrapper.cols = cols;
        this->fekete = std::move(Wrapper.asPostMeshMatrix());
    #endif
    }

    ALWAYS_INLINE void ReturnModifiedMeshPoints(Real *points)
    {
        // RETURN MODIFIED MESH POINTS - INVOLVES DEEP COPY
        Eigen::Map<decltype(this->mesh_points)>(points,
                                                this->mesh_points.rows(),
                                                this->mesh_points.cols()) = this->mesh_points/this->scale;
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
    std::vector<Real> ObtainGeomVertices();

    ALWAYS_INLINE Integer NbPoints()
    {
        return this->geometry_points.size();
    }

    ALWAYS_INLINE Integer NbCurves()
    {
        return this->geometry_curves.size();
    }

    ALWAYS_INLINE Integer NbSurfaces()
    {
        return this->geometry_surfaces.size();
    }

    void ComputeProjectionCriteria();


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
    std::vector<TopoDS_Edge> topo_edges;
    std::vector<TopoDS_Face> topo_faces;
    Eigen::MatrixR displacements_BC;
    Eigen::MatrixI index_nodes;
    Eigen::MatrixUI nodes_dir;
    Eigen::MatrixR fekete;


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

