#ifndef OCC_FRONTEND_HPP
#define OCC_FRONTEND_HPP

#include <std_inc.hpp>
#include <eigen_inc.hpp>
#include <occ_inc.hpp>
#include <cnp_funcs.hpp>
#include <aux_funcs.hpp>



// OCC_FrontEnd CLASS
class OCC_FrontEnd
{
private:
    Eigen::MatrixXi projection_ID;
    Eigen::MatrixXdr projection_U;
    Eigen::MatrixXdr projection_V;
    Eigen::MatrixXi dirichlet_edges;
    Eigen::MatrixXi dirichlet_faces;
    std::vector<int> listedges;
    std::vector<int> listfaces;
    Standard_Integer no_dir_edges;
    Standard_Integer no_dir_faces;
    Eigen::MatrixXi unique_edges;
    Eigen::MatrixXi unique_faces;
    Eigen::MatrixXd fekete_1d;
    Eigen::MatrixXi boundary_points_order;
public:
    // CONSTRUCTOR
    OCC_FrontEnd() {
    }
    OCC_FrontEnd(std::string &element_type,int64_t &ndim){
        this->mesh_element_type = element_type;
        this->ndim = ndim;
        this->condition = 0.;
        this->scale = 1000.;
    }
    ~OCC_FrontEnd(){}

    // members of OCC_FrontEnd
    std::string mesh_element_type;
    int ndim;
//    Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> mesh_elements;
    Eigen::MatrixXir mesh_elements;
    Eigen::MatrixXdr mesh_points;
    Eigen::MatrixXir mesh_edges;
    Eigen::MatrixXir mesh_faces;
    TopoDS_Shape imported_shape;
    Standard_Integer no_of_shapes;
    std::vector<Handle_Geom_Curve> geometry_edges;
    std::vector<Handle_Geom_Surface> geometry_faces;
    std::vector<Handle_Geom_BSplineCurve> geometry_edges_bspline;
    std::vector<Handle_Geom_BSplineSurface> geometry_faces_bspline;
    Standard_Real condition;
    Standard_Real scale;
    Eigen::MatrixXd displacements_BC;
    Eigen::MatrixXi index_nodes;
    std::string projection_methoed;


    // methods of occ_backend
    void Init(std::string &element_type,int &ndim);
    void SetCondition(Standard_Real &condition);
    void SetDimension(int &ndim);
    void SetElementType(std::string &type);
    void SetElements(Eigen::MatrixXir &arr);
    void SetPoints(Eigen::MatrixXdr &arr);
    void SetEdges(Eigen::MatrixXir &arr);
    void SetFaces(Eigen::MatrixXir &arr);
    std::string GetElementType();
    void ReadMeshConnectivityFile(std::string &filename, char delim);
    void ReadMeshCoordinateFile(std::string &filename, char delim);
    void ReadMeshEdgesFile(std::string &filename, char delim);
    void ReadMeshFacesFile(std::string &filename, char delim);
    void ReadUniqueEdges(std::string &filename);
    void ReadUniqueFaces(std::string &filename);
    Eigen::MatrixXi Read(std::string &filename);
    void CheckMesh();
    void ReadIGES(std::string & filename);
    void GetGeomEdges();
    void GetGeomFaces();
    void ProjectMeshOnCurve(std::string &method);
    void ProjectMeshOnSurface();
    void RepairDualProjectedParameters();
    void RepairDualProjectedParameters_Old();
    void CurvesToBsplineCurves();
    void SurfacesToBsplineSurfaces();
    void MeshPointInversionCurve();
    void MeshPointInversionSurface();
    void FeketePoints1D();
    Eigen::MatrixXd FeketePointsOnCurve(Handle_Geom_Curve &curve,Standard_Real &u1,Standard_Real &u2);
};




#endif // OCC_FRONTEND_HPP

