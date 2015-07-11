//#define NDEBUG
//#define EIGEN_NO_DEBUG

#include <assert.h>
#include <stdlib.h>

#include <occ_frontend.hpp>


//PyArrayObject *CreatePyArrayObject(Integer *data, npy_intp size)
//{
////    _import_array();
//    PyObject *py_array = PyArray_SimpleNewFromData(1,&size,NPY_INT64,data);
//    return (PyArrayObject*)py_array;
//}


using namespace std;
// A syntactic sugar equivalent to "import to numpy as np"
namespace cnp = cpp_numpy;




int main()
{

    clock_t begin = clock();


    std::string elem_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Annular_Circle_Nurbs/elements_circle_p2.dat";
    std::string point_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Annular_Circle_Nurbs/points_circle_p2.dat";
    std::string edge_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Annular_Circle_Nurbs/edges_circle_p2.dat";
    std::string unique_edge_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Annular_Circle_Nurbs/unique_edges_circle_p2.dat";


    std::string element_type = "tri";
    Integer ndim = 2;
    OCC_FrontEnd occ_interface = OCC_FrontEnd(element_type,ndim);

    // READ ELEMENT CONNECTIVITY, NODAL COORDINATES, EDGES & FACES
    occ_interface.SetElementType(element_type);
    occ_interface.ReadMeshConnectivityFile(elem_file,',');
    occ_interface.ReadMeshCoordinateFile(point_file,',');
    occ_interface.ReadMeshEdgesFile(edge_file,',');
    occ_interface.ReadUniqueEdges(unique_edge_file);
    occ_interface.CheckMesh();

    Standard_Real condition = 2000.;
//    Standard_Real condition = 2.;
    occ_interface.SetCondition(condition);
    occ_interface.mesh_points *= 1000.0;

    // READ THE GEOMETRY FROM THE IGES FILE
    //std::string filename = "/home/roman/Dropbox/Python/Problems/FiniteElements/RAE2822/rae2822.igs";
//    std::string filename = "/home/roman/Dropbox/2015_HighOrderMeshing/examples/mechanical2d.igs";
    std::string filename = "/home/roman/Dropbox/Python/Problems/FiniteElements/Annular_Circle_Nurbs/Circle.igs";

//    std::string filename = "/home/roman/Dropbox/OCC_Geometry_Checks/Line.igs";
//    std::string filename = "/home/roman/Dropbox/OCC_Geometry_Checks/Sphere.igs";
    occ_interface.ReadIGES(filename);

    // EXTRACT GEOMETRY INFORMATION FROM THE IGES FILE
    occ_interface.GetGeomEdges();
    occ_interface.GetGeomFaces();

//    cout << occ_interface.geometry_edges.size() << " " << occ_interface.geometry_faces.size() << endl;

    // PROJECT ALL BOUNDARY POINTS FROM THE MESH TO THE CURVE
    //std::string method = "Newton";
    std::string method = "bisection";
    occ_interface.ProjectMeshOnCurve(method);
    // CONVERT CURVES TO BSPLINE CURVES
    occ_interface.CurvesToBsplineCurves();
    // FIX IAMGES AND ANTI IMAGES IN PERIODIC CURVES/SURFACES
    //occ_interface.RepairDualProjectedParameters_Old();
    occ_interface.RepairDualProjectedParameters();

    // PERFORM POINT INVERTION FOR THE INTERIOR POINTS
    occ_interface.MeshPointInversionCurve();


//    Py_Initialize();
//    _import_array();
//    npy_intp pyshape = occ_interface.nodes_dir.rows();
//    PyObject *py_array = PyArray_SimpleNewFromData(1,&pyshape,NPY_INT64,occ_interface.nodes_dir.data());

//    PyArrayObject *X = (PyArrayObject *)py_array;

//    cout << py_array << " " << X << endl;

//    cout << xxx[1] << endl;
//    PyArray_Descr *dtype;
//    Integer **dataptr;
//    NpyIter *iter;
//    NpyIter_IterNextFunc *iternext;
//    dtype = PyArray_DescrFromType(NPY_INT64);
//    iter = NpyIter_New(X, NPY_ITER_READONLY, NPY_KEEPORDER, NPY_NO_CASTING, dtype);
//    iternext = NpyIter_GetIterNext(iter, NULL);
//    dataptr = (Integer **) NpyIter_GetDataPtrArray(iter);
//    cout << **dataptr << endl;

//    PyList_GetItem(py_array,pyshape);
//    cout << PyArray_DATA(py_array) << endl;
//    cout << PyArray_DIM(py_array,0) << endl;
//    (PyArrayObject*)pyshape;
//    cout << xxx << endl;

//    PyArrayObject py_array2 = (PyArrayObject)py_array;
//    PyArray_GetPtr(py_array, &pyshape);
//    cout << pyshape << endl;
//    cout << py_array[0] << endl;
//    cout << PyArray_FLAGS(py_array) << endl;


//    Py_Finalize();
//    cout << py_array << endl;

    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << endl << "Total time elapsed was " << elapsed_secs << " seconds" << endl;

    return 0;
}
