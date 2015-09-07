#include <iostream>
#include <fstream>
#include </home/roman/Dropbox/eigen-devel/Eigen/Dense>
#include </home/roman/Dropbox/eigen-devel/Eigen/Core>

using namespace std;
using namespace Eigen;

typedef double Real;
typedef long long Integer;
typedef unsigned long long UInteger;

namespace Eigen {
typedef Eigen::Matrix<Real,-1,-1,Eigen::RowMajor> MatrixR;
typedef Eigen::Matrix<Integer,-1,-1,Eigen::RowMajor> MatrixI;
typedef Eigen::Matrix<UInteger,-1,-1,Eigen::RowMajor> MatrixUI;
}


//! AUXILARY FUNCTIONS FOR POSTMESH
inline std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    //! SPLIT STRINGS
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

inline std::vector<std::string> split(const std::string &s, char delim) {
    //! SPLIT STRINGS
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}

//template<class T>
Eigen::MatrixUI ReadI(std::string &filename, char delim)
{
    /*Reading 2D integer row major arrays */
    std::vector<std::string> arr;
    arr.clear();
    std::string temp;

    std::ifstream datafile;
    datafile.open(filename.c_str());

    if(!datafile)
    {
        cout << "Unable to read file" << endl;
    }
    while(datafile)
    {
        datafile >> temp;
        temp += "";
        arr.push_back(temp);
    }

    datafile.close();


    const Integer rows = arr.size();
    const Integer cols = (split(arr[0], delim)).size();


    Eigen::MatrixUI out_arr = Eigen::MatrixUI::Zero(rows,cols);

    for(Integer i=0 ; i<rows;i++)
    {
        std::vector<std::string> elems;
        elems = split(arr[i], delim);
        for(Integer j=0 ; j<cols;j++)
        {
            out_arr(i,j) = std::atof(elems[j].c_str());
        }
    }


    // CHECK IF LAST LINE IS READ CORRECTLY
    bool duplicate_rows = false;
    for (Integer j=0; j<cols; ++j)
    {
        if ( out_arr(out_arr.rows()-2,j)==out_arr(out_arr.rows()-1,j) )
        {
            duplicate_rows = true;
        }
        else
        {
            duplicate_rows = false;
        }
    }
    if (duplicate_rows==true)
    {
        out_arr = out_arr.block(0,0,out_arr.rows()-1,out_arr.cols()).eval();
    }

    return out_arr;
}

struct Mesh {
    Eigen::MatrixUI elements;
    Eigen::MatrixUI edges;
    Eigen::MatrixUI faces;
    Eigen::MatrixR points;
};

void HighOrderMeshTri(const Integer C, Mesh mesh)
{
    Eigen::MatrixUI &elements = mesh.elements;
    Eigen::MatrixUI &edges = mesh.edges;
    Eigen::MatrixUI &faces = mesh.elements;
    Eigen::MatrixR &points = mesh.points;

    Integer nodeperelem = 3;
    Integer renodeperelem = Integer((C+2.)*(C+3.)/2.);
    Integer left_over_nodes = renodeperelem - nodeperelem;

    Eigen::MatrixUI reelements = -1*Eigen::MatrixUI::Ones(mesh.elements.rows(),renodeperelem);
    reelements.block(0,0,reelements.rows(),3) = mesh.elements;

}

int main()
{
    string filename = "/home/roman/Dropbox/handy_codes/PlanarMeshGenerator/elements_circle_p2.dat";

    Mesh mesh;
    mesh.elements = ReadI(filename,',');


    Eigen::Matrix<Real,6,2> fekete;
    Integer C = 1;
    fekete << -1.,-1., 1.,-1., -1.,1., 0.,-1., -1.,0., 0.,0.;



    cout << mesh.elements  << endl;

//    MatrixR m;
//    m.setRandom(3,3);
//    MatrixR &n = m;
//    MatrixR o; o.setRandom(4,4);

//    cout << m << endl<<endl;
//    n(0,0) = 2.;
////    n=o;
//    cout << m << endl<<endl;
//    cout << n << endl<<endl;


//    HighOrderMeshTri(C,mesh);
//    cout << fekete << endl;
    return 0;
}

