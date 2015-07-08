//#ifndef MATRIXXDPY
#include </usr/local/include/Eigen/Dense>
#include </usr/local/include/Eigen/Core>

using namespace Eigen;

class MatrixXdPy: public MatrixXd
{
public:
    MatrixXdPy() : MatrixXd() {}
    MatrixXdPy(int rows,int cols) : MatrixXd(rows,cols) {}
    MatrixXdPy(const MatrixXd other) : MatrixXd(other) { }
};
