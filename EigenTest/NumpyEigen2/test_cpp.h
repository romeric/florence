#ifndef TEST_CPP_H
#define TEST_CPP_H

#include "matrices.h"

using namespace Eigen;

class Test {
public:
    int test1;
    Test();
    Test(int test1);
    ~Test();
    Test operator+(const Test& other);
    Test operator-(const Test& other);
    MatrixXdPy getMatrixXd(int d1, int d2);
};

#endif // TEST_CPP_H

