#include "test_cpp.h"

Test::Test() {
    test1 = 0;
}

Test::Test(int test1) {
    this->test1 = test1;
}

Test::~Test() { }

Test Test::operator+(const Test& other) {
    return Test(test1 += other.test1);
}

Test Test::operator-(const Test& other) {
    return Test(test1 -= other.test1);
}

MatrixXdPy Test::getMatrixXd(int d1, int d2) {
    MatrixXfPy matrix = (MatrixXd)MatrixXdPy::Zero(d1,d2);
    matrix(0,0) = -10.0101003; // some manipulation, to show it carries over
    return matrix;
}
