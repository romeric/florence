cdef extern from "cpp_matrixxfpy.h":
    cdef cppclass MatrixXdPy:
        MatrixXdPy()
        MatrixXdPy(int d1, int d2)
        MatrixXdPy(MatrixXfPy other)
        int rows()
        int cols()
        double coeff(int, int)
