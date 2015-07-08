#ifndef AUX_FUNCS_HPP
#define AUX_FUNCS_HPP


#ifndef EIGEN_INC_HPP
#include <eigen_inc.hpp>
#endif

// auxilary functions
// split strings
inline std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}
inline std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}

//// print functions
//template<typename Derived> void print(Eigen::MatrixBase<Derived> &A)
//{
//    int rows = A.rows();
//    int cols = A.cols();
//    for(int i=0 ; i<rows;i++)
//    {
//        for (int j=0;j<cols; j++)
//        {
//            cout << A(i,j) << " ";
//        }
//        cout << endl;
//    }
//}

//template<typename Derived> void print(Derived A)
//{
//    std::cout << A << std::endl;
//}

//void print(const char* fmt...)
//{
//    va_list args;
//    va_start(args, fmt);

//    while (*fmt != '\0') {
//        if (*fmt == 'd') {
//            int i = va_arg(args, int);
//            std::cout << i << ", ";
//        } else if (*fmt == 'c') {
//            // note automatic conversion to integral type
//            int c = va_arg(args, int);
//            std::cout << static_cast<char>(c) << ", ";
//        } else if (*fmt == 'f') {
//            double d = va_arg(args, double);
//            std::cout << d << ", ";
//        }
//        ++fmt;
//    }

//    va_end(args);
//    std::cout << std::endl;
//}
//// end of print functions





#endif // AUX_FUNCS_HPP

