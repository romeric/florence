#ifndef AUX_FUNCS_HPP
#define AUX_FUNCS_HPP


#ifndef EIGEN_INC_HPP
#include <EIGEN_INC.hpp>
#endif

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


template<typename T>
inline void print(std::vector<T> arr)
{
    //! PRINT FUNCTION OVERLOADED FOR STL VECTORS
    //! EASIER TO BIND EVERYTHING WITH PRINT FUNCTION
    //! RATHER THAN BINDING WITH OPERATOR <<

    //typeid(arr[0]).name();
    std::cout << std::endl;
    for (typename std::vector<T>::const_iterator i=arr.begin(); i<arr.end();++i)
    {
        std::cout <<  *i << std::endl;
    }
    std::cout << std::endl;
}

template <typename T>
inline void print(const T last)
{
    //! PRINT FUNCTION OVERLOADED FOR GENERIC TYPE T
    std::cout << last << std::endl;
}

template <typename U, typename... T>
inline void print(const U first, const T... rest)
{
    //! PRINT FUNCTION OVERLOADED USING VARIADIC TEMPLATE ARGUMENTS
    std::cout << first << " ";
    print(rest...);
}

template <typename T>
inline void warn(const T last)
{
    //! WARN FUNCTION FOR GENERIC TYPE T
    std::cerr << last << std::endl;
}

template <typename U, typename... T>
inline void warn(const U first, const T... rest)
{
    //! WARN FUNCTION OVERLOADED USING VARIADIC TEMPLATE ARGUMENTS
    std::cerr << first << " ";
    warn(rest...);
}



#endif // AUX_FUNCS_HPP

