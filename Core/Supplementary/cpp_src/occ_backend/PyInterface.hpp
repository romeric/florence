#ifndef PYINTERFACE_HPP
#define PYINTERFACE_HPP

#include <STL_INC.hpp>

// struct to pass to Python
struct PassToPython
{
    std::vector<Real> displacement_BC_stl;
    std::vector<Integer> nodes_dir_out_stl;
    Integer nodes_dir_size;
};

#endif // PYINTERFACE_HPP

