#ifndef PY_TO_OCC_BACKEND_HPP
#define PY_TO_OCC_BACKEND_HPP

#include <occ_frontend.hpp>

struct to_python_structs
{
    std::vector<Real> displacement_BC_stl;
    std::vector<Integer> nodes_dir_out_stl;
    Integer nodes_dir_size;
};

to_python_structs PyCppInterface(const char *iges_filename, Real scale, Real* points_array, Integer points_rows, Integer points_cols,
                                   Integer *elements_array, Integer element_rows, Integer element_cols,
                                   Integer *edges_array, Integer edges_rows, Integer edges_cols,
                                   Integer *faces_array, Integer faces_rows, Integer faces_cols, Real condition,
                                 Real *boundary_fekete, Integer fekete_rows, Integer fekete_cols, const char *projection_method);

#endif // PY_TO_OCC_BACKEND_HPP

