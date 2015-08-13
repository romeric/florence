#ifndef PY_TO_OCC_BACKEND_HPP
#define PY_TO_OCC_BACKEND_HPP

#include <OCCPlugin.hpp>

PassToPython PyCppInterface(const char *iges_filename, Real scale, Real* points_array, const Integer points_rows, const Integer points_cols,
                                   UInteger *elements_array, const Integer element_rows, const Integer element_cols,
                                   UInteger *edges_array, const Integer &edges_rows, const Integer &edges_cols,
                                   UInteger *faces_array, const Integer &faces_rows, const Integer &faces_cols, Real condition,
                                 Real *boundary_fekete, const Integer fekete_rows, const Integer fekete_cols,
                                 UInteger *criteria, const Integer criteria_rows, const Integer criteria_cols, const char *projection_method);


#endif // PY_TO_OCC_BACKEND_HPP

