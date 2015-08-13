#include "PostMeshSurface.hpp"

PostMeshSurface::PostMeshSurface()
{
    this->ndim = 3;
    this->mesh_element_type = "tet";
    this->scale = 1.0;
    this->condition = 1.0e10;
}

void PostMeshSurface::Init()
{
    this->ndim = 3;
    this->mesh_element_type = "tet";
    this->scale = 1.0;
    this->condition = 1.0e10;
}

