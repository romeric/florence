import numpy as np
from Florence import *


def simple_laplace():
    """An example of solving simple laplace equation using
        fourth order hexahedral elements on a cube
    """

    mesh = Mesh()
    mesh.Cube(element_type="hex",nx=6,ny=6,nz=6)
    mesh.GetHighOrderMesh(p=4)

    def DirichletFunc(mesh):
        boundary_data = np.zeros((mesh.points.shape[0]))+np.NAN
        # potential at left (Y=0)
        Y_0 = np.isclose(mesh.points[:,1],0)
        boundary_data[Y_0] = 0.
        # potential at right (Y=1)
        Y_1 = np.isclose(mesh.points[:,1],mesh.points[:,1].max())
        boundary_data[Y_1] = 10.
        return boundary_data


    boundary_condition = BoundaryCondition()
    boundary_condition.SetDirichletCriteria(DirichletFunc, mesh)

    material = IdealDielectric(mesh.InferSpatialDimension(),eps_1=2.35)
    formulation = LaplacianFormulation(mesh)
    fem_solver = FEMSolver(optimise=True)

    solution = fem_solver.Solve(boundary_condition=boundary_condition,
                                material=material,
                                formulation=formulation,
                                mesh=mesh)

    # Write results to vtk file
    # solution.WriteVTK("results", quantity=4)


if __name__ == "__main__":
    simple_laplace()