import os, sys
sys.path.insert(1,'/home/roman/Dropbox/florence')

import numpy as np 
from Florence import *
from Florence.VariationalPrinciple import *
from Florence.Tensor import makezero




def ProblemData_Mesh(*args, **kwargs):

    p = kwargs['p']
    ndim=2

    # material = MooneyRivlin(ndim,youngs_modulus=1000.0,poissons_ratio=0.3)
    material = IncrementalLinearElastic(ndim,youngs_modulus=1000.0,poissons_ratio=0.3)


    ProblemPath = PWD(__file__)
    filename = ProblemPath + '/QuadCyl_76.dat'

    mesh = Mesh()
    mesh.Reader(filename=filename)
    mesh.GetHighOrderMesh(p=p)
    makezero(mesh.points)
    mesh.points = np.copy(mesh.points,'c')
    # print mesh.points.shape[0], mesh.nelem
    # mesh.SimplePlot()




    boundary_condition = BoundaryCondition()
    cad_file = ProblemPath + '/QuadCyl.iges'
    boundary_condition = BoundaryCondition()
    boundary_condition.SetCADProjectionParameters(cad_file,projection_type='arc_length',
        nodal_spacing='fekete',scale=1000.0,condition=20000.0)
    boundary_condition.GetProjectionCriteria(mesh)

    formulation = DisplacementFormulation(mesh)

    fem_solver = FEMSolver(number_of_load_increments=30,analysis_type="static",
        analysis_nature="linear",parallelise=False, compute_mesh_qualities=False,
        newton_raphson_tolerance=1.0e-05)

    solution = fem_solver.Solve(formulation=formulation, mesh=mesh, 
            material=material, boundary_condition=boundary_condition)

    mesh.WriteHDF5(filename.split(".")[0]+"_P"+str(p),{"TotalDisp":solution.sol})

    # solution.CurvilinearPlot(plot_points=True)



def ProblemData(*args, **kwargs):

    p = kwargs['p']
    ndim=2

    material = IsotropicElectroMechanics_0(ndim,youngs_modulus=1e4,poissons_ratio=0.3, eps_1=1.0)


    ProblemPath = PWD(__file__)
    filename = ProblemPath + '/QuadCyl_76_P'+str(p)+'.mat'
    
    mesh = Mesh()
    mesh.ReadHDF5(filename)
    # mesh.points = mesh.points + mesh.TotalDisp[:,:,-1]
    makezero(mesh.points, tol=1e-12)
    # print(dir(mesh))

    # print mesh.points
    # mesh.SimplePlot()
    # mesh.PlotMeshNumbering()
    # exit()



    boundary_condition = BoundaryCondition()

    def DirichletFunc(mesh):
        boundary_data = np.zeros((mesh.points.shape[0],material.nvar))+np.NAN

        # FIX ALL MECHANICS
        boundary_data[:,:2] = 0.

        # r = np.sqrt(mesh.points[:,0]**2 + mesh.points[:,1]**2)
        # # APPLY POTENTIAL IN INNER AND OUTER BOUNDARY
        # boundary_data[np.isclose(r,1.),2] = 10.
        # boundary_data[np.isclose(r,2.),2] = 100.

        un_edges = np.unique(mesh.edges)
        bpoints = mesh.points[un_edges]
        r = np.sqrt(bpoints[:,0]**2 + bpoints[:,1]**2)
        boundary_data[un_edges,2] = 90./np.log(2)*np.log(r) + 10

        return boundary_data


    boundary_condition.dirichlet_flags = DirichletFunc(mesh)
    formulation = DisplacementPotentialFormulation(mesh)

    fem_solver = FEMSolver(number_of_load_increments=1,analysis_type="static",
        analysis_nature="nonlinear",parallelise=False, compute_mesh_qualities=False,
        newton_raphson_tolerance=1.0e-05)

    solution = fem_solver.Solve(formulation=formulation, mesh=mesh, 
            material=material, boundary_condition=boundary_condition)

    # solution.Animate(configuration="deformed")
    # solution.StressRecovery()
    # solution.WriteVTK("/home/roman/zzchecker/HHH", quantity=1)
    # solution.Plot(configuration="original", quantity=2)





    def Analytical(mesh):
        r = np.sqrt(mesh.points[:,0]**2 + mesh.points[:,1]**2)
        exact_sol = 90./np.log(2)*np.log(r) + 10
        return exact_sol


    exact_sol = Analytical(mesh)
    pot = solution.sol[:,2,-1]
    error = np.abs((np.linalg.norm(pot) - np.linalg.norm(exact_sol))/np.linalg.norm(exact_sol))
    # error = np.abs(np.linalg.norm(pot - exact_sol)/np.linalg.norm(exact_sol))
    # print(pot)
    # print(exact_sol)
    # print(pot - exact_sol)
    print error, mesh.points.shape[0] 


    # curved
    # 0.00085312685699 56
    # 1.57877446987e-06 187
    # 2.51041301952e-08 394
    # 1.02987136909e-09 677
    # 1.05924350928e-08 1036
    # 7.12221691746e-10 1471

    # straight
    # 0.00085312685699 56
    # 1.38374649714e-06 187
    # 7.67518101837e-08 394
    # 5.12872062961e-10 677
    # 2.89802810933e-10 1036
    # 3.46638614227e-09 1471

    p = [1,2,3,4,5,6]
    e = [0.00085312685699, 1.57877446987e-06, 2.51041301952e-08, 1.02987136909e-09, 1.005924350928e-10, 7.12221691746e-12]
    ndof = [56,187,394,677,1036,1471]
    import matplotlib.pyplot as plt
    # plt.semilogy(p,e,'-ko')
    plt.semilogy(np.sqrt(ndof),e,'-ko')
    plt.grid('on')
    plt.show()

if __name__ == "__main__":
    
    # for p in range(1,10):
    #     ProblemData_Mesh(p=p)

    p=2
    # ProblemData_Mesh(p=p)
    
    ProblemData(p=p)
