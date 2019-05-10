import os, sys
import numpy as np
from copy import deepcopy
from warnings import warn
from .Mesh import Mesh
from .GeometricPath import *
from Florence.Tensor import totuple, unique2d


__all__ = ['HarvesterPatch', 'SubdivisionArc', 'SubdivisionCircle', 'QuadBall',
'QuadBallSphericalArc']

"""
A series of custom meshes
"""


def HarvesterPatch(ndisc=20, nradial=4, show_plot=False):
    """A custom mesh for an energy harvester patch. [Not to be modified]
        ndisc:              [int] number of discretisation in c
        ndradial:           [int] number of discretisation in radial directions for different
                            components of harevester
    """


    center = np.array([30.6979,20.5])
    p1     = np.array([30.,20.])
    p2     = np.array([30.,21.])
    p1line = p1 - center
    p2line = p2 - center
    radius = np.linalg.norm(p1line)
    pp = np.array([center[0],center[1]+radius])
    y_line = pp - center
    start_angle = -np.pi/2. - np.arccos(np.linalg.norm(y_line*p1line)/np.linalg.norm(y_line)/np.linalg.norm(p1line))
    end_angle   = np.pi/2. + np.arccos(np.linalg.norm(y_line*p1line)/np.linalg.norm(y_line)/np.linalg.norm(p1line))
    points = np.array([p1,p2,center])

    # nradial = 4
    mesh = Mesh()
    mesh.Arc(element_type="quad", radius=radius, start_angle=start_angle,
        end_angle=end_angle, nrad=nradial, ncirc=ndisc, center=(center[0],center[1]), refinement=True)

    mesh1 = Mesh()
    mesh1.Triangle(element_type="quad",npoints=nradial, c1=totuple(center), c2=totuple(p1), c3=totuple(p2))

    mesh += mesh1

    mesh_patch = Mesh()
    mesh_patch.HollowArc(ncirc=ndisc, nrad=nradial, center=(-7.818181,44.22727272),
        start_angle=np.arctan(44.22727272/-7.818181), end_angle=np.arctan(-24.22727272/37.818181),
        element_type="quad", inner_radius=43.9129782, outer_radius=44.9129782)

    mesh3 = Mesh()
    mesh3.Triangle(element_type="quad",npoints=nradial, c2=totuple(p1), c3=totuple(p2), c1=(mesh_patch.points[0,0], mesh_patch.points[0,1]))

    mesh += mesh3
    mesh += mesh_patch


    mesh.Extrude(nlong=ndisc,length=40)

    if show_plot:
        mesh.SimplePlot()

    return mesh


def CurvedPlate(ncirc=2, nlong=20, show_plot=False):
    """Custom mesh for plate with curved edges
        ncirc           discretisation around circular fillets
        nlong           discretisation along the length - X
    """

    mesh_arc = Mesh()
    mesh_arc.Arc(element_type="quad",nrad=ncirc,ncirc=ncirc, radius=5)

    mesh_arc1 = deepcopy(mesh_arc)
    mesh_arc1.points[:,1] += 15
    mesh_arc1.points[:,0] += 95
    mesh_arc2 = deepcopy(mesh_arc)
    mesh_arc2.points[:,1] +=15
    mesh_arc2.points[:,0] *= -1.
    mesh_arc2.points[:,0] += 5.

    mesh_plate1 = Mesh()
    mesh_plate1.Rectangle(element_type="quad",lower_left_point=(5,15),upper_right_point=(95,20),ny=ncirc, nx=nlong)

    mesh_plate2 = deepcopy(mesh_plate1)
    mesh_plate2.points[:,1] -= 5.

    mesh_square1 = Mesh()
    mesh_square1.Square(element_type="quad",lower_left_point=(0,10), side_length=5,nx=ncirc,ny=ncirc)

    mesh_square2 = deepcopy(mesh_square1)
    mesh_square2.points[:,0] += 95

    mesh = mesh_plate1 + mesh_plate2 + mesh_arc1 + mesh_arc2 + mesh_square1 + mesh_square2

    mesh.Extrude(length=0.5,nlong=1)

    mesh2 = deepcopy(mesh)
    mesh2.points[:,2] += 0.5
    mesh += mesh2

    if show_plot:
        mesh.SimplePlot()

    return mesh



def SubdivisionArc(center=(0.,0.), radius=1., nrad=16, ncirc=40,
        start_angle=0., end_angle=np.pi/2., element_type="tri", refinement=False, refinement_level=2):
    """Creating a mesh on circle using midpoint subdivision.
        This function is internally called from Mesh.Circle if
        'midpoint_subdivision' algorithm is selected
    """

    if start_angle!=0. and end_angle!=np.pi/2.:
        raise ValueError("Subdivision based arc only produces meshes for a quarter-circle arc for now")

    r = float(radius)
    h_r = float(radius)/2.
    nx = int(ncirc/4.)
    ny = int(nrad/2.)

    if nx < 3:
        warn("Number of division in circumferential direction too low")

    mesh = Mesh()
    mesh.Rectangle(element_type="quad", lower_left_point=(-1.,-1.),
        upper_right_point=(1.,1.), nx=nx, ny=ny)

    uv = np.array([
        [-1.,-1],
        [1.,-1],
        [1.,1],
        [-1.,1],
        ])

    t = np.pi/4.
    end_points = np.array([
        [0.,h_r*np.sin(t)],
        [h_r*np.cos(t),h_r*np.sin(t)],
        [r*np.cos(t),r*np.sin(t)],
        [0.,radius],
        ])

    edge_points = mesh.points[np.unique(mesh.edges),:]

    new_end_points = []
    new_end_points.append(end_points[0,:])
    new_end_points.append(end_points[1,:])
    new_end_points.append(end_points[2,:])

    tt = np.linspace(np.pi/4,np.pi/2,nx)
    x = r*np.cos(tt)
    y = r*np.sin(tt)
    interp_p = np.vstack((x,y)).T


    for i in range(1,len(x)-1):
        new_end_points.append([x[i], y[i]])
    new_end_points.append(end_points[3,:])
    new_end_points = np.array(new_end_points)

    new_uv = []
    new_uv.append(uv[0,:])
    new_uv.append(uv[1,:])
    new_uv.append(uv[2,:])

    L = 0.
    for i in range(1,interp_p.shape[0]):
        L += np.linalg.norm(interp_p[i,:] - interp_p[i-1,:])

    interp_uv = []
    last_uv = uv[2,:]
    for i in range(1,interp_p.shape[0]-1):
        val = (uv[3,:] - uv[2,:])*np.linalg.norm(interp_p[i,:] - interp_p[i-1,:])/L + last_uv
        last_uv = np.copy(val)
        interp_uv.append(val)
    interp_uv = np.array(interp_uv)

    new_uv = np.array(new_uv)
    if interp_uv.shape[0] !=0:
        new_uv = np.vstack((new_uv,interp_uv))
    new_uv = np.vstack((new_uv,uv[3,:]))

    from Florence.FunctionSpace import MeanValueCoordinateMapping
    new_points = np.zeros_like(mesh.points)
    # All nodes barring the ones lying on the arc
    for i in range(mesh.nnode - nx - 1):
        point = MeanValueCoordinateMapping(mesh.points[i,:], new_uv, new_end_points)
        new_points[i,:] = point
    # The nodes on the arc are not exactly on the arc
    # so they need to be snapped/clipped
    tt = np.linspace(np.pi/4,np.pi/2,nx+1)[::-1]
    x = r*np.cos(tt)
    y = r*np.sin(tt)
    new_points[mesh.nnode-nx-1:,:] = np.vstack((x,y)).T
    mesh.points = new_points

    rmesh = deepcopy(mesh)
    rmesh.points = mesh.Rotate(angle=-np.pi/2., copy=True)
    rmesh.points[:,1] *= -1.
    mesh += rmesh

    mesh.LaplacianSmoothing(niter=10)
    qmesh = Mesh()
    qmesh.Rectangle(element_type="quad", lower_left_point=(0.0,0.0),
        upper_right_point=(h_r*np.cos(t),h_r*np.sin(t)),
        nx=nx,
        ny=nx)
    mesh += qmesh

    # mesh.LaplacianSmoothing(niter=20)
    NodeSliderSmootherArc(mesh, niter=20)

    mesh.points[:,0] += center[0]
    mesh.points[:,1] += center[1]

    if refinement:
        mesh.Refine(level=refinement_level)

    if element_type == "tri":
        sys.stdout = open(os.devnull, "w")
        mesh.ConvertQuadsToTris()
        sys.stdout = sys.__stdout__

    return mesh




def SubdivisionCircle(center=(0.,0.), radius=1., nrad=16, ncirc=40,
        element_type="tri", refinement=False, refinement_level=2):
    """Creating a mesh on circle using midpoint subdivision.
        This function is internally called from Mesh.Circle if
        'midpoint_subdivision' algorithm is selected
    """

    r = float(radius)
    h_r = float(radius)/2.
    nx = int(ncirc/4.)
    ny = int(nrad/2.)

    if nx < 3:
        warn("Number of division in circumferential direction too low")

    mesh = Mesh()
    mesh.Rectangle(element_type="quad", lower_left_point=(-1.,-1.),
        upper_right_point=(1.,1.), nx=nx, ny=ny)

    uv = np.array([
        [-1.,-1],
        [1.,-1],
        [1.,1],
        [-1.,1],
        ])

    t = np.pi/4
    end_points = np.array([
        [-h_r*np.cos(t),h_r*np.sin(t)],
        [h_r*np.cos(t),h_r*np.sin(t)],
        [r*np.cos(t),r*np.sin(t)],
        [-r*np.cos(t),r*np.sin(t)],
        ])

    edge_points = mesh.points[np.unique(mesh.edges),:]

    new_end_points = []
    new_end_points.append(end_points[0,:])
    new_end_points.append(end_points[1,:])
    new_end_points.append(end_points[2,:])

    tt = np.linspace(np.pi/4,3*np.pi/4,nx)
    x = r*np.cos(tt)
    y = r*np.sin(tt)
    interp_p = np.vstack((x,y)).T


    for i in range(1,len(x)-1):
        new_end_points.append([x[i], y[i]])
    new_end_points.append(end_points[3,:])
    new_end_points = np.array(new_end_points)

    new_uv = []
    new_uv.append(uv[0,:])
    new_uv.append(uv[1,:])
    new_uv.append(uv[2,:])

    L = 0.
    for i in range(1,interp_p.shape[0]):
        L += np.linalg.norm(interp_p[i,:] - interp_p[i-1,:])

    interp_uv = []
    last_uv = uv[2,:]
    for i in range(1,interp_p.shape[0]-1):
        val = (uv[3,:] - uv[2,:])*np.linalg.norm(interp_p[i,:] - interp_p[i-1,:])/L + last_uv
        last_uv = np.copy(val)
        interp_uv.append(val)
    interp_uv = np.array(interp_uv)

    new_uv = np.array(new_uv)
    if interp_uv.shape[0] !=0:
        new_uv = np.vstack((new_uv,interp_uv))
    new_uv = np.vstack((new_uv,uv[3,:]))

    from Florence.FunctionSpace import MeanValueCoordinateMapping
    new_points = np.zeros_like(mesh.points)
    for i in range(mesh.nnode):
        point = MeanValueCoordinateMapping(mesh.points[i,:], new_uv, new_end_points)
        new_points[i,:] = point
    mesh.points = new_points

    rmesh = deepcopy(mesh)
    rmesh.points = mesh.Rotate(angle=np.pi/2., copy=True)
    mesh += rmesh
    rmesh.points = rmesh.Rotate(angle=np.pi/2., copy=True)
    mesh += rmesh
    rmesh.points = rmesh.Rotate(angle=np.pi/2., copy=True)
    mesh += rmesh

    mesh.LaplacianSmoothing(niter=10)
    qmesh = Mesh()
    qmesh.Rectangle(element_type="quad", lower_left_point=(-h_r*np.cos(t),-h_r*np.sin(t)),
        upper_right_point=(h_r*np.cos(t),h_r*np.sin(t)),
        nx=nx,
        ny=nx)
    mesh += qmesh

    mesh.LaplacianSmoothing(niter=20)

    mesh.points[:,0] += center[0]
    mesh.points[:,1] += center[1]

    if refinement:
        mesh.Refine(level=refinement_level)

    if element_type == "tri":
        sys.stdout = open(os.devnull, "w")
        mesh.ConvertQuadsToTris()
        sys.stdout = sys.__stdout__

    return mesh





def QuadBall(center=(0.,0.,0.), radius=1., n=10, element_type="hex"):
    """Creates a fully hexahedral mesh on sphere using midpoint subdivision algorithm
        by creating a cube and spherifying it using PostMesh's projection schemes

        inputs:

            n:                          [int] number of divsion in every direction.
                                        Given that this implementation is based on
                                        high order bases different divisions in
                                        different directions is not possible
    """


    try:
        from Florence import Mesh, BoundaryCondition, DisplacementFormulation, FEMSolver, LinearSolver
        from Florence import LinearElastic, NeoHookean
        from Florence.Tensor import prime_number_factorisation
    except ImportError:
        raise ImportError("This function needs Florence's core support")

    n = int(n)
    if n > 50:
        # Values beyond this result in >1M DoFs due to internal prime factoristaion splitting
        raise ValueError("The value of n={} (division in each direction) is too high".format(str(n)))

    if not isinstance(center,tuple):
        raise ValueError("The center of the circle should be given in a tuple with two elements (x,y,z)")
    if len(center) != 3:
        raise ValueError("The center of the circle should be given in a tuple with two elements (x,y,z)")

    if n == 2 or n==3 or n==5 or n==7:
        ps = [n]
    else:
        def factorise_all(n):
            if n < 2:
                n = 2
            factors = prime_number_factorisation(n)
            if len(factors) == 1 and n > 2:
                n += 1
                factors = prime_number_factorisation(n)
            return factors

        factors = factorise_all(n)
        ps = []
        for factor in factors:
            ps +=factorise_all(factor)

    # Do high ps first
    ps = np.sort(ps)[::-1].tolist()
    niter = len(ps)

    # IGS file for sphere with radius 1000.
    sphere_igs_file_content = SphereIGS()

    with open("sphere_cad_file.igs", "w") as f:
        f.write(sphere_igs_file_content)

    sys.stdout = open(os.devnull, "w")

    ndim = 3
    scale = 1000.
    condition = 1.e020

    mesh = Mesh()
    material = LinearElastic(ndim, mu=1., lamb=4.)
    # Keep the solver iterative for low memory consumption. All boundary points are Dirichlet BCs
    # so they will be exact anyway
    solver = LinearSolver(linear_solver="iterative", linear_solver_type="cg2",
        dont_switch_solver=True, iterative_solver_tolerance=1e-9)

    for it in range(niter):

        if it == 0:
            mesh.Parallelepiped(element_type="hex", nx=1, ny=1, nz=1, lower_left_rear_point=(-0.5,-0.5,-0.5),
                upper_right_front_point=(0.5,0.5,0.5))
        mesh.GetHighOrderMesh(p=ps[it], equally_spaced=True)

        boundary_condition = BoundaryCondition()
        boundary_condition.SetCADProjectionParameters(
            "sphere_cad_file.igs",
            scale=scale,condition=condition, project_on_curves=True, solve_for_planar_faces=True,
            modify_linear_mesh_on_projection=True, fix_dof_elsewhere=False
            )
        boundary_condition.GetProjectionCriteria(mesh)

        formulation = DisplacementFormulation(mesh)
        fem_solver = FEMSolver(
            number_of_load_increments=1,
            analysis_nature="linear",
            force_not_computing_mesh_qualities=True,
            report_log_level=0,
            optimise=True)

        solution = fem_solver.Solve(formulation=formulation, mesh=mesh,
                material=material, boundary_condition=boundary_condition, solver=solver)

        mesh.points += solution.sol[:,:,-1]
        mesh = mesh.ConvertToLinearMesh()

    os.remove("sphere_cad_file.igs")


    if not np.isclose(radius,1):
        mesh.points *= radius

    mesh.points[:,0] += center[0]
    mesh.points[:,1] += center[1]
    mesh.points[:,2] += center[2]

    if element_type == "tet":
        mesh.ConvertHexesToTets()

    sys.stdout = sys.__stdout__

    return mesh



def QuadBallSurface(center=(0.,0.,0.), radius=1., n=10, element_type="quad"):
    """Creates a surface quad mesh on sphere using midpoint subdivision algorithm
        by creating a cube and spherifying it using PostMesh's projection schemes.
        Unlike the volume QuadBall method there is no restriction on number of divisions
        here as no system of equations is solved

        inputs:

            n:                          [int] number of divsion in every direction.
                                        Given that this implementation is based on
                                        high order bases different divisions in
                                        different directions is not possible
    """

    try:
        from Florence import Mesh, BoundaryCondition, DisplacementFormulation, FEMSolver, LinearSolver
        from Florence import LinearElastic, NeoHookean
        from Florence.Tensor import prime_number_factorisation
    except ImportError:
        raise ImportError("This function needs Florence's core support")

    n = int(n)
    if not isinstance(center,tuple):
        raise ValueError("The center of the circle should be given in a tuple with two elements (x,y,z)")
    if len(center) != 3:
        raise ValueError("The center of the circle should be given in a tuple with two elements (x,y,z)")

    if n == 2 or n==3 or n==5 or n==7:
        ps = [n]
    else:
        def factorise_all(n):
            if n < 2:
                n = 2
            factors = prime_number_factorisation(n)
            if len(factors) == 1 and n > 2:
                n += 1
                factors = prime_number_factorisation(n)
            return factors

        factors = factorise_all(n)
        ps = []
        for factor in factors:
            ps +=factorise_all(factor)

    # Do high ps first
    ps = np.sort(ps)[::-1].tolist()
    niter = len(ps)

    sphere_igs_file_content = SphereIGS()
    with open("sphere_cad_file.igs", "w") as f:
        f.write(sphere_igs_file_content)

    sys.stdout = open(os.devnull, "w")

    ndim = 3
    scale = 1000.
    condition = 1.e020

    mesh = Mesh()
    material = LinearElastic(ndim, mu=1., lamb=4.)

    for it in range(niter):

        if it == 0:
            mesh.Parallelepiped(element_type="hex", nx=1, ny=1, nz=1, lower_left_rear_point=(-0.5,-0.5,-0.5),
                upper_right_front_point=(0.5,0.5,0.5))
            mesh = mesh.CreateSurface2DMeshfrom3DMesh()
            mesh.GetHighOrderMesh(p=ps[it], equally_spaced=True)
            mesh = mesh.CreateDummy3DMeshfrom2DMesh()
            formulation = DisplacementFormulation(mesh)
        else:
            mesh.GetHighOrderMesh(p=ps[it], equally_spaced=True)
            mesh = mesh.CreateDummy3DMeshfrom2DMesh()

        boundary_condition = BoundaryCondition()
        boundary_condition.SetCADProjectionParameters(
            "sphere_cad_file.igs",
            scale=scale,condition=condition,
            project_on_curves=True,
            solve_for_planar_faces=True,
            modify_linear_mesh_on_projection=True,
            fix_dof_elsewhere=False
            )
        boundary_condition.GetProjectionCriteria(mesh)
        nodesDBC, Dirichlet = boundary_condition.PostMeshWrapper(formulation, mesh, None, None, FEMSolver())

        mesh.points[nodesDBC.ravel(),:] += Dirichlet
        mesh = mesh.CreateSurface2DMeshfrom3DMesh()
        mesh = mesh.ConvertToLinearMesh()


    os.remove("sphere_cad_file.igs")


    if not np.isclose(radius,1):
        mesh.points *= radius

    mesh.points[:,0] += center[0]
    mesh.points[:,1] += center[1]
    mesh.points[:,2] += center[2]

    if element_type == "tri":
        mesh.ConvertQuadsToTris()

    sys.stdout = sys.__stdout__

    return mesh



def QuadBallSphericalArc(center=(0.,0.,0.), inner_radius=9., outer_radius=10., n=10, nthick=1,
    element_type="hex", cut_threshold=None, portion=1./8.):
    """Similar to QuadBall but hollow and creates only 1/8th or 1/4th or 1/2th of the sphere.
        Starting and ending angles are not supported. Radial division (nthick: to be consistent
        with SphericalArc method of Mesh class) is supported

        input:
            cut_threshold               [float] cutting threshold for element removal since this function is based
                                        QuadBall. Ideal value is zero, so prescribe a value as close to zero
                                        as possible, however that might not always be possible as the cut
                                        might take remove some wanted elements [default = -0.01]
            portion                     [float] portion of the sphere to take. Can only be 1/8., 1/4., 1/2.
    """

    assert inner_radius < outer_radius

    mm = QuadBallSurface(n=n, element_type=element_type)

    offset = outer_radius*2.
    if cut_threshold is None:
        cut_threshold = -0.01
    if portion == 1./8.:
        mm.RemoveElements(np.array([ [ cut_threshold, cut_threshold, cut_threshold], [ offset, offset,  offset]]))
    elif portion == 1./4.:
        mm.RemoveElements(np.array([ [ cut_threshold, cut_threshold, -offset], [ offset, offset,  offset]]))
    elif portion == 1./2.:
        mm.RemoveElements(np.array([ [ cut_threshold, -offset, -offset], [ offset, offset,  offset]]))
    else:
        raise ValueError("The value of portion can only be 1/8., 1/4. or 1/2.")

    radii = np.linspace(inner_radius, outer_radius, nthick+1)

    mesh = Mesh()
    mesh.element_type = "hex"
    mesh.nelem = 0
    mesh.nnode = 0

    for i in range(nthick):
        mm1, mm2 = deepcopy(mm), deepcopy(mm)
        if not np.isclose(radii[i],1):
            mm1.points *= radii[i]
        if not np.isclose(radii[i+1],1):
            mm2.points *= radii[i+1]

        if i == 0:
            elements = np.hstack((mm1.elements, mm1.nnode + mm2.elements)).astype(np.int64)
            mesh.elements = np.copy(elements)
            mesh.points = np.vstack((mm1.points, mm2.points))
        else:
            elements = np.hstack((mesh.elements[(i-1)*mm2.nelem:i*mm2.nelem,4:],
                mesh.nnode + mm2.elements)).astype(np.int64)
            mesh.elements = np.vstack((mesh.elements, elements))
            mesh.points = np.vstack((mesh.points, mm2.points))
        mesh.nelem = mesh.elements.shape[0]
        mesh.nnode = mesh.points.shape[0]

    mesh.elements = np.ascontiguousarray(mesh.elements, dtype=np.int64)
    mesh.nelem = mesh.elements.shape[0]
    mesh.nnode = mesh.points.shape[0]
    mesh.GetBoundaryFaces()
    mesh.GetBoundaryEdges()

    mesh.points[:,0] += center[0]
    mesh.points[:,1] += center[1]
    mesh.points[:,2] += center[2]

    return mesh



def Torus(show_plot=False):
    """Custom mesh for torus
    """

    raise NotImplementedError("Not fully implemented yet")

    # MAKE TORUS WORK
    from copy import deepcopy
    from numpy.linalg import norm
    mesh = Mesh()
    mesh.Circle(element_type="quad", ncirc=2, nrad=2)
    tmesh = deepcopy(mesh)
    arc = GeometricArc(start=(10,10,8),end=(10,10,-8))
    # arc.GeometricArc()
    nlong = 10
    points = mesh.Extrude(path=arc, nlong=nlong)
    # mesh.SimplePlot()
    # print points

    # elem_nodes = tmesh.elements[0,:]
    # p1 = tmesh.points[elem_nodes[0],:]
    # p2 = tmesh.points[elem_nodes[1],:]
    # p3 = tmesh.points[elem_nodes[2],:]
    # p4 = tmesh.points[elem_nodes[3],:]

    # E1 = np.append(p2 - p1, 0.0)
    # E2 = np.append(p4 - p1, 0.0)
    # E3 = np.array([0,0,1.])

    # E1 /= norm(E1)
    # E2 /= norm(E2)

    # # print E1,E2,E3

    # elem_nodes = mesh.elements[0,:]
    # p1 = mesh.points[elem_nodes[0],:]
    # p2 = mesh.points[elem_nodes[1],:]
    # p3 = mesh.points[elem_nodes[2],:]
    # p4 = mesh.points[elem_nodes[3],:]
    # p5 = mesh.points[elem_nodes[4],:]
    # e1 = p2 - p1
    # e2 = p4 - p1
    # e3 = p5 - p1

    # e1 /= norm(e1)
    # e2 /= norm(e2)
    # e3 /= norm(e3)
    # # print e1,e2,e3


    # # TRANSFORMATION MATRIX
    # Q = np.array([
    #     [np.einsum('i,i',e1,E1), np.einsum('i,i',e1,E2), np.einsum('i,i',e1,E3)],
    #     [np.einsum('i,i',e2,E1), np.einsum('i,i',e2,E2), np.einsum('i,i',e2,E3)],
    #     [np.einsum('i,i',e3,E1), np.einsum('i,i',e3,E2), np.einsum('i,i',e3,E3)]
    #     ])
    # mesh.points = np.dot(mesh.points,Q.T)
    # points = np.dot(points,Q)
    # E1 = np.array([1,0,0.])
    E3 = np.array([0.,0.,1.])
    nnode_2D = tmesh.points.shape[0]
    for i in range(nlong+1):
        # e1 = points[i,:][None,:]/norm(points[i,:])
        # Q = np.dot(E1[:,None],e1)
        # vpoints = np.dot(points,Q)

        e3 = points[i+1,:] - points[i,:]; e3 /= norm(e3)
        Q = np.dot(e3[:,None],E3[None,:])
        # print Q
        # print np.dot(Q,points[i,:][:,None])
        vpoints = np.dot(points,Q)

        # print current_points
        mesh.points[nnode_2D*i:nnode_2D*(i+1),:2] = tmesh.points + points[i,:2]
        mesh.points[nnode_2D*i:nnode_2D*(i+1), 2] = vpoints[i,2]


    # print Q
    # print tmesh.points

    # mesh = Mesh.HexahedralProjection()
    if show_plot:
        mesh.SimplePlot()

    return mesh









def NodeSliderSmootherArc(mesh, niter=10):
    """This is less than half-baked node slider smoother that only works
        for arc type meshes
    """

    if mesh.element_type != "quad":
        raise RuntimeError("Only implemented for quads")

    un_edges = np.unique(mesh.edges)
    points = mesh.points[un_edges,:]

    radius = mesh.Bounds[1,1]

    # For all x==0
    idx = np.where(np.isclose(mesh.points[:,0], 0.0)==True)[0]
    idx_sort = np.lexsort((mesh.points[idx,1],mesh.points[idx,0]))
    mesh.points[idx[idx_sort],1] = np.linspace(0.,radius, idx_sort.shape[0])

    # For all y==0
    idx = np.where(np.isclose(mesh.points[:,1], 0.0)==True)[0]
    idx_sort = np.lexsort((mesh.points[idx,0],mesh.points[idx,1]))
    mesh.points[idx[idx_sort],0] = np.linspace(0.,radius, idx_sort.shape[0])

    mesh.LaplacianSmoothing(niter)






























# -----------------------------------------------------------------------------------------
def SphereIGS():
    # IGS file for sphere with radius 1000.
    sphere_igs_file_content ="""
                                                                        S0000001
,,31HOpen CASCADE IGES processor 6.7,13HFilename.iges,                  G0000001
16HOpen CASCADE 6.7,31HOpen CASCADE IGES processor 6.7,32,308,15,308,15,G0000002
,1.,6,1HM,1,0.00001,15H20150628.043945,1E-07,1.007104,5Hroman,,11,0,    G0000003
15H20150628.043945,;                                                    G0000004
     186       1       0       0       0       0       0       000000000D0000001
     186       0       0       1       0                               0D0000002
     514       2       0       0       0       0       0       000010000D0000003
     514       0       0       1       1                               0D0000004
     510       3       0       0       0       0       0       000010000D0000005
     510       0       0       1       1                               0D0000006
     196       4       0       0       0       0       0       000010000D0000007
     196       0       0       1       1                               0D0000008
     116       5       0       0       0       0       0       000010400D0000009
     116       0       0       1       0                               0D0000010
     123       6       0       0       0       0       0       000010200D0000011
     123       0       0       1       0                               0D0000012
     123       7       0       0       0       0       0       000010200D0000013
     123       0       0       1       0                               0D0000014
     508       8       0       0       0       0       0       000010000D0000015
     508       0       0       2       1                               0D0000016
     502      10       0       0       0       0       0       000010000D0000017
     502       0       0       2       1                               0D0000018
     110      12       0       0       0       0       0       000010000D0000019
     110       0       0       1       0                               0D0000020
     504      13       0       0       0       0       0       000010001D0000021
     504       0       0       1       1                               0D0000022
     100      14       0       0       0       0      25       000010000D0000023
     100       0       0       1       0                               0D0000024
     124      15       0       0       0       0       0       000000000D0000025
     124       0       0       2       0                               0D0000026
     110      17       0       0       0       0       0       000010000D0000027
     110       0       0       1       0                               0D0000028
     110      18       0       0       0       0       0       000010000D0000029
     110       0       0       1       0                               0D0000030
     110      19       0       0       0       0       0       000010000D0000031
     110       0       0       1       0                               0D0000032
186,3,1,0;                                                       0000001P0000001
514,1,5,1;                                                       0000003P0000002
510,7,1,1,15;                                                    0000005P0000003
196,9,1.,11,13;                                                  0000007P0000004
116,0.,0.,0.,0;                                                  0000009P0000005
123,0.,0.,1.;                                                    0000011P0000006
123,1.,0.,-0.;                                                   0000013P0000007
508,4,1,17,1,0,1,0,19,0,21,1,0,1,0,27,1,17,2,1,1,0,29,0,21,1,1,  0000015P0000008
1,0,31;                                                          0000015P0000009
502,2,6.123233996E-17,-1.499759783E-32,1.,6.123233996E-17,       0000017P0000010
-1.499759783E-32,-1.;                                            0000017P0000011
110,360.,90.,0.,0.,90.,0.;                                       0000019P0000012
504,1,23,17,2,17,1;                                              0000021P0000013
100,0.,0.,0.,-1.836970199E-16,-1.,3.061616998E-16,1.;            0000023P0000014
124,1.,0.,-2.449293598E-16,0.,-2.449293598E-16,0.,-1.,0.,0.,1.,  0000025P0000015
0.,0.;                                                           0000025P0000016
110,0.,90.,-0.,0.,-90.,-0.;                                      0000027P0000017
110,0.,-90.,0.,360.,-90.,0.;                                     0000029P0000018
110,360.,-90.,0.,360.,90.,0.;                                    0000031P0000019
S      1G      4D     32P     19                                        T0000001
    """

    return sphere_igs_file_content