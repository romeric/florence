import numpy as np
from copy import deepcopy
from warnings import warn
from .Mesh import Mesh
from .GeometricPath import *
from Florence.Tensor import totuple


__all__ = ['HarvesterPatch']

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

