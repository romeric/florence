
from __future__ import division
try:
    import meshpy.triangle as triangle
except ImportError:
    meshpy = None
    # print 'meshpy module was not found!'
import numpy as np
import numpy.linalg as la
from NormalDistance import NormalDistance

class MeshGeneration(object):
    """ class info """
    def __init__(self):
        super(MeshGeneration, self).__init__()
        self.arg = []



    def GenerateMesh(self,points,facets,ref_func=None,max_vol=None,gen_faces=False):

	   info = triangle.MeshInfo()
	   info.set_points(points)
	   info.set_facets(facets)

	   return triangle.build(info, max_volume=max_vol,generate_faces=gen_faces,refinement_func=ref_func)



    # Find if edges are at the boundary
    def BoundaryEdges(self,points,facets,mesh_points,mesh_edges):
	    # """This function returns an array containing boundary edges of the computational mesh"""
	    #  Find slope of geometry edges
	    geo_edges = np.array(facets); geo_points = np.array(points)
	    # Allocate
	    mesh_boundary_edges = np.zeros((1,2),dtype=int)
	    # Loop over geometry lines
	    for i in range(0,geo_edges.shape[0]):
		    # Each edge has two nodes - get their coordinates
		    geo_edge_coord = geo_points[geo_edges[i]]
		    geo_x1 = geo_edge_coord[0,0];		geo_y1 = geo_edge_coord[0,1]
		    geo_x2 = geo_edge_coord[1,0];		geo_y2 = geo_edge_coord[1,1]

		    # Find slope of this line 
		    geo_angle = np.arctan((geo_y2-geo_y1)/(geo_x2-geo_x1)) #*180/np.pi

		    # Now for each of these geometry lines loop over all mesh edges
		    for j in range(0,mesh_edges.shape[0]):
			    mesh_edge_coord = mesh_points[mesh_edges[j]]
			    mesh_x1 = mesh_edge_coord[0,0];			mesh_y1 = mesh_edge_coord[0,1]
			    mesh_x2 = mesh_edge_coord[1,0];			mesh_y2 = mesh_edge_coord[1,1]

			    # Find slope of this line 
			    mesh_angle = np.arctan((mesh_y2-mesh_y1)/(mesh_x2-mesh_x1))

			    # Check if geometry and mesh edges are parallel
			    if np.allclose(geo_angle,mesh_angle,atol=1e-12):
				    # If parallel then find the normal distance between them
				    P1 = np.array([geo_x1,geo_y1,0]);				P2 = np.array([geo_x2,geo_y2,0])		# 1st line's coordinates
				    P3 = np.array([mesh_x1,mesh_y1,0]);				P4 = np.array([mesh_x2,mesh_y2,0])		# 2nd line's coordinates

				    dist = NormalDistance(P1,P2,P3,P4)
				    # If normal distance is zero then the mesh line is on the boundary
				    if np.allclose(dist,0,atol=1e-14):
					    mesh_boundary_edges = np.append(mesh_boundary_edges,mesh_edges[j].reshape(1,2),axis=0)

	    # Costly Computation (find a better way to do this, the problem is allocating mesh_boundary_edges in the first place) 
	    return np.delete(mesh_boundary_edges,0,0)


    def BEM_Mesh_from_2D_Mesh(self,mesh_points,mesh_boundary_edges,C):
    	# This technique is used for boundary-element-only problems (not coupled fem-bem)  
        counter = 1
        # Do not assign one variable to another - make a copy instead if you need the original
        new_mesh_boundary_edges = np.copy(mesh_boundary_edges) 
        mapp = np.zeros((mesh_boundary_edges.shape[0],2))
        for i in range(0,mesh_boundary_edges.shape[0]):
            for j in range(0,mesh_boundary_edges.shape[1]):
            	if new_mesh_boundary_edges[i,j] >= 0:
            		xco, yco =  np.nonzero(new_mesh_boundary_edges==new_mesh_boundary_edges[i,j])
            		mapp[counter-1,0] = mesh_boundary_edges[xco[0],yco[0]]
            		mapp[counter-1,1] = counter-1
                	for k in range(0,2):
               			new_mesh_boundary_edges[xco[k],yco[k]] = -counter

                	counter+=1
                else:
                	break
        new_mesh_boundary_edges = -1*(new_mesh_boundary_edges+1)

        # Don't do it this way
        # # new_mesh_boundary_edges is for linear basis, for higher order basis 
        # # we need another mapping
        # if C>0:
        # 	# Take the first element of new_mesh_boundary (0,0) which should always 
        # 	# be zero and don't change it, then find the largest element of the array
        # 	max_elem = np.max(new_mesh_boundary_edges)
        # 	# Find number of elements
        # 	nelem = new_mesh_boundary_edges.shape[0]
        # 	pass

        return new_mesh_boundary_edges, mapp




# if __name__ == '__main__':

# from ProblemData import ProblemData_BEM_1, ProblemData_BEM_2
# C, points, facets = ProblemData_BEM_2()

# mesh = MeshGeneration().GenerateMesh(points,facets,max_vol=0.02,gen_faces=True)  # Calling from instance
# # mesh = GenerateMesh(points,facets,max_vol=0.2,gen_faces=True)
# mesh_points = np.array(mesh.points)
# mesh_tris = np.array(mesh.elements)
# mesh_edges = np.array(mesh.faces)

# import matplotlib.pyplot as pt
# pt.triplot(mesh_points[:, 0], mesh_points[:, 1], mesh_tris)

# mesh_boundary_edges = MeshGeneration().BoundaryEdges(points,facets,mesh_points,mesh_edges)
# # print mesh_boundary_edges

# new_mesh_boundary_edges, mapp =  MeshGeneration().BEM_Mesh_from_2D_Mesh(mesh_points,mesh_boundary_edges,C)
# # print new_mesh_boundary_edges

# # print mesh_points[mesh_boundary_edges]
# # print mesh_points.shape, mesh_edges.shape, mesh_boundary_edges.shape

# # Now build an algorithm such that it takes nodes of the edge line
# # remove the redandency (each node appears exactly twice) and builds 
# # two element connectivity and nodal coordinates one for linear and and one 
# # for higher basis functions

# # Note: the edge connectivity is basically element connectivity in 1D just renumber
# # it and then find the corresponding nodal coordinates and the linear part is done 


# pt.show()