from __future__ import division, print_function
from warnings import warn
import numpy as np
from Florence.Tensor import makezero, makezero3d, unique2d


class GeometricPath(object):
    """Construct a geometric path for mesh extrusions
    """

    def __init__(self):
        pass

    def ComputeExtrusion(self,base_mesh):
        pass





class GeometricLine(GeometricPath):

    def __init__(self, start=(0.,0.,0.), end=(1.,2.,3)):
        """Constructs a line given the start and ending points 
        """

        self.start = np.array(start)
        self.end = np.array(end)
        self.length = np.linalg.norm(self.start - self.end)

    
    def ComputeExtrusion(self, nlong=10):
        """Computes extrusion of base_mesh along the self (line)
            using equal spacing 

            input:
                nlong:                      [int] number of discretisation along
                                            the arc
            returns:
                points:                     [1D array] of discretisation along arc
        """


        from numpy.linalg import norm

        npoints = nlong + 1
        x = np.linspace(self.start[0],self.end[0],npoints)
        y = np.linspace(self.start[1],self.end[1],npoints)
        z = np.linspace(self.start[2],self.end[2],npoints)

        points = np.array([x,y,z]).T.copy()
        # IF ALL THE Z-COORDIDATES ARE ZERO, THEN EXTRUSION NOT POSSIBLE
        if np.allclose(points[:,2],0.0):
            warn("Third dimension is zero. Extrusion will fail, unless axes are swapped")

        return points






class GeometricArc(GeometricPath):

    def __init__(self, center=(0.,0.,0.), start=(-1.,-2.,-3.), end=(3.,2.,1.)):
        """Constructs an arc given the center of the arc and start
            and ending points of the arc
        """
        
        from numpy.linalg import norm

        self.center = np.array(center)
        self.start = np.array(start)
        self.end = np.array(end)

        # COMPUTE ARC'S RADIUS
        self.radius = norm(self.start - self.center)
        if not np.isclose(self.radius, norm(self.end - self.center)):
            raise ValueError("An arc could not be constructed with the given points")

        # COMPUTE ANGLE OF ARC
        line0 = self.start - self.center
        line1 = self.end   - self.center
        self.angle = np.arccos(norm(line0*line1)/norm(line0)/norm(line1))
        # self.angle = np.arccos(np.abs(np.dot(line0,line1))/norm(line0)/norm(line1))


    def Construct(self, center=(0.,0.), start=(-2.,2.), end=(2.,2.)):
        """Constructs an arc given the center of the arc and start
            and ending points of the arc
        """
        self.init(center=center, start=start, end=end)


    def SetAngles(self, start_angle_phi=None, end_angle_phi=None,
        start_angle_theta=None, end_angle_theta=None):
        """Set the arc angle in radians
        """

        if start_angle_phi is not None:
            self.start_angle_phi = start_angle_phi
        if end_angle_phi is not None:
            self.end_angle_phi = end_angle_phi
        if start_angle_theta is not None:
            self.start_angle_theta = start_angle_theta
        if end_angle_theta is not None:
            self.end_angle_theta = end_angle_theta


    def ComputeExtrusion(self, nlong=10):
        """Computes extrusion of a base mesh along an arc

            input:
                nlong:                      [int] number of discretisation along
                                            the arc
            returns:
                points:                     [1D array] of discretisation along arc
        """

        from numpy.linalg import norm

        npoints = nlong + 1

        line0 = self.start - self.center
        line1 = self.end   - self.center

        # CHECK IF ALL THE POINTS LIE ON A LINE
        if np.isclose(np.abs(self.angle),0.0) or \
            np.isclose(np.abs(self.angle),np.pi) or \
            np.isclose(np.abs(self.angle),2.*np.pi):
            self.angle += 1e-08
            warn("The arc points are nearly colinear")

        t = np.linspace(0,self.angle,npoints+1)

        points = (np.einsum("i,j",np.sin(self.angle - t),line0) + \
            np.einsum("i,j",np.sin(t),line1))/np.sin(self.angle) + self.center
        makezero(points)

        # IF ALL THE Z-COORDIDATES ARE ZERO, THEN EXTRUSION NOT POSSIBLE
        if np.allclose(points[:,2],0.0):
            warn("Third dimension is zero. Extrusion will fail, unless axes are swapped")

        return points




        # # print(tx*self.angle)
        # # FIND NORMAL TO THE PLANE OF ARC
        # normal_to_arc_plane = np.cross(self.start - self.center, self.end - self.center)
        # # FIND TWO ORTHOGONAL VECTORS IN THE PLANE OF ARC
        # X = self.start/norm(self.start)
        # Y = np.cross(normal_to_arc_plane, self.start) / norm(np.cross(normal_to_arc_plane, self.start))
        # # print(normal_to_arc_plane, self.start)
        # # t = np.linspace(0, self.angle, npoints+1)
        # t = np.linspace(0., self.angle, npoints+1)
        # # t = np.linspace(0, self.angle+0.2284, npoints+1)
        # # x = norm(np.dot(self.start - self.center, self.end - self.center))
        # # print(np.arccos(x))
        # # print(self.angle)

        # points = np.einsum("i,j",self.radius*np.cos(t),X) + np.einsum("i,j",self.radius*np.sin(t),Y)
        # # points = np.einsum("i,j",self.radius*np.sin(t),X) + np.einsum("i,j",self.radius*np.cos(t),Y)
        # # y = np.einsum("i,j",self.radius*np.sin(t),Y)
        # # print(X,Y,t)
        # print(points)
        # # print(np.dot(X,Y))

        # # t= np.linspace(-np.pi/2,np.pi,100)
        # # X = np.array([0,1])
        # # Y = np.array([1,0])
        # # # points = np.einsum("i,j",self.radius*np.cos(t),X) + np.einsum("i,j",self.radius*np.sin(t),Y)
        # # points = np.einsum("i,j",self.radius*np.sin(t),X) + np.einsum("i,j",self.radius*np.cos(t),Y)
        # # makezero(points)
        # # print(points)
        # exit()

        # import matplotlib.pyplot as plt
        # plt.plot(points[:,0],points[:,1],'ro')
        # plt.axis('equal')
        # plt.show()

        # import os
        # os.environ['ETS_TOOLKIT'] = 'qt4'
        # from mayavi import mlab

        # figure = mlab.figure(bgcolor=(1,1,1),fgcolor=(1,1,1),size=(800,600))
        # mlab.plot3d(points[:,0],points[:,1],points[:,2])
        # mlab.show()
        # exit()

        # print(self.start_angle, self.end_angle)
        # x = self.radius*np.cos(t)
        # y = self.radius*np.sin(t)


        # points = np.array([x,y]).T.copy()
        # makezero(points)

        # t = np.linspace(0,40.,npoints+1)