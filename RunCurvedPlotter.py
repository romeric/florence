import numpy as np
from scipy.io import savemat, loadmat

from Core import Mesh
from Core.FiniteElements.PostProcess import *

def RunCurvedPlotter(filename):
    DictOutput = loadmat(filename)

    mesh = Mesh()
    mesh.elements = DictOutput['elements']
    mesh.element_type = DictOutput['element_type']
    mesh.points = DictOutput['points']
    TotalDisp = DictOutput['TotalDisp']
    C = DictOutput['C']
    ScaledJacobian = DictOutput['ScaledJacobian']
    ProjFlags = DictOutput['ProjFlags']
    mesh.nelem = mesh.elements.shape[0]

    PostProcess.HighOrderCurvedPatchPlot(mesh,TotalDisp,QuantityToPlot=ScaledJacobian.flatten(),
            ProjectionFlags=ProjFlags,InterpolationDegree=40)


if __name__ == '__main__':

    directory = "/home/roman/Dropbox/"
    # filename = directory+"Almond3D_P2.mat"
    filename = directory+"Almond3D_P3.mat"
    # filename = directory+"Almond3D_P4.mat"
    # filename = directory+"Almond3D_H2_P4.mat"
    # filename = directory+"Sphere.mat"

    RunCurvedPlotter(filename)



    # ######################################################################
    # mesh = Mesh()
    # mesh.ReadGIDMesh("/home/roman/Dropbox/Florence/Examples/FiniteElements/Falcon3D/falcon_iso.dat","tet",0)
    # # mesh.ReadGIDMesh("/home/roman/Dropbox/Florence/Examples/FiniteElements/Almond3D/almond_H2.dat","tet",0)

    # mesh.GetFacesTet()
    # mesh.GetEdgesTet()
    # face_flags = mesh.GetInteriorFacesTet()
    # mesh.GetElementsFaceNumberingTet()
    # boundary_face_to_element = mesh.GetElementsWithBoundaryFacesTet()

    # Dict = {'points':mesh.points, 'elements':mesh.elements, 
    #     'element_type':mesh.element_type, 'faces':mesh.faces,
    #     'edges':mesh.edges, 'all_faces':mesh.all_faces, 'all_edges':mesh.all_edges,
    #     'face_flags':face_flags,'face_to_element':mesh.face_to_element,
    #     'boundary_face_to_element':boundary_face_to_element}
    # savemat('/home/roman/Dropbox/Florence/Examples/FiniteElements/Falcon3D/Falcon3DIso.mat',Dict)