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

    directory = "/home/roman/"
    # filename = directory+"Almond3D.mat"
    filename = directory+"Almond3D_P3.mat"
    # filename = directory+"Sphere.mat"

    RunCurvedPlotter(filename)