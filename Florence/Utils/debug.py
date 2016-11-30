from __future__ import print_function
import numpy as np

def FEMDebugger(function_space, quadrature_rule, mesh):
    """Generic florence debugger for finite elements"""


    Domain = function_space
    Quadrature = quadrature_rule

    ndim = mesh.points.shape[1]

    tick = u'\u2713'.encode('utf8')+' : ' 
    cross = u'\u2717'.encode('utf8')+' : '

    # CHECK GAUSS POINTS
    sum_bases = np.sum(Domain.Bases,axis=0)
    sum_gbasesx = np.sum(Domain.gBasesx,axis=0)
    sum_gbasesy = np.sum(Domain.gBasesy,axis=0)
    sum_gbasesz = np.array([])
    if ndim==3:
        sum_gbasesz = np.sum(Domain.gBasesz,axis=0)
    sum_gauss_weights = np.sum(Quadrature.weights)
    
    isones = np.ones_like(sum_bases)
    iszeros = np.zeros_like(sum_gbasesx)
    
    # CHECK QUADRATURE RULE
    element_area = {'tri':2.0,'tet':4.0/3.0,'quad':4,'hex':8}
    fem_element_area = {'tri':sum_gauss_weights,'tet':sum_gauss_weights,
        'quad':sum_gauss_weights**2,'hex':sum_gauss_weights**3}
    if np.isclose(element_area[mesh.element_type],fem_element_area[mesh.element_type]):
        print(tick, 'Summation of all quadrature weights equals area of the parent element')
    else:
        print(cross, 'Summation of all quadrature weights does NOT equal area of the parent element')

    # CHECK BASIS FUNCTIONS
    if np.allclose(sum_bases,isones):
        print(tick, 'Summation of all interpolation functions at every Gauss point is one')
    else:
        print(cross, 'Summation of all interpolation functions at every Gauss point is NOT one')
    if np.allclose(sum_gbasesx,iszeros):
        print(tick, 'Summation of X-gradient of interpolation functions at every Gauss point is zero')
    else:
        print(cross, 'Summation of X-gradient of interpolation functions at every Gauss point is NOT zero')
    if np.allclose(sum_gbasesy,iszeros):
        print(tick, 'Summation of Y-gradient of interpolation functions at every Gauss point is zero')
    else:
        print(cross, 'Summation of Y-gradient of interpolation functions at every Gauss point is NOT zero')
    if ndim == 3:
        if np.allclose(sum_gbasesx,iszeros):
            print(tick, 'Summation of Z-gradient of interpolation functions at every Gauss point is zero')
        else:
            print(cross, 'Summation of Z-gradient of interpolation functions at every Gauss point is NOT zero')

    # CHECK MESH
    mesh.CheckNodeNumbering()


def NonlinearMechanicsDebugger(mesh,formulation,material):

    tick = u'\u2713'.encode('utf8')+' : ' 
    cross = u'\u2717'.encode('utf8')+' : '
    
    from Florence.FiniteElements.ElementalMatrices.KinematicMeasures import KinematicMeasures
    StrainTensors = KinematicMeasures(np.asarray([np.eye(self.ndim,self.ndim)]*2),"nonlinear")
    stress = material.CauchyStress(StrainTensors,np.zeros((self.ndim,1)))

    if formulation.fields == "electro-mechanics":
        D = material.ElectricDisplacementx(StrainTensors,np.zeros((self.ndim,1)))

    if not np.allclose(stress,0.0): # and not formulation.has_prestress
        print(tick, 'Cauchy stress at the origin is zero')
    else:
        print(cross, 'Cauchy stress at the origin is NOT zero')

    if formulation.fields == "electro-mechanics":
        if not np.allclose(D,0.0):
            print(tick, 'Electric displacement at the origin is zero')
        else:
            print(cross, 'Electric displacement at the origin is NOT zero')