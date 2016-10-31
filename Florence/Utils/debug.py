import numpy as np

def debug(function_space, quadrature_rule,mesh,TotalDisp):
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
    
    if np.allclose(sum_bases,isones):
        print tick, 'Summation of all interpolation functions at every Gauss point is one'
    else:
        print cross, 'Summation of all interpolation functions at every Gauss point is NOT one'
    if np.allclose(sum_gbasesx,iszeros):
        print tick, 'Summation of X-gradient of interpolation functions at every Gauss point is zero'
    else:
        print cross, 'Summation of X-gradient of interpolation functions at every Gauss point is NOT zero'
    if np.allclose(sum_gbasesy,iszeros):
        print tick, 'Summation of Y-gradient of interpolation functions at every Gauss point is zero'
    else:
        print cross, 'Summation of Y-gradient of interpolation functions at every Gauss point is NOT zero'
    if ndim == 3:
        if np.allclose(sum_gbasesx,iszeros):
            print tick, 'Summation of Z-gradient of interpolation functions at every Gauss point is zero'
        else:
            print cross, 'Summation of Z-gradient of interpolation functions at every Gauss point is NOT zero'

    
    element_area = {'tri':2.0,'tet':4.0/3.0,'quad':4,'hex':8}
    if np.isclose(element_area[mesh.element_type],sum_gauss_weights):
        print tick, 'Summation of all quadrature weights equals area of the parent element'
    else:
        print tick, 'Summation of all quadrature weights does NOT equal area of the parent element'