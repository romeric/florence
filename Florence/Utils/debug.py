import numpy as np

def debug(MainData,mesh,TotalDisp):
    """Generic florence debugger"""


    # FINITE ELEMENTS DEBUGGER

    tick = u'\u2713'.encode('utf8')+' : ' 
    cross = u'\u2717'.encode('utf8')+' : '

    # CHECK GAUSS POINTS
    sum_bases = np.sum(MainData.Domain.Bases,axis=0)
    sum_gbasesx = np.sum(MainData.Domain.gBasesx,axis=0)
    sum_gbasesy = np.sum(MainData.Domain.gBasesy,axis=0)
    sum_gbasesz = np.array([])
    if MainData.ndim==3:
        sum_gbasesz = np.sum(MainData.Domain.gBasesz,axis=0)
    sum_gauss_weights = np.sum(MainData.Quadrature.weights)
    
    # from Core.Supplementary.Tensors import makezero
    # sum_gbasesx = makezero(sum_gbasesx[:,None])[:,0]
    # sum_gbasesy = makezero(sum_gbasesy[:,None])[:,0]
    
    isones = np.ones_like(sum_bases)
    iszeros = np.zeros_like(sum_gbasesx)
    # print sum_bases
    # print sum_gbasesx
    # print sum_gbasesy
    # print sum_gauss_weights
    
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
    if MainData.ndim == 3:
        if np.allclose(sum_gbasesx,iszeros):
            print tick, 'Summation of Z-gradient of interpolation functions at every Gauss point is zero'
        else:
            print cross, 'Summation of Z-gradient of interpolation functions at every Gauss point is NOT zero'

    
    element_area = {'tri':2.0,'tet':4.0/3.0,'quad':4,'hex':8}
    if np.isclose(element_area[mesh.element_type],sum_gauss_weights):
        print tick, 'Summation of all quadrature weights equals area of the parent element'
    else:
        print tick, 'Summation of all quadrature weights does NOT equal area of the parent element'