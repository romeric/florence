import numpy as np

def KinematicMeasures(F,analysis_nature):
    """Computes the kinematic measures at all Gauss points for a given element.

    input:

        F:                  [ndarray (nofGauss x ndim x ndim)] Deformation gradient tensor evaluated at all integration points
        analysis_nature:       [str] type of analysis i.e. linear or non-linear 


    returns:

        StrainTensors:      [dict] a dictionary containing kinematic measures such as F, J, b etc evaluated at all integration points   
    """
    
    assert F.ndim == 3


    StrainTensors = {'F':F, 'J':np.linalg.det(F), 'b':np.einsum('ijk,ilk->ijl',F,F),
    'I':np.eye(F.shape[1],F.shape[1],dtype=np.float64)}


    # if analysis_nature == 'Nonlinear':
    #   # ADDITIONAL POLYCONVEX MEASURES - ACTIVATE IF NECESSARY
    #   # StrainTensors['H'] = np.einsum('i,ijk->ijk',StrainTensors['J'],np.einsum('ikj',np.linalg.inv(StrainTensors['F'])))
    #   # StrainTensors['H'] = np.einsum('i,ikj->ijk',StrainTensors['J'],np.linalg.inv(StrainTensors['F'])) # or
    #   # StrainTensors['g'] = np.einsum('ijk,ilk->ijl',StrainTensors['H'],StrainTensors['H'])
    #   # StrainTensors['C'] = np.einsum('ikj,ikl->ijl',StrainTensors['F'],StrainTensors['F']) 
    #   # StrainTensors['G'] = np.einsum('ikj,ikl->ijl',StrainTensors['H'],StrainTensors['H'])
    #   # StrainTensors['detC'] = np.einsum('i,i->i',StrainTensors['J'],StrainTensors['J'])
    #   pass


    # if analysis_nature=='linear':
    #     # LINEARISED KINEMATICS
    #     # MATERIAL GRADIENT OF DISPLACEMENT
    #     StrainTensors['Gradu'] = F - StrainTensors['I']
    #     # SMALL STRAIN TENSOR IS THE LINEARISED VERSION OF GREEN-LAGRANGE STRAIN TENSOR
    #     StrainTensors['strain'] = 0.5*(StrainTensors['Gradu'] + np.einsum('ikj',StrainTensors['Gradu']))


    # LINEARISED KINEMATICS
    # MATERIAL GRADIENT OF DISPLACEMENT
    StrainTensors['Gradu'] = F - StrainTensors['I']
    # SMALL STRAIN TENSOR IS THE LINEARISED VERSION OF GREEN-LAGRANGE STRAIN TENSOR
    StrainTensors['strain'] = 0.5*(StrainTensors['Gradu'] + np.einsum('ikj',StrainTensors['Gradu']))

        
    return StrainTensors









def KinematicMeasures_NonVectorised(F,AnalysisType,ncounter):
    """Computes the kinematic measures at all Gauss points for a given element.

    input:

        F:                  [ndarray (nofGauss x ndim x ndim)] Deformation gradient tensor evaluated at all integration points
        AnalysisType:       [str] type of analysis i.e. linear or non-linear 
        ncounter:           [int] no of Gauss points


    returns:

        StrainTensors:      [dict] a dictionary containing kinematic measures such as J, b etc evaluated at all integration points  
    """
    
    assert len(F.shape) == 2


    StrainTensors = {'F':[F]*ncounter, 'J':[np.linalg.det(F)]*ncounter, 'b':[np.dot(F,F.T)]*ncounter,
    'I':np.eye(F.shape[1],F.shape[1],dtype=np.float64)}


    if AnalysisType == 'Nonlinear':
        # ADDITIONAL POLYCONVEX MEASURES - ACTIVATE IF NECESSARY
        # StrainTensors['H'] = [StrainTensors['J'][0]*np.linalg.inv(F)]*ncounter
        pass


    elif AnalysisType=='Linear':
        # LINEARISED KINEMATICS
        # MATERIAL GRADIENT OF DISPLACEMENT
        StrainTensors['Gradu'] = F - StrainTensors['I']
        # SMALL STRAIN TENSOR IS THE LINEARISED VERSION OF GREEN-LAGRANGE STRAIN TENSOR
        StrainTensors['strain'] = [0.5*(StrainTensors['Gradu'] + StrainTensors['Gradu'].T)]*ncounter
        # RE-ASSIGN
        StrainTensors['Gradu'] = [F - StrainTensors['I']]*ncounter


    return StrainTensors






#                   OLD VERSION - KEPT FOR LEGACY/DEBUG
#--------------------------------------------------------------------------------#
# class KinematicMeasures(object):
#   """docstring for KinematicMeasures"""
#   def __init__(self, F):
#       super(KinematicMeasures, self).__init__()
#       self.F = F

#   def Compute(self,AnalysisType):

#       # F = self.F
#       self.J = np.linalg.det(self.F)
#       self.I = np.eye(self.F.shape[0],self.F.shape[0],dtype=np.float64)

#       self.b = np.dot(self.F,self.F.T)
#       # self.C = np.dot(self.F.T,self.F)

#       # self.Gradu = self.F - self.I
#       # self.strain = 0.5*(self.Gradu + self.Gradu.T)

#       if AnalysisType=='Nonlinear':
#           # self.C = np.dot(self.F.T,self.F)
#           # self.b = np.dot(self.F,self.F.T)
#           # self.E = 0.5*(self.C-self.I)
#           # self.e = 0.5*(self.I-np.linalg.inv(self.b)) 

#           # self.Gradu = self.F - self.I
#           # self.strain = 0.5*(self.Gradu + self.Gradu.T)
            

#           # POLYCONVEX KINEMATICS MEASURE
#           self.H = self.J*np.linalg.inv(self.F).T
#           # self.G = self.H.T*self.H
#       elif AnalysisType=='Linear':
#           # LINEARISED KINEMATICS
#           # MATERIAL GRADIENT OF DISPLACEMENT
#           self.Gradu = self.F - self.I
#           # SMALL STRAIN TENSOR IS THE LINEARISED VERSION OF GREEN-LAGRANGE STRAIN TENSOR
#           self.strain = 0.5*(self.Gradu + self.Gradu.T)

#       return self


