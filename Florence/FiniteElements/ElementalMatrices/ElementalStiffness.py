import numpy as np 
import numpy.linalg as la
from KinematicMeasures import *
from Florence.Tensor import issymetric



#-------------------------------------------------------------------------------------------------------------------#
#                   VECTORISED ELEMENTAL MATRIX COMPUTATION USING EINSUM
#-------------------------------------------------------------------------------------------------------------------#
def Stiffness(function_space, formulation, material, fem_solver, 
    LagrangeElemCoords, EulerELemCoords, ElectricPotentialElem, elem):

    nvar = formulation.nvar
    ndim = formulation.ndim
    Domain = function_space

    inv = np.linalg.inv
    det = np.linalg.det

    # ALLOCATE
    stiffness = np.zeros((Domain.Bases.shape[0]*nvar,Domain.Bases.shape[0]*nvar),dtype=np.float64)
    tractionforce = np.zeros((Domain.Bases.shape[0]*nvar,1),dtype=np.float64)
    # B = np.zeros((MainData.Domain.Bases.shape[0]*nvar,MainData.MaterialArgs.H_VoigtSize),dtype=np.float64)
    B = np.zeros((Domain.Bases.shape[0]*nvar,material.H_VoigtSize),dtype=np.float64)

    
    # COMPUTE KINEMATIC MEASURES AT ALL INTEGRATION POINTS USING EINSUM (AVOIDING THE FOR LOOP)
    # MAPPING TENSOR [\partial\vec{X}/ \partial\vec{\varepsilon} (ndim x ndim)]
    ParentGradientX = np.einsum('ijk,jl->kil',Domain.Jm,LagrangeElemCoords)
    # MATERIAL GRADIENT TENSOR IN PHYSICAL ELEMENT [\nabla_0 (N)]
    MaterialGradient = np.einsum('ijk,kli->ijl',inv(ParentGradientX),Domain.Jm)
    # DEFORMATION GRADIENT TENSOR [\vec{x} \otimes \nabla_0 (N)]
    F = np.einsum('ij,kli->kjl',EulerELemCoords,MaterialGradient)

    # COMPUTE REMAINING KINEMATIC MEASURES
    StrainTensors = KinematicMeasures(F,fem_solver.analysis_nature)
    
    # UPDATE/NO-UPDATE GEOMETRY
    if fem_solver.requires_geometry_update:
        # MAPPING TENSOR [\partial\vec{X}/ \partial\vec{\varepsilon} (ndim x ndim)]
        ParentGradientx = np.einsum('ijk,jl->kil',Domain.Jm,EulerELemCoords)
        # SPATIAL GRADIENT TENSOR IN PHYSICAL ELEMENT [\nabla (N)]
        SpatialGradient = np.einsum('ijk,kli->ilj',inv(ParentGradientx),Domain.Jm)
        # COMPUTE ONCE detJ (GOOD SPEEDUP COMPARED TO COMPUTING TWICE)
        detJ = np.einsum('i,i,i->i',Domain.AllGauss[:,0],np.abs(det(ParentGradientX)),np.abs(StrainTensors['J']))
    else:
        # SPATIAL GRADIENT AND MATERIAL GRADIENT TENSORS ARE EQUAL
        SpatialGradient = np.einsum('ikj',MaterialGradient)
        # COMPUTE ONCE detJ (GOOD SPEEDUP COMPARED TO COMPUTING TWICE)
        detJ = np.einsum('i,i->i',Domain.AllGauss[:,0],np.abs(det(ParentGradientX)))


    # LOOP OVER GAUSS POINTS
    for counter in range(Domain.AllGauss.shape[0]): 

        if formulation.fields == 'electro_mechanics':
            # MATERIAL ELECTRIC FIELD  
            # ElectricFieldX = - np.dot(ElectricPotentialElem.T,MaterialGradient.T)
            # SPATIAL ELECTRIC FIELD
            ElectricFieldx = - np.dot(SpatialGradient[counter,:,:].T,ElectricPotentialElem)
            # COMPUTE SPATIAL ELECTRIC DISPLACEMENT
            ElectricDisplacementx = fem_solver.ElectricDisplacementx(material,StrainTensors,ElectricFieldx)
        else:
            ElectricFieldx, ElectricDisplacementx = [],[]


        # COMPUTE THE HESSIAN AT THIS GAUSS POINT
        # H_Voigt = MainData.Hessian(MainData.MaterialArgs,StrainTensors,ElectricFieldx,elem,counter)
        # H_Voigt = MainData.Hessian(StrainTensors,ElectricFieldx,elem,counter)[0]
        H_Voigt = material.Hessian(StrainTensors,ElectricFieldx,elem,counter)

        # COMPUTE CAUCHY STRESS TENSOR
        CauchyStressTensor = []
        if fem_solver.requires_geometry_update:
            CauchyStressTensor = material.CauchyStress(StrainTensors,ElectricFieldx,elem,counter)

        # COMPUTE THE TANGENT STIFFNESS MATRIX
        # BDB_1, t = MainData().ConstitutiveStiffnessIntegrand(B,nvar,ndim,MainData.AnalysisType,
            # MainData.Prestress,SpatialGradient[counter,:,:],CauchyStressTensor,ElectricDisplacementx,H_Voigt)
        BDB_1, t = formulation.ConstitutiveStiffnessIntegrand(B,nvar,ndim,fem_solver.analysis_nature,
            fem_solver.has_prestress,SpatialGradient[counter,:,:],CauchyStressTensor,ElectricDisplacementx,H_Voigt)
        
        
        # COMPUTE GEOMETRIC STIFFNESS MATRIX
        if fem_solver.requires_geometry_update:
            # BDB_1 += MainData().GeometricStiffnessIntegrand(SpatialGradient[counter,:,:],CauchyStressTensor,nvar,ndim)
            BDB_1 += formulation.GeometricStiffnessIntegrand(SpatialGradient[counter,:,:],CauchyStressTensor,nvar,ndim)
            # stiffness += (BDB_1)*detJ[counter]
            # INTEGRATE TRACTION FORCE
            tractionforce += t*detJ[counter]

        # INTEGRATE STIFFNESS
        stiffness += BDB_1*detJ[counter]


    # CHECK FOR SYMMETRY OF STIFFNESS MATRIX
    if fem_solver.debug:
        symmetric = issymetric(stiffness)
        if not symmetric:
            print u'\u2717'.encode('utf8')+' : ', 'Elemental stiffness matrix is not symmetric.',
            print ' Is this meant to be?'
        if issym and (fem_solver.IsSymmetricityComputed is False):
            print u'\u2713'.encode('utf8')+' : ', 'Elemental stiffness matrix is symmetric'
            fem_solver.IsSymmetricityComputed = True


    return stiffness, tractionforce 
#-------------------------------------------------------------------------------------------------------------------#






#-------------------------------------------------------------------------------------------------------------------#
# THIS METHOD IS KEPT FOR DEBUGGING PURPOSES AS IT RETAINS MUCH OF THE ORIGINAL NON-VECTORISED ALGORITHM, HOWEVER IT 
# IS COSTLIER THAN THE ORIGINAL AS IT MAKES DUPICATE COPIES OF KINEMATIC MEASURES. THERE ARE ONLY TWO LINES WHICH HAVE 
# CHANGED FROM THE ORIGINAL VERSION ONE IS THE LINE WHERE KinematicMeasures_NonVectorised IS CALLED AND THE OTHER ONE 
# IS COMPUTING detJ FOR NONLINEAR PROBLEMS AS detJ IS NOW A VECTOR NOT A SCALAR 
#-------------------------------------------------------------------------------------------------------------------#
def Stiffness_NonVectorised(MainData,material,LagrangeElemCoords,EulerELemCoords,ElectricPotentialElem,elem):

    nvar = MainData.nvar
    ndim = MainData.ndim
    # w = MainData.Quadrature.weights

    # ALLOCATE
    stiffness = np.zeros((MainData.Domain.Bases.shape[0]*nvar,MainData.Domain.Bases.shape[0]*nvar))
    tractionforce = np.zeros((MainData.Domain.Bases.shape[0]*nvar,1))
    B = np.zeros((MainData.Domain.Bases.shape[0]*nvar,MainData.MaterialArgs.H_VoigtSize))
    # Volume = 0

    # LOOP OVER GAUSS POINTS
    # for counter in range(0,MainData.Domain.AllGauss.shape[0]):
    for counter in range(MainData.Domain.AllGauss.shape[0]): 
        # GRADIENT TENSOR IN PARENT ELEMENT [\nabla_\varepsilon (N)]
        Jm = MainData.Domain.Jm[:,:,counter]
        # MAPPING TENSOR [\partial\vec{X}/ \partial\vec{\varepsilon} (ndim x ndim)]
        ParentGradientX=np.dot(Jm,LagrangeElemCoords) #
        # MATERIAL GRADIENT TENSOR IN PHYSICAL ELEMENT [\nabla_0 (N)]
        MaterialGradient = np.dot(la.inv(ParentGradientX),Jm)
        # MaterialGradient = la.solve(ParentGradientX,Jm) # A TINY BIT SLOWER
        # DEFORMATION GRADIENT TENSOR [\vec{x} \otimes \nabla_0 (N)]
        F = np.dot(EulerELemCoords.T,MaterialGradient.T)
        # COMPUTE REMAINING KINEMATIC MEASURES
        StrainTensors = KinematicMeasures_NonVectorised(F,MainData.AnalysisType,MainData.Domain.AllGauss.shape[0])

        # UPDATE/NO-UPDATE GEOMETRY
        if MainData.GeometryUpdate:
        # if MainData.GeometryUpdate or MainData.Prestress:
            # MAPPING TENSOR [\partial\vec{X}/ \partial\vec{\varepsilon} (ndim x ndim)]
            ParentGradientx=np.dot(Jm,EulerELemCoords)
            # SPATIAL GRADIENT TENSOR IN PHYSICAL ELEMENT [\nabla (N)]
            SpatialGradient = np.dot(la.inv(ParentGradientx),Jm).T
        else:
            SpatialGradient = MaterialGradient.T

        if MainData.Fields == 'ElectroMechanics':
            # MATERIAL ELECTRIC FIELD  
            # ElectricFieldX = - np.dot(ElectricPotentialElem.T,MaterialGradient.T)
            # SPATIAL ELECTRIC FIELD
            ElectricFieldx = - np.dot(SpatialGradient.T,ElectricPotentialElem)
            # COMPUTE SPATIAL ELECTRIC DISPLACEMENT
            ElectricDisplacementx = MainData.ElectricDisplacementx(MainData.MaterialArgs,StrainTensors,ElectricFieldx)
        else:
            ElectricFieldx, ElectricDisplacementx = [],[]

        # COMPUTE CAUCHY STRESS TENSOR
        CauchyStressTensor = []
        if MainData.AnalysisType == 'Nonlinear':
            CauchyStressTensor = MainData.CauchyStress(MainData.MaterialArgs,StrainTensors,ElectricFieldx,elem,counter)
        elif MainData.Prestress:
            CauchyStressTensor, LastCauchyStressTensor = MainData.CauchyStress(MainData.MaterialArgs,StrainTensors,ElectricFieldx,elem,counter)

        # COMPUTE THE HESSIAN AT THIS GAUSS POINT
        H_Voigt = MainData.Hessian(MainData.MaterialArgs,ndim,StrainTensors,ElectricFieldx,elem,counter)
        # COMPUTE THE TANGENT STIFFNESS MATRIX
        BDB_1, t = MainData().ConstitutiveStiffnessIntegrand(B,nvar,ndim,MainData.AnalysisType,MainData.Prestress,
                    SpatialGradient,CauchyStressTensor,ElectricDisplacementx,H_Voigt)


        if MainData.GeometryUpdate:
            # COMPUTE GEOMETRIC STIFFNESS MATRIX
            BDB_2 = MainData().GeometricStiffnessIntegrand(SpatialGradient,CauchyStressTensor,nvar,ndim)
            # COMPUTE ONCE detJ (GOOD SPEEDUP COMPARED TO COMPUTING TWICE)
            detJ = MainData.Domain.AllGauss[counter,0]*np.abs(la.det(ParentGradientX))*np.abs(StrainTensors['J'][counter])
            # INTEGRATE STIFFNESS
            stiffness += (BDB_1+BDB_2)*detJ
            # INTEGRATE TRACTION FORCE
            tractionforce += t*detJ
        else:
            # COMPUTE ONCE detJ (GOOD SPEEDUP COMPARED TO COMPUTING TWICE)
            detJ = MainData.Domain.AllGauss[counter,0]*np.abs(la.det(ParentGradientX))
            # detJ = MainData.Domain.AllGauss[counter,0]*np.abs(la.det(ParentGradientX))*np.abs(StrainTensors.J)
            if MainData.Prestress:
                # COMPUTE GEOMETRIC STIFFNESS MATRIX
                # BDB_2 = MainData().GeometricStiffnessIntegrand(SpatialGradient,LastCauchyStressTensor,nvar,ndim)
                # BDB_1 += BDB_2
                BDB_1 += MainData().GeometricStiffnessIntegrand(SpatialGradient,LastCauchyStressTensor,nvar,ndim)
                # BDB_1 = BDB_1 + BDB_2
            # INTEGRATE STIFFNESS
            stiffness += (BDB_1)*detJ
            if MainData.AnalysisType == 'Nonlinear' or MainData.Prestress:
                # INTEGRATE TRACTION FORCE
                tractionforce += t*detJ



    # # CHECK FOR SYMMETRY OF STIFFNESS MATRIX
    # for i in range(0,stiffness.shape[0]):
    #   for j in range(0,stiffness.shape[0]):
    #       if ~np.allclose(stiffness[i,j],stiffness[j,i]):
    #           print i,j

    return stiffness, tractionforce 















#-------------------------------------------------------------------------------------------------------------------#
#                       THE ORIGINAL OLDER NON-VECTORISED VERSION - KEPT FOR LEGACY/DEBUG
#-------------------------------------------------------------------------------------------------------------------#
# def Stiffness(MainData,LagrangeElemCoords,EulerELemCoords,ElectricPotentialElem,elem):

    # nvar = MainData.nvar
    # ndim = MainData.ndim
    # # w = MainData.Quadrature.weights

    # # ALLOCATE
    # stiffness = np.zeros((MainData.Domain.Bases.shape[0]*nvar,MainData.Domain.Bases.shape[0]*nvar))
    # tractionforce = np.zeros((MainData.Domain.Bases.shape[0]*nvar,1))
    # B = np.zeros((MainData.Domain.Bases.shape[0]*nvar,MainData.MaterialArgs.H_VoigtSize))
    # # Volume = 0

    # # LOOP OVER GAUSS POINTS
    # # for counter in range(0,MainData.Domain.AllGauss.shape[0]):
    # for counter in xrange(MainData.Domain.AllGauss.shape[0]): 
    #   # GRADIENT TENSOR IN PARENT ELEMENT [\nabla_\varepsilon (N)]
    #   Jm = MainData.Domain.Jm[:,:,counter]
    #   # MAPPING TENSOR [\partial\vec{X}/ \partial\vec{\varepsilon} (ndim x ndim)]
    #   ParentGradientX=np.dot(Jm,LagrangeElemCoords) #
    #   # MATERIAL GRADIENT TENSOR IN PHYSICAL ELEMENT [\nabla_0 (N)]
    #   MaterialGradient = np.dot(la.inv(ParentGradientX),Jm)
    #   # MaterialGradient = la.solve(ParentGradientX,Jm) # A TINY BIT SLOWER
    #   # DEFORMATION GRADIENT TENSOR [\vec{x} \otimes \nabla_0 (N)]
    #   F = np.dot(EulerELemCoords.T,MaterialGradient.T)
    #   # COMPUTE REMAINING KINEMATIC MEASURES
    #   StrainTensors = KinematicMeasures(F).Compute(MainData.AnalysisType)

    #   # UPDATE/NO-UPDATE GEOMETRY
    #   if MainData.GeometryUpdate:
    #   # if MainData.GeometryUpdate or MainData.Prestress:
    #       # MAPPING TENSOR [\partial\vec{X}/ \partial\vec{\varepsilon} (ndim x ndim)]
    #       ParentGradientx=np.dot(Jm,EulerELemCoords)
    #       # SPATIAL GRADIENT TENSOR IN PHYSICAL ELEMENT [\nabla (N)]
    #       SpatialGradient = np.dot(la.inv(ParentGradientx),Jm).T
    #   else:
    #       SpatialGradient = MaterialGradient.T

    #   if MainData.Fields == 'ElectroMechanics':
    #       # MATERIAL ELECTRIC FIELD  
    #       # ElectricFieldX = - np.dot(ElectricPotentialElem.T,MaterialGradient.T)
    #       # SPATIAL ELECTRIC FIELD
    #       ElectricFieldx = - np.dot(SpatialGradient.T,ElectricPotentialElem)
    #       # COMPUTE SPATIAL ELECTRIC DISPLACEMENT
    #       ElectricDisplacementx = MainData.ElectricDisplacementx(MainData.MaterialArgs,StrainTensors,ElectricFieldx)
    #   else:
    #       ElectricFieldx, ElectricDisplacementx = [],[]

    #   # COMPUTE CAUCHY STRESS TENSOR
    #   CauchyStressTensor = []
    #   if MainData.AnalysisType == 'Nonlinear':
    #       CauchyStressTensor = MainData.CauchyStress(MainData.MaterialArgs,StrainTensors,ElectricFieldx,elem,counter)
    #   elif MainData.Prestress:
    #       CauchyStressTensor, LastCauchyStressTensor = MainData.CauchyStress(MainData.MaterialArgs,StrainTensors,ElectricFieldx,elem,counter)

    #   # COMPUTE THE HESSIAN AT THIS GAUSS POINT
    #   H_Voigt = MainData.Hessian(MainData.MaterialArgs,ndim,StrainTensors,ElectricFieldx,elem,counter)
    #   # COMPUTE THE TANGENT STIFFNESS MATRIX
    #   BDB_1, t = MainData().ConstitutiveStiffnessIntegrand(B,nvar,ndim,MainData.AnalysisType,MainData.Prestress,
                            # SpatialGradient,CauchyStressTensor,ElectricDisplacementx,H_Voigt)


    #   if MainData.GeometryUpdate:
    #       # COMPUTE GEOMETRIC STIFFNESS MATRIX
    #       BDB_2 = MainData().GeometricStiffnessIntegrand(SpatialGradient,CauchyStressTensor,nvar,ndim)
    #       # COMPUTE ONCE detJ (GOOD SPEEDUP COMPARED TO COMPUTING TWICE)
    #       detJ = MainData.Domain.AllGauss[counter,0]*np.abs(la.det(ParentGradientX))*np.abs(StrainTensors.J)
    #       # INTEGRATE STIFFNESS
    #       stiffness += (BDB_1+BDB_2)*detJ
    #       # INTEGRATE TRACTION FORCE
    #       tractionforce += t*detJ
    #   else:
    #       # COMPUTE ONCE detJ (GOOD SPEEDUP COMPARED TO COMPUTING TWICE)
    #       detJ = MainData.Domain.AllGauss[counter,0]*np.abs(la.det(ParentGradientX))
    #       # detJ = MainData.Domain.AllGauss[counter,0]*np.abs(la.det(ParentGradientX))*np.abs(StrainTensors.J)
    #       if MainData.Prestress:
    #           # COMPUTE GEOMETRIC STIFFNESS MATRIX
    #           # BDB_2 = MainData().GeometricStiffnessIntegrand(SpatialGradient,LastCauchyStressTensor,nvar,ndim)
    #           # BDB_1 += BDB_2
    #           BDB_1 += MainData().GeometricStiffnessIntegrand(SpatialGradient,LastCauchyStressTensor,nvar,ndim)
    #           # BDB_1 = BDB_1 + BDB_2
    #       # INTEGRATE STIFFNESS
    #       stiffness += (BDB_1)*detJ
    #       if MainData.AnalysisType == 'Nonlinear' or MainData.Prestress:
    #           # INTEGRATE TRACTION FORCE
    #           tractionforce += t*detJ






    # # # CHECK FOR SYMMETRY OF STIFFNESS MATRIX
    # # for i in range(0,stiffness.shape[0]):
    # #     for j in range(0,stiffness.shape[0]):
    # #         if ~np.allclose(stiffness[i,j],stiffness[j,i]):
    # #             print i,j


    # return stiffness, tractionforce 
#-------------------------------------------------------------------------------------------------------------------#
