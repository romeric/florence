import numpy as np 
import numpy.linalg as la
from Florence.FiniteElements.ElementalMatrices.KinematicMeasures import *


def ComputeErrorNorms(MainData,mesh,nmesh,AnalyticalSolution,Domain,Quadrature,MaterialArgs):

    # AT THE MOMENT THE ERROR NORMS ARE COMPUTED FOR LINEAR PROBLEMS ONLY
    if MainData.GeometryUpdate:
        raise ValueError('The error norms are computed for linear problems only.')

    # INITIATE/ALLOCATE
    C = MainData.C 
    ndim = MainData.ndim
    nvar = MainData.nvar
    elements = nmesh.elements
    points = nmesh.points
    nelem = elements.shape[0]
    nodeperelem = elements.shape[1]
    w = Quadrature.weights

    # TotalDisp & TotalPot ARE BOTH THIRD ORDER TENSOR (3RD DIMENSION FOR INCREMENTS) - TRANCATE THEM UNLESS REQUIRED
    TotalDisp = MainData.TotalDisp[:,:,-1]
    TotalPot  = MainData.TotalPot[:,:,-1]

    # # print TotalDisp
    # TotalDispa = np.zeros(TotalDisp.shape)
    # uxa = (points[:,1]*np.cos(points[:,0])); uxa=np.zeros(uxa.shape)
    # uya = (points[:,0]*np.sin(points[:,1]))
    # TotalDispa[:,0]=uxa
    # TotalDispa[:,1]=uya
    # # print TotalDispa
    # # print TotalDisp
    # print np.concatenate((TotalDispa,TotalDisp),axis=1)
    # # print points


    # ALLOCATE
    B = np.zeros((Domain.Bases.shape[0]*nvar,MaterialArgs.H_VoigtSize))
    E_nom = 0; E_denom = 0; L2_nom = 0; L2_denom = 0
    # LOOP OVER ELEMENTS
    for elem in range(0,nelem):
        xycoord = points[elements[elem,:],:]    
        # GET THE NUMERICAL SOLUTION WITHIN THE ELEMENT (AT NODES)
        ElementalSol = np.zeros((nodeperelem,nvar))
        ElementalSol[:,:ndim] = TotalDisp[elements[elem,:],:]
        ElementalSol[:,ndim]  = TotalPot[elements[elem,:],:].reshape(nodeperelem)
        # GET THE ANALYTICAL SOLUTION WITHIN THE ELEMENT (AT NODES)
        AnalyticalSolution.Args.node = xycoord
        AnalyticalSol = AnalyticalSolution().Get(AnalyticalSolution.Args)

        # AnalyticalSol[:,0] = ElementalSol[:,0]
        # print np.concatenate((AnalyticalSol[:,:2], ElementalSol[:,:2]),axis=1)
        # print

        # print points
        # print points[elements[elem,:],:]
        # print AnalyticalSol

        # ALLOCATE
        nvarBasis = np.zeros((Domain.Bases.shape[0],nvar))
        ElementalSolGauss  = np.zeros((Domain.AllGauss.shape[0],nvar)); AnalyticalSolGauss  = np.copy(ElementalSolGauss)
        dElementalSolGauss = np.zeros((Domain.AllGauss.shape[0],nvar)); dAnalyticalSolGauss = np.copy(ElementalSolGauss)
        # LOOP OVER GAUSS POINTS
        for counter in range(0,Domain.AllGauss.shape[0]):
            # GET THE NUMERICAL SOLUTION WITHIN THE ELEMENT (AT QUADRATURE POINTS)
            ElementalSolGauss[counter,:] = np.dot(Domain.Bases[:,counter].reshape(1,nodeperelem),ElementalSol)
            # GET THE ANALYTICAL SOLUTION WITHIN THE ELEMENT (AT QUADRATURE POINTS)
            AnalyticalSolution.Args.node = np.dot(Domain.Bases[:,counter],xycoord)
            AnalyticalSolGauss[counter,:] = AnalyticalSolution().Get(AnalyticalSolution.Args)

            # print AnalyticalSolGauss, ElementalSolGauss[:,0]
            # AnalyticalSolGauss[:,0] = ElementalSolGauss[:,0] # REMOVE
            # AnalyticalSolGauss[:,1] = ElementalSolGauss[:,1] # REMOVE
            # print np.concatenate((AnalyticalSolGauss[:,:2], ElementalSolGauss[:,:2]),axis=1)#, AnalyticalSolution.Args.node
            # print 
            # GRADIENT TENSOR IN PARENT ELEMENT [\nabla_\varepsilon (N)]
            Jm = Domain.Jm[:,:,counter]
            # MAPPING TENSOR [\partial\vec{X}/ \partial\vec{\varepsilon} (ndim x ndim)]
            ParentGradientX=np.dot(Jm,xycoord) #
            # MATERIAL GRADIENT TENSOR IN PHYSICAL ELEMENT [\nabla_0 (N)]
            MaterialGradient = np.dot(la.inv(ParentGradientX),Jm)
            # DEFORMATION GRADIENT TENSOR [\vec{x} \otimes \nabla_0 (N)]
            # F = np.dot(EulerELemCoords.T,MaterialGradient.T)
            F = np.eye(ndim,ndim)
            # COMPUTE REMAINING KINEMATIC MEASURES
            StrainTensors = KinematicMeasures(F).Compute()
            # UPDATE/NO-UPDATE GEOMETRY
            if MainData.GeometryUpdate:
                # MAPPING TENSOR [\partial\vec{X}/ \partial\vec{\varepsilon} (ndim x ndim)]
                ParentGradientx=np.dot(Jm,EulerELemCoords)
                # SPATIAL GRADIENT TENSOR IN PHYSICAL ELEMENT [\nabla (N)]
                SpatialGradient = np.dot(la.inv(ParentGradientx),Jm).T
            else:
                SpatialGradient = MaterialGradient.T

            # SPATIAL ELECTRIC FIELD
            ElectricFieldx = - np.dot(SpatialGradient.T,ElementalSol[:,ndim])
            # COMPUTE SPATIAL ELECTRIC DISPLACEMENT
            ElectricDisplacementx = MainData.ElectricDisplacementx(MaterialArgs,StrainTensors,ElectricFieldx)
            # COMPUTE CAUCHY STRESS TENSOR
            CauchyStressTensor = MainData.CauchyStress(MaterialArgs,StrainTensors,ElectricFieldx)
            # COMPUTE THE HESSIAN AT THIS GAUSS POINT
            H_Voigt = MainData.Hessian(MaterialArgs,ndim,StrainTensors,ElectricFieldx)
            # COMPUTE THE TANGENT STIFFNESS MATRIX
            BDB_1, t = MainData().ConstitutiveStiffnessIntegrand(Domain,B,MaterialGradient,nvar,SpatialGradient,
                ndim,CauchyStressTensor,ElectricDisplacementx,MaterialArgs,H_Voigt)

            # L2 NORM
            L2_nom   += np.linalg.norm((AnalyticalSolGauss - ElementalSolGauss)**2)*Domain.AllGauss[counter,0]
            L2_denom += np.linalg.norm((AnalyticalSolGauss)**2)*Domain.AllGauss[counter,0]

            # ENERGY NORM
            DiffSol = (AnalyticalSol - ElementalSol).reshape(ElementalSol.shape[0]*ElementalSol.shape[1],1)
            E_nom   += np.linalg.norm(DiffSol**2)*Domain.AllGauss[counter,0]
            E_denom += np.linalg.norm(AnalyticalSol.reshape(AnalyticalSol.shape[0]*AnalyticalSol.shape[1],1)**2)*Domain.AllGauss[counter,0]

    L2Norm = np.sqrt(L2_nom)/np.sqrt(L2_denom)
    EnergyNorm = np.sqrt(E_nom)/np.sqrt(E_denom)
    print L2Norm, EnergyNorm
    return L2Norm, EnergyNorm




def CheapNorm(MainData,mesh,TotalDisp):

    if MainData.AssemblyParameters.FailedToConverge is True:
        MainData.L2NormX = np.NAN
        MainData.L2Normx = np.NAN
        MainData.DoF = mesh.points.shape[0]*MainData.nvar
        MainData.NELEM = mesh.elements.shape[0]

        return


    # mesh.points = np.array([
    #   [2.,2.],
    #   [5.,2.],
    #   [5.,4.],
    #   [2.,4.]
    #   ])
    # mesh.elements = np.array([
    #   [0,1,3],
    #   [1,2,3]
    #   ])
    # mesh.edges = np.array([
    #   [0,1],
    #   [1,3],
    #   [3,0],
    #   [1,2],
    #   [2,3]
    #   ])

    # mesh.GetHighOrderMesh(MainData.C)

    # print mesh.points
    # print mesh.elements
    # mesh.SimplePlot()
    # mesh.PlotMeshNumberingTri()
    # return

    # func = lambda x,y : np.sin(x)*np.cos(y)
    # func = lambda x,y : x*np.sin(y)+y*np.cos(x)
    # func = lambda x,y : x**3+(y+1)**3
    # func = lambda x,y : (x+y)**11 ##
    func = lambda x,y : (x+y)**7 ##
    # func = lambda x,y : x**3*(y+1)**5

    nodeperelem = mesh.elements.shape[1]
    # print MainData.Domain.Bases[:,0].shape
    # print nodeperelem
    # print MainData.Domain.Bases[:,0].reshape(1,nodeperelem)
    # print MainData.Domain.Bases.shape
    # print MainData.Domain.Jm.shape

    # print MainData.Quadrature.points, "\n"

    # print MainData.Domain.Bases.shape, MainData.Domain.Jm.shape
    # exit()
    

    L2_normX = 0; L2_denormX = 0
    L2_normx = 0; L2_denormx = 0
    for elem in range(mesh.nelem):
        LagrangeElemCoords = mesh.points[mesh.elements[elem,:],:]
        vpoints = mesh.points+TotalDisp[:,:,-1]
        EulerElemCoords = vpoints[mesh.elements[elem,:],:]
        # print LagrangeElemCoords.shape


        ParentGradientX = np.einsum('ijk,jl->kil',MainData.Domain.Jm,LagrangeElemCoords)
        ElementalSolGaussX = np.einsum('ij,ik',MainData.Domain.Bases,LagrangeElemCoords)

        AnalyticalSolNodesX = func(LagrangeElemCoords[:,0],LagrangeElemCoords[:,1])
        AnalyticalSolGaussX = func(ElementalSolGaussX[:,0],ElementalSolGaussX[:,1])
        NumericalSolGaussX = np.einsum('i,ij',AnalyticalSolNodesX,MainData.Domain.Bases)

        # Deformed
        ParentGradientx = np.einsum('ijk,jl->kil',MainData.Domain.Jm,EulerElemCoords)
        ElementalSolGaussx = np.einsum('ij,ik',MainData.Domain.Bases,EulerElemCoords)

        AnalyticalSolNodesx = func(EulerElemCoords[:,0],EulerElemCoords[:,1])
        AnalyticalSolGaussx = func(ElementalSolGaussx[:,0],ElementalSolGaussx[:,1])
        NumericalSolGaussx = np.einsum('i,ij',AnalyticalSolNodesx,MainData.Domain.Bases)

        # L2_norm = (AnalyticalSolGauss - NumericalSolGauss)

        # MATERIAL GRADIENT TENSOR IN PHYSICAL ELEMENT [\nabla_0 (N)]
        MaterialGradient = np.einsum('ijk,kli->ijl',la.inv(ParentGradientX),MainData.Domain.Jm)


        # L2_norm += np.linalg.norm((AnalyticalSolGauss - NumericalSolGauss)**2)
        # L2_denorm += np.linalg.norm((AnalyticalSolGauss)**2)
        # print L2_norm, L2_denorm
        # print AnalyticalSolGauss -  NumericalSolGauss

        for counter in range(0,MainData.Domain.AllGauss.shape[0]):
            # L2_normX += np.linalg.norm((AnalyticalSolGaussX - NumericalSolGaussX)**2)*MainData.Domain.AllGauss[counter,0]
            # L2_denormX += np.linalg.norm((AnalyticalSolGaussX)**2)*MainData.Domain.AllGauss[counter,0]

            # L2_normx += np.linalg.norm((AnalyticalSolGaussx - NumericalSolGaussx)**2)*MainData.Domain.AllGauss[counter,0]
            # L2_denormx += np.linalg.norm((AnalyticalSolGaussx)**2)*MainData.Domain.AllGauss[counter,0]

            L2_normX += (AnalyticalSolGaussX - NumericalSolGaussX)**2*MainData.Domain.AllGauss[counter,0]
            L2_denormX += (AnalyticalSolGaussX)**2*MainData.Domain.AllGauss[counter,0]

            L2_normx += (AnalyticalSolGaussx - NumericalSolGaussx)**2*MainData.Domain.AllGauss[counter,0]
            L2_denormx += (AnalyticalSolGaussx)**2*MainData.Domain.AllGauss[counter,0]

            # L2_normX += (AnalyticalSolGaussX - NumericalSolGaussX)*MainData.Domain.AllGauss[counter,0]
            # L2_denormX += (AnalyticalSolGaussX)*MainData.Domain.AllGauss[counter,0]

            # L2_normx += (AnalyticalSolGaussx - NumericalSolGaussx)*MainData.Domain.AllGauss[counter,0]
            # L2_denormx += (AnalyticalSolGaussx)*MainData.Domain.AllGauss[counter,0]


    # L2NormX = np.sqrt(L2_normX)/np.sqrt(L2_denormX)
    # L2NormX = np.linalg.norm(L2NormX)

    # L2Normx = np.sqrt(L2_normx)/np.sqrt(L2_denormx)
    # L2Normx = np.linalg.norm(L2Normx)

    L2NormX = np.sqrt(np.sum(L2_normX))/np.sqrt(np.sum(L2_denormX))
    L2Normx = np.sqrt(np.sum(L2_normx))/np.sqrt(np.sum(L2_denormx))


    # L2NormX = L2_normX/L2_denormX
    # L2Normx = L2_normx/L2_denormx
    print L2NormX, L2Normx
    # print np.linalg.norm(L2NormX)

    MainData.L2NormX = L2NormX
    MainData.L2Normx = L2Normx
    MainData.DoF = mesh.points.shape[0]*MainData.nvar
    MainData.NELEM = mesh.elements.shape[0]