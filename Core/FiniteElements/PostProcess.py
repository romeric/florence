import numpy as np 
import numpy.linalg as la

from Base import JacobianError, IllConditionedError

import Core.InterpolationFunctions.TwoDimensional.Quad.QuadLagrangeGaussLobatto as TwoD
import Core.InterpolationFunctions.ThreeDimensional.Hexahedral.HexLagrangeGaussLobatto as ThreeD
# Modal Bases
# import Core.InterpolationFunctions.TwoDimensional.Tri.hpModal as Tri 
# import Core.InterpolationFunctions.ThreeDimensional.Tetrahedral.hpModal as Tet 
# Nodal Bases
import Core.InterpolationFunctions.TwoDimensional.Tri.hpNodal as Tri 
import Core.InterpolationFunctions.ThreeDimensional.Tetrahedral.hpNodal as Tet 

from ElementalMatrices.KinematicMeasures import *
from Core.MeshGeneration import vtk_writer

class PostProcess(object):
    """PostProcess"""

    def __init__(self):
        self.is_scaledjacobian_computed = False


    @staticmethod
    def TotalComponentSol(MainData,sol,ColumnsIn,ColumnsOut,AppliedDirichletInc,Iter,fsize):

        nvar = MainData.nvar
        ndim = MainData.ndim
        Analysis = MainData.Analysis
        TotalSol = np.zeros((fsize,1))

        # GET TOTAL SOLUTION
        if MainData.AnalysisType == 'Nonlinear':
            if Analysis =='Static':
                TotalSol[ColumnsIn,0] = sol
                if Iter==0:
                    TotalSol[ColumnsOut,0] = AppliedDirichletInc
            if Analysis !='Static':
                TotalSol = np.copy(sol)
                TotalSol[ColumnsOut,0] = AppliedDirichletInc

        elif MainData.AnalysisType == 'Linear':
                TotalSol[ColumnsIn,0] = sol
                TotalSol[ColumnsOut,0] = AppliedDirichletInc
                

        # RE-ORDER SOLUTION COMPONENTS
        dU = TotalSol.reshape(TotalSol.shape[0]/nvar,nvar)


        return dU



    @staticmethod
    def StressRecovery(MainData,mesh,TotalDisp):

        # KEEP THE IMPORTS LOCAL AS MATPLOTLIB IMPORT IS SLOW
        import matplotlib.pyplot as plt
        from scipy import io 

        C = MainData.C
        nvar = MainData.nvar
        ndim = MainData.ndim
        LoadIncrement = MainData.AssemblyParameters.LoadIncrements
        w = MainData.Quadrature.weights
        z = MainData.Quadrature.points

        ns=[]; Basis=[]; gBasisx=[]; gBasisy=[]; gBasisz=[]
        if mesh.element_type=='hex':
            ns = (C+2)**ndim
            # GET THE BASES AT NODES INSTEAD OF GAUSS POINTS
            Basis = np.zeros((ns,w.shape[0]**ndim))
            gBasisx = np.zeros((ns,w.shape[0]**ndim))
            gBasisy = np.zeros((ns,w.shape[0]**ndim))
            gBasisz = np.zeros((ns,w.shape[0]**ndim))
        elif mesh.element_type=='tet':
            p=C+1
            ns = (p+1)*(p+2)*(p+3)/6
            # GET BASES AT NODES INSTEAD OF GAUSS POINTS
            # BE CAREFUL TAHT 4 STANDS FOR 4 VERTEX NODES (FOR HIGHER C CHECK THIS)
            Basis = np.zeros((ns,4))
            gBasisx = np.zeros((ns,4))
            gBasisy = np.zeros((ns,4))
            gBasisz = np.zeros((ns,4))
        elif mesh.element_type =='tri':
            p=C+1
            ns = (p+1)*(p+2)/2
            # GET BASES AT NODES INSTEAD OF GAUSS POINTS
            # BE CAREFUL TAHT 3 STANDS FOR 3 VERTEX NODES (FOR HIGHER C CHECK THIS)
            Basis = np.zeros((ns,3))
            gBasisx = np.zeros((ns,3))
            gBasisy = np.zeros((ns,3))


        eps=[]
        if mesh.element_type == 'hex':
            counter = 0
            eps = ThreeD.LagrangeGaussLobatto(C,0,0,0)[1]
            for i in range(0,eps.shape[0]):
                ndummy = ThreeD.LagrangeGaussLobatto(C,eps[i,0],eps[i,1],eps[i,2],arrange=1)[0]
                Basis[:,counter] = ndummy[:,0]
                dummy = ThreeD.GradLagrangeGaussLobatto(C,eps[i,0],eps[i,1],eps[i,2],arrange=1)
                gBasisx[:,counter] = dummy[:,0]
                gBasisy[:,counter] = dummy[:,1]
                gBasisz[:,counter] = dummy[:,2]
                counter+=1
        elif mesh.element_type == 'tet':
            counter = 0
            eps = np.array([
                [-1.,-1.,-1.],
                [1.,-1.,-1.],
                [-1.,1.,-1.],
                [-1.,-1.,1.]
                ])
            for i in range(0,eps.shape[0]):
                ndummy, dummy = Tet.hpBases(C,eps[i,0],eps[i,1],eps[i,2],1,1)
                ndummy = ndummy.reshape(ndummy.shape[0],1)
                Basis[:,counter] = ndummy[:,0]
                gBasisx[:,counter] = dummy[:,0]
                gBasisy[:,counter] = dummy[:,1]
                gBasisz[:,counter] = dummy[:,2]
                counter+=1
        elif mesh.element_type == 'tri':
            eps = np.array([
                [-1.,-1.],
                [1.,-1.],
                [-1.,1.]
                ])
            for i in range(0,eps.shape[0]):
                ndummy, dummy = Tri.hpBases(C,eps[i,0],eps[i,1],1,1)
                ndummy = ndummy.reshape(ndummy.shape[0],1)
                Basis[:,i] = ndummy[:,0]
                gBasisx[:,i] = dummy[:,0]
                gBasisy[:,i] = dummy[:,1]

        # elements = mesh.elements[:,:eps.shape[0]]
        # points = mesh.points[:np.max(elements),:]
        # TotalDisp = TotalDisp[:np.max(elements),:,:]  # BECAREFUL TOTALDISP IS CHANGING HERE
        elements = mesh.elements
        points = mesh.points
        nelem = elements.shape[0]; npoint = points.shape[0]
        nodeperelem = elements.shape[1]


        # FOR AVERAGING SECONDARY VARIABLES AT NODES WE NEED TO KNOW WHICH ELEMENTS SHARE THE SAME NODES
        # GET UNIQUE NODES
        unique_nodes = np.unique(elements)
        # shared_elements = -1*np.ones((unique_nodes.shape[0],nodeperelem))
        shared_elements = -1*np.ones((unique_nodes.shape[0],50)) # This number (50) is totally arbitrary
        position = np.copy(shared_elements)
        for inode in range(0,unique_nodes.shape[0]):
            shared_elems, pos = np.where(elements==unique_nodes[inode])
            for i in range(0,shared_elems.shape[0]):
                shared_elements[inode,i] = shared_elems[i]
                position[inode,i] = pos[i]

        MainData.MainDict['CauchyStress'] = np.zeros((ndim,ndim,npoint,LoadIncrement))
        MainData.MainDict['DeformationGradient'] = np.zeros((ndim,ndim,npoint,LoadIncrement))
        MainData.MainDict['ElectricField'] = np.zeros((ndim,1,npoint,LoadIncrement))
        MainData.MainDict['ElectricDisplacement'] = np.zeros((ndim,1,npoint,LoadIncrement))
        MainData.MainDict['SmallStrain'] = np.zeros((ndim,ndim,npoint,LoadIncrement))
        CauchyStressTensor = np.zeros((ndim,ndim,nodeperelem,nelem))
        ElectricFieldx = np.zeros((ndim,1,nodeperelem,nelem))
        ElectricDisplacementx = np.zeros((ndim,1,nodeperelem,nelem))
        F = np.zeros((ndim,ndim,nodeperelem,nelem))
        strain = np.zeros((ndim,ndim,nodeperelem,nelem))

        
        for Increment in range(0,LoadIncrement):
            # LOOP OVER ELEMENTS
            for elem in range(0,elements.shape[0]):
                # GET THE FIELDS AT THE ELEMENT LEVEL
                Eulerx = points + TotalDisp[:,:,Increment]
                LagrangeElemCoords = points[elements[elem,:],:]
                EulerElemCoords = Eulerx[elements[elem,:],:]
                # if MainData.Fields == 'ElectroMechanics':
                    # ElectricPotentialElem =  MainData.TotalPot[elements[elem,:],:,Increment] 

                # LOOP OVER ELEMENTS
                for g in range(0,eps.shape[0]):
                    # GRADIENT TENSOR IN PARENT ELEMENT [\nabla_\varepsilon (N)]
                    Jm = np.zeros((ndim,MainData.Domain.Bases.shape[0]))    
                    if ndim==3:
                        Jm[0,:] = gBasisx[:,g]
                        Jm[1,:] = gBasisy[:,g]
                        Jm[2,:] = gBasisz[:,g]
                    if ndim==2:
                        Jm[0,:] = gBasisx[:,g]
                        Jm[1,:] = gBasisy[:,g]
                    # MAPPING TENSOR [\partial\vec{X}/ \partial\vec{\varepsilon} (ndim x ndim)]
                    ParentGradientX=np.dot(Jm,LagrangeElemCoords)
                    # MATERIAL GRADIENT TENSOR IN PHYSICAL ELEMENT [\nabla_0 (N)]
                    MaterialGradient = np.dot(la.inv(ParentGradientX).T,Jm)

                    # DEFORMATION GRADIENT TENSOR [\vec{x} \otimes \nabla_0 (N)]
                    F[:,:,g,elem] = np.dot(EulerElemCoords.T,MaterialGradient.T)
                    # UPDATE/NO-UPDATE GEOMETRY
                    if MainData.GeometryUpdate:
                        # MAPPING TENSOR [\partial\vec{X}/ \partial\vec{\varepsilon} (ndim x ndim)]
                        ParentGradientx=np.dot(Jm,EulerElemCoords)
                        # SPATIAL GRADIENT TENSOR IN PHYSICAL ELEMENT [\nabla (N)]
                        SpatialGradient = np.dot(la.inv(ParentGradientx),Jm).T
                    else:
                        SpatialGradient = MaterialGradient.T

                    if MainData.Fields == 'ElectroMechanics':
                        # SPATIAL ELECTRIC FIELD
                        ElectricFieldx[:,:,g,elem] = - np.dot(SpatialGradient.T,ElectricPotentialElem)
                        # COMPUTE SPATIAL ELECTRIC DISPLACEMENT
                        ElectricDisplacementx[:,:,g,elem] = MainData.ElectricDisplacementx(MaterialArgs,
                            StrainTensors,ElectricFieldx[:,:,g,elem])

                    # Compute remaining kinematic/deformation measures
                    # StrainTensors = KinematicMeasures(F[:,:,g,elem])
                    StrainTensors = KinematicMeasures_NonVectorised(F[:,:,g,elem],MainData.AnalysisType,
                        MainData.Domain.AllGauss.shape[0])
                    if MainData.GeometryUpdate is False:
                        strain[:,:,g,elem] = StrainTensors['strain'][g]

                    # COMPUTE CAUCHY STRESS TENSOR
                    if MainData.Prestress:
                        CauchyStressTensor[:,:,g,elem]= MainData.CauchyStress(MainData.MaterialArgs,
                            StrainTensors,ElectricFieldx[:,:,g,elem])[0] 
                    else:
                        CauchyStressTensor[:,:,g,elem] = MainData.CauchyStress(MainData.MaterialArgs,
                            StrainTensors,ElectricFieldx[:,:,g,elem])
                    


            # AVERAGE THE QUANTITIES OVER NODES
            for inode in range(0,unique_nodes.shape[0]):

                x = np.where(shared_elements[inode,:]==-1)[0]
                if x.shape[0]!=0:
                    myrange=np.linspace(0,x[0],x[0]+1)
                else:
                    myrange = np.linspace(0,elements.shape[1]-1,elements.shape[1])

                for j in myrange:
                    MainData.MainDict['DeformationGradient'][:,:,inode,Increment] += \
                        F[:,:,position[inode,j],shared_elements[inode,j]]
                    MainData.MainDict['CauchyStress'][:,:,inode,Increment] += \
                        CauchyStressTensor[:,:,position[inode,j],shared_elements[inode,j]]
                    MainData.MainDict['ElectricField'][:,:,inode,Increment] += \
                        ElectricFieldx[:,:,position[inode,j],shared_elements[inode,j]]
                    MainData.MainDict['ElectricDisplacement'][:,:,inode,Increment] += \
                        ElectricDisplacementx[:,:,position[inode,j],shared_elements[inode,j]]
                    if ~MainData.GeometryUpdate:
                        MainData.MainDict['SmallStrain'][:,:,inode,Increment] += \
                            strain[:,:,position[inode,j],shared_elements[inode,j]] 

                MainData.MainDict['DeformationGradient'][:,:,inode,Increment] = \
                    MainData.MainDict['DeformationGradient'][:,:,inode,Increment]/(1.0*len(myrange))
                MainData.MainDict['CauchyStress'][:,:,inode,Increment] = \
                    MainData.MainDict['CauchyStress'][:,:,inode,Increment]/(1.0*len(myrange))
                MainData.MainDict['ElectricField'][:,:,inode,Increment] = \
                MainData.MainDict['ElectricField'][:,:,inode,Increment]/(1.0*len(myrange))
                MainData.MainDict['ElectricDisplacement'][:,:,inode,Increment] = \
                MainData.MainDict['ElectricDisplacement'][:,:,inode,Increment]/(1.0*len(myrange))
                if ~MainData.GeometryUpdate:
                        MainData.MainDict['SmallStrain'][:,:,inode,Increment] = \
                            MainData.MainDict['SmallStrain'][:,:,inode,Increment]/(1.0*len(myrange))


        # FOR PLOTTING PURPOSES ONLY COMPUTE THE EXTERIOR SOLUTION
        if C>0:
            ns_0 = np.max(elements[:,:eps.shape[0]]) + 1
            MainData.MainDict['DeformationGradient'] = MainData.MainDict['DeformationGradient'][:,:,:ns_0,:]
            MainData.MainDict['CauchyStress'] = MainData.MainDict['CauchyStress'][:,:,:ns_0,:]
            if ~MainData.GeometryUpdate:
                MainData.MainDict['SmallStrain'] = MainData.MainDict['SmallStrain'][:,:,:ns_0,:]
            if MainData.Fields == 'ElectroMechanics':
                MainData.MainDict['ElectricField'] = MainData.MainDict['ElectricField'][:,:,:ns_0,:]
                MainData.MainDict['ElectricDisplacement'] = MainData.MainDict['ElectricDisplacement'][:,:,:ns_0,:]


        # NEWTON-RAPHSON CONVERGENCE PLOT
        if MainData.nrplot[0]:
            if MainData.nrplot[1] == 'first':
                # First increment convergence
                plt.semilogy(MainData.NRConvergence['Increment_0'],'-ko')       
            elif MainData.nrplot[1] == 'last':
                # Last increment convergence
                plt.plot(np.log10(MainData.NRConvergence['Increment_'+str(len(MainData.NRConvergence)-1)]),'-ko')   
            else:
                # Arbitrary increment convergence
                plt.plot(np.log10(MainData.NRConvergence['Increment_'+str(MainData.nrplot[1])]),'-ko')  

            axis_font = {'size':'18'}
            plt.xlabel(r'$No\, of\, Iteration$', **axis_font)
            plt.ylabel(r'$log_{10}|Residual|$', **axis_font)
            # Save plot
            # plt.savefig(MainData.Path.ProblemResults+MainData.Path.Analysis+\
            #   MainData.Path.MaterialModel+'/NR_convergence_'+MaterialArgs.Type+'.eps',
            #   format='eps', dpi=1000)
            # Display plot
            plt.show()


        #-----------------------------------------------------------------------------------------------------------#
        #-----------------------------------------------------------------------------------------------------------#
        # Save the dictionary in .mat file
        # MainData.MainDict['Solution'] = MainData.TotalDisp
        # Write to Matlab .mat dictionary
        # io.savemat(MainData.Path.ProblemResults+MainData.Path.ProblemResultsFileNameMATLAB,MainData.MainDict)
        #-----------------------------------------------------------------------------------------------------------#
        #-----------------------------------------------------------------------------------------------------------#


        # WRITE IN VTK FILE 
        if MainData.write:
            if mesh.element_type =='tri':
                cellflag = 5
            elif mesh.element_type =='quad':
                cellflag = 9
            if mesh.element_type =='tet':
                cellflag = 10
            elif mesh.element_type == 'hex':
                cellflag = 12
            for incr in range(0,MainData.AssemblyParameters.LoadIncrements):
                # PLOTTING ON THE DEFORMED MESH
                elements = mesh.elements[:,:eps.shape[0]]
                points = mesh.points[:np.max(elements)+1,:]
                TotalDisp = TotalDisp[:np.max(elements)+1,:,:] # BECAREFUL TOTALDISP IS CHANGING HERE
                points[:,:nvar] += TotalDisp[:,:nvar,incr]

                # OLDER APPROACH
                # points[:,0] += TotalDisp[:,0,incr] 
                # points[:,1] += TotalDisp[:,1,incr] 
                # if ndim==3:
                    # points[:,2] += MainData.TotalDisp[:,2,incr] 

                # WRITE DISPLACEMENTS
                vtk_writer.write_vtu(Verts=points, Cells={cellflag:elements}, pdata=TotalDisp[:,:,incr],
                fname=MainData.Path.ProblemResults+MainData.Path.Analysis+MainData.Path.MaterialModel+\
                MainData.Path.ProblemResultsFileNameVTK+'_U_'+str(incr)+'.vtu')
                
                
                if MainData.Fields == 'ElectroMechanics':
                    # WRITE ELECTRIC POTENTIAL
                    vtk_writer.write_vtu(Verts=points, Cells={cellflag:elements}, pdata=MainData.TotalPot[:,:,incr],
                    fname=MainData.Path.ProblemResults+MainData.Path.Analysis+MainData.Path.MaterialModel+\
                    MainData.Path.ProblemResultsFileNameVTK+'_Phi_'+str(incr)+'.vtu')

            # FOR LINEAR STATIC ANALYSIS
            # vtk_writer.write_vtu(Verts=vmesh.points, Cells={12:vmesh.elements}, 
            #   pdata=MainData.TotalDisp[:,:,incr], fname=MainData.Path.ProblemResults+'/Results.vtu')


            for incr in range(0,MainData.AssemblyParameters.LoadIncrements):
                # PLOTTING ON THE DEFORMED MESH
                elements = mesh.elements[:,:eps.shape[0]]
                points = mesh.points[:np.max(elements)+1,:]
                TotalDisp = TotalDisp[:np.max(elements)+1,:,:] # BECAREFUL TOTALDISP IS CHANGING HERE
                points[:,:nvar] += TotalDisp[:,:nvar,incr]
                npoint = points.shape[0]

                #----------------------------------------------------------------------------------------------------#
                # CAUCHY STRESS
                vtk_writer.write_vtu(Verts=points, Cells={cellflag:elements}, 
                    pdata=MainData.MainDict['CauchyStress'][:,0,:,incr].reshape(npoint,ndim),
                    fname=MainData.Path.ProblemResults+MainData.Path.Analysis+MainData.Path.MaterialModel+\
                    MainData.Path.ProblemResultsFileNameVTK+'_S_i0_'+str(incr)+'.vtu')

                vtk_writer.write_vtu(Verts=points, Cells={cellflag:elements}, 
                    pdata=MainData.MainDict['CauchyStress'][:,1,:,incr].reshape(npoint,ndim),
                    fname=MainData.Path.ProblemResults+MainData.Path.Analysis+MainData.Path.MaterialModel+\
                    MainData.Path.ProblemResultsFileNameVTK+'_S_i1_'+str(incr)+'.vtu')

                if ndim==3:
                    vtk_writer.write_vtu(Verts=points, Cells={cellflag:elements}, 
                        pdata=MainData.MainDict['CauchyStress'][:,2,:,incr].reshape(npoint,ndim),
                        fname=MainData.Path.ProblemResults+MainData.Path.Analysis+MainData.Path.MaterialModel+\
                        MainData.Path.ProblemResultsFileNameVTK+'_S_i2_'+str(incr)+'.vtu')
                #-------------------------------------------------------------------------------------------------------#


                #-------------------------------------------------------------------------------------------------------#
                if MainData.Fields == 'ElectroMechanics':
                    # ELECTRIC FIELD
                    vtk_writer.write_vtu(Verts=points, Cells={cellflag:elements}, 
                        pdata=MainData.MainDict['ElectricField'][:,0,:,incr].reshape(npoint,ndim),
                        fname=MainData.Path.ProblemResults+MainData.Path.Analysis+MainData.Path.MaterialModel+\
                        MainData.Path.ProblemResultsFileNameVTK+'_E_'+str(incr)+'.vtu')
                        #-----------------------------------------------------------------------------------------------------------#
                        #-----------------------------------------------------------------------------------------------------------#
                    # ELECTRIC DISPLACEMENT
                    vtk_writer.write_vtu(Verts=points, Cells={cellflag:elements}, 
                        pdata=MainData.MainDict['ElectricDisplacement'][:,0,:,incr].reshape(npoint,ndim),
                        fname=MainData.Path.ProblemResults+MainData.Path.Analysis+MainData.Path.MaterialModel+\
                        MainData.Path.ProblemResultsFileNameVTK+'_D_'+str(incr)+'.vtu')
                #-----------------------------------------------------------------------------------------------------------#

                #-----------------------------------------------------------------------------------------------------------#
                # STRAIN/KINEMATICS
                if ~MainData.GeometryUpdate:
                    # SMALL STRAINS
                    vtk_writer.write_vtu(Verts=points, Cells={cellflag:elements}, 
                        pdata=MainData.MainDict['SmallStrain'][:,0,:,incr].reshape(npoint,ndim),
                        fname=MainData.Path.ProblemResults+MainData.Path.Analysis+MainData.Path.MaterialModel+\
                        MainData.Path.ProblemResultsFileNameVTK+'_Strain_i0_'+str(incr)+'.vtu')

                    vtk_writer.write_vtu(Verts=points, Cells={cellflag:elements}, 
                        pdata=MainData.MainDict['SmallStrain'][:,1,:,incr].reshape(npoint,ndim),
                        fname=MainData.Path.ProblemResults+MainData.Path.Analysis+MainData.Path.MaterialModel+\
                        MainData.Path.ProblemResultsFileNameVTK+'_Strain_i1_'+str(incr)+'.vtu')

                    if ndim==3:
                        vtk_writer.write_vtu(Verts=points, Cells={cellflag:elements}, 
                            pdata=MainData.MainDict['SmallStrain'][:,2,:,incr].reshape(npoint,ndim),
                            fname=MainData.Path.ProblemResults+MainData.Path.Analysis+MainData.Path.MaterialModel+\
                            MainData.Path.ProblemResultsFileNameVTK+'_Strain_i2_'+str(incr)+'.vtu')
                else:
                    # DEFORMATION GRADIENT
                    vtk_writer.write_vtu(Verts=points, Cells={cellflag:elements}, 
                        pdata=MainData.MainDict['DeformationGradient'][:,0,:,incr].reshape(npoint,ndim),
                        fname=MainData.Path.ProblemResults+MainData.Path.Analysis+MainData.Path.MaterialModel+\
                        MainData.Path.ProblemResultsFileNameVTK+'_F_i0_'+str(incr)+'.vtu')

                    vtk_writer.write_vtu(Verts=points, Cells={cellflag:elements}, 
                        pdata=MainData.MainDict['DeformationGradient'][:,1,:,incr].reshape(npoint,ndim),
                        fname=MainData.Path.ProblemResults+MainData.Path.Analysis+MainData.Path.MaterialModel+\
                        MainData.Path.ProblemResultsFileNameVTK+'_F_i1_'+str(incr)+'.vtu')

                    if ndim==3:
                        vtk_writer.write_vtu(Verts=points, Cells={cellflag:elements}, 
                            pdata=MainData.MainDict['DeformationGradient'][:,2,:,incr].reshape(npoint,ndim),
                            fname=MainData.Path.ProblemResults+MainData.Path.Analysis+MainData.Path.MaterialModel+\
                            MainData.Path.ProblemResultsFileNameVTK+'_F_i2_'+str(incr)+'.vtu')
                #-------------------------------------------------------------------------------------------------------#




    
    def MeshQualityMeasures(self,MainData,mesh,TotalDisp,show_plot=True):

        if self.is_scaledjacobian_computed is True:
            raise AssertionError('Scaled Jacobian seems to be already computed. Re-Computing it may return incorrect results')
        if MainData.isScaledJacobianComputed is True:
            raise AssertionError('Scaled Jacobian seems to be already computed. Re-Computing it may return incorrect results')

        PostDomain = MainData.PostDomain
        points = mesh.points
        vpoints = np.copy(mesh.points)
        # vpoints   += TotalDisp[:,:,-1]
        vpoints = vpoints + TotalDisp[:,:,-1]
        # vpoints = mesh.points + np.sum(TotalDisp,axis=2)

        elements = mesh.elements

        MainData.ScaledJacobian = np.zeros(elements.shape[0])
        MainData.ScaledFF = np.zeros(elements.shape[0])
        MainData.ScaledHH = np.zeros(elements.shape[0])

        MainData.ScaledFNFN = np.zeros(elements.shape[0])
        MainData.ScaledCNCN = np.zeros(elements.shape[0])
        # MainData.Jacobian = np.zeros(elements.shape[0])
        # MainData.ScaledJacobian = []
        # MainData.ScaledJacobianElem = []


        JMax =[]; JMin=[]
        for elem in range(mesh.nelem):
            LagrangeElemCoords = points[elements[elem,:],:]
            EulerElemCoords = vpoints[elements[elem,:],:]

            # COMPUTE KINEMATIC MEASURES AT ALL INTEGRATION POINTS USING EINSUM (AVOIDING THE FOR LOOP)
            # MAPPING TENSOR [\partial\vec{X}/ \partial\vec{\varepsilon} (ndim x ndim)]
            ParentGradientX = np.einsum('ijk,jl->kil',MainData.PostDomain.Jm,LagrangeElemCoords)
            # MAPPING TENSOR [\partial\vec{x}/ \partial\vec{\varepsilon} (ndim x ndim)]
            ParentGradientx = np.einsum('ijk,jl->kil',MainData.PostDomain.Jm,EulerElemCoords)
            # MATERIAL GRADIENT TENSOR IN PHYSICAL ELEMENT [\nabla_0 (N)]
            MaterialGradient = np.einsum('ijk,kli->ijl',la.inv(ParentGradientX),MainData.PostDomain.Jm)
            # DEFORMATION GRADIENT TENSOR [\vec{x} \otimes \nabla_0 (N)]
            F = np.einsum('ij,kli->kjl',EulerElemCoords,MaterialGradient)
            # JACOBIAN OF DEFORMATION GRADIENT TENSOR
            detF = np.abs(np.linalg.det(F))
            # COFACTOR OF DEFORMATION GRADIENT TENSOR
            H = np.einsum('ijk,k->ijk',np.linalg.inv(F).T,detF)

            # FIND JACOBIAN OF SPATIAL GRADIENT
            # USING ISOPARAMETRIC
            Jacobian = np.abs(np.linalg.det(ParentGradientx))
            # USING DETERMINANT OF DEFORMATION GRADIENT TENSOR
            # Jacobian = detF
            # USING INVARIANT F:F
            Q1 = np.sqrt(np.einsum('kij,lij->kl',F,F)).diagonal()
            # USING INVARIANT H:H
            Q2 = np.sqrt(np.einsum('ijk,ijl->kl',H,H)).diagonal()
            # print np.isnan(Q3).any()
            Directions = getattr(MainData.MaterialArgs,"AnisotropicOrientations",None)
            if Directions != None and MainData.MaterialArgs.Type == "BonetTranservselyIsotropicHyperElastic":
                Q4 = np.einsum('ijk,k',F,Directions[elem,:])
                Q4 = np.sqrt(np.dot(Q4,Q4.T)).diagonal()

                C = np.einsum('ikj,ikl->ijl',F,F)
                Q5 = np.einsum('ijk,k',C,Directions[elem,:]) 
                Q5 = np.sqrt(np.dot(Q5,Q5.T)).diagonal()
                # print Q4
                # print F.shape
                # Q4 = np.sqrt()
                # print Q4

            # FIND MIN AND MAX VALUES
            JMin = np.min(Jacobian); JMax = np.max(Jacobian)

            # MainData.ScaledJacobian[elem] = 1.0*JMin/JMax
            # MainData.ScaledJacobian = np.append(MainData.ScaledJacobian,1.0*JMin/JMax)
            # MainData.ScaledJacobianElem = np.append(MainData.ScaledJacobianElem,elem) 

            MainData.ScaledJacobian[elem] = 1.0*JMin/JMax
            # MainData.Jacobian[elem] = np.min(Jacobian)
            MainData.ScaledFF[elem] = 1.0*np.min(Q1)/np.max(Q1)
            MainData.ScaledHH[elem] = 1.0*np.min(Q2)/np.max(Q2)
            # MainData.Jacobian[elem] = np.min(detF)

            if Directions != None and MainData.MaterialArgs.Type == "BonetTranservselyIsotropicHyperElastic":
                MainData.ScaledFNFN[elem] = 1.0*np.min(Q4)/np.max(Q4)
                MainData.ScaledCNCN[elem] = 1.0*np.min(Q5)/np.max(Q5)

        if np.isnan(MainData.ScaledJacobian).any():
            raise JacobianError()

        # exit(0)
        

        print 'Minimum ScaledJacobian value is', np.min(MainData.ScaledJacobian), \
        'corresponding to element', np.where(np.min(MainData.ScaledJacobian)==MainData.ScaledJacobian)[0][0]

        print 'Minimum ScaledFF value is', np.min(MainData.ScaledFF), \
        'corresponding to element', np.where(np.min(MainData.ScaledFF)==MainData.ScaledFF)[0][0]

        print 'Minimum ScaledHH value is', np.min(MainData.ScaledHH), \
        'corresponding to element', np.where(np.min(MainData.ScaledHH)==MainData.ScaledHH)[0][0]

        if Directions != None and MainData.MaterialArgs.Type == "BonetTranservselyIsotropicHyperElastic":
            print 'Minimum ScaledFNFN value is', np.min(MainData.ScaledFNFN), \
            'corresponding to element', np.where(np.min(MainData.ScaledFNFN)==MainData.ScaledFNFN)[0][0]

            print 'Minimum ScaledCNCN value is', np.min(MainData.ScaledCNCN), \
            'corresponding to element', np.where(np.min(MainData.ScaledCNCN)==MainData.ScaledCNCN)[0][0]


        if show_plot == True:

            import matplotlib.pyplot as plt
                
            fig = plt.figure()
            # plt.bar(np.linspace(0,elements.shape[0]-1,elements.shape[0]),
            #   MainData.ScaledJacobian,width=1.,color='#FE6F5E',alpha=0.8)

            plt.bar(np.linspace(0,elements.shape[0]-1,elements.shape[0]),MainData.ScaledJacobian,width=1.,alpha=0.4)
            plt.xlabel(r'$Elements$',fontsize=18)
            plt.ylabel(r'$Scaled\, Jacobian$',fontsize=18)

            # plt.bar(np.linspace(0,MainData.ScaledJacobianElem.shape[0]-1,
            #   MainData.ScaledJacobianElem.shape[0]),MainData.ScaledJacobian,width=1.,alpha=0.4)
            # plt.xlabel(r'$Elements$',fontsize=18)
            # plt.ylabel(r'$Scaled\, Jacobian$',fontsize=18)


    @staticmethod   
    def HighOrderPatchPlot(MainData,mesh,TotalDisp):

        import matplotlib.pyplot as plt
        
        fig = plt.figure()
        ax = fig.axes

        # TotalDisp = np.loadtxt('/home/roman/Desktop/step11.dat',delimiter=',')
        # TotalDisp = TotalDisp[:,:,None]
        # import sys; sys.exit(0)
        # TotalDisp = np.zeros_like(TotalDisp)
        # MainData.ScaledJacobian = np.ones_like(MainData.ScaledJacobian)

        # print TotalDisp[:,0,-1]
        # MainData.ScaledJacobian = np.zeros_like(MainData.ScaledJacobian)+1
        vpoints = np.copy(mesh.points)
        # print TotalDisp[:,:MainData.ndim,-1]
        vpoints += TotalDisp[:,:MainData.ndim,-1]

        dum1=[]; dum2=[]; dum3 = []; ddum=np.array([0,1,2,0])
        for i in range(0,MainData.C):
            dum1=np.append(dum1,i+3)
            dum2 = np.append(dum2, 2*MainData.C+3 +i*MainData.C -i*(i-1)/2 )
            dum3 = np.append(dum3,MainData.C+3 +i*(MainData.C+1) -i*(i-1)/2 )

        if MainData.C>0:
            ddum = (np.append(np.append(np.append(np.append(np.append(np.append(0,dum1),1),dum2),2),
                np.fliplr(dum3.reshape(1,dum3.shape[0]))),0) ).astype(np.int32)

        x_avg = []; y_avg = []
        for i in range(mesh.nelem):
            dum = vpoints[mesh.elements[i,:],:]

            plt.plot(dum[ddum,0],dum[ddum,1],alpha=0.02)
            # plt.fill(dum[ddum,0],dum[ddum,1],'#A4DDED')
            plt.fill(dum[ddum,0],dum[ddum,1],color=(0.75,MainData.ScaledJacobian[i],0.35))
            # afig = plt.fill(dum[ddum,0],dum[ddum,1],color=(0.75,MainData.Jacobian[i]/4.,0.35))    
            # afig= plt.fill(dum[ddum,0],dum[ddum,1])   

            # plt.fill(dum[ddum,0],dum[ddum,1],color=(0.75,1.0*i/mesh.elements.shape[0],0.35))  
            # plt.fill(dum[ddum,0],dum[ddum,1],color=(MainData.ScaledJacobian[i],0,1-MainData.ScaledJacobian[i]))   

            plt.plot(dum[ddum,0],dum[ddum,1],'#000000')
            
            # WRITE JACOBIAN VALUES ON ELEMENTS
            # coord = mesh.points[mesh.elements[i,:],:]
            # x_avg.append(np.sum(coord[:,0])/mesh.elements.shape[1])
            # y_avg.append(np.sum(coord[:,1])/mesh.elements.shape[1])
            # plt.text(x_avg[i],y_avg[i],np.around(MainData.ScaledJacobian[i],decimals=3))


        plt.plot(vpoints[:,0],vpoints[:,1],'o',color='#F88379',markersize=5) 

        plt.axis('equal')
        # plt.xlim([-0.52,-0.43])
        # plt.ylim([-0.03,0.045])
        plt.axis('off')

        # ax = plt.gca()
        # PCM=ax.get_children()[2]
        # plt.colorbar(afig)



    @staticmethod
    def HighOrderPatchPlot3D(MainData,mesh,TotalDisp=0):
        """ This 3D patch plot works but the elements at background are
        also shown
        """
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        import matplotlib.pyplot as plt

        C = MainData.C
        a1,a2,a3,a4 = [],[],[],[]
        if C==1:
            a1 = [1, 5, 2, 9, 4, 8, 1]
            a2 = [1, 5, 2, 7, 3, 6, 1]
            a3 = [1, 6, 3, 10, 4, 8, 1]
            a4 = [2, 7, 3, 10, 4, 9, 2]
        elif C==2:
            a1 = [1, 5, 6, 2, 9, 11, 3, 10, 7, 1]
            a2 = [1, 5, 6, 2, 14, 19, 4, 18, 12, 1]
            a3 = [2, 9, 11, 3, 17, 20, 4, 19, 14, 2]
            a4 = [1, 12, 18, 4, 20, 17, 3, 10, 7, 1]
        elif C==3:
            a1 = [1, 5, 6, 7, 2, 20, 29, 34, 4, 33, 27, 17, 1]
            a2 = [1, 8, 12, 15, 3, 16, 14, 11, 2, 7, 6, 5, 1]
            a3 = [2, 11, 14, 16, 3, 26, 32, 35, 4, 34, 29, 20, 2]
            a4 = [1, 8, 12, 15, 3, 26, 32, 35, 4, 33, 27, 17, 1]
        elif C==4:
            a1 = [1, 5, 6, 7, 8, 2, 27, 41, 50, 55, 4, 54, 48, 38, 23, 1]
            a2 = [1, 9, 14, 18, 21, 3, 22, 20, 17, 13, 2, 8, 7, 6, 5, 1]
            a3 = [2, 13, 17, 20, 22, 3, 37, 47, 53, 56, 4, 55, 50, 41, 27, 2]
            a4 = [1, 9, 14, 18, 21, 3, 37, 47, 53, 56, 4, 54, 48, 38, 23, 1]

        a1 = np.asarray(a1); a2 = np.asarray(a2); a3 = np.asarray(a3); a4 = np.asarray(a4)
        a1 -= 1;    a2 -= 1;    a3 -= 1;    a4 -= 1
        a_list = [a1,a2,a3,a4]

        fig = plt.figure()
        ax = Axes3D(fig)

        # face_elements = mesh.GetElementsWithBoundaryFacesTet()
        # elements = mesh.elements[face_elements,:]
        # print mesh.faces 
        # mesh.ArrangeFacesTet() 

        # for elem in range(elements.shape[0]):
        for elem in range(mesh.nelem):
        # for elem in range(1):

            # LOOP OVER TET FACES
            num_faces = 4
            for iface in range(num_faces):
                a = a_list[iface]

                x = mesh.points[mesh.elements[elem,a],0]
                y = mesh.points[mesh.elements[elem,a],1]
                z = mesh.points[mesh.elements[elem,a],2]

                # x = mesh.points[elements[elem,a],0]
                # y = mesh.points[elements[elem,a],1]
                # z = mesh.points[elements[elem,a],2]

                vertices = [zip(x,y,z)]
                poly_object = Poly3DCollection(vertices)
                poly_object.set_linewidth(1)
                poly_object.set_linestyle('solid')
                poly_object.set_facecolor((0.75,1,0.35)) 
                ax.add_collection3d(poly_object)


        # ax.autoscale(enable=True, axis=u'both', tight=None)
        ax.plot(mesh.points[:,0],mesh.points[:,1],mesh.points[:,2],'o',color='#F88379')

        plt.axis('equal')
        plt.axis('off')
        # plt.savefig('/home/roman/Desktop/destination_path.eps', format='eps', dpi=1000)
        plt.show()



    @staticmethod
    def HighOrderCurvedPatchPlot(MainData,mesh,TotalDisp,
        InterpolationDegree=40,EquallySpacedPoints=False,TriSurf=False,colorbar=False,
        PlotActualCurve=False):

        """High order curved triangular mesh plots, based on high order nodal FEM.
            The equally spaced FEM points do not work as good as the Fekete points 
        """

        from Core.QuadratureRules.FeketePointsTri import FeketePointsTri
        from Core.QuadratureRules.EquallySpacedPoints import EquallySpacedPointsTri
        from Core.QuadratureRules.NumericIntegrator import GaussLobattoQuadrature, GaussQuadrature
        import Core.InterpolationFunctions.TwoDimensional.Tri.hpNodal as Tri 
        from Core.InterpolationFunctions.OneDimensional.BasisFunctions import LagrangeGaussLobatto, Lagrange
        from Core.FiniteElements.GetBases import GetBases

        from copy import deepcopy
        from scipy.spatial import Delaunay
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.colors import LightSource
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
        import matplotlib.cm as cm
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        

        C = InterpolationDegree
        p = C+1
        nsize = int((p+1)*(p+2)/2.)
        nsize_2 = int((MainData.C+2)*(MainData.C+3)/2.)

        FeketePointsTri = FeketePointsTri(C)
        if EquallySpacedPoints is True:
            FeketePointsTri = EquallySpacedPointsTri(C)

        # BUILD DELAUNAY TRIANGULATION OF REFERENCE ELEMENTS
        TrianglesFunc = Delaunay(FeketePointsTri)
        Triangles = TrianglesFunc.simplices.copy()
        # plt.triplot(FeketePointsTri[:,0], FeketePointsTri[:,1], Triangles); plt.axis('off')


        # GET EQUALLY-SPACED/GAUSS-LOBATTO POINTS FOR THE EDGES
        if EquallySpacedPoints is False:
            GaussLobattoPointsOneD = GaussLobattoQuadrature(C+2)[0]
        else:
            GaussLobattoPointsOneD = Lagrange(C,0)[-1]

        BasesTri = np.zeros((nsize_2,FeketePointsTri.shape[0]),dtype=np.float64)
        for i in range(FeketePointsTri.shape[0]):
            BasesTri[:,i] = Tri.hpBases(MainData.C,FeketePointsTri[i,0],FeketePointsTri[i,1],
                EvalOpt=1,EquallySpacedPoints=EquallySpacedPoints,Transform=1)[0]

        BasesOneD = np.zeros((MainData.C+2,GaussLobattoPointsOneD.shape[0]),dtype=np.float64)
        for i in range(GaussLobattoPointsOneD.shape[0]):
            BasesOneD[:,i] = LagrangeGaussLobatto(MainData.C,GaussLobattoPointsOneD[i])[0]

        smesh = deepcopy(mesh)
        smesh.elements = mesh.elements[:,:MainData.ndim+1]
        smesh.edges = mesh.edges[:,:MainData.ndim]
        nmax = np.max(smesh.elements)+1
        smesh.points = mesh.points[:nmax,:]
        all_edges = smesh.GetEdgesTri()
        edge_elements = smesh.GetElementsEdgeNumberingTri()


        edge0 = []; edge1 = []; edge2 = []
        for i in range(0,MainData.C):
            edge0 = np.append(edge0,i+3)
            edge1 = np.append(edge1, 2*MainData.C+3 +i*MainData.C -i*(i-1)/2 )
            edge2 = np.append(edge2,MainData.C+3 +i*(MainData.C+1) -i*(i-1)/2 )

        edge0 = np.append(np.append(0,edge0),1)
        edge1 = np.append(np.append(1,edge1),2)
        edge2 = np.append(np.append(2,edge2[::-1]),0)

        ref_edges = np.concatenate((edge0[None,:],edge1[None,:],edge2[None,:]),axis=0).astype(np.int64)

        vpoints = np.copy(mesh.points)
        vpoints = mesh.points + TotalDisp[:,:,-1]

        x_edges = np.zeros((C+2,all_edges.shape[0]))
        y_edges = np.zeros((C+2,all_edges.shape[0]))

        for iedge in range(all_edges.shape[0]):

            ielem = edge_elements[iedge,0]
            edge = mesh.elements[ielem,ref_edges[edge_elements[iedge,1],:]]
            coord_edge = vpoints[edge,:]
            x_edges[:,iedge], y_edges[:,iedge] = np.dot(coord_edge.T,BasesOneD)


        # MAKE FIGURE
        fig = plt.figure()
        ls = LightSource(azdeg=315, altdeg=45)
        if TriSurf is True:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

        # PLOT CURVED EDGES
        ax.plot(x_edges,y_edges,'k')
        

        nnode = nsize*mesh.nelem
        nelem = Triangles.shape[0]*mesh.nelem

        Xplot = np.zeros((nnode,2),dtype=np.float64)
        Tplot = np.zeros((nelem,3),dtype=np.int64)
        Uplot = np.zeros(nnode,dtype=np.float64)

        # FOR CURVED ELEMENTS
        for ielem in range(mesh.nelem):
            Xplot[ielem*nsize:(ielem+1)*nsize,:] = np.dot(BasesTri.T, vpoints[mesh.elements[ielem,:],:])
            Tplot[ielem*TrianglesFunc.nsimplex:(ielem+1)*TrianglesFunc.nsimplex,:] = Triangles + ielem*nsize
            Uplot[ielem*nsize:(ielem+1)*nsize] = MainData.ScaledJacobian[ielem]

        # PLOT CURVED ELEMENTS
        if TriSurf is True:
            # ax.plot_trisurf(Tplot,Xplot[:,0], Xplot[:,1], Xplot[:,1]*0)
            triang = mtri.Triangulation(Xplot[:,0], Xplot[:,1],Tplot)
            ax.plot_trisurf(triang,Xplot[:,0]*0, edgecolor="none",facecolor="#ffddbb")
            ax.view_init(90,-90)
            ax.dist = 7
        else:
            # plt.tricontourf(Xplot[:,0], Xplot[:,1], Tplot, np.ones(Xplot.shape[0]), 100,alpha=0.8)
            plt.tricontourf(Xplot[:,0], Xplot[:,1], Tplot, Uplot, 100,alpha=0.8)
            # plt.tricontourf(Xplot[:,0], Xplot[:,1], Tplot[:4,:], np.ones(Xplot.shape[0]),alpha=0.8,origin='lower')
            pass

        # PLOT CURVED POINTS
        # plt.plot(vpoints[:,0],vpoints[:,1],'o',markersize=3,color='#F88379')
        plt.plot(vpoints[:,0],vpoints[:,1],'o',markersize=3,color='k')


        plt.set_cmap('viridis')
        plt.clim(0,1)
        

        if colorbar is True:
            ax_cbar = mpl.colorbar.make_axes(plt.gca(), shrink=1.0)[0]
            cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=cm.viridis,
                               norm=mpl.colors.Normalize(vmin=-0, vmax=1))
            cbar.set_clim(0, 1)
            divider = make_axes_locatable(ax_cbar)
            cax = divider.append_axes("right", size="1%", pad=0.005)
        
        if PlotActualCurve is True:
            ActualCurve = getattr(MainData,'ActualCurve',None)
            if ActualCurve is not None:
                for i in range(len(MainData.ActualCurve)):
                    actual_curve_points = MainData.ActualCurve[i]
                    plt.plot(actual_curve_points[:,0],actual_curve_points[:,1],'-r',linewidth=3)
            else:
                raise KeyError("You have not computed the CAD curve points")

        plt.axis('equal')
        plt.axis('off')







    @staticmethod
    def HighOrderCurvedPatchPlotTet(MainData,mesh,TotalDisp,
        InterpolationDegree=10,EquallySpacedPoints=False,TriSurf=False,colorbar=False,
        PlotActualCurve=False):

        """High order curved triangular mesh plots, based on high order nodal FEM.
            The equally spaced FEM points do not work as good as the Fekete points 
        """

        from Core.QuadratureRules.FeketePointsTri import FeketePointsTri
        from Core.QuadratureRules.EquallySpacedPoints import EquallySpacedPointsTri
        from Core.QuadratureRules.NumericIntegrator import GaussLobattoQuadrature, GaussQuadrature
        import Core.InterpolationFunctions.TwoDimensional.Tri.hpNodal as Tri 
        from Core.InterpolationFunctions.OneDimensional.BasisFunctions import LagrangeGaussLobatto, Lagrange
        from Core.FiniteElements.GetBases import GetBases

        from copy import deepcopy
        from scipy.spatial import Delaunay
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.colors import LightSource
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
        import matplotlib.cm as cm
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        

        C = InterpolationDegree
        p = C+1
        nsize = int((p+1)*(p+2)/2.)
        nsize_2 = int((MainData.C+2)*(MainData.C+3)/2.)

        FeketePointsTri = FeketePointsTri(C)
        if EquallySpacedPoints is True:
            FeketePointsTri = EquallySpacedPointsTri(C)

        # BUILD DELAUNAY TRIANGULATION OF REFERENCE ELEMENTS
        TrianglesFunc = Delaunay(FeketePointsTri)
        Triangles = TrianglesFunc.simplices.copy()
        # plt.triplot(FeketePointsTri[:,0], FeketePointsTri[:,1], Triangles); plt.axis('off')


        # GET EQUALLY-SPACED/GAUSS-LOBATTO POINTS FOR THE EDGES
        if EquallySpacedPoints is False:
            GaussLobattoPointsOneD = GaussLobattoQuadrature(C+2)[0]
        else:
            GaussLobattoPointsOneD = Lagrange(C,0)[-1]

        BasesTri = np.zeros((nsize_2,FeketePointsTri.shape[0]),dtype=np.float64)
        for i in range(FeketePointsTri.shape[0]):
            BasesTri[:,i] = Tri.hpBases(MainData.C,FeketePointsTri[i,0],FeketePointsTri[i,1],
                EvalOpt=1,EquallySpacedPoints=EquallySpacedPoints,Transform=1)[0]

        BasesOneD = np.zeros((MainData.C+2,GaussLobattoPointsOneD.shape[0]),dtype=np.float64)
        for i in range(GaussLobattoPointsOneD.shape[0]):
            BasesOneD[:,i] = LagrangeGaussLobatto(MainData.C,GaussLobattoPointsOneD[i])[0]

        smesh = deepcopy(mesh)
        smesh.elements = mesh.elements[:,:MainData.ndim+1]
        smesh.edges = mesh.edges[:,:MainData.ndim]
        nmax = np.max(smesh.elements)+1
        smesh.points = mesh.points[:nmax,:]
        all_edges = smesh.GetEdgesTri()
        edge_elements = smesh.GetElementsEdgeNumberingTri()


        edge0 = []; edge1 = []; edge2 = []
        for i in range(0,MainData.C):
            edge0 = np.append(edge0,i+3)
            edge1 = np.append(edge1, 2*MainData.C+3 +i*MainData.C -i*(i-1)/2 )
            edge2 = np.append(edge2,MainData.C+3 +i*(MainData.C+1) -i*(i-1)/2 )

        edge0 = np.append(np.append(0,edge0),1)
        edge1 = np.append(np.append(1,edge1),2)
        edge2 = np.append(np.append(2,edge2[::-1]),0)

        ref_edges = np.concatenate((edge0[None,:],edge1[None,:],edge2[None,:]),axis=0).astype(np.int64)

        vpoints = np.copy(mesh.points)
        vpoints = mesh.points + TotalDisp[:,:,-1]

        x_edges = np.zeros((C+2,all_edges.shape[0]))
        y_edges = np.zeros((C+2,all_edges.shape[0]))
        z_edges = np.zeros((C+2,all_edges.shape[0]))

        for iedge in range(all_edges.shape[0]):

            ielem = edge_elements[iedge,0]
            edge = mesh.elements[ielem,ref_edges[edge_elements[iedge,1],:]]
            coord_edge = vpoints[edge,:]
            # x_edges[:,iedge], y_edges[:,iedge] = np.dot(coord_edge.T,BasesOneD)
            x_edges[:,iedge], y_edges[:,iedge], z_edges[:,iedge] = np.dot(coord_edge.T,BasesOneD)


        # MAKE FIGURE
        fig = plt.figure()
        ls = LightSource(azdeg=315, altdeg=45)
        if TriSurf is True:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

        # PLOT CURVED EDGES
        ax.plot(x_edges,y_edges,'k')
        
        nface = mesh.faces.shape[0]

        nnode = nsize*nface
        nelem = Triangles.shape[0]*nface

        Xplot = np.zeros((nnode,3),dtype=np.float64)
        Tplot = np.zeros((nelem,3),dtype=np.int64)
        Uplot = np.zeros(nnode,dtype=np.float64)

        # FOR CURVED ELEMENTS
        for ielem in range(nface):
            Xplot[ielem*nsize:(ielem+1)*nsize,:] = np.dot(BasesTri.T, vpoints[mesh.faces[ielem,:],:])
            Tplot[ielem*TrianglesFunc.nsimplex:(ielem+1)*TrianglesFunc.nsimplex,:] = Triangles + ielem*nsize
            Uplot[ielem*nsize:(ielem+1)*nsize] = MainData.ScaledJacobian[ielem]

        # PLOT CURVED ELEMENTS
        if TriSurf is True:
            # ax.plot_trisurf(Tplot,Xplot[:,0], Xplot[:,1], Xplot[:,1]*0)
            triang = mtri.Triangulation(Xplot[:,0], Xplot[:,1],Tplot)            
            ax.plot_trisurf(triang,Xplot[:,0]*0, edgecolor="none",facecolor="#ffddbb")
            # ax.view_init(90,-90)
            # ax.dist = 7
        else:
            # plt.tricontourf(Xplot[:,0], Xplot[:,1], Tplot, np.ones(Xplot.shape[0]), 100,alpha=0.8)
            plt.tricontourf(Xplot[:,0], Xplot[:,1], Tplot, Uplot, 100,alpha=0.8)
            # plt.tricontourf(Xplot[:,0], Xplot[:,1], Tplot[:4,:], np.ones(Xplot.shape[0]),alpha=0.8,origin='lower')
            pass

        # PLOT CURVED POINTS
        # plt.plot(vpoints[:,0],vpoints[:,1],'o',markersize=3,color='#F88379')
        plt.plot(vpoints[:,0],vpoints[:,1],vpoints[:,0],'o',markersize=3,color='k')


        if TriSurf is False:        
            plt.set_cmap('viridis')
            plt.clim(0,1)        

        if colorbar is True:
            ax_cbar = mpl.colorbar.make_axes(plt.gca(), shrink=1.0)[0]
            cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=cm.viridis,
                               norm=mpl.colors.Normalize(vmin=-0, vmax=1))
            cbar.set_clim(0, 1)
            divider = make_axes_locatable(ax_cbar)
            cax = divider.append_axes("right", size="1%", pad=0.005)
        
        if PlotActualCurve is True:
            ActualCurve = getattr(MainData,'ActualCurve',None)
            if ActualCurve is not None:
                for i in range(len(MainData.ActualCurve)):
                    actual_curve_points = MainData.ActualCurve[i]
                    plt.plot(actual_curve_points[:,0],actual_curve_points[:,1],'-r',linewidth=3)
            else:
                raise KeyError("You have not computed the CAD curve points")

        plt.axis('equal')
        plt.axis('off')        







