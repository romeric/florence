import numpy as np 
import numpy.linalg as la
import gc
from warnings import warn
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
    """Post-process class for finite element solvers"""

    def __init__(self,ndim,nvar):

        self.domain_bases = None
        self.postdomain_bases = None
        self.boundary_bases = None
        self.ndim = ndim
        self.nvar = nvar
        self.analysis_type = None
        self.analysis_nature = None
        self.material_type = None

        self.is_scaledjacobian_computed = False
        self.is_material_anisotropic = False


    def SetBases(self,domain=None,postdomain=None,boundary=None):
        """Sets bases for all integration points for 'domain', 'postdomain' or 'boundary'
        """

        if domain is None and postdomain is None and boundary is None:
            warn("Nothing to be set") 

        self.domain_bases = domain
        self.postdomain_bases = postdomain
        self.boundary_bases = boundary

    def SetSolution(self,sol):
        self.sol = sol

    def SetAnalysis(self,AnalysisType,AnalysisNature):
        self.analysis_type = AnalysisType
        self.analysis_nature = AnalysisNature


    def TotalComponentSol(self,sol,ColumnsIn,ColumnsOut,AppliedDirichletInc,Iter,fsize):

        nvar = self.nvar
        ndim = self.ndim
        TotalSol = np.zeros((fsize,1))

        # GET TOTAL SOLUTION
        if self.analysis_type == 'Nonlinear':
            if self.analysis_nature =='Static':
                TotalSol[ColumnsIn,0] = sol
                if Iter==0:
                    TotalSol[ColumnsOut,0] = AppliedDirichletInc
            if self.analysis_nature !='Static':
                TotalSol = np.copy(sol)
                TotalSol[ColumnsOut,0] = AppliedDirichletInc

        elif self.analysis_type == 'Linear':
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




    
    def MeshQualityMeasures(self, mesh, TotalDisp, plot=True, show_plot=True):
        """Computes mesh quality measures, Q_1, Q_2, Q_3

            input:
                mesh:                   [Mesh] an instance of class mesh can be any mesh type

        """

        if self.is_scaledjacobian_computed is None:
            self.is_scaledjacobian_computed = False
        if self.is_material_anisotropic is None:
            self.is_material_anisotropic = False

        if self.is_scaledjacobian_computed is True:
            raise AssertionError('Scaled Jacobian seems to be already computed. Re-Computing it may return incorrect results')

        PostDomain = self.postdomain_bases

        vpoints = mesh.points
        if TotalDisp.ndim == 3:
            vpoints = vpoints + TotalDisp[:,:,-1]
        elif TotalDisp.ndim == 2:
            vpoints = vpoints + TotalDisp
        else:
            raise AssertionError("mesh points and displacment arrays are incompatible")

        elements = mesh.elements

        ScaledJacobian = np.zeros(elements.shape[0])
        ScaledFF = np.zeros(elements.shape[0])
        ScaledHH = np.zeros(elements.shape[0])

        if self.is_material_anisotropic:
            ScaledFNFN = np.zeros(elements.shape[0])
            ScaledCNCN = np.zeros(elements.shape[0])

        JMax =[]; JMin=[]
        for elem in range(mesh.nelem):
            LagrangeElemCoords = mesh.points[elements[elem,:],:]
            EulerElemCoords = vpoints[elements[elem,:],:]

            # COMPUTE KINEMATIC MEASURES AT ALL INTEGRATION POINTS USING EINSUM (AVOIDING THE FOR LOOP)
            # MAPPING TENSOR [\partial\vec{X}/ \partial\vec{\varepsilon} (ndim x ndim)]
            ParentGradientX = np.einsum('ijk,jl->kil',PostDomain.Jm,LagrangeElemCoords)
            # MAPPING TENSOR [\partial\vec{x}/ \partial\vec{\varepsilon} (ndim x ndim)]
            ParentGradientx = np.einsum('ijk,jl->kil',PostDomain.Jm,EulerElemCoords)
            # MATERIAL GRADIENT TENSOR IN PHYSICAL ELEMENT [\nabla_0 (N)]
            MaterialGradient = np.einsum('ijk,kli->ijl',la.inv(ParentGradientX),PostDomain.Jm)
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
            Jacobian = detF
            # USING INVARIANT F:F
            Q1 = np.sqrt(np.einsum('kij,lij->kl',F,F)).diagonal()
            # USING INVARIANT H:H
            Q2 = np.sqrt(np.einsum('ijk,ijl->kl',H,H)).diagonal()

            if self.is_material_anisotropic:
                Q4 = np.einsum('ijk,k',F,Directions[elem,:])
                Q4 = np.sqrt(np.dot(Q4,Q4.T)).diagonal()

                C = np.einsum('ikj,ikl->ijl',F,F)
                Q5 = np.einsum('ijk,k',C,Directions[elem,:]) 
                Q5 = np.sqrt(np.dot(Q5,Q5.T)).diagonal()

            # FIND MIN AND MAX VALUES
            JMin = np.min(Jacobian); JMax = np.max(Jacobian)
            ScaledJacobian[elem] = 1.0*JMin/JMax
            ScaledFF[elem] = 1.0*np.min(Q1)/np.max(Q1)
            ScaledHH[elem] = 1.0*np.min(Q2)/np.max(Q2)
            # Jacobian[elem] = np.min(detF)

            if self.is_material_anisotropic:
                ScaledFNFN[elem] = 1.0*np.min(Q4)/np.max(Q4)
                ScaledCNCN[elem] = 1.0*np.min(Q5)/np.max(Q5)

        if np.isnan(ScaledJacobian).any():
            warn("Jacobian of mapping is close to zero")

        print 'Minimum ScaledJacobian value is', ScaledJacobian.min(), \
        'corresponding to element', ScaledJacobian.argmin()

        print 'Minimum ScaledFF value is', ScaledFF.min(), \
        'corresponding to element', ScaledFF.argmin()

        print 'Minimum ScaledHH value is', ScaledHH.min(), \
        'corresponding to element', ScaledHH.argmin()

        if self.is_material_anisotropic:
            print 'Minimum ScaledFNFN value is', ScaledFNFN.min(), \
            'corresponding to element', ScaledFNFN.argmin()

            print 'Minimum ScaledCNCN value is', ScaledCNCN.min(), \
            'corresponding to element', ScaledCNCN.argmin()


        if plot:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            plt.bar(np.linspace(0,elements.shape[0]-1,elements.shape[0]),ScaledJacobian,width=1.,alpha=0.4)
            plt.xlabel(r'$Elements$',fontsize=18)
            plt.ylabel(r'$Scaled\, Jacobian$',fontsize=18)
            if show_plot:
                plt.show()

        # SET COMPUTED TO TRUE
        self.is_scaledjacobian_computed = True

        if not self.is_material_anisotropic:
            return self.is_scaledjacobian_computed, ScaledFF, ScaledHH, ScaledJacobian
        else:
            return self.is_scaledjacobian_computed, ScaledFF, ScaledHH, ScaledJacobian, ScaledFNFN, ScaledCNCN




    @staticmethod   
    def HighOrderPatchPlot(MainData,mesh,TotalDisp):

        import matplotlib.pyplot as plt
        
        fig = plt.figure()
        ax = fig.axes

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


    def HighOrderCurvedPatchPlot(self,*args,**kwargs):
        mesh = args[0]
        if mesh.element_type == "tri":
            self.HighOrderCurvedPatchPlotTri(*args,**kwargs)
        elif mesh.element_type == "tet":
            self.HighOrderCurvedPatchPlotTet(*args,**kwargs)
        else:
            raise ValueError("Unknown mesh type")


    @staticmethod
    def HighOrderCurvedPatchPlotTri(mesh,TotalDisp,QuantityToPlot=None,
        ProjectionFlags=None,InterpolationDegree=40,EquallySpacedPoints=False,
        TriSurf=False,colorbar=False,PlotActualCurve=False,plot_points=False,save=False,filename=None):

        """High order curved triangular mesh plots, based on high order nodal FEM.
            The equally spaced FEM points do not work as good as the Fekete points 
        """


        from Core.QuadratureRules.FeketePointsTri import FeketePointsTri
        from Core.QuadratureRules.EquallySpacedPoints import EquallySpacedPointsTri
        from Core.QuadratureRules.NumericIntegrator import GaussLobattoQuadrature
        from Core.QuadratureRules.NodeArrangement import NodeArrangementTri
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
        
        # SINCE THIS IS A 2D PLOT
        ndim = 2

        C = InterpolationDegree
        p = C+1
        nsize = int((p+1)*(p+2)/2.)
        CActual = mesh.InferPolynomialDegree() - 1 
        nsize_2 = int((CActual+2)*(CActual+3)/2.)

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
            BasesTri[:,i] = Tri.hpBases(CActual,FeketePointsTri[i,0],FeketePointsTri[i,1],
                EvalOpt=1,EquallySpacedPoints=EquallySpacedPoints,Transform=1)[0]

        BasesOneD = np.zeros((CActual+2,GaussLobattoPointsOneD.shape[0]),dtype=np.float64)
        for i in range(GaussLobattoPointsOneD.shape[0]):
            BasesOneD[:,i] = LagrangeGaussLobatto(CActual,GaussLobattoPointsOneD[i])[0]

        smesh = deepcopy(mesh)
        smesh.elements = mesh.elements[:,:ndim+1]
        nmax = np.max(smesh.elements)+1
        smesh.points = mesh.points[:nmax,:]
        smesh.GetEdgesTri()
        edge_elements = smesh.GetElementsEdgeNumberingTri()


        # GET EDGE ORDERING IN THE REFERENCE ELEMENT
        reference_edges = NodeArrangementTri(CActual)[0]
        reference_edges = np.concatenate((reference_edges,reference_edges[:,1,None]),axis=1)
        reference_edges = np.delete(reference_edges,1,1)

        # GET EULERIAN GEOMETRY
        if TotalDisp.ndim==3:
            vpoints = mesh.points + TotalDisp[:,:,-1]
        else:
            vpoints = mesh.points + TotalDisp

        # GET X & Y OF CURVED EDGES
        x_edges = np.zeros((C+2,smesh.all_edges.shape[0]))
        y_edges = np.zeros((C+2,smesh.all_edges.shape[0]))

        for iedge in range(smesh.all_edges.shape[0]):
            ielem = edge_elements[iedge,0]
            edge = mesh.elements[ielem,reference_edges[edge_elements[iedge,1],:]]
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
            Uplot[ielem*nsize:(ielem+1)*nsize] = QuantityToPlot[ielem]

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

        # PLOT CURVED POINTS
        if plot_points:
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

        if save:
            if filename is None:
                raise ValueError("No filename given. Supply one with extension")
            else:
                plt.savefig("filename",format="eps",dpi=300)

        plt.axis('equal')
        plt.axis('off')







    @staticmethod
    def HighOrderCurvedPatchPlotTet(mesh,TotalDisp,QuantityToPlot=None,
        ProjectionFlags=None,InterpolationDegree=20,EquallySpacedPoints=False,PlotActualCurve=False,
        plot_points=False,point_radius=0.1,colorbar=False,color=None,figure=None,
        show_plot=True,save=False,filename=None):

        """High order curved tetrahedral surfaces mesh plots, based on high order nodal FEM.
            The equally spaced FEM points do not work as good as the Fekete points 
        """



        from Core.QuadratureRules.FeketePointsTri import FeketePointsTri
        from Core.QuadratureRules.EquallySpacedPoints import EquallySpacedPointsTri
        from Core.QuadratureRules.NumericIntegrator import GaussLobattoQuadrature
        from Core.QuadratureRules.NodeArrangement import NodeArrangementTri
        import Core.InterpolationFunctions.TwoDimensional.Tri.hpNodal as Tri 
        from Core.InterpolationFunctions.OneDimensional.BasisFunctions import LagrangeGaussLobatto, Lagrange
        from Core.FiniteElements.GetBases import GetBases
        from Core import Mesh

        from copy import deepcopy
        from scipy.spatial import Delaunay
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.colors import LightSource
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
        import matplotlib.cm as cm
        from matplotlib.colors import ColorConverter
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        import os
        os.environ['ETS_TOOLKIT'] = 'qt4'
        from mayavi import mlab

        # SINCE THIS IS A 3D PLOT
        ndim=3

        C = InterpolationDegree
        p = C+1
        nsize = int((p+1)*(p+2)/2.)
        CActual = mesh.InferPolynomialDegree() - 1
        nsize_2 = int((CActual+2)*(CActual+3)/2.)

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
            BasesTri[:,i] = Tri.hpBases(CActual,FeketePointsTri[i,0],FeketePointsTri[i,1],
                EvalOpt=1,EquallySpacedPoints=EquallySpacedPoints,Transform=1)[0]

        BasesOneD = np.zeros((CActual+2,GaussLobattoPointsOneD.shape[0]),dtype=np.float64)
        for i in range(GaussLobattoPointsOneD.shape[0]):
            BasesOneD[:,i] = LagrangeGaussLobatto(CActual,GaussLobattoPointsOneD[i])[0]


        # GET ONLY THE FACES WHICH NEED TO BE PLOTTED 
        if ProjectionFlags is None:
            faces_to_plot_flag = np.ones(mesh.faces.shape[0])
        else:
            faces_to_plot_flag = ProjectionFlags.flatten()

        # CHECK IF ALL FACES NEED TO BE PLOTTED OR ONLY BOUNDARY FACES
        if faces_to_plot_flag.shape[0] > mesh.faces.shape[0]:
            # ALL FACES
            corr_faces = mesh.all_faces
            # FOR MAPPING DATA E.G. SCALED JACOBIAN FROM ELEMENTS TO FACES
            face_elements = mesh.GetElementsFaceNumberingTet()

        elif faces_to_plot_flag.shape[0] == mesh.faces.shape[0]:
            # ONLY BOUNDARY FACES
            corr_faces = mesh.faces
            # FOR MAPPING DATA E.G. SCALED JACOBIAN FROM ELEMENTS TO FACES
            face_elements = mesh.GetElementsWithBoundaryFacesTet()
        else:
            # raise ValueError("I do not understand what you want to plot")
            corr_faces = mesh.all_faces
            face_elements = mesh.GetElementsFaceNumberingTet()

        faces_to_plot = corr_faces[faces_to_plot_flag.flatten()==1,:]

        if QuantityToPlot is not None:
            quantity_to_plot = QuantityToPlot[face_elements[faces_to_plot_flag.flatten()==1,0]]

        # faces_to_plot = np.zeros_like(corr_faces)
        # quantity_to_plot = np.zeros(corr_faces.shape[0])
        # counter = 0
        # for i in range(corr_faces.shape[0]):
        #     if faces_to_plot_flag[i]==1:
        #         faces_to_plot[counter,:] = corr_faces[i,:]
        #         quantity_to_plot[counter] = QuantityToPlot[face_elements[i,0]]
        #         counter +=1
        # faces_to_plot = faces_to_plot[:counter,:]
        # quantity_to_plot = quantity_to_plot[:counter]


        # BUILD MESH OF SURFACE
        smesh = Mesh()
        smesh.element_type = "tri"
        # smesh.elements = np.copy(corr_faces)
        smesh.elements = np.copy(faces_to_plot)
        smesh.nelem = smesh.elements.shape[0]
        smesh.points = mesh.points[np.unique(smesh.elements),:]


        # MAP         
        unique_elements, inv = np.unique(smesh.elements,return_inverse=True)
        mapper = np.arange(unique_elements.shape[0])
        smesh.elements = mapper[inv].reshape(smesh.elements.shape)

        # nmin, nmax = np.min(smesh.elements), np.max(smesh.elements)
        # nrange = np.arange(nmin,nmax+1,dtype=np.int64)
        # counter = 0
        # for i in nrange:
        #     # rows, cols = np.where(smesh.elements==nrange[i])
        #     rows, cols = np.where(smesh.elements==i)
        #     if rows.shape[0]!=0:
        #         smesh.elements[rows,cols]=counter
        #         counter +=1


        smesh.GetBoundaryEdgesTri()
        smesh.GetEdgesTri()
        edge_elements = smesh.GetElementsEdgeNumberingTri()

        # color = mpl.colors.hex2color('#F88379')
        # linewidth = 100.2
        # nmax = np.max(smesh.elements[:,:3])+1
        # print np.max(smesh.elements[:,:3]), np.min(smesh.elements[:,:3])
        # trimesh_h = mlab.triangular_mesh(smesh.points[:nmax,0], 
        #         smesh.points[:nmax,1], smesh.points[:nmax,2], smesh.elements[:,:3],
        #         line_width=linewidth,tube_radius=linewidth,color=(0,0.6,0.4),
        #         representation='surface')
        # mlab.show()
        # return
        
        # GET EDGE ORDERING IN THE REFERENCE ELEMENT
        reference_edges = NodeArrangementTri(CActual)[0]
        reference_edges = np.concatenate((reference_edges,reference_edges[:,1,None]),axis=1)
        reference_edges = np.delete(reference_edges,1,1)

        # GET EULERIAN GEOMETRY
        if TotalDisp.ndim == 3:
            vpoints = mesh.points + TotalDisp[:,:,-1]
        elif TotalDisp.ndim == 2:
            vpoints = mesh.points + TotalDisp
        else:
            raise AssertionError("mesh points and displacment arrays are incompatible")

        # svpoints = vpoints[np.unique(mesh.faces),:]
        svpoints = vpoints[np.unique(faces_to_plot),:]
        del vpoints
        gc.collect()

        # GET X, Y & Z OF CURVED EDGES  
        x_edges = np.zeros((C+2,smesh.all_edges.shape[0]))
        y_edges = np.zeros((C+2,smesh.all_edges.shape[0]))
        z_edges = np.zeros((C+2,smesh.all_edges.shape[0]))

        for iedge in range(smesh.all_edges.shape[0]):
            ielem = edge_elements[iedge,0]
            edge = smesh.elements[ielem,reference_edges[edge_elements[iedge,1],:]]
            coord_edge = svpoints[edge,:]
            x_edges[:,iedge], y_edges[:,iedge], z_edges[:,iedge] = np.dot(coord_edge.T,BasesOneD)


        # MAKE A FIGURE
        if figure is None:
            figure = mlab.figure(bgcolor=(1,1,1),fgcolor=(1,1,1),size=(800,600))
        # PLOT CURVED EDGES
        # edge_width = .0016
        # edge_width = .08
        # edge_width = 0.0003
        # edge_width = 0.75
        edge_width = .03

        
        connections_elements = np.arange(x_edges.size).reshape(x_edges.shape[1],x_edges.shape[0])
        connections = np.zeros((x_edges.size,2),dtype=np.int64)
        for i in range(connections_elements.shape[0]):
            connections[i*(x_edges.shape[0]-1):(i+1)*(x_edges.shape[0]-1),0] = connections_elements[i,:-1]
            connections[i*(x_edges.shape[0]-1):(i+1)*(x_edges.shape[0]-1),1] = connections_elements[i,1:]
        connections = connections[:(i+1)*(x_edges.shape[0]-1),:]
        # print connenctions
        # point_cloulds = np.concatenate((x_edges.flatten()[:,None],y_edges.flatten()[:,None],z_edges.flatten()[:,None]),axis=1)
        
        figure.scene.disable_render = True
        # src = mlab.pipeline.scalar_scatter(x_edges.flatten(), y_edges.flatten(), z_edges.flatten())
        src = mlab.pipeline.scalar_scatter(x_edges.T.copy().flatten(), y_edges.T.copy().flatten(), z_edges.T.copy().flatten())
        src.mlab_source.dataset.lines = connections
        lines = mlab.pipeline.stripper(src)
        mlab.pipeline.surface(lines, color = (0,0,0), line_width=2)

        # for i in range(x_edges.shape[1]):
        #     mlab.plot3d(x_edges[:,i],y_edges[:,i],z_edges[:,i],color=(0,0,0),tube_radius=edge_width)
        

        nface = smesh.elements.shape[0]
        nnode = nsize*nface
        nelem = Triangles.shape[0]*nface

        Xplot = np.zeros((nnode,3),dtype=np.float64)
        Tplot = np.zeros((nelem,3),dtype=np.int64)

        # FOR CURVED ELEMENTS
        for ielem in range(nface):
            Xplot[ielem*nsize:(ielem+1)*nsize,:] = np.dot(BasesTri.T, svpoints[smesh.elements[ielem,:],:])
            Tplot[ielem*TrianglesFunc.nsimplex:(ielem+1)*TrianglesFunc.nsimplex,:] = Triangles + ielem*nsize

        if QuantityToPlot is not None:
            Uplot = np.zeros(nnode,dtype=np.float64)
            for ielem in range(nface):
                Uplot[ielem*nsize:(ielem+1)*nsize] = quantity_to_plot[ielem]

            # if face_elements[ielem,0] == 70:
            #     print ielem*TrianglesFunc.nsimplex,(ielem+1)*TrianglesFunc.nsimplex
            #     Uplot[ielem*nsize:(ielem+1)*nsize] = 0
            # else:
            #     Uplot[ielem*nsize:(ielem+1)*nsize] = 0.5

        # Tplot2 = Tplot[68921:70602,:]
        # Tplot3 = Tplot[21853:23534,:]
        # Tplot4 = Tplot[65559:67240,:]

        point_line_width = .002
        # point_line_width = 0.5
        # point_line_width = .0008
        # point_line_width = 2.
        # point_line_width = .045
        # point_line_width = .015 # F6


        if color is None:
            color=(197/255.,241/255.,197/255.)

        if QuantityToPlot is None:
            trimesh_h = mlab.triangular_mesh(Xplot[:,0], Xplot[:,1], Xplot[:,2], Tplot,
                line_width=point_line_width,color=color)
        else:
            trimesh_h = mlab.triangular_mesh(Xplot[:,0], Xplot[:,1], Xplot[:,2], Tplot, scalars = Uplot,
                line_width=point_line_width,colormap='summer')


        # if mesh.dd == 1:
        #     trimesh_h = mlab.triangular_mesh(Xplot[:,0], Xplot[:,1], Xplot[:,2], Tplot, scalars=Uplot,
        #         line_width=point_line_width,color=(197/255.,241/255.,197/255.))
        # else:
        #     trimesh_h = mlab.triangular_mesh(Xplot[:,0], Xplot[:,1], Xplot[:,2], Tplot, scalars=Uplot,
        #     line_width=point_line_width,color=(254/255., 111/255., 94/255.))

        # trimesh_h = mlab.triangular_mesh(Xplot[:,0], Xplot[:,1], Xplot[:,2], Tplot, scalars=Uplot,
        #     line_width=point_line_width,color=(197/255.,241/255.,197/255.))

        # trimesh_h = mlab.triangular_mesh(Xplot[:,0], Xplot[:,1], Xplot[:,2], Tplot2, scalars=Uplot,
        #     line_width=point_line_width,color=(73/255.,89/255.,133/255.))
        # trimesh_h = mlab.triangular_mesh(Xplot[:,0], Xplot[:,1], Xplot[:,2], Tplot3, scalars=Uplot,
        #     line_width=point_line_width,color=(254/255., 111/255., 94/255.))
        # trimesh_h = mlab.triangular_mesh(Xplot[:,0], Xplot[:,1], Xplot[:,2], Tplot4, scalars=Uplot,
        #     line_width=point_line_width,color=(254/255., 111/255., 94/255.))

        # trimesh_h = mlab.triangular_mesh(Xplot[:,0], Xplot[:,1], Xplot[:,2], Tplot, scalars=Uplot,line_width=point_line_width,colormap='summer')

        # PLOT POINTS ON CURVED MESH
        if plot_points:
            # mlab.points3d(svpoints[:,0],svpoints[:,1],svpoints[:,2],color=(0,0,0),mode='sphere',scale_factor=2.5*point_line_width)
            mlab.points3d(svpoints[:,0],svpoints[:,1],svpoints[:,2],color=(0,0,0),mode='sphere',scale_factor=point_radius)

        figure.scene.disable_render = False

        if QuantityToPlot is not None:
            # CHANGE LIGHTING OPTION
            trimesh_h.actor.property.interpolation = 'phong'
            trimesh_h.actor.property.specular = 0.1
            trimesh_h.actor.property.specular_power = 5

            # MAYAVI MLAB DOES NOT HAVE VIRIDIS AS OF NOW SO 
            # GET VIRIDIS COLORMAP FROM MATPLOTLIB
            color_func = ColorConverter()
            rgba_lower = color_func.to_rgba_array(cm.viridis.colors)
            # rgba_lower = color_func.to_rgba_array(cm.viridis_r.colors)
            RGBA_higher = np.round(rgba_lower*255).astype(np.int64)
            # UPDATE LUT OF THE COLORMAP
            trimesh_h.module_manager.scalar_lut_manager.lut.table = RGBA_higher 

        # SAVEFIG
        if save:
            if filename is None:
                raise ValueError("No filename given. Supply one with extension")
            else:
                mlab.savefig(filename,magnification="auto")


        # CONTROL CAMERA VIEW
        # mlab.view(azimuth=45, elevation=50, distance=80, focalpoint=None,
        #         roll=0, reset_roll=True, figure=None)

        # Falcon3D
        # mlab.view(azimuth=-140, elevation=50, distance=22, focalpoint=None,
        #         roll=60, reset_roll=True, figure=None)

        # F6
        # mlab.view(azimuth=-120, elevation=60, distance=52, focalpoint=None,
        #         roll=60, reset_roll=True, figure=None)

    
        if show_plot is True:
            # FORCE UPDATE MLAB TO UPDATE COLORMAP
            mlab.draw()
            mlab.show()     