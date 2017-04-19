import numpy as np 
import numpy.linalg as la
import gc
from warnings import warn
from Florence import QuadratureRule, FunctionSpace
from Florence.Base import JacobianError, IllConditionedError
from Florence.Utils import PWD, RSWD

# Modal Bases
# import Florence.FunctionSpace.TwoDimensional.Tri.hpModal as Tri 
# import Florence.FunctionSpace.ThreeDimensional.Tet.hpModal as Tet 
# Nodal Bases
from Florence.FunctionSpace import Tri
from Florence.FunctionSpace import Tet
from Florence.FunctionSpace import Quad
from Florence.FunctionSpace import Hex

from Florence.FiniteElements.ElementalMatrices.KinematicMeasures import *
from Florence import Mesh
from Florence.MeshGeneration import vtk_writer
from Florence.Utils import constant_camera_view


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
        self.directions = None

        self.mesh = None
        self.sol = None
        self.recovered_fields = None

        self.formulation = None
        self.material = None
        self.fem_solver = None


    def SetBases(self,domain=None,postdomain=None,boundary=None):
        """Sets bases for all integration points for 'domain', 'postdomain' or 'boundary'
        """

        if domain is None and postdomain is None and boundary is None:
            warn("Nothing to be set") 

        self.domain_bases = domain
        self.postdomain_bases = postdomain
        self.boundary_bases = boundary

    def SetMesh(self,mesh):
        """Set initial (undeformed) mesh"""
        self.mesh = mesh

    def SetSolution(self,sol):
        self.sol = sol

    def SetAnalysis(self,analysis_type=None, analysis_nature=None):
        self.analysis_type = analysis_type
        self.analysis_nature = analysis_nature

    def SetFormulation(self,formulation):
        self.formulation = formulation

    def SetMaterial(self,material):
        self.material = material

    def SetFEMSolver(self,fem_solver):
        self.fem_solver = fem_solver

    def SetAnisotropicOrientations(self,Directions):
        self.directions = Directions


    def TotalComponentSol(self, sol, ColumnsIn, ColumnsOut, AppliedDirichletInc, Iter, fsize):

        nvar = self.nvar
        ndim = self.ndim
        TotalSol = np.zeros((fsize,1))

        # GET TOTAL SOLUTION
        if self.analysis_nature == 'nonlinear':
            if self.analysis_type =='static':
                TotalSol[ColumnsIn,0] = sol
                if Iter==0:
                    TotalSol[ColumnsOut,0] = AppliedDirichletInc
            if self.analysis_type !='static':
                TotalSol = np.copy(sol)
                TotalSol[ColumnsOut,0] = AppliedDirichletInc

        elif self.analysis_nature == 'linear':
                TotalSol[ColumnsIn,0] = sol
                TotalSol[ColumnsOut,0] = AppliedDirichletInc
                
        # RE-ORDER SOLUTION COMPONENTS
        dU = TotalSol.reshape(TotalSol.shape[0]/nvar,nvar)

        return dU


    def StressRecovery(self):

        if self.mesh is None:
            raise ValueError("Mesh not set for post-processing")
        if self.sol is None:
            raise ValueError("Solution not set for post-processing")
        if self.formulation is None:
            raise ValueError("formulation not set for post-processing")
        if self.material is None:
            raise ValueError("mesh not set for post-processing")
        if self.fem_solver is None:
            raise ValueError("FEM solver not set for post-processing")

        if self.sol.shape[1] > self.nvar:
            return 

        mesh = self.mesh

        # GET THE UNDERLYING LINEAR MESH
        # lmesh = mesh.GetLinearMesh()
        C = mesh.InferPolynomialDegree() - 1
        ndim = mesh.InferSpatialDimension()


         # GET QUADRATURE
        norder = 2*C
        if norder == 0:
            norder=1
        Domain = FunctionSpace(mesh, p=C+1, evaluate_at_nodes=True)
        # w = Domain.AllGauss[:,0]

        fem_solver = self.fem_solver
        formulation = self.formulation
        material = self.material

        det = np.linalg.det
        inv = np.linalg.inv
        Jm = Domain.Jm
        AllGauss = Domain.AllGauss

        # exit()


        elements = mesh.elements
        points = mesh.points
        nelem = elements.shape[0]; npoint = points.shape[0]
        nodeperelem = elements.shape[1]
        LoadIncrement = fem_solver.number_of_load_increments
        requires_geometry_update = fem_solver.requires_geometry_update
        TotalDisp = self.sol[:,:]
        # TotalDisp = self.sol[:mesh.nnode,:]


        F = np.zeros((nelem,nodeperelem,ndim,ndim))
        CauchyStressTensor = np.zeros((nelem,nodeperelem,ndim,ndim))
        # DEFINE FOR MECH AND ELECTROMECH FORMULATIONS
        ElectricFieldx = np.zeros((nelem,nodeperelem,ndim))
        ElectricDisplacementx = np.zeros((nelem,nodeperelem,ndim))


        MainDict = {}
        MainDict['F'] = np.zeros((LoadIncrement,npoint,ndim,ndim))
        MainDict['CauchyStress'] = np.zeros((LoadIncrement,npoint,ndim,ndim))
        if formulation.fields == 'electro_mechanics':
            MainDict['ElectricFieldx'] = np.zeros((LoadIncrement,npoint,ndim))
            MainDict['ElectricDisplacementx'] = np.zeros((LoadIncrement,npoint,ndim))

        for Increment in range(LoadIncrement):
            Eulerx = points + TotalDisp[:,:ndim,Increment]
            if self.formulation.fields == 'electro_mechanics':
                Eulerp = TotalDisp[:,ndim,Increment]

            # LOOP OVER ELEMENTS
            for elem in range(nelem):
                # GET THE FIELDS AT THE ELEMENT LEVEL
                LagrangeElemCoords = points[elements[elem,:],:]
                EulerELemCoords = Eulerx[elements[elem,:],:]
                if self.formulation.fields == 'electro_mechanics':
                    ElectricPotentialElem =  Eulerp[elements[elem,:]]


                # GAUSS LOOP IN VECTORISED FORM
                ParentGradientX = np.einsum('ijk,jl->kil', Jm, LagrangeElemCoords)
                # MATERIAL GRADIENT TENSOR IN PHYSICAL ELEMENT [\nabla_0 (N)]
                MaterialGradient = np.einsum('ijk,kli->ijl', inv(ParentGradientX), Jm)
                # DEFORMATION GRADIENT TENSOR [\vec{x} \otimes \nabla_0 (N)]
                F[elem,:,:,:] = np.einsum('ij,kli->kjl', EulerELemCoords, MaterialGradient)
                # COMPUTE REMAINING KINEMATIC MEASURES
                StrainTensors = KinematicMeasures(F[elem,:,:,:], fem_solver.analysis_nature)

                # UPDATE/NO-UPDATE GEOMETRY
                if fem_solver.requires_geometry_update:
                    # MAPPING TENSOR [\partial\vec{X}/ \partial\vec{\varepsilon} (ndim x ndim)]
                    ParentGradientx = np.einsum('ijk,jl->kil',Jm,EulerELemCoords)
                    # SPATIAL GRADIENT TENSOR IN PHYSICAL ELEMENT [\nabla (N)]
                    SpatialGradient = np.einsum('ijk,kli->ilj',inv(ParentGradientx),Jm)
                    # COMPUTE ONCE detJ (GOOD SPEEDUP COMPARED TO COMPUTING TWICE)
                    detJ = np.einsum('i,i,i->i',AllGauss[:,0],np.abs(det(ParentGradientX)),
                        np.abs(StrainTensors['J']))
                else:
                    # SPATIAL GRADIENT AND MATERIAL GRADIENT TENSORS ARE EQUAL
                    SpatialGradient = np.einsum('ikj',MaterialGradient)
                    # COMPUTE ONCE detJ
                    detJ = np.einsum('i,i->i',AllGauss[:,0],np.abs(det(ParentGradientX)))


                # LOOP OVER GAUSS POINTS
                for counter in range(AllGauss.shape[0]):

                    if self.formulation.fields == 'electro_mechanics':
                        # GET ELECTRIC FILED
                        ElectricFieldx[elem,counter,:] = - np.dot(SpatialGradient[counter,:,:].T,
                            ElectricPotentialElem)

                        # COMPUTE ELECTRIC DISPLACEMENT
                        ElectricDisplacementx[elem,counter,:] = (material.ElectricDisplacementx(StrainTensors, 
                            ElectricFieldx[elem,counter,:], elem, counter))[:,0]
                    # else:
                        # ElectricFieldx, ElectricDisplacementx = [], []

                    if material.energy_type == "enthalpy":
                        
                        # COMPUTE CAUCHY STRESS TENSOR
                        if fem_solver.requires_geometry_update:
                            CauchyStressTensor[elem,counter,:] = material.CauchyStress(StrainTensors,
                                ElectricFieldx[elem,counter,:],elem,counter)

                    elif material.energy_type == "internal_energy":
                        # COMPUTE THE HESSIAN AT THIS GAUSS POINT
                        # H_Voigt = material.Hessian(StrainTensors,ElectricDisplacementx[elem,counter,:], elem, counter)
                        
                        # COMPUTE CAUCHY STRESS TENSOR
                        if fem_solver.requires_geometry_update:
                            CauchyStressTensor[elem,counter,:] = material.CauchyStress(StrainTensors,
                                ElectricDisplacementx[elem,counter,:],elem,counter)
            


            for inode in np.unique(elements):
                Els, Pos = np.where(elements==inode)
                ncommon_nodes = Els.shape[0]
                for uelem in range(ncommon_nodes):
                    MainDict['F'][Increment,inode,:,:] += F[Els[uelem],Pos[uelem],:,:]
                    if formulation.fields == "electro_mechanics":
                        MainDict['ElectricFieldx'][Increment,inode,:] += ElectricFieldx[Els[uelem],Pos[uelem],:]
                        MainDict['ElectricDisplacementx'][Increment,inode,:] += ElectricDisplacementx[Els[uelem],Pos[uelem],:]
                    MainDict['CauchyStress'][Increment,inode,:,:] += CauchyStressTensor[Els[uelem],Pos[uelem],:,:]

                # AVERAGE OUT
                MainDict['F'][Increment,inode,:,:] /= ncommon_nodes
                if formulation.fields == "electro_mechanics":
                    MainDict['ElectricFieldx'][Increment,inode,:] /= ncommon_nodes
                    MainDict['ElectricDisplacementx'][Increment,inode,:] /= ncommon_nodes
                MainDict['CauchyStress'][Increment,inode,:,:] /= ncommon_nodes


        self.recovered_fields = MainDict
        return


    def GetAugmentedSolution(self):
        """Computes all recovered variable and puts them in one big nd.array including with primary variables
            The following numbering convention is used for storing variables:

            Variables (quantity) numbering:

                quantity    mechanics 2D    mechanics 3D    electro_mechanics 2D    electro_mechanics 3D
                ----------------------------------------------------------------------------------------
                0           ux              ux              ux                      ux
                ----------------------------------------------------------------------------------------
                1           uy              uy              uy                      uy
                ----------------------------------------------------------------------------------------
                2           F_xx            uz              phi                     uz
                ----------------------------------------------------------------------------------------
                3           F_xy            F_xx            F_xx                    phi
                ----------------------------------------------------------------------------------------
                4           F_yx            F_xy            F_xy                    F_xx
                ----------------------------------------------------------------------------------------
                5           F_yy            F_xz            F_yx                    F_xy
                ----------------------------------------------------------------------------------------
                6           H_xx            F_yx            F_yy                    F_xz
                ----------------------------------------------------------------------------------------
                7           H_xy            F_yy            H_xx                    F_yx
                ----------------------------------------------------------------------------------------
                8           H_yx            F_yz            H_xy                    F_yy
                ----------------------------------------------------------------------------------------
                9           H_yy            F_zx            H_yx                    F_yz
                ----------------------------------------------------------------------------------------
                10          J               F_zy            H_yy                    F_zx
                ----------------------------------------------------------------------------------------
                11          C_xx            F_zz            J                       F_zy
                ----------------------------------------------------------------------------------------
                12          C_xy            H_xx            C_xx                    F_zz
                ----------------------------------------------------------------------------------------
                13          C_yy            H_xy            C_xy                    H_xx
                ----------------------------------------------------------------------------------------
                14          G_xx            H_xz            C_yy                    H_xy
                ----------------------------------------------------------------------------------------
                15          G_xy            H_yx            G_xx                    H_xz
                ----------------------------------------------------------------------------------------
                16          G_yy            H_yy            G_xy                    H_yx
                ----------------------------------------------------------------------------------------
                17          detC            H_yz            G_yy                    H_yy
                ----------------------------------------------------------------------------------------
                18          S_xx            H_zx            detC                    H_yz
                ----------------------------------------------------------------------------------------
                19          S_xy            H_zy            S_xx                    H_zx
                ----------------------------------------------------------------------------------------
                20          S_yy            H_zz            S_xy                    H_zy
                ----------------------------------------------------------------------------------------
                21                          J               S_yy                    H_zz
                ----------------------------------------------------------------------------------------
                22                          C_xx            E_x                     J
                ----------------------------------------------------------------------------------------
                23                          C_xy            E_y                     C_xx
                ----------------------------------------------------------------------------------------
                24                          C_xz            D_x                     C_xy
                ----------------------------------------------------------------------------------------
                25                          C_yy            D_y                     C_xz
                ----------------------------------------------------------------------------------------
                26                          C_yz                                    C_yy
                ----------------------------------------------------------------------------------------
                27                          C_zz                                    C_yz
                ----------------------------------------------------------------------------------------
                28                          G_xx                                    C_zz
                ----------------------------------------------------------------------------------------
                29                          G_xy                                    G_xx
                ----------------------------------------------------------------------------------------
                30                          G_xz                                    G_xy
                ----------------------------------------------------------------------------------------
                31                          G_yy                                    G_xz
                ----------------------------------------------------------------------------------------
                32                          G_yz                                    G_yx
                ----------------------------------------------------------------------------------------
                33                          G_zz                                    G_yy
                ----------------------------------------------------------------------------------------
                34                          detC                                    G_zz
                ----------------------------------------------------------------------------------------
                35                          S_xx                                    detC
                ----------------------------------------------------------------------------------------
                36                          S_xy                                    S_xx
                ----------------------------------------------------------------------------------------
                37                          S_xz                                    S_xy
                ----------------------------------------------------------------------------------------
                37                          S_yy                                    S_xz
                ----------------------------------------------------------------------------------------
                39                          S_yz                                    S_yy
                ----------------------------------------------------------------------------------------
                40                          S_zz                                    S_yz
                ----------------------------------------------------------------------------------------
                41                                                                  S_zz
                ----------------------------------------------------------------------------------------
                42                                                                  E_x
                ----------------------------------------------------------------------------------------
                43                                                                  E_y
                ----------------------------------------------------------------------------------------
                44                                                                  E_z
                ----------------------------------------------------------------------------------------
                45                                                                  D_x
                ----------------------------------------------------------------------------------------
                46                                                                  D_y
                ----------------------------------------------------------------------------------------
                47                                                                  D_z
                ----------------------------------------------------------------------------------------




            where S represents Cauchy stress tensor, E the electric field and D the electric
            displacements

            This function modifies self.sol to augmented_sol and returns the augmented solution 
            augmented_sol
        

        """

        if self.sol.shape[1] > self.nvar:
            return self.sol

        # GET RECOVERED VARIABLES ALL VARIABLE CHECKS ARE DONE IN STRESS RECOVERY
        self.StressRecovery()

        ndim = self.formulation.ndim
        fields = self.formulation.fields
        nnode = self.mesh.points.shape[0]
        increments = self.sol.shape[2]

        F = self.recovered_fields['F']
        J = np.linalg.det(F)
        H = np.einsum('ij,ijlk->ijkl',J,np.linalg.inv(F))
        C = np.einsum('ijlk,ijkm->ijlm',np.einsum('ijlk',F),F)
        detC = J**2
        G = np.einsum('ijlk,ijkm->ijlm',np.einsum('ijlk',H),H)
        Cauchy = self.recovered_fields['CauchyStress']

        if self.formulation.fields == "electro_mechanics":
            ElectricFieldx = self.recovered_fields['ElectricFieldx']
            ElectricDisplacementx = self.recovered_fields['ElectricDisplacementx']

        F = np.einsum('lijk',F).reshape(nnode,ndim**2,increments)
        H = np.einsum('lijk',H).reshape(nnode,ndim**2,increments)
        J = J.reshape(nnode,increments)
        C = np.einsum('lijk',C).reshape(nnode,ndim**2,increments)
        G = np.einsum('lijk',G).reshape(nnode,ndim**2,increments)
        detC = detC.reshape(nnode,increments)
        Cauchy = np.einsum('lijk',Cauchy).reshape(nnode,ndim**2,increments)

        if self.formulation.fields == "electro_mechanics":
            ElectricFieldx = ElectricFieldx.reshape(nnode,ndim,increments)
            ElectricDisplacementx = ElectricDisplacementx.reshape(nnode,ndim,increments)


        if ndim == 2:
            C = C[:,[0,1,3],:]
            G = G[:,[0,1,3],:]
            Cauchy = Cauchy[:,[0,1,3],:]
        elif ndim == 3:
            C = C[:,[0,1,2,4,5,8],:]
            G = G[:,[0,1,2,4,5,8],:]
            Cauchy = Cauchy[:,[0,1,2,4,5,8],:]


        if fields == "mechanics" and ndim == 2:

            augmented_sol = np.zeros((nnode,21,increments),dtype=np.float64)
            augmented_sol[:,:2,:]     = self.sol
            augmented_sol[:,2:6,:]    = F
            augmented_sol[:,6:10,:]   = H
            augmented_sol[:,10,:]     = J
            augmented_sol[:,11:14,:]  = C
            augmented_sol[:,14:17,:]  = G
            augmented_sol[:,17,:]     = detC
            augmented_sol[:,18:21,:]  = Cauchy

        elif fields == "mechanics" and ndim == 3:

            augmented_sol = np.zeros((nnode,41,increments),dtype=np.float64)
            augmented_sol[:,:3,:]     = self.sol
            augmented_sol[:,3:12,:]   = F
            augmented_sol[:,12:21,:]  = H
            augmented_sol[:,21,:]     = J
            augmented_sol[:,22:28,:]  = C
            augmented_sol[:,28:34,:]  = G
            augmented_sol[:,34,:]     = detC
            augmented_sol[:,35:41,:]  = Cauchy


        elif fields == "electro_mechanics" and ndim == 2:

            augmented_sol = np.zeros((nnode,26,increments),dtype=np.float64)
            augmented_sol[:,:3,:]     = self.sol
            augmented_sol[:,3:7,:]    = F
            augmented_sol[:,7:11,:]   = H
            augmented_sol[:,11,:]     = J
            augmented_sol[:,12:15,:]  = C
            augmented_sol[:,15:18,:]  = G
            augmented_sol[:,18,:]     = detC
            augmented_sol[:,19:22,:]  = Cauchy
            augmented_sol[:,22:24,:]  = ElectricFieldx
            augmented_sol[:,24:26,:]  = ElectricDisplacementx


        elif fields == "electro_mechanics" and ndim == 3:
            augmented_sol = np.zeros((nnode,48,increments),dtype=np.float64)

            augmented_sol[:,:4,:]     = self.sol
            augmented_sol[:,4:13,:]   = F
            augmented_sol[:,13:22,:]  = H
            augmented_sol[:,22,:]     = J
            augmented_sol[:,23:29,:]  = C
            augmented_sol[:,29:35,:]  = G
            augmented_sol[:,35,:]     = detC
            augmented_sol[:,36:42,:]  = Cauchy
            augmented_sol[:,42:45,:]  = ElectricFieldx
            augmented_sol[:,45:48,:]  = ElectricDisplacementx

        
        self.sol = augmented_sol
        return augmented_sol


    def QuantityNamer(self, num):
        """Returns the quantity (for augmented solution i.e. primary and recovered variables) 
            name given its number (from numbering order)
        """

        namer = None
        if num > 47:
            print('Quantity corresponds to ' + str(namer))
            return namer

        lines = []
        with open(__file__) as f:
            lines.append(f.readlines())
        lines = lines[0]

        line_number = len(lines)+1

        for counter, line in enumerate(lines):
            line = line.strip()
            if "quantity" in line and "mechanics" in line and "2D" in line and "3D" in line:
                line_number = counter
            if counter > line_number+1 and counter < line_number+100:
                spl = filter(None, line.split(" "))
                if spl[0] == str(num):
                    if self.nvar == 2 and self.ndim==2:
                        namer = spl[1]
                    elif self.nvar == 3 and self.ndim==2:
                        namer = spl[3]
                    elif self.nvar == 3 and self.ndim==3:
                        namer = spl[2]
                    elif self.nvar == 4:
                        namer = spl[4]
                    break

        print('Quantity corresponds to ' + str(namer))
        return namer



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

        AverageJacobian = np.zeros(elements.shape[0])

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
            # THIS GIVES A RESULT MUCH CLOSER TO ONE
            Jacobian = detF
            # USING INVARIANT F:F
            Q1 = np.sqrt(np.einsum('kij,lij->kl',F,F)).diagonal()
            # USING INVARIANT H:H
            Q2 = np.sqrt(np.einsum('ijk,ijl->kl',H,H)).diagonal()

            if self.is_material_anisotropic:
                Q4 = np.einsum('ijk,k',F,self.directions[elem,:])
                Q4 = np.sqrt(np.dot(Q4,Q4.T)).diagonal()

                C = np.einsum('ikj,ikl->ijl',F,F)
                Q5 = np.einsum('ijk,k',C,self.directions[elem,:]) 
                Q5 = np.sqrt(np.dot(Q5,Q5.T)).diagonal()

            # FIND MIN AND MAX VALUES
            JMin = np.min(Jacobian); JMax = np.max(Jacobian)
            ScaledJacobian[elem] = 1.0*JMin/JMax
            ScaledFF[elem] = 1.0*np.min(Q1)/np.max(Q1)
            ScaledHH[elem] = 1.0*np.min(Q2)/np.max(Q2)
            # Jacobian[elem] = np.min(detF)
            # print(np.min(Jacobian), np.max(Jacobian))
            AverageJacobian[elem] = np.mean(Jacobian)

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
        self.AverageJacobian = AverageJacobian

        if not self.is_material_anisotropic:
            return self.is_scaledjacobian_computed, ScaledFF, ScaledHH, ScaledJacobian
        else:
            return self.is_scaledjacobian_computed, ScaledFF, ScaledHH, ScaledJacobian, ScaledFNFN, ScaledCNCN





    def WriteVTK(self,filename=None, quantity="all", configuration="deformed", write_curved_mesh=True):
        """Writes results to a VTK file for Paraview

            quantity = "all" means write all solution fields, otherwise specific quantities 
            would be written based on augmented solution numbering order
        """

        if isinstance(quantity,int):
            if quantity>=self.sol.shape[1]:
                self.GetAugmentedSolution()
                if quantity >= self.sol.shape[1]:
                    raise ValueError('Plotting quantity not understood') 
            iterator = range(quantity,quantity+1)
        elif isinstance(quantity,str):
            if quantity=="all":
                self.GetAugmentedSolution()
                iterator = range(self.sol.shape[1])
            else:
                raise ValueError('Plotting quantity not understood')
        else:
            raise ValueError('Plotting quantity not understood')


        if filename is None:
            warn("file name not specified. I am going to write in the current directory")
            filename = PWD(__file__) + "/output.vtu"
        elif filename is not None:
            if isinstance(filename,str) is False:
                raise ValueError("file name should be a string")

        C = self.mesh.InferPolynomialDegree() - 1
        if C == 0:
            write_curved_mesh = False


        # GET LINEAR MESH & SOLUTION 
        lmesh = self.mesh.GetLinearMesh()
        sol = self.sol[:lmesh.nnode,:,:]

        if lmesh.element_type =='tri':
            cellflag = 5
        elif lmesh.element_type =='quad':
            cellflag = 9
        if lmesh.element_type =='tet':
            cellflag = 10
        elif lmesh.element_type == 'hex':
            cellflag = 12

        ndim = lmesh.points.shape[1]
        LoadIncrement = self.sol.shape[2]


        if write_curved_mesh:

            if lmesh.element_type =='tet':
                cellflag = 5
                tmesh = PostProcess.TessellateTets(self.mesh, np.zeros_like(self.mesh.points), 
                    QuantityToPlot=self.sol[:,0,0], plot_on_faces=False, plot_points=True,
                    interpolation_degree=10)
            elif lmesh.element_type =='hex':
                cellflag = 5 
                tmesh = PostProcess.TessellateHexes(self.mesh, np.zeros_like(self.mesh.points), 
                    QuantityToPlot=self.sol[:,0,0], plot_on_faces=False, plot_points=True,
                    interpolation_degree=10)
            else:
                raise ValueError('Not implemented yet. Use in-built visualiser for 2D problems')

            nsize = tmesh.nsize
            nface = tmesh.nface
            ssol = self.sol[np.unique(tmesh.faces_to_plot),:,:]

            for Increment in range(LoadIncrement):

                extrapolated_sol = np.zeros((tmesh.points.shape[0], self.sol.shape[1]))
                for ielem in range(nface):
                    extrapolated_sol[ielem*nsize:(ielem+1)*nsize,:] = np.dot(tmesh.bases_2, 
                        ssol[tmesh.smesh.elements[ielem,:],:, Increment])

                svpoints = self.mesh.points[np.unique(tmesh.faces_to_plot),:] + ssol[:,:ndim,Increment]

                for iedge in range(tmesh.smesh.all_edges.shape[0]):
                    ielem = tmesh.edge_elements[iedge,0]
                    edge = tmesh.smesh.elements[ielem,tmesh.reference_edges[tmesh.edge_elements[iedge,1],:]]
                    coord_edge = svpoints[edge,:]
                    tmesh.x_edges[:,iedge], tmesh.y_edges[:,iedge], tmesh.z_edges[:,iedge] = np.dot(coord_edge.T,tmesh.bases_1)

                edge_coords = np.concatenate((tmesh.x_edges.T.copy().flatten()[:,None], 
                    tmesh.y_edges.T.copy().flatten()[:,None],
                    tmesh.z_edges.T.copy().flatten()[:,None]),axis=1)

                vtk_writer.write_vtu(Verts=edge_coords, 
                    Cells={3:tmesh.connections},
                    fname=filename.split('.')[0]+'_curved_lines_increment_'+str(Increment)+'.vtu')

                vtk_writer.write_vtu(Verts=svpoints,
                    Cells={1:np.arange(svpoints.shape[0])},
                    fname=filename.split('.')[0]+'_curved_points_increment_'+str(Increment)+'.vtu')

                for quant in iterator:
                    vtk_writer.write_vtu(Verts=tmesh.points+extrapolated_sol[:,:ndim], 
                        Cells={cellflag:tmesh.elements}, pdata=extrapolated_sol[:,quant],
                        fname=filename.split('.')[0]+'_curved_quantity_'+str(quant)+'_increment_'+str(Increment)+'.vtu')

        else:

            if configuration == "original":
                for Increment in range(LoadIncrement):
                    for quant in iterator:
                        vtk_writer.write_vtu(Verts=lmesh.points, 
                            Cells={cellflag:lmesh.elements}, pdata=sol[:,quant,Increment],
                            fname=filename.split('.')[0]+'_quantity_'+str(quant)+'_increment_'+str(Increment)+'.vtu')
            elif configuration == "deformed":
                for Increment in range(LoadIncrement):
                    for quant in iterator:
                        vtk_writer.write_vtu(Verts=lmesh.points+sol[:,:ndim,Increment], 
                            Cells={cellflag:lmesh.elements}, pdata=sol[:,quant,Increment],
                            fname=filename.split('.')[0]+'_quantity_'+str(quant)+'_increment_'+str(Increment)+'.vtu')

        return


    def WriteHDF5(self, filename=None, compute_recovered_fields=True, dict_wise=False):
        """Writes the solution data to a HDF5 file. Give the extension name while providing filename

            Input:
                dict_wise:                  saves the dictionary of recovered variables as they are
                                            computed in StressRecovery
        """

        if compute_recovered_fields:
            self.GetAugmentedSolution()
        if filename is None:
            warn("file name not specified. I am going to write in the current directory")
            filename = PWD(__file__) + '/output.mat'
        elif filename is not None :
            if isinstance(filename,str) is False:
                raise ValueError("file name should be a string")

        from scipy.io import savemat

        if compute_recovered_fields is False:
            MainDict = {}
            MainDict['Solution'] = self.sol
            savemat(filename,MainDict,do_compression=True)
        else:
            if dict_wise:
                MainDict = self.recovered_fields
            else:
                MainDict = {}
            MainDict['Solution'] = self.sol
            savemat(filename,MainDict,do_compression=True)
        



    def PlotNewtonRaphsonConvergence(self, increment=None, save=False, filename=None):
        """Plots convergence of Newton-Raphson for a given increment"""

        if increment == None:
            increment = len(self.newton_raphson_convergence)-1

        import matplotlib.pyplot as plt
        # NEWTON-RAPHSON CONVERGENCE PLOT
        plt.plot(np.log10(self.newton_raphson_convergence['Increment_'+str(increment)]),'-ko') 
        axis_font = {'size':'18'}
        plt.xlabel(r'$No\;\; of\;\; Iterations$', **axis_font)
        plt.ylabel(r'$log_{10}|Residual|$', **axis_font)
        plt.grid('on')

        # SAVE
        if save:
            if filename is None:
                warn("No filename provided. I am going to write one in the current directory")
                filename = PWD(__file__) + '/output.eps'

            plt.savefig(filename, format='eps', dpi=500)

        plt.show()



    def Plot(self, figure=None, quantity=0, configuration="original", increment=-1, colorbar=True, axis_type=None, 
        plot_points=False, point_radius=0.5, plot_edges=True, plot_on_curvilinear_mesh=True, show_plot=True, save=False, filename=None):
        """ 

            Input:
                configuration:                  [str] to plot on original or deformed configuration 
                increment:                      [int] if results at specific increment needs to be plotted. 
        """


        if self.sol is None:
            raise ValueError("Solution not set for post-processing")
        if configuration != "deformed" and configuration != "original":
            raise ValueError("configuration can only be 'original' or 'deformed'")

        # CHECKS ARE DONE HERE
        if quantity>=self.sol.shape[1]:
            self.GetAugmentedSolution()
            if quantity >= self.sol.shape[1]:
                raise ValueError('Plotting quantity not understood')


        if save:
            if filename is None:
                warn("file name not specified. I am going to write in the current directory")
                filename = PWD(__file__) + "/output.eps"
            elif filename is not None :
                if isinstance(filename,str) is False:
                    raise ValueError("file name should be a string")

            # DO NOT SAVE THESE IMAGES IN .eps FORMAT 
            if len(filename.split("."))>1:
                if filename.split(".")[-1] != "png":
                    filename = filename.split(".")[0] + ".png"
            else:
                filename += ".png"


        C = self.mesh.InferPolynomialDegree()
        if C==0:
            plot_on_curvilinear_mesh = False

        # GET LINEAR MESH
        mesh = self.mesh.GetLinearMesh()
        # GET LINEAR SOLUTION 
        sol = np.copy(self.sol[:mesh.nnode,:,:])

        if self.mesh.element_type == "tri":

            from Florence.QuadratureRules.FeketePointsTri import FeketePointsTri
            from Florence.QuadratureRules.EquallySpacedPoints import EquallySpacedPointsTri
            from Florence.QuadratureRules.NumericIntegrator import GaussLobattoQuadrature
            from Florence.QuadratureRules.NodeArrangement import NodeArrangementTri
            from Florence.FunctionSpace import Tri 
            from Florence.FunctionSpace.OneDimensional.Line import LagrangeGaussLobatto, Lagrange
            from Florence.FunctionSpace.GetBases import GetBases

            
            from scipy.spatial import Delaunay
            from mpl_toolkits.mplot3d import Axes3D
            import matplotlib as mpl
            import matplotlib.pyplot as plt
            import matplotlib.tri as mtri
            import matplotlib.cm as cm
            from mpl_toolkits.axes_grid1 import make_axes_locatable

            # FIX FOR TRI MESHES
            point_radius = 2.

            # IF PLOT ON CURVED MESH IS ACTIVATED 
            if plot_on_curvilinear_mesh:

                if figure is None:
                    figure = plt.figure()

                if configuration=="original":
                    incr = 0
                    tmesh = PostProcess.CurvilinearPlotTri(self.mesh, np.zeros_like(self.sol), 
                        interpolation_degree=20, show_plot=False, figure=figure, 
                        save_tessellation=True, plot_points=plot_points, plot_edges=plot_edges)[-1]
                else:
                    incr = -1
                    tmesh = PostProcess.CurvilinearPlotTri(self.mesh, self.sol, 
                        interpolation_degree=20, show_plot=False, figure=figure, 
                        save_tessellation=True, plot_points=plot_points, plot_edges=plot_edges)[-1]

                nsize = tmesh.nsize
                nnode = tmesh.nnode
                extrapolated_sol = np.zeros((nnode,self.sol.shape[1]),dtype=np.float64)

                for ielem in range(self.mesh.nelem):
                    extrapolated_sol[ielem*nsize:(ielem+1)*nsize,:] = np.dot(tmesh.bases_2, self.sol[self.mesh.elements[ielem,:],:,incr])

                triang = mtri.Triangulation(tmesh.points[:,0], tmesh.points[:,1], tmesh.elements)
                h_fig = plt.tripcolor(triang, extrapolated_sol[:,quantity], shading='gouraud', cmap=cm.viridis)
                # h_fig = plt.tripcolor(triang, extrapolated_sol[:,quantity], shading='flat', cmap=cm.viridis)

                if save:
                    plt.savefig(filename, format="png", dpi=100, bbox_inches='tight',pad_inches=0.01)

                if colorbar:
                    plt.colorbar(h_fig,shrink=0.5)

                if show_plot:
                    plt.show()

                return


            # OTHERWISE PLOT ON PLANAR MESH 
            if figure is None:
                fig = plt.figure()

            if configuration == "original":
                plt.triplot(mesh.points[:,0],mesh.points[:,1], mesh.elements[:,:3],color='k')

                triang = mtri.Triangulation(mesh.points[:,0], mesh.points[:,1], mesh.elements)
                plt.tripcolor(triang, sol[:,quantity,-1], shading='gouraud', cmap=cm.viridis)

                if plot_points:
                    plt.plot(mesh.points[:,0], mesh.points[:,1],'o',markersize=point_radius,color='k')
            else:
                plt.triplot(mesh.points[:,0]+sol[:,0,-1], mesh.points[:,1]+sol[:,1,-1], mesh.elements[:,:3],color='k')

                triang = mtri.Triangulation(mesh.points[:,0]+sol[:,0,-1], 
                    mesh.points[:,1]+sol[:,1,-1], mesh.elements)
                plt.tripcolor(triang, sol[:,quantity,-1], shading='gouraud', cmap=cm.viridis)
                
                if plot_points:
                    plt.plot(mesh.points[:,0]+sol[:,0,-1], 
                        mesh.points[:,1]+sol[:,1,-1],'o',markersize=point_radius,color='k')

            plt.axis('equal')
            plt.axis('off')
            
            if colorbar:
                plt.colorbar()

            if save:
                plt.savefig(filename, format="png", dpi=100, bbox_inches='tight',pad_inches=0.01)

            if show_plot:
                plt.show()



        elif self.mesh.element_type == "tet":

            ndim = 3

            import os
            os.environ['ETS_TOOLKIT'] = 'qt4'
            from mayavi import mlab
            from matplotlib.colors import ColorConverter
            import matplotlib.cm as cm

            if figure is None:
                figure = mlab.figure(bgcolor=(1,1,1),fgcolor=(0,0,0),size=(800,600))


            if plot_on_curvilinear_mesh:

                if configuration == "original":
                    PostProcess.CurvilinearPlotTet(self.mesh, np.zeros_like(self.mesh.points),
                        QuantityToPlot = self.sol[:,quantity,increment],
                        figure=figure, show_plot=show_plot, plot_on_faces=False, 
                        plot_points=plot_points, point_radius=point_radius, plot_edges=plot_edges, 
                        colorbar=colorbar, save=save, filename=filename)

                elif configuration=="deformed":

                    PostProcess.CurvilinearPlotTet(self.mesh, self.sol[:,:ndim,-1], 
                        QuantityToPlot= self.sol[:,quantity,increment],
                        figure=figure, show_plot=show_plot, plot_on_faces=False, 
                        plot_points=plot_points, point_radius=point_radius, plot_edges=plot_edges, 
                        colorbar=colorbar, save=save, filename=filename)

                return 
            
            if configuration == "original":
                trimesh_h = mlab.triangular_mesh(mesh.points[:,0], mesh.points[:,1], mesh.points[:,2], 
                    mesh.faces, scalars=self.sol[:,0,-1])

                if plot_edges:
                    mlab.triangular_mesh(mesh.points[:,0], mesh.points[:,1], mesh.points[:,2], 
                        mesh.faces, representation="wireframe", color=(0,0,0)) # representation="mesh"

                if plot_points:
                    mlab.points3d(self.mesh.points[:,0],self.mesh.points[:,1],
                        self.mesh.points[:,2],color=(0,0,0),mode='sphere',scale_factor=point_radius)

            elif configuration == "deformed":
                trimesh_h = mlab.triangular_mesh(mesh.points[:,0]+sol[:,0,-1], 
                    mesh.points[:,1]+sol[:,1,-1], mesh.points[:,2]+sol[:,2,-1], 
                    mesh.faces, scalars=sol[:,quantity,-1])

                if plot_edges:
                    mlab.triangular_mesh(mesh.points[:,0]+sol[:,0,-1], 
                        mesh.points[:,1]+sol[:,1,-1], mesh.points[:,2]+sol[:,2,-1], 
                        mesh.faces, representation="wireframe", color=(0,0,0))

                if plot_points:
                    mlab.points3d(self.mesh.points[:,0]+self.sol[:,0,-1],self.mesh.points[:,1]+self.sol[:,1,-1],
                        self.mesh.points[:,2]+self.sol[:,2,-1],color=(0,0,0),mode='sphere',scale_factor=point_radius)


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


            if colorbar:
                cbar = mlab.colorbar(object=trimesh_h, title=self.QuantityNamer(quantity), 
                    orientation="horizontal",label_fmt="%9.2f")

            mlab.draw()
            mlab.show()

        elif self.mesh.element_type == "quad":

            ndim = 2
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            import matplotlib.tri as mtri


            if plot_on_curvilinear_mesh:

                if configuration=="original":
                    PostProcess.CurvilinearPlotQuad(self.mesh, np.zeros_like(self.sol), 
                        QuantityToPlot=self.sol[:,quantity,increment],
                        interpolation_degree=20, show_plot=show_plot, figure=figure, 
                        save_tessellation=True, plot_points=plot_points, plot_edges=plot_edges,
                        colorbar=colorbar, plot_on_faces=False)
                else:
                    PostProcess.CurvilinearPlotQuad(self.mesh, self.sol, 
                        QuantityToPlot=self.sol[:,quantity,increment],
                        interpolation_degree=20, show_plot=show_plot, figure=figure, 
                        save_tessellation=True, plot_points=plot_points, plot_edges=plot_edges, 
                        colorbar=colorbar, plot_on_faces=False)

                return 

            # PLOT ON STRAIGHT MESH
            fig = plt.figure()

            elements = np.concatenate((mesh.elements[:,:3],mesh.elements[:,[0,1,3]],
                mesh.elements[:,[0,2,3]],mesh.elements[:,[1,2,3]]),axis=0)

            if configuration=="original":
                triang = mtri.Triangulation(mesh.points[:,0], mesh.points[:,1], elements)
                h_fig = plt.tripcolor(triang, sol[:,quantity,increment], shading='gouraud', cmap=cm.viridis)

                if plot_points:
                    h_points = plt.plot(mesh.points[:,0], mesh.points[:,1], 'o', 
                        markersize=point_radius,color='k')

                if plot_edges:
                    round_elements = np.concatenate((mesh.elements,mesh.elements[:,0][:,None]),axis=1)
                    x_edges = mesh.points[round_elements,0]
                    y_edges = mesh.points[round_elements,1]
                    h_edges = plt.plot(x_edges.T, y_edges.T, 'k')

            elif configuration=="deformed":
                triang = mtri.Triangulation(mesh.points[:,0]+sol[:,0,-1], mesh.points[:,1]+sol[:,1,-1], elements)
                h_fig = plt.tripcolor(triang, sol[:,quantity,increment], shading='gouraud', cmap=cm.viridis)

                if plot_points:
                    h_points = plt.plot(mesh.points[:,0]+sol[:,0,-1], 
                        mesh.points[:,1]+sol[:,1,-1],'o',markersize=point_radius,color='k')

                if plot_edges:
                    round_elements = np.concatenate((mesh.elements,mesh.elements[:,0][:,None]),axis=1)
                    vpoints = mesh.points + sol[:,:ndim,-1]
                    x_edges = vpoints[round_elements,0]
                    y_edges = vpoints[round_elements,1]
                    h_edges = plt.plot(x_edges.T, y_edges.T, 'k')

                
            plt.axis('equal')
            plt.axis('off')

            if save:
                plt.savefig(filename, format="png", dpi=100, bbox_inches='tight',pad_inches=0.01)

            if colorbar:
                plt.colorbar(h_fig,shrink=0.5)

            if show_plot:
                plt.show()


        elif self.mesh.element_type == "hex":

            ndim = 3

            if configuration=="original":
                PostProcess.CurvilinearPlotHex(self.mesh, np.zeros_like(self.sol), 
                    QuantityToPlot=self.sol[:,quantity,increment],
                    interpolation_degree=20, show_plot=show_plot, figure=figure, 
                    plot_points=plot_points, point_radius=point_radius, plot_edges=plot_edges,
                    colorbar=colorbar, plot_on_faces=False, save=save, filename=filename)
            else:
                PostProcess.CurvilinearPlotHex(self.mesh, self.sol, 
                    QuantityToPlot=self.sol[:,quantity,increment],
                    interpolation_degree=20, show_plot=show_plot, figure=figure, 
                    plot_points=plot_points, point_radius=point_radius, plot_edges=plot_edges, 
                    colorbar=colorbar, plot_on_faces=False, save=save, filename=filename)




    def Animate(self, figure=None, quantity=0, configuration="original", increment=0, colorbar=True, axis_type=None, 
        plot_points=False, point_radius=0.5, plot_edges=True, plot_on_curvilinear_mesh=True, show_plot=True, save=False, 
        filename=None):
        """ 

            Input:
                configuration:                  [str] to plot on original or deformed configuration 
                increment:                      [int] if results at specific increment needs to be plotted. 
        """


        if self.sol is None:
            raise ValueError("Solution not set for post-processing")
        if configuration != "deformed" and configuration != "original":
            raise ValueError("configuration can only be 'original' or 'deformed'")

        # ALL CHECKS ARE DONE HERE
        if quantity>=self.sol.shape[1]:
            self.GetAugmentedSolution()
            if quantity >= self.sol.shape[1]:
                raise ValueError('Plotting quantity not understood')


        if save:
            if filename is None:
                warn("file name not specified. I am going to write in the current directory")
                filename = PWD(__file__) + "/output.mp4"
            elif filename is not None:
                if isinstance(filename,str) is False:
                    raise ValueError("file name should be a string")


        C = self.mesh.InferPolynomialDegree()
        if C==0:
            plot_on_curvilinear_mesh = False

        # GET LINEAR MESH
        mesh = self.mesh.GetLinearMesh()
        # GET LINEAR SOLUTION 
        sol = np.copy(self.sol[:mesh.nnode,:,:])

        if self.mesh.element_type == "tri":

            from copy import deepcopy
            from scipy.spatial import Delaunay
            import matplotlib as mpl
            from mpl_toolkits.mplot3d import Axes3D
            from matplotlib.colors import LightSource
            import matplotlib.pyplot as plt
            import matplotlib.tri as mtri
            import matplotlib.cm as cm
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            import matplotlib.animation as animation
            
            # FIX FOR TRI MESHES
            point_radius = 2.
            ndim = 2
            pp = .1

            if configuration=="deformed":
                vpoints = self.mesh.points+self.sol[:,:ndim,-1] 
            else:
                vpoints = self.mesh.points

            x_min = min(np.min(self.mesh.points[:,0]),np.min((self.mesh.points+self.sol[:,:ndim,-1])[:,0]))
            y_min = min(np.min(self.mesh.points[:,1]),np.min((self.mesh.points+self.sol[:,:ndim,-1])[:,1]))
            x_max = max(np.max(self.mesh.points[:,0]),np.max((self.mesh.points+self.sol[:,:ndim,-1])[:,0]))
            y_max = max(np.max(self.mesh.points[:,1]),np.max((self.mesh.points+self.sol[:,:ndim,-1])[:,1]))

            # IF PLOT ON CURVED MESH IS ACTIVATED 
            if plot_on_curvilinear_mesh:

                figure, ax = plt.subplots()

                tmesh = PostProcess.CurvilinearPlotTri(self.mesh, np.zeros_like(self.sol), 
                        interpolation_degree=10, show_plot=False, 
                        save_tessellation=True)[-1]
                plt.close()

                triang = mtri.Triangulation(tmesh.points[:,0], tmesh.points[:,1], tmesh.elements)
                nsize = tmesh.nsize
                nnode = tmesh.nnode
                extrapolated_sol = np.zeros((nnode,self.sol.shape[1]),dtype=np.float64)

                for ielem in range(self.mesh.nelem):
                    extrapolated_sol[ielem*nsize:(ielem+1)*nsize,:] = np.dot(tmesh.bases_2, self.sol[self.mesh.elements[ielem,:],:,0])


                def init_animation():
                    
                    ax.set_xlim([x_min - pp*np.abs(x_min), x_max + pp*np.abs(x_max)])
                    ax.set_ylim([y_min - pp*np.abs(y_min), y_max + pp*np.abs(y_max)])

                    # ax.set_xlim([-1, 7.5])
                    # ax.set_ylim([-9.5, 9.5])

                    if plot_points:
                        self.h_points, = ax.plot(self.mesh.points[:,0], self.mesh.points[:,1],'o',markersize=point_radius,color='k')

                    if plot_edges:
                        self.h_edges = ax.plot(tmesh.x_edges,tmesh.y_edges,'k')

                    self.h_fig = ax.tripcolor(triang, extrapolated_sol[:,quantity], shading='gouraud', cmap=cm.viridis)
                    self.h_fig.set_clim(extrapolated_sol[:,quantity].min(),extrapolated_sol[:,quantity].max())

                    ax.set_aspect('equal',anchor='C')
                    ax.set_axis_off()

                    if colorbar:
                        self.cbar = figure.colorbar(self.h_fig, shrink=0.5)
                        self.cbar.set_clim(extrapolated_sol[:,quantity].min(),extrapolated_sol[:,quantity].max())



                def animator(incr):

                    if plot_points and configuration=="deformed":
                        self.h_points.set_xdata(self.mesh.points[:,0]+self.sol[:,0,incr])
                        self.h_points.set_ydata(self.mesh.points[:,1]+self.sol[:,1,incr])

                    # PLOT EDGES
                    if plot_edges and configuration=="deformed":
                        for iedge in range(tmesh.x_edges.shape[1]):
                            ielem = tmesh.edge_elements[iedge,0]
                            edge = self.mesh.elements[ielem,tmesh.reference_edges[tmesh.edge_elements[iedge,1],:]]
                            coord_edge = (self.mesh.points + self.sol[:,:ndim,incr])[edge,:]
                            tmesh.x_edges[:,iedge], tmesh.y_edges[:,iedge] = np.dot(coord_edge.T,tmesh.bases_1)

                            self.h_edges[iedge].set_xdata(tmesh.x_edges[:,iedge])
                            self.h_edges[iedge].set_ydata(tmesh.y_edges[:,iedge])

                    extrapolated_sol = np.zeros((nnode,self.sol.shape[1]),dtype=np.float64)
                    for ielem in range(self.mesh.nelem):
                        extrapolated_sol[ielem*nsize:(ielem+1)*nsize,:] = np.dot(tmesh.bases_2, self.sol[self.mesh.elements[ielem,:],:,incr])

                    if configuration=="deformed":
                        triang.x = extrapolated_sol[:,0] + tmesh.points[:,0]
                        triang.y = extrapolated_sol[:,1] + tmesh.points[:,1]
                    self.h_fig.set_array(extrapolated_sol[:,quantity])

                    self.h_fig.set_clim(extrapolated_sol[:,quantity].min(),extrapolated_sol[:,quantity].max())
                    if colorbar:
                        self.cbar.set_clim(extrapolated_sol[:,quantity].min(),extrapolated_sol[:,quantity].max())

                interval = 0
                ani = animation.FuncAnimation(figure, animator, frames=range(0,self.sol.shape[2]), init_func=init_animation)


                if save:
                    # Writer = animation.writers['imagemagick']
                    # writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
                    if filename.split('.')[-1] == "gif":
                        ani.save(filename,fps=5, writer="imagemagick", savefig_kwargs={'pad_inches':0.01})
                    else:
                        ani.save(filename,fps=5, savefig_kwargs={'bbox_inches':'tight','pad_inches':0.01})
                    # convert column_bending_p04.gif -coalesce -repage 0x0 -crop -250-110 +repage column_bending_p004.gif

                if show_plot:
                    ax.clear()
                    if colorbar and save:
                        self.cbar.ax.cla()
                        self.cbar.ax.clear()
                        del self.cbar
                    plt.show()

                return




            # MAKE FIGURE
            figure, ax = plt.subplots()

            triang = mtri.Triangulation(mesh.points[:,0], 
                mesh.points[:,1], mesh.elements)

            def init_animation():

                ax.set_xlim([x_min - pp*np.abs(x_min), x_max + pp*np.abs(x_max)])
                ax.set_ylim([y_min - pp*np.abs(y_min), y_max + pp*np.abs(y_max)])

                if plot_edges:
                    self.h_edges = ax.triplot(triang,color='k')
                self.h_fig = ax.tripcolor(triang, sol[:,quantity,0], shading='gouraud', cmap=cm.viridis)
                self.h_fig.set_clim(sol[:,quantity,0].min(),sol[:,quantity,0].max())

                if plot_points:
                    self.h_points, = ax.plot(mesh.points[:,0], mesh.points[:,1],
                        'o', markersize=point_radius, color='k')

                if colorbar:
                    self.cbar = figure.colorbar(self.h_fig, shrink=0.5)
                    self.cbar.set_clim(sol[:,quantity,0].min(),sol[:,quantity,0].max())
                
                ax.set_aspect('equal',anchor='C')
                ax.set_axis_off()


            def animator(incr):

                ax.clear()
                    
                if configuration=="deformed":
                    vpoints = mesh.points + sol[:,:ndim,incr]
                    triang.x = vpoints[:,0]
                    triang.y = vpoints[:,1]
                else:
                    vpoints = mesh.points

                if plot_points:
                    self.h_points, = ax.plot(vpoints[:,0], vpoints[:,1],
                        'o', markersize=point_radius, color='k')

                if plot_edges:
                    self.h_edges = ax.triplot(triang,color='k')

                self.h_fig = ax.tripcolor(triang, sol[:,quantity,incr], shading='gouraud', cmap=cm.viridis)

                self.h_fig.set_clim(sol[:,quantity,incr].min(),sol[:,quantity,incr].max())
                
                if colorbar:
                    # self.cbar = figure.colorbar(self.h_fig, shrink=0.5)
                    self.cbar.set_clim(sol[:,quantity,incr].min(),sol[:,quantity,incr].max())


                ax.set_aspect('equal',anchor='C')
                ax.set_axis_off()

 
            ani = animation.FuncAnimation(figure,func=animator, 
                frames=range(self.sol.shape[2]), init_func=init_animation)
            
            if save:
                if filename.split('.')[-1] == "gif":
                    ani.save(filename,fps=5, writer="imagemagick", savefig_kwargs={'pad_inches':0.01})
                else:
                    ani.save(filename,fps=5, savefig_kwargs={'bbox_inches':'tight','pad_inches':0.01})

            if show_plot:
                plt.show()



        elif self.mesh.element_type == "tet":

            import os
            os.environ['ETS_TOOLKIT'] = 'qt4'
            from mayavi import mlab
            from matplotlib.colors import ColorConverter
            import matplotlib.cm as cm

            ndim = 3

            if figure is None:
                figure = mlab.figure(bgcolor=(1,1,1),fgcolor=(0,0,0),size=(800,600))

            if plot_on_curvilinear_mesh:

                tmesh = PostProcess.CurvilinearPlotTet(self.mesh, 
                    np.zeros_like(self.mesh.points),
                    QuantityToPlot = self.sol[:,quantity,increment],
                    show_plot=False, plot_on_faces=False, 
                    plot_points=plot_points, point_radius=point_radius, plot_edges=plot_edges, 
                    colorbar=colorbar, save_tessellation=True)[-1]

                nsize = tmesh.nsize
                nface = tmesh.nface

                extrapolated_sol = np.zeros((tmesh.points.shape[0], self.sol.shape[1]))
                ssol = self.sol[np.unique(tmesh.faces_to_plot),:,:]
                for ielem in range(nface):
                    extrapolated_sol[ielem*nsize:(ielem+1)*nsize,:] = np.dot(tmesh.bases_2, 
                        ssol[tmesh.smesh.elements[ielem,:],:, 0])

                trimesh_h = mlab.triangular_mesh(tmesh.points[:,0], tmesh.points[:,1], 
                    tmesh.points[:,2], tmesh.elements, scalars = extrapolated_sol[:,quantity],
                    line_width=0.5)

                if plot_edges:
                    src = mlab.pipeline.scalar_scatter(tmesh.x_edges.T.copy().flatten(), 
                        tmesh.y_edges.T.copy().flatten(), tmesh.z_edges.T.copy().flatten())
                    src.mlab_source.dataset.lines = tmesh.connections
                    lines = mlab.pipeline.stripper(src)
                    h_edges = mlab.pipeline.surface(lines, color = (0,0,0), line_width=2)

                if plot_points:
                    svpoints = self.mesh.points[np.unique(tmesh.faces_to_plot),:]
                    h_points = mlab.points3d(svpoints[:,0], svpoints[:,1], svpoints[:,2],
                        color=(0,0,0), mode='sphere', scale_factor=point_radius)

                # mlab.view(azimuth=45, elevation=50, distance=90, focalpoint=None,
                    # roll=0, reset_roll=True, figure=None)


                m_trimesh = trimesh_h.mlab_source
                if plot_edges:
                    m_wire = h_edges.mlab_source
                if plot_points:
                    m_points = h_points.mlab_source

                @mlab.animate(delay=100) 
                def animator():
                    # fig = mlab.gcf()

                    # ssol = self.sol[np.unique(tmesh.faces_to_plot),:,:]

                    for i in range(0, self.sol.shape[2]):

                        # GET SOLUTION AT THIS INCREMENT
                        extrapolated_sol = np.zeros((tmesh.points.shape[0], self.sol.shape[1]))
                        for ielem in range(nface):
                            extrapolated_sol[ielem*nsize:(ielem+1)*nsize,:] = np.dot(tmesh.bases_2, 
                                ssol[tmesh.smesh.elements[ielem,:],:, i])

                        svpoints = self.mesh.points[np.unique(tmesh.faces_to_plot),:] + ssol[:,:ndim,i]

                        if configuration == "deformed":

                            m_trimesh.reset(x=tmesh.points[:,0]+extrapolated_sol[:,0], 
                                y=tmesh.points[:,1]+extrapolated_sol[:,1],
                                z=tmesh.points[:,2]+extrapolated_sol[:,2], 
                                scalars=extrapolated_sol[:,quantity])
                            
                            # GET UPDATED EDGE COORDINATES AT THIS INCREMENT
                            if plot_edges:
                                for iedge in range(tmesh.smesh.all_edges.shape[0]):
                                    ielem = tmesh.edge_elements[iedge,0]
                                    edge = tmesh.smesh.elements[ielem,tmesh.reference_edges[tmesh.edge_elements[iedge,1],:]]
                                    coord_edge = svpoints[edge,:]
                                    tmesh.x_edges[:,iedge], tmesh.y_edges[:,iedge], tmesh.z_edges[:,iedge] = np.dot(coord_edge.T,tmesh.bases_1)

                                m_wire.reset(x=tmesh.x_edges.T.copy().flatten(), 
                                    y=tmesh.y_edges.T.copy().flatten(),
                                    z=tmesh.z_edges.T.copy().flatten())

                            if plot_points:
                                m_points.reset(x=svpoints[:,0], y=svpoints[:,1], z=svpoints[:,2])

                        else:

                            m_trimesh.reset(scalars=extrapolated_sol[:,quantity])

                        
                        if colorbar:

                            cbar = mlab.colorbar(object=trimesh_h, title=self.QuantityNamer(quantity), 
                                orientation="horizontal",label_fmt="%9.2f")

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
                        

                        figure.scene.reset_zoom()
                        # fig.scene.reset_zoom()

                        # SAVE EACH FRAME USING AN EXTERNAL TOOL
                        if save:
                            mlab.savefig(filename.split(".")[0]+"_increment_"+str(i)+".png")
                        
                        yield

                with constant_camera_view():
                    animator()

                # mlab.view(azimuth=45, elevation=50, distance=90, focalpoint=None,
                    # roll=0, reset_roll=True, figure=None)
                

                if show_plot:
                    # mlab.draw()
                    mlab.show()


                if save:
                    # mlab.close()
                    import subprocess
                    # fname = os.path.basename(filename).split(".")[0]
                    # ex = os.path.basename(filename).split(".")[-1]
                    # os.path.join(PWD(filename), os.path.basename(filename))

                    # REMOVE OLD FILE WITH THE SAME NAME
                    p = subprocess.Popen('rm -f ' + filename, shell=True)
                    p.wait()

                    fps = 25
                    p = subprocess.Popen('ffmpeg -framerate ' +str(fps)+ ' -i ' + \
                        filename.split('.')[0] + '_increment_%00d.png' + \
                        ' -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p '+ filename, 
                        shell=True)
                    p.wait()
                    # REMOVE TEMPS
                    p = subprocess.Popen('rm -rf ' + filename.split(".")[0]+"_increment_*", 
                        shell=True)
                    p.wait()
                    # ffmpeg -framerate 25 -i yy_increment_%0d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output.mp4
                    # convert -delay 10 -loop 0 plate_with_holes_frame.00*.png plate_with_holes_wireframe.gif


                # del tmesh

            return 


            # FOR PLANAR MESHES
            if configuration == "original":
                trimesh_h = mlab.triangular_mesh(mesh.points[:,0], mesh.points[:,1], mesh.points[:,2], 
                    mesh.faces, scalars=self.sol[:,quantity,0])

                wire_h = mlab.triangular_mesh(mesh.points[:,0], mesh.points[:,1], mesh.points[:,2], 
                    mesh.faces, representation="mesh", color=(0,0,0))

                points_h = mlab.points3d(self.mesh.points[:,0],self.mesh.points[:,1],
                    self.mesh.points[:,2],color=(0,0,0),mode='sphere',scale_factor=point_radius)

            elif configuration == "deformed":
                trimesh_h = mlab.triangular_mesh(mesh.points[:,0]+sol[:,0,0], 
                    mesh.points[:,1]+sol[:,1,0], mesh.points[:,2]+sol[:,2,0], 
                    mesh.faces, scalars=sol[:,quantity,0])

                wire_h = mlab.triangular_mesh(mesh.points[:,0]+sol[:,0,0], 
                    mesh.points[:,1]+sol[:,1,0], mesh.points[:,2]+sol[:,2,0], 
                    mesh.faces, representation="mesh", color=(0,0,0))

                points_h = mlab.points3d(self.mesh.points[:,0]+self.sol[:,0,0],self.mesh.points[:,1]+self.sol[:,1,0],
                    self.mesh.points[:,2]+self.sol[:,2,0],color=(0,0,0),mode='sphere',scale_factor=point_radius)


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


            if colorbar:
                cbar = mlab.colorbar(object=trimesh_h, orientation="vertical",label_fmt="%9.2f")


            @mlab.animate(delay=1000) 
            def animator():
                fig = mlab.gcf()
                m_trimesh = trimesh_h.mlab_source
                m_wire = wire_h.mlab_source
                m_points = points_h.mlab_source
                for i in range(1,sol.shape[2]):
                    m_trimesh.reset(x=mesh.points[:,0]+sol[:,0,i], y=mesh.points[:,1]+sol[:,1,i],
                        z=mesh.points[:,2]+sol[:,2,i], scalars=sol[:,quantity,i])

                    m_wire.reset(x=mesh.points[:,0]+sol[:,0,i], y=mesh.points[:,1]+sol[:,1,i],
                        z=mesh.points[:,2]+sol[:,2,i])

                    m_points.reset(x=mesh.points[:,0]+sol[:,0,i], y=mesh.points[:,1]+sol[:,1,i],
                        z=mesh.points[:,2]+sol[:,2,i])

                    fig.scene.reset_zoom()
                    yield

            animator()
            mlab.show()


        if self.mesh.element_type == "quad":

            from copy import deepcopy
            from scipy.spatial import Delaunay
            import matplotlib as mpl
            from mpl_toolkits.mplot3d import Axes3D
            from matplotlib.colors import LightSource
            import matplotlib.pyplot as plt
            import matplotlib.tri as mtri
            import matplotlib.cm as cm
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            import matplotlib.animation as animation
            
            # FIX FOR TRI MESHES
            point_radius = 2.
            ndim = 2
            pp = .1

            if configuration=="deformed":
                vpoints = self.mesh.points+self.sol[:,:ndim,-1] 
            else:
                vpoints = self.mesh.points

            x_min = min(np.min(self.mesh.points[:,0]),np.min((self.mesh.points+self.sol[:,:ndim,-1])[:,0]))
            y_min = min(np.min(self.mesh.points[:,1]),np.min((self.mesh.points+self.sol[:,:ndim,-1])[:,1]))
            x_max = max(np.max(self.mesh.points[:,0]),np.max((self.mesh.points+self.sol[:,:ndim,-1])[:,0]))
            y_max = max(np.max(self.mesh.points[:,1]),np.max((self.mesh.points+self.sol[:,:ndim,-1])[:,1]))

            # IF PLOT ON CURVED MESH IS ACTIVATED 
            if plot_on_curvilinear_mesh:

                figure, ax = plt.subplots()

                tmesh = PostProcess.TessellateQuads(self.mesh, np.zeros_like(self.sol), 
                        interpolation_degree=10, plot_points=plot_points, plot_edges=plot_edges,
                        plot_on_faces=False)

                triang = mtri.Triangulation(tmesh.points[:,0], tmesh.points[:,1], tmesh.elements)
                nsize = tmesh.nsize
                nnode = tmesh.nnode
                extrapolated_sol = np.zeros((nnode,self.sol.shape[1]),dtype=np.float64)

                for ielem in range(self.mesh.nelem):
                    extrapolated_sol[ielem*nsize:(ielem+1)*nsize,:] = np.dot(tmesh.bases_2, self.sol[self.mesh.elements[ielem,:],:,0])


                def init_animation():
                    
                    ax.set_xlim([x_min - pp*np.abs(x_min), x_max + pp*np.abs(x_max)])
                    ax.set_ylim([y_min - pp*np.abs(y_min), y_max + pp*np.abs(y_max)])

                    if plot_points:
                        self.h_points, = ax.plot(self.mesh.points[:,0], self.mesh.points[:,1],'o',markersize=point_radius,color='k')

                    if plot_edges:
                        self.h_edges = ax.plot(tmesh.x_edges,tmesh.y_edges,'k')

                    self.h_fig = ax.tripcolor(triang, extrapolated_sol[:,quantity], shading='gouraud', cmap=cm.viridis)
                    self.h_fig.set_clim(extrapolated_sol[:,quantity].min(),extrapolated_sol[:,quantity].max())

                    ax.set_aspect('equal',anchor='C')
                    ax.set_axis_off()

                    if colorbar:
                        self.cbar = figure.colorbar(self.h_fig, shrink=0.5)
                        self.cbar.set_clim(extrapolated_sol[:,quantity].min(),extrapolated_sol[:,quantity].max())



                def animator(incr):

                    if plot_points and configuration=="deformed":
                        self.h_points.set_xdata(self.mesh.points[:,0]+self.sol[:,0,incr])
                        self.h_points.set_ydata(self.mesh.points[:,1]+self.sol[:,1,incr])

                    # PLOT EDGES
                    if plot_edges and configuration=="deformed":
                        for iedge in range(tmesh.x_edges.shape[1]):
                            ielem = tmesh.edge_elements[iedge,0]
                            edge = self.mesh.elements[ielem,tmesh.reference_edges[tmesh.edge_elements[iedge,1],:]]
                            coord_edge = (self.mesh.points + self.sol[:,:ndim,incr])[edge,:]
                            tmesh.x_edges[:,iedge], tmesh.y_edges[:,iedge] = np.dot(coord_edge.T,tmesh.bases_1)

                            self.h_edges[iedge].set_xdata(tmesh.x_edges[:,iedge])
                            self.h_edges[iedge].set_ydata(tmesh.y_edges[:,iedge])

                    extrapolated_sol = np.zeros((nnode,self.sol.shape[1]),dtype=np.float64)
                    for ielem in range(self.mesh.nelem):
                        extrapolated_sol[ielem*nsize:(ielem+1)*nsize,:] = np.dot(tmesh.bases_2, self.sol[self.mesh.elements[ielem,:],:,incr])

                    if configuration=="deformed":
                        triang.x = extrapolated_sol[:,0] + tmesh.points[:,0]
                        triang.y = extrapolated_sol[:,1] + tmesh.points[:,1]
                    self.h_fig.set_array(extrapolated_sol[:,quantity])

                    self.h_fig.set_clim(extrapolated_sol[:,quantity].min(),extrapolated_sol[:,quantity].max())
                    if colorbar:
                        self.cbar.set_clim(extrapolated_sol[:,quantity].min(),extrapolated_sol[:,quantity].max())

                interval = 0
                ani = animation.FuncAnimation(figure, animator, frames=range(0,self.sol.shape[2]), init_func=init_animation)


                if save:
                    # Writer = animation.writers['imagemagick']
                    # writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
                    if filename.split('.')[-1] == "gif":
                        ani.save(filename,fps=5, writer="imagemagick", savefig_kwargs={'pad_inches':0.01})
                    else:
                        ani.save(filename,fps=5, savefig_kwargs={'bbox_inches':'tight','pad_inches':0.01})

                if show_plot:
                    ax.clear()
                    if colorbar and save:
                        self.cbar.ax.cla()
                        self.cbar.ax.clear()
                        del self.cbar
                    plt.show()

                return



        elif self.mesh.element_type == "hex":

            import os
            os.environ['ETS_TOOLKIT'] = 'qt4'
            from mayavi import mlab
            from matplotlib.colors import ColorConverter
            import matplotlib.cm as cm

            ndim = 3

            if figure is None:
                figure = mlab.figure(bgcolor=(1,1,1),fgcolor=(0,0,0),size=(800,600))

            if plot_on_curvilinear_mesh:

                tmesh = PostProcess.TessellateHexes(self.mesh, 
                    np.zeros_like(self.mesh.points),
                    QuantityToPlot = self.sol[:,quantity,increment],
                    plot_on_faces=False, plot_points=plot_points, 
                    plot_edges=plot_edges, plot_surfaces=True)

                nsize = tmesh.nsize
                nface = tmesh.nface

                extrapolated_sol = np.zeros((tmesh.points.shape[0], self.sol.shape[1]))
                ssol = self.sol[np.unique(tmesh.faces_to_plot),:,:]
                for ielem in range(nface):
                    extrapolated_sol[ielem*nsize:(ielem+1)*nsize,:] = np.dot(tmesh.bases_2, 
                        ssol[tmesh.smesh.elements[ielem,:],:, 0])

                trimesh_h = mlab.triangular_mesh(tmesh.points[:,0], tmesh.points[:,1], 
                    tmesh.points[:,2], tmesh.elements, scalars = extrapolated_sol[:,quantity],
                    line_width=0.5)

                if plot_edges:
                    src = mlab.pipeline.scalar_scatter(tmesh.x_edges.T.copy().flatten(), 
                        tmesh.y_edges.T.copy().flatten(), tmesh.z_edges.T.copy().flatten())
                    src.mlab_source.dataset.lines = tmesh.connections
                    lines = mlab.pipeline.stripper(src)
                    h_edges = mlab.pipeline.surface(lines, color = (0,0,0), line_width=2)

                if plot_points:
                    svpoints = self.mesh.points[np.unique(tmesh.faces_to_plot),:]
                    h_points = mlab.points3d(svpoints[:,0], svpoints[:,1], svpoints[:,2],
                        color=(0,0,0), mode='sphere', scale_factor=point_radius)

                # mlab.view(azimuth=45, elevation=50, distance=90, focalpoint=None,
                    # roll=0, reset_roll=True, figure=None)


                m_trimesh = trimesh_h.mlab_source
                if plot_edges:
                    m_wire = h_edges.mlab_source
                if plot_points:
                    m_points = h_points.mlab_source

                @mlab.animate(delay=100) 
                def animator():
                    # fig = mlab.gcf()

                    # ssol = self.sol[np.unique(tmesh.faces_to_plot),:,:]

                    for i in range(0, self.sol.shape[2]):

                        # GET SOLUTION AT THIS INCREMENT
                        extrapolated_sol = np.zeros((tmesh.points.shape[0], self.sol.shape[1]))
                        for ielem in range(nface):
                            extrapolated_sol[ielem*nsize:(ielem+1)*nsize,:] = np.dot(tmesh.bases_2, 
                                ssol[tmesh.smesh.elements[ielem,:],:, i])

                        svpoints = self.mesh.points[np.unique(tmesh.faces_to_plot),:] + ssol[:,:ndim,i]

                        if configuration == "deformed":

                            m_trimesh.reset(x=tmesh.points[:,0]+extrapolated_sol[:,0], 
                                y=tmesh.points[:,1]+extrapolated_sol[:,1],
                                z=tmesh.points[:,2]+extrapolated_sol[:,2], 
                                scalars=extrapolated_sol[:,quantity])
                            
                            # GET UPDATED EDGE COORDINATES AT THIS INCREMENT
                            if plot_edges:
                                for iedge in range(tmesh.smesh.all_edges.shape[0]):
                                    ielem = tmesh.edge_elements[iedge,0]
                                    edge = tmesh.smesh.elements[ielem,tmesh.reference_edges[tmesh.edge_elements[iedge,1],:]]
                                    coord_edge = svpoints[edge,:]
                                    tmesh.x_edges[:,iedge], tmesh.y_edges[:,iedge], tmesh.z_edges[:,iedge] = np.dot(coord_edge.T,tmesh.bases_1)

                                m_wire.reset(x=tmesh.x_edges.T.copy().flatten(), 
                                    y=tmesh.y_edges.T.copy().flatten(),
                                    z=tmesh.z_edges.T.copy().flatten())

                            if plot_points:
                                m_points.reset(x=svpoints[:,0], y=svpoints[:,1], z=svpoints[:,2])

                        else:

                            m_trimesh.reset(scalars=extrapolated_sol[:,quantity])

                        
                        if colorbar:

                            cbar = mlab.colorbar(object=trimesh_h, title=self.QuantityNamer(quantity), 
                                orientation="horizontal",label_fmt="%9.2f")

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
                        

                        figure.scene.reset_zoom()
                        # fig.scene.reset_zoom()

                        # SAVE EACH FRAME USING AN EXTERNAL TOOL
                        if save:
                            mlab.savefig(filename.split(".")[0]+"_increment_"+str(i)+".png")
                        
                        yield

                with constant_camera_view():
                    animator()

                # mlab.view(azimuth=45, elevation=50, distance=90, focalpoint=None,
                #     roll=0, reset_roll=True, figure=None)
                

                if show_plot:
                    # mlab.draw()
                    mlab.show()


                if save:
                    # mlab.close()
                    import subprocess
                    # fname = os.path.basename(filename).split(".")[0]
                    # ex = os.path.basename(filename).split(".")[-1]
                    # os.path.join(PWD(filename), os.path.basename(filename))

                    # REMOVE OLD FILE WITH THE SAME NAME
                    p = subprocess.Popen('rm -f ' + filename, shell=True)
                    p.wait()

                    fps = 25
                    p = subprocess.Popen('ffmpeg -framerate ' +str(fps)+ ' -i ' + \
                        filename.split('.')[0] + '_increment_%00d.png' + \
                        ' -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p '+ filename, 
                        shell=True)
                    p.wait()
                    # REMOVE TEMPS
                    p = subprocess.Popen('rm -rf ' + filename.split(".")[0]+"_increment_*", 
                        shell=True)
                    p.wait()
                    # ffmpeg -framerate 25 -i yy_increment_%0d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output.mp4
                    # convert -delay 10 -loop 0 plate_with_holes_frame.00*.png plate_with_holes_wireframe.gif


    def CurvilinearPlot(self,*args,**kwargs):
        """Curvilinear (or linear) plots for high order finite elements"""
        if len(args) == 0:
            if self.mesh is None:
                raise ValueError("Mesh not set for post-processing")
            else:
                mesh = self.mesh
        else:
            mesh = args[0]

        if len(args) > 1:
            TotalDisp = args[1]
        else:
            if self.sol is None:
                raise ValueError("Solution not set for post-processing")
            else:
                TotalDisp = self.sol

        if mesh.element_type == "tri":
            return self.CurvilinearPlotTri(mesh,TotalDisp,**kwargs)
        elif mesh.element_type == "quad":
            return self.CurvilinearPlotQuad(mesh,TotalDisp,**kwargs)
        elif mesh.element_type == "tet":
            return self.CurvilinearPlotTet(mesh,TotalDisp,**kwargs)
        elif mesh.element_type == "hex":
            return self.CurvilinearPlotHex(mesh,TotalDisp,**kwargs)
        else:
            raise ValueError("Unknown mesh type")


    @staticmethod
    def CurvilinearPlotTri(mesh, TotalDisp, QuantityToPlot=None,
        ProjectionFlags=None, interpolation_degree=30, EquallySpacedPoints=False,
        TriSurf=False, colorbar=False, PlotActualCurve=False, point_radius = 3, color="#C5F1C5",
        plot_points=False, plot_edges=True, save=False, filename=None, figure=None, show_plot=True, 
        save_tessellation=False):

        """High order curved triangular mesh plots, based on high order nodal FEM.
            The equally spaced FEM points do not work as good as the Fekete points 
        """


        from Florence.QuadratureRules.FeketePointsTri import FeketePointsTri
        from Florence.QuadratureRules.EquallySpacedPoints import EquallySpacedPointsTri
        from Florence.QuadratureRules.NumericIntegrator import GaussLobattoQuadrature
        from Florence.QuadratureRules.NodeArrangement import NodeArrangementTri
        from Florence.FunctionSpace import Tri 
        from Florence.FunctionSpace.OneDimensional.Line import LagrangeGaussLobatto, Lagrange
        from Florence.FunctionSpace.GetBases import GetBases

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

        C = interpolation_degree
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


        # GET EQUALLY-SPACED/GAUSS-LOBATTO POINTS FOR THE EDGES
        if EquallySpacedPoints is False:
            GaussLobattoPointsOneD = GaussLobattoQuadrature(C+2)[0].flatten()
        else:
            GaussLobattoPointsOneD = Lagrange(C,0)[-1]

        BasesTri = np.zeros((nsize_2,FeketePointsTri.shape[0]),dtype=np.float64)
        hpBases = Tri.hpNodal.hpBases
        for i in range(FeketePointsTri.shape[0]):
            BasesTri[:,i] = hpBases(CActual,FeketePointsTri[i,0],FeketePointsTri[i,1],
                EvalOpt=1,EquallySpacedPoints=EquallySpacedPoints,Transform=1)[0]

        BasesOneD = np.zeros((CActual+2,GaussLobattoPointsOneD.shape[0]),dtype=np.float64)
        for i in range(GaussLobattoPointsOneD.shape[0]):
            BasesOneD[:,i] = LagrangeGaussLobatto(CActual,GaussLobattoPointsOneD[i])[0]

        smesh = deepcopy(mesh)
        smesh.elements = mesh.elements[:,:ndim+1]
        nmax = int(np.max(smesh.elements)+1)
        smesh.points = mesh.points[:nmax,:]
        smesh.GetEdgesTri()
        edge_elements = smesh.GetElementsEdgeNumberingTri()


        # GET EDGE ORDERING IN THE REFERENCE ELEMENT
        reference_edges = NodeArrangementTri(CActual)[0]
        reference_edges = np.concatenate((reference_edges,reference_edges[:,1,None]),axis=1)
        reference_edges = np.delete(reference_edges,1,1)

        # GET EULERIAN GEOMETRY
        if TotalDisp.ndim==3:
            vpoints = mesh.points + TotalDisp[:,:ndim,-1]
        else:
            vpoints = mesh.points + TotalDisp[:,:ndim]

        # GET X & Y OF CURVED EDGES
        if plot_edges:
            x_edges = np.zeros((C+2,smesh.all_edges.shape[0]))
            y_edges = np.zeros((C+2,smesh.all_edges.shape[0]))

            for iedge in range(smesh.all_edges.shape[0]):
                ielem = edge_elements[iedge,0]
                edge = mesh.elements[ielem,reference_edges[edge_elements[iedge,1],:]]
                coord_edge = vpoints[edge,:]
                x_edges[:,iedge], y_edges[:,iedge] = np.dot(coord_edge.T,BasesOneD)


        # MAKE FIGURE
        if figure is None:
            fig = plt.figure()
        else:
            fig = figure

        h_surfaces, h_edges, h_points = None, None, None
        # ls = LightSource(azdeg=315, altdeg=45)
        if TriSurf is True:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

        if plot_edges:
            # PLOT CURVED EDGES
            h_edges = ax.plot(x_edges,y_edges,'k')
        
        mesh.nelem = int(mesh.nelem)
        nnode = int(nsize*mesh.nelem)
        nelem = int(Triangles.shape[0]*mesh.nelem)

        Xplot = np.zeros((nnode,2),dtype=np.float64)
        Tplot = np.zeros((nelem,3),dtype=np.int64)
        Uplot = np.zeros(nnode,dtype=np.float64)

        if QuantityToPlot is None:
            quantity_to_plot = np.zeros(mesh.nelem)
        else:
            quantity_to_plot = QuantityToPlot

        # FOR CURVED ELEMENTS
        for ielem in range(mesh.nelem):
            Xplot[ielem*nsize:(ielem+1)*nsize,:] = np.dot(BasesTri.T, vpoints[mesh.elements[ielem,:],:])
            Tplot[ielem*TrianglesFunc.nsimplex:(ielem+1)*TrianglesFunc.nsimplex,:] = Triangles + ielem*nsize
            Uplot[ielem*nsize:(ielem+1)*nsize] = quantity_to_plot[ielem]

        # PLOT CURVED ELEMENTS
        if TriSurf is True:
            # ax.plot_trisurf(Tplot,Xplot[:,0], Xplot[:,1], Xplot[:,1]*0)
            triang = mtri.Triangulation(Xplot[:,0], Xplot[:,1],Tplot)
            ax.plot_trisurf(triang,Xplot[:,0]*0, edgecolor="none",facecolor="#ffddbb")
            ax.view_init(90,-90)
            ax.dist = 7
        else:
            # plt.tricontourf(Xplot[:,0], Xplot[:,1], Tplot, np.ones(Xplot.shape[0]), 100,alpha=0.8)
            # plt.tricontourf(Xplot[:,0], Xplot[:,1], Tplot, Uplot, 100,alpha=0.8)
            # plt.tricontourf(Xplot[:,0], Xplot[:,1], Tplot[:4,:], np.ones(Xplot.shape[0]),alpha=0.8,origin='lower')
            if QuantityToPlot is None:
                h_surfaces = plt.tricontourf(Xplot[:,0], Xplot[:,1], Tplot, Uplot, colors="#C5F1C5")
            else:
                h_surfaces = plt.tricontourf(Xplot[:,0], Xplot[:,1], Tplot, Uplot, 100,alpha=0.8)

        # PLOT CURVED POINTS
        if plot_points:
            # plt.plot(vpoints[:,0],vpoints[:,1],'o',markersize=3,color='#F88379')
            h_points = plt.plot(vpoints[:,0],vpoints[:,1],'o',markersize=point_radius,color='k')

        if QuantityToPlot is not None:
            plt.set_cmap('viridis')
            # plt.set_cmap('viridis_r')
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

        if save:
            if filename is None:
                raise ValueError("No filename given. Supply one with extension")
            else:
                plt.savefig(filename, format="eps",dpi=300, bbox_inches='tight')


        if show_plot:
            plt.show()

        if save_tessellation:

            tmesh = Mesh()
            tmesh.element_type = "tri"
            tmesh.elements = Tplot
            tmesh.points = Xplot
            tmesh.nelem = nelem
            tmesh.nnode = nnode
            tmesh.nsize = nsize
            tmesh.bases_1 = BasesOneD
            tmesh.bases_2 = BasesTri.T

            if plot_edges:
                tmesh.x_edges = x_edges
                tmesh.y_edges = y_edges
                tmesh.edge_elements = edge_elements
                tmesh.reference_edges = reference_edges

            return h_surfaces, h_edges, h_points, tmesh


        return h_surfaces, h_edges, h_points







    @staticmethod
    def CurvilinearPlotTet(mesh, TotalDisp, QuantityToPlot=None, plot_on_faces=True,
        ProjectionFlags=None, interpolation_degree=20, EquallySpacedPoints=False, PlotActualCurve=False,
        plot_points=False, plot_edges=True, plot_surfaces=True, point_radius=0.02, colorbar=False, color=None, figure=None,
        show_plot=True, save=False, filename=None, save_tessellation=False):

        """High order curved tetrahedral surfaces mesh plots, based on high order nodal FEM.
            The equally spaced FEM points do not work as good as the Fekete points 
        """



        from Florence.QuadratureRules.FeketePointsTri import FeketePointsTri
        from Florence.QuadratureRules.EquallySpacedPoints import EquallySpacedPointsTri
        from Florence.QuadratureRules.NumericIntegrator import GaussLobattoQuadrature
        from Florence.QuadratureRules.NodeArrangement import NodeArrangementTri
        from Florence.FunctionSpace import Tri 
        from Florence.FunctionSpace.OneDimensional.Line import LagrangeGaussLobatto, Lagrange
        from Florence.FunctionSpace.GetBases import GetBases

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

        C = interpolation_degree
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
            GaussLobattoPointsOneD = GaussLobattoQuadrature(C+2)[0].flatten()
        else:
            GaussLobattoPointsOneD = Lagrange(C,0)[-1].flatten()

        BasesTri = np.zeros((nsize_2,FeketePointsTri.shape[0]),dtype=np.float64)
        hpBases = Tri.hpNodal.hpBases
        for i in range(FeketePointsTri.shape[0]):
            BasesTri[:,i] = hpBases(CActual,FeketePointsTri[i,0],FeketePointsTri[i,1],
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

        if QuantityToPlot is not None and plot_on_faces:
            quantity_to_plot = QuantityToPlot[face_elements[faces_to_plot_flag.flatten()==1,0]]

        # BUILD MESH OF SURFACE
        smesh = Mesh()
        smesh.element_type = "tri"
        # smesh.elements = np.copy(corr_faces)
        smesh.elements = np.copy(faces_to_plot)
        smesh.nelem = smesh.elements.shape[0]
        smesh.points = mesh.points[np.unique(smesh.elements),:]


        # MAP TO ORIGIN    
        unique_elements, inv = np.unique(smesh.elements,return_inverse=True)
        mapper = np.arange(unique_elements.shape[0])
        smesh.elements = mapper[inv].reshape(smesh.elements.shape)
 
        smesh.GetBoundaryEdgesTri()
        smesh.GetEdgesTri()
        edge_elements = smesh.GetElementsEdgeNumberingTri()

        
        # GET EDGE ORDERING IN THE REFERENCE ELEMENT
        reference_edges = NodeArrangementTri(CActual)[0]
        reference_edges = np.concatenate((reference_edges,reference_edges[:,1,None]),axis=1)
        reference_edges = np.delete(reference_edges,1,1)

        # GET EULERIAN GEOMETRY
        if TotalDisp.ndim == 3:
            vpoints = mesh.points + TotalDisp[:,:ndim,-1]
        elif TotalDisp.ndim == 2:
            vpoints = mesh.points + TotalDisp[:,:ndim]
        else:
            raise AssertionError("mesh points and displacment arrays are incompatible")

        # svpoints = vpoints[np.unique(mesh.faces),:]
        svpoints = vpoints[np.unique(faces_to_plot),:]
        del vpoints
        gc.collect()

        # MAKE A FIGURE
        if figure is None:
            figure = mlab.figure(bgcolor=(1,1,1),fgcolor=(1,1,1),size=(800,600))
        figure.scene.disable_render = True

        h_points, h_edges, trimesh_h = None, None, None

        if plot_edges:
            # GET X, Y & Z OF CURVED EDGES  
            x_edges = np.zeros((C+2,smesh.all_edges.shape[0]))
            y_edges = np.zeros((C+2,smesh.all_edges.shape[0]))
            z_edges = np.zeros((C+2,smesh.all_edges.shape[0]))

            for iedge in range(smesh.all_edges.shape[0]):
                ielem = edge_elements[iedge,0]
                edge = smesh.elements[ielem,reference_edges[edge_elements[iedge,1],:]]
                coord_edge = svpoints[edge,:]
                x_edges[:,iedge], y_edges[:,iedge], z_edges[:,iedge] = np.dot(coord_edge.T,BasesOneD)

            
            # PLOT CURVED EDGES
            connections_elements = np.arange(x_edges.size).reshape(x_edges.shape[1],x_edges.shape[0])
            connections = np.zeros((x_edges.size,2),dtype=np.int64)
            for i in range(connections_elements.shape[0]):
                connections[i*(x_edges.shape[0]-1):(i+1)*(x_edges.shape[0]-1),0] = connections_elements[i,:-1]
                connections[i*(x_edges.shape[0]-1):(i+1)*(x_edges.shape[0]-1),1] = connections_elements[i,1:]
            connections = connections[:(i+1)*(x_edges.shape[0]-1),:]
            
            src = mlab.pipeline.scalar_scatter(x_edges.T.copy().flatten(), y_edges.T.copy().flatten(), z_edges.T.copy().flatten())
            src.mlab_source.dataset.lines = connections
            lines = mlab.pipeline.stripper(src)
            h_edges = mlab.pipeline.surface(lines, color = (0,0,0), line_width=2)
            # h_edges = mlab.pipeline.surface(lines, color = (0,0,0), line_width=1)
            # mlab.pipeline.surface(lines, color = (0.72,0.72,0.72), line_width=2)

            # OLDER VERSION
            # for i in range(x_edges.shape[1]):
            #     mlab.plot3d(x_edges[:,i],y_edges[:,i],z_edges[:,i],color=(0,0,0),tube_radius=edge_width)
        

        # CURVED SURFACES
        if plot_surfaces:

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
                if plot_on_faces:
                    for ielem in range(nface):
                        Uplot[ielem*nsize:(ielem+1)*nsize] = quantity_to_plot[ielem]
                else:
                    # IF QUANTITY IS DEFINED ON NODES
                    quantity = QuantityToPlot[np.unique(faces_to_plot)]
                    for ielem in range(nface):
                        Uplot[ielem*nsize:(ielem+1)*nsize] = np.dot(BasesTri.T, quantity[smesh.elements[ielem,:]])


            point_line_width = .002
            # point_line_width = 0.5
            # point_line_width = .0008
            # point_line_width = 2.
            # point_line_width = .045
            # point_line_width = .015 # F6


            if color is None:
                color=(197/255.,241/255.,197/255.) # green
                # color=( 0, 150/255., 187/255.)   # blue

            # PLOT SURFACES (CURVED ELEMENTS)
            if QuantityToPlot is None:
                trimesh_h = mlab.triangular_mesh(Xplot[:,0], Xplot[:,1], Xplot[:,2], Tplot,
                    line_width=point_line_width,color=color)

            else:
                trimesh_h = mlab.triangular_mesh(Xplot[:,0], Xplot[:,1], Xplot[:,2], Tplot, scalars = Uplot,
                    line_width=point_line_width,colormap='summer')
                


            # PLOT POINTS ON CURVED MESH
            if plot_points:
                # mlab.points3d(svpoints[:,0],svpoints[:,1],svpoints[:,2],color=(0,0,0),mode='sphere',scale_factor=2.5*point_line_width)
                h_points = mlab.points3d(svpoints[:,0],svpoints[:,1],svpoints[:,2],color=(0,0,0),mode='sphere',scale_factor=point_radius)

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


        if colorbar and plot_surfaces:
            cbar = mlab.colorbar(object=trimesh_h, orientation="horizontal",label_fmt="%9.2f")


        # CONTROL CAMERA VIEW
        # mlab.view(azimuth=45, elevation=50, distance=80, focalpoint=None,
        #         roll=0, reset_roll=True, figure=None)

        mlab.view(azimuth=45, elevation=50, distance=80, focalpoint=None,
            roll=0, reset_roll=True, figure=None)

        # SAVEFIG
        if save:
            if filename is None:
                raise ValueError("No filename given. Supply one with extension")
            else:
                mlab.savefig(filename,magnification="auto")
    
        if show_plot is True:
            # FORCE UPDATE MLAB TO UPDATE COLORMAP
            mlab.draw()
            mlab.show()


        if save_tessellation:

            # THIS IS NOT A FLORENCE MESH COMPLIANT MESH
            tmesh = Mesh()
            tmesh.element_type = "tri"
            tmesh.elements = Tplot
            tmesh.points = Xplot
            tmesh.nelem = nelem
            tmesh.nnode = nnode
            tmesh.nsize = nsize
            tmesh.bases_1 = BasesOneD
            tmesh.bases_2 = BasesTri.T

            tmesh.nface = nface
            tmesh.smesh = smesh
            tmesh.faces_to_plot = faces_to_plot

            if plot_edges:
                tmesh.x_edges = x_edges
                tmesh.y_edges = y_edges
                tmesh.z_edges = z_edges
                tmesh.connections = connections
                tmesh.edge_elements = edge_elements
                tmesh.reference_edges = reference_edges

            mlab.close()

            return trimesh_h, h_edges, h_points, tmesh

        return trimesh_h, h_edges, h_points



    @staticmethod
    def CurvilinearPlotQuad(mesh, TotalDisp, QuantityToPlot=None,
        ProjectionFlags=None, interpolation_degree=30, EquallySpacedPoints=False,
        TriSurf=False, colorbar=False, PlotActualCurve=False, point_radius = 3, color="#C5F1C5",
        plot_points=False, plot_edges=True, save=False, filename=None, figure=None, show_plot=True, 
        save_tessellation=False, plot_on_faces=True):

        """High order curved quad mesh plots, based on high order nodal FEM.
        """

        from copy import deepcopy
        from scipy.spatial import Delaunay
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
        import matplotlib.cm as cm
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        tmesh = PostProcess.TessellateQuads(mesh, TotalDisp, QuantityToPlot=QuantityToPlot,
            ProjectionFlags=ProjectionFlags, interpolation_degree=interpolation_degree, 
            EquallySpacedPoints=EquallySpacedPoints, plot_points=plot_points, 
            plot_edges=plot_edges, plot_on_faces=plot_on_faces)

        # UNPACK
        x_edges = tmesh.x_edges
        y_edges = tmesh.y_edges
        nnode = tmesh.nnode
        nelem = tmesh.nelem
        nsize = tmesh.nsize

        Xplot = tmesh.points
        Tplot = tmesh.elements
        Uplot = tmesh.quantity
        vpoints = tmesh.vpoints
        BasesQuad = tmesh.bases_2.T

        # MAKE FIGURE
        if figure is None:
            fig = plt.figure()
        else:
            fig = figure

        h_surfaces, h_edges, h_points = None, None, None
        # ls = LightSource(azdeg=315, altdeg=45)
        if TriSurf is True:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

        if plot_edges:
            # PLOT CURVED EDGES
            h_edges = ax.plot(x_edges,y_edges,'k')


        # PLOT CURVED ELEMENTS
        if QuantityToPlot is None:
            h_surfaces = plt.tricontourf(Xplot[:,0], Xplot[:,1], Tplot, Uplot, colors="#C5F1C5")
        else:
            h_surfaces = plt.tricontourf(Xplot[:,0], Xplot[:,1], Tplot, Uplot, 100,alpha=0.8)

        # PLOT CURVED POINTS
        if plot_points:
            h_points = plt.plot(vpoints[:,0],vpoints[:,1],'o',markersize=point_radius,color='k')

        if QuantityToPlot is not None:
            plt.set_cmap('viridis')
            # plt.set_cmap('viridis_r')
            if plot_on_faces:
                plt.clim(0,1)
        

        if colorbar is True:
            if plot_on_faces:
                ax_cbar = mpl.colorbar.make_axes(plt.gca(), shrink=1.0)[0]
                cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=cm.viridis,
                                   norm=mpl.colors.Normalize(vmin=-0, vmax=1))
                cbar.set_clim(0, 1)
                divider = make_axes_locatable(ax_cbar)
                cax = divider.append_axes("right", size="1%", pad=0.005)
            else:
                plt.colorbar(shrink=0.5,orientation="vertical")
        
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

        if save:
            if filename is None:
                warn("file name not provided. I am going to write one in the current directory")
                filename = PWD(__file__) + "/output.eps"
            else:
                plt.savefig(filename, format="eps",dpi=300, bbox_inches='tight')


        if show_plot:
            plt.show()


        return h_surfaces, h_edges, h_points




    @staticmethod
    def CurvilinearPlotHex(mesh, TotalDisp, QuantityToPlot=None, plot_on_faces=True,
        ProjectionFlags=None, interpolation_degree=20, EquallySpacedPoints=False, PlotActualCurve=False,
        plot_points=False, plot_edges=True, plot_surfaces=True, point_radius=0.02, colorbar=False, color=None, figure=None,
        show_plot=True, save=False, filename=None, save_tessellation=False):

        """High order curved hexahedral surfaces mesh plots, based on high order nodal FEM.
            The equally spaced FEM points do not work as good as the Fekete points 
        """

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

        tmesh = PostProcess.TessellateHexes(mesh, TotalDisp, QuantityToPlot=QuantityToPlot, 
            plot_on_faces=plot_on_faces, ProjectionFlags=ProjectionFlags, 
            interpolation_degree=interpolation_degree, 
            EquallySpacedPoints=EquallySpacedPoints,
            plot_points=plot_points, plot_edges=plot_edges, plot_surfaces=plot_surfaces)

        # SINCE THIS IS A 3D PLOT
        ndim=3

        C = interpolation_degree
        p = C+1
        nsize = int((C+2)**2)

        faces_to_plot = tmesh.faces_to_plot        
        svpoints = tmesh.svpoints

        if plot_surfaces:
            Uplot = tmesh.quantity
            Xplot = tmesh.points
            Tplot = tmesh.elements
        # bases_1 = tmesh.bases_1
        # bases_2 = tmesh.bases_2

        # MAKE A FIGURE
        if figure is None:
            figure = mlab.figure(bgcolor=(1,1,1),fgcolor=(1,1,1),size=(800,600))
        figure.scene.disable_render = True

        h_points, h_edges, trimesh_h = None, None, None

        if plot_edges:
            x_edges = tmesh.x_edges
            y_edges = tmesh.y_edges
            z_edges = tmesh.z_edges
            connections = tmesh.connections
            
            src = mlab.pipeline.scalar_scatter(x_edges.T.copy().flatten(), y_edges.T.copy().flatten(), z_edges.T.copy().flatten())
            src.mlab_source.dataset.lines = connections
            lines = mlab.pipeline.stripper(src)
            h_edges = mlab.pipeline.surface(lines, color = (0,0,0), line_width=2)


        # CURVED SURFACES
        if plot_surfaces:

            point_line_width = .002

            if color is None:
                color=(197/255.,241/255.,197/255.)

            # PLOT SURFACES (CURVED ELEMENTS)
            if QuantityToPlot is None:
                trimesh_h = mlab.triangular_mesh(Xplot[:,0], Xplot[:,1], Xplot[:,2], Tplot,
                    line_width=point_line_width,color=color)

            else:
                trimesh_h = mlab.triangular_mesh(Xplot[:,0], Xplot[:,1], Xplot[:,2], Tplot, scalars = Uplot,
                    line_width=point_line_width,colormap='summer')


        # PLOT POINTS ON CURVED MESH
        if plot_points:
            h_points = mlab.points3d(svpoints[:,0],svpoints[:,1],svpoints[:,2],color=(0,0,0),mode='sphere',scale_factor=point_radius)

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


        if colorbar and plot_surfaces:
            cbar = mlab.colorbar(object=trimesh_h, orientation="horizontal",label_fmt="%9.2f")


        # CONTROL CAMERA VIEW
        # mlab.view(azimuth=45, elevation=50, distance=80, focalpoint=None,
            # roll=0, reset_roll=True, figure=None)

        mlab.view(azimuth=15, elevation=17, distance=80, focalpoint=None,
            roll=20, reset_roll=True, figure=None)
        # cam,foc = mlab.move()
        # print cam
        # print foc
        mlab.move(forward=-16, right=-20, up=-20)


        # SAVEFIG
        if save:
            if filename is None:
                warn("No filename given. I am going to write one in the current directory")
                filename = PWD(__file__) + "/output.png"
            else:
                mlab.savefig(filename,magnification="auto")
    
        if show_plot is True:
            # FORCE UPDATE MLAB TO UPDATE COLORMAP
            mlab.draw()
            mlab.show()

        # mlab.close()


        if save_tessellation:
            return trimesh_h, h_edges, h_points, tmesh

        return trimesh_h, h_edges, h_points





    #-----------------------------------------------------------------------------#
    @staticmethod
    def TessellateTris(mesh, TotalDisp, QuantityToPlot=None,
        ProjectionFlags=None, interpolation_degree=30, EquallySpacedPoints=False,
        plot_points=False, plot_edges=True):

        """High order curved triangular mesh plots, based on high order nodal FEM.
            The equally spaced FEM points do not work as good as the Fekete points 
        """


        from Florence.QuadratureRules.FeketePointsTri import FeketePointsTri
        from Florence.QuadratureRules.EquallySpacedPoints import EquallySpacedPointsTri
        from Florence.QuadratureRules.NumericIntegrator import GaussLobattoQuadrature
        from Florence.QuadratureRules.NodeArrangement import NodeArrangementTri
        from Florence.FunctionSpace import Tri 
        from Florence.FunctionSpace.OneDimensional.Line import LagrangeGaussLobatto, Lagrange
        from Florence.FunctionSpace.GetBases import GetBases

        from copy import deepcopy
        from scipy.spatial import Delaunay
        # SINCE THIS IS A 2D PLOT
        ndim = 2

        C = interpolation_degree
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


        # GET EQUALLY-SPACED/GAUSS-LOBATTO POINTS FOR THE EDGES
        if EquallySpacedPoints is False:
            GaussLobattoPointsOneD = GaussLobattoQuadrature(C+2)[0].flatten()
        else:
            GaussLobattoPointsOneD = Lagrange(C,0)[-1].flatten()

        BasesTri = np.zeros((nsize_2,FeketePointsTri.shape[0]),dtype=np.float64)
        hpBases = Tri.hpNodal.hpBases
        for i in range(FeketePointsTri.shape[0]):
            BasesTri[:,i] = hpBases(CActual,FeketePointsTri[i,0],FeketePointsTri[i,1],
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
            vpoints = mesh.points + TotalDisp[:,:ndim,-1]
        else:
            vpoints = mesh.points + TotalDisp[:,:ndim]

        # GET X & Y OF CURVED EDGES
        if plot_edges:
            x_edges = np.zeros((C+2,smesh.all_edges.shape[0]))
            y_edges = np.zeros((C+2,smesh.all_edges.shape[0]))

            for iedge in range(smesh.all_edges.shape[0]):
                ielem = edge_elements[iedge,0]
                edge = mesh.elements[ielem,reference_edges[edge_elements[iedge,1],:]]
                coord_edge = vpoints[edge,:]
                x_edges[:,iedge], y_edges[:,iedge] = np.dot(coord_edge.T,BasesOneD)


        nnode = nsize*mesh.nelem
        nelem = Triangles.shape[0]*mesh.nelem

        Xplot = np.zeros((nnode,2),dtype=np.float64)
        Tplot = np.zeros((nelem,3),dtype=np.int64)
        Uplot = np.zeros(nnode,dtype=np.float64)

        if QuantityToPlot is None:
            quantity_to_plot = np.zeros(mesh.nelem)
        else:
            quantity_to_plot = QuantityToPlot

        # FOR CURVED ELEMENTS
        for ielem in range(mesh.nelem):
            Xplot[ielem*nsize:(ielem+1)*nsize,:] = np.dot(BasesTri.T, vpoints[mesh.elements[ielem,:],:])
            Tplot[ielem*TrianglesFunc.nsimplex:(ielem+1)*TrianglesFunc.nsimplex,:] = Triangles + ielem*nsize
            Uplot[ielem*nsize:(ielem+1)*nsize] = quantity_to_plot[ielem]



        tmesh = Mesh()
        tmesh.element_type = "tri"
        tmesh.elements = Tplot
        tmesh.points = Xplot
        tmesh.nelem = nelem
        tmesh.nnode = nnode
        tmesh.nsize = nsize
        tmesh.bases_1 = BasesOneD
        tmesh.bases_2 = BasesTri.T

        if plot_edges:
            tmesh.x_edges = x_edges
            tmesh.y_edges = y_edges
            tmesh.edge_elements = edge_elements
            tmesh.reference_edges = reference_edges


        return tmesh







    @staticmethod
    def TessellateQuads(mesh, TotalDisp, QuantityToPlot=None,
        ProjectionFlags=None, interpolation_degree=30, EquallySpacedPoints=False,
        plot_points=False, plot_edges=True, plot_on_faces=True):

        """High order curved quadrilaterial tessellation, based on high order nodal FEM.
        """


        from Florence.QuadratureRules.GaussLobattoPoints import GaussLobattoPointsQuad
        from Florence.QuadratureRules.NumericIntegrator import GaussLobattoQuadrature
        from Florence.QuadratureRules.NodeArrangement import NodeArrangementQuad
        from Florence.FunctionSpace import Quad
        from Florence.FunctionSpace.OneDimensional.Line import LagrangeGaussLobatto, Lagrange

        from copy import deepcopy
        from scipy.spatial import Delaunay
        
        # SINCE THIS IS A 2D PLOT
        ndim = 2

        C = interpolation_degree
        p = C+1
        nsize = int((C+2)**ndim)
        CActual = mesh.InferPolynomialDegree() - 1 
        nsize_2 = int((CActual+2)**ndim)

        GaussLobattoPoints = GaussLobattoPointsQuad(C)

        # BUILD DELAUNAY TRIANGULATION OF REFERENCE ELEMENTS
        TrianglesFunc = Delaunay(GaussLobattoPoints)
        Triangles = TrianglesFunc.simplices.copy()

        # GET EQUALLY-SPACED/GAUSS-LOBATTO POINTS FOR THE EDGES
        GaussLobattoPointsOneD = GaussLobattoQuadrature(C+2)[0].flatten()

        BasesQuad = np.zeros((nsize_2,GaussLobattoPoints.shape[0]),dtype=np.float64)
        hpBases = Quad.LagrangeGaussLobatto
        for i in range(GaussLobattoPoints.shape[0]):
            BasesQuad[:,i] = hpBases(CActual,GaussLobattoPoints[i,0],GaussLobattoPoints[i,1])[:,0]

        BasesOneD = np.zeros((CActual+2,GaussLobattoPointsOneD.shape[0]),dtype=np.float64)
        for i in range(GaussLobattoPointsOneD.shape[0]):
            BasesOneD[:,i] = LagrangeGaussLobatto(CActual,GaussLobattoPointsOneD[i])[0]

        smesh = deepcopy(mesh)
        smesh.elements = mesh.elements[:,:4]
        nmax = np.max(smesh.elements)+1
        smesh.points = mesh.points[:nmax,:]
        smesh.GetEdgesQuad()
        edge_elements = smesh.GetElementsEdgeNumberingQuad()


        # GET EDGE ORDERING IN THE REFERENCE ELEMENT
        reference_edges = NodeArrangementQuad(CActual)[0]
        reference_edges = np.concatenate((reference_edges,reference_edges[:,1,None]),axis=1)
        reference_edges = np.delete(reference_edges,1,1)

        # GET EULERIAN GEOMETRY
        if TotalDisp.ndim==3:
            vpoints = mesh.points + TotalDisp[:,:ndim,-1]
        else:
            vpoints = mesh.points + TotalDisp[:,:ndim]


        # GET X & Y OF CURVED EDGES
        if plot_edges:
            x_edges = np.zeros((C+2,smesh.all_edges.shape[0]))
            y_edges = np.zeros((C+2,smesh.all_edges.shape[0]))

            for iedge in range(smesh.all_edges.shape[0]):
                ielem = edge_elements[iedge,0]
                edge = mesh.elements[ielem,reference_edges[edge_elements[iedge,1],:]]
                coord_edge = vpoints[edge,:]
                x_edges[:,iedge], y_edges[:,iedge] = np.dot(coord_edge.T,BasesOneD)


        nnode = nsize*mesh.nelem
        nelem = Triangles.shape[0]*mesh.nelem

        Xplot = np.zeros((nnode,2),dtype=np.float64)
        Tplot = np.zeros((nelem,3),dtype=np.int64)
        Uplot = np.zeros(nnode,dtype=np.float64)

        if QuantityToPlot is None:
            if plot_on_faces:
                quantity_to_plot = np.zeros(mesh.nelem)
            else:
                quantity_to_plot = np.zeros(mesh.points.shape[0])
        else:
            quantity_to_plot = QuantityToPlot

        # FOR CURVED ELEMENTS
        for ielem in range(mesh.nelem):
            Xplot[ielem*nsize:(ielem+1)*nsize,:] = np.dot(BasesQuad.T, vpoints[mesh.elements[ielem,:],:])
            Tplot[ielem*TrianglesFunc.nsimplex:(ielem+1)*TrianglesFunc.nsimplex,:] = Triangles + ielem*nsize
            # Uplot[ielem*nsize:(ielem+1)*nsize] = quantity_to_plot[ielem]
            if plot_on_faces:
                Uplot[ielem*nsize:(ielem+1)*nsize] = quantity_to_plot[ielem]
            else:
                # IF QUANTITY IS DEFINED ON NODES
                Uplot[ielem*nsize:(ielem+1)*nsize] = np.dot(BasesQuad.T, quantity_to_plot[mesh.elements[ielem,:]]).flatten()

        # SAVE TESSELLATION
        tmesh = Mesh()
        tmesh.element_type = "tri"
        tmesh.elements = Tplot
        tmesh.points = Xplot
        tmesh.quantity = Uplot
        tmesh.vpoints = vpoints
        tmesh.nelem = nelem
        tmesh.nnode = nnode
        tmesh.nsize = nsize
        tmesh.bases_1 = BasesOneD
        tmesh.bases_2 = BasesQuad.T

        if plot_edges:
            tmesh.x_edges = x_edges
            tmesh.y_edges = y_edges
            tmesh.edge_elements = edge_elements
            tmesh.reference_edges = reference_edges

            return tmesh




    @staticmethod
    def TessellateTets(mesh, TotalDisp, QuantityToPlot=None, plot_on_faces=True,
        ProjectionFlags=None, interpolation_degree=20, EquallySpacedPoints=False,
        plot_points=False, plot_edges=True, plot_surfaces=True):

        """High order curved tetrahedral surfaces tessellation, based on high order nodal FEM.
        """

        from Florence.QuadratureRules.FeketePointsTri import FeketePointsTri
        from Florence.QuadratureRules.EquallySpacedPoints import EquallySpacedPointsTri
        from Florence.QuadratureRules.NumericIntegrator import GaussLobattoQuadrature
        from Florence.QuadratureRules.NodeArrangement import NodeArrangementTri
        from Florence.FunctionSpace import Tri 
        from Florence.FunctionSpace.OneDimensional.Line import LagrangeGaussLobatto, Lagrange

        from scipy.spatial import Delaunay

        # SINCE THIS IS A 3D PLOT
        ndim=3

        C = interpolation_degree
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


        # GET EQUALLY-SPACED/GAUSS-LOBATTO POINTS FOR THE EDGES
        if EquallySpacedPoints is False:
            GaussLobattoPointsOneD = GaussLobattoQuadrature(C+2)[0].flatten()
        else:
            GaussLobattoPointsOneD = Lagrange(C,0)[-1].flatten()

        BasesTri = np.zeros((nsize_2,FeketePointsTri.shape[0]),dtype=np.float64)
        hpBases = Tri.hpNodal.hpBases
        for i in range(FeketePointsTri.shape[0]):
            BasesTri[:,i] = hpBases(CActual,FeketePointsTri[i,0],FeketePointsTri[i,1],
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

        if QuantityToPlot is not None and plot_on_faces:
            quantity_to_plot = QuantityToPlot[face_elements[faces_to_plot_flag.flatten()==1,0]]

        # BUILD MESH OF SURFACE
        smesh = Mesh()
        smesh.element_type = "tri"
        # smesh.elements = np.copy(corr_faces)
        smesh.elements = np.copy(faces_to_plot)
        smesh.nelem = smesh.elements.shape[0]
        smesh.points = mesh.points[np.unique(smesh.elements),:]


        # MAP TO ORIGIN    
        unique_elements, inv = np.unique(smesh.elements,return_inverse=True)
        mapper = np.arange(unique_elements.shape[0])
        smesh.elements = mapper[inv].reshape(smesh.elements.shape)
 
        smesh.GetBoundaryEdgesTri()
        smesh.GetEdgesTri()
        edge_elements = smesh.GetElementsEdgeNumberingTri()

        
        # GET EDGE ORDERING IN THE REFERENCE ELEMENT
        reference_edges = NodeArrangementTri(CActual)[0]
        reference_edges = np.concatenate((reference_edges,reference_edges[:,1,None]),axis=1)
        reference_edges = np.delete(reference_edges,1,1)

        # GET EULERIAN GEOMETRY
        if TotalDisp.ndim == 3:
            vpoints = mesh.points + TotalDisp[:,:ndim,-1]
        elif TotalDisp.ndim == 2:
            vpoints = mesh.points + TotalDisp[:,:ndim]
        else:
            raise AssertionError("mesh points and displacment arrays are incompatible")

        # svpoints = vpoints[np.unique(mesh.faces),:]
        svpoints = vpoints[np.unique(faces_to_plot),:]
        del vpoints
        gc.collect()

        if plot_edges:
            # GET X, Y & Z OF CURVED EDGES  
            x_edges = np.zeros((C+2,smesh.all_edges.shape[0]))
            y_edges = np.zeros((C+2,smesh.all_edges.shape[0]))
            z_edges = np.zeros((C+2,smesh.all_edges.shape[0]))

            for iedge in range(smesh.all_edges.shape[0]):
                ielem = edge_elements[iedge,0]
                edge = smesh.elements[ielem,reference_edges[edge_elements[iedge,1],:]]
                coord_edge = svpoints[edge,:]
                x_edges[:,iedge], y_edges[:,iedge], z_edges[:,iedge] = np.dot(coord_edge.T,BasesOneD)

            
            # PLOT CURVED EDGES
            connections_elements = np.arange(x_edges.size).reshape(x_edges.shape[1],x_edges.shape[0])
            connections = np.zeros((x_edges.size,2),dtype=np.int64)
            for i in range(connections_elements.shape[0]):
                connections[i*(x_edges.shape[0]-1):(i+1)*(x_edges.shape[0]-1),0] = connections_elements[i,:-1]
                connections[i*(x_edges.shape[0]-1):(i+1)*(x_edges.shape[0]-1),1] = connections_elements[i,1:]
            connections = connections[:(i+1)*(x_edges.shape[0]-1),:]

        # CURVED SURFACES
        if plot_surfaces:

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
                if plot_on_faces:
                    for ielem in range(nface):
                        Uplot[ielem*nsize:(ielem+1)*nsize] = quantity_to_plot[ielem]
                else:
                    # IF QUANTITY IS DEFINED ON NODES
                    quantity = QuantityToPlot[np.unique(faces_to_plot)]
                    for ielem in range(nface):
                        Uplot[ielem*nsize:(ielem+1)*nsize] = np.dot(BasesTri.T, quantity[smesh.elements[ielem,:]])


        # THIS IS NOT A FLORENCE MESH COMPLIANT MESH
        tmesh = Mesh()
        tmesh.element_type = "tri"
        tmesh.elements = Tplot
        tmesh.points = Xplot
        tmesh.nelem = nelem
        tmesh.nnode = nnode
        tmesh.nsize = nsize
        tmesh.bases_1 = BasesOneD
        tmesh.bases_2 = BasesTri.T

        tmesh.nface = nface
        tmesh.smesh = smesh
        tmesh.faces_to_plot = faces_to_plot

        if plot_edges:
            tmesh.x_edges = x_edges
            tmesh.y_edges = y_edges
            tmesh.z_edges = z_edges
            tmesh.connections = connections
            tmesh.edge_elements = edge_elements
            tmesh.reference_edges = reference_edges



        return tmesh



    @staticmethod
    def TessellateHexes(mesh, TotalDisp, QuantityToPlot=None, plot_on_faces=True,
        ProjectionFlags=None, interpolation_degree=20, EquallySpacedPoints=False,
        plot_points=False, plot_edges=True, plot_surfaces=True):

        """High order curved hexahedral surfaces mesh plots, based on high order nodal FEM.
            The equally spaced FEM points do not work as good as the Fekete points 
        """



        from Florence.QuadratureRules import GaussLobattoPointsQuad
        from Florence.QuadratureRules.NumericIntegrator import GaussLobattoQuadrature
        from Florence.QuadratureRules.NodeArrangement import NodeArrangementQuad
        from Florence.FunctionSpace import Quad
        from Florence.FunctionSpace.OneDimensional.Line import LagrangeGaussLobatto, Lagrange

        from copy import deepcopy
        from scipy.spatial import Delaunay

        # SINCE THIS IS A 3D PLOT
        ndim=3

        C = interpolation_degree
        p = C+1
        nsize = int((C+2)**2)
        CActual = mesh.InferPolynomialDegree() - 1
        nsize_2 = int((CActual+2)**2)

        GaussLobattoPoints = GaussLobattoPointsQuad(C)

        # BUILD DELAUNAY TRIANGULATION OF REFERENCE ELEMENTS
        TrianglesFunc = Delaunay(GaussLobattoPoints)
        Triangles = TrianglesFunc.simplices.copy()

        # GET EQUALLY-SPACED/GAUSS-LOBATTO POINTS FOR THE EDGES
        GaussLobattoPointsOneD = GaussLobattoQuadrature(C+2)[0].flatten()

        BasesQuad = np.zeros((nsize_2,GaussLobattoPoints.shape[0]),dtype=np.float64)
        hpBases = Quad.LagrangeGaussLobatto
        for i in range(GaussLobattoPoints.shape[0]):
            BasesQuad[:,i] = hpBases(CActual,GaussLobattoPoints[i,0],GaussLobattoPoints[i,1])[:,0]

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
            face_elements = mesh.GetElementsFaceNumberingHex()

        elif faces_to_plot_flag.shape[0] == mesh.faces.shape[0]:
            # ONLY BOUNDARY FACES
            corr_faces = mesh.faces
            # FOR MAPPING DATA E.G. SCALED JACOBIAN FROM ELEMENTS TO FACES
            face_elements = mesh.GetElementsWithBoundaryFacesHex()
        else:
            # raise ValueError("I do not understand what you want to plot")
            corr_faces = mesh.all_faces
            face_elements = mesh.GetElementsFaceNumberingHex()

        faces_to_plot = corr_faces[faces_to_plot_flag.flatten()==1,:]

        if QuantityToPlot is not None and plot_on_faces:
            quantity_to_plot = QuantityToPlot[face_elements[faces_to_plot_flag.flatten()==1,0]]

        # BUILD MESH OF SURFACE
        smesh = Mesh()
        smesh.element_type = "quad"
        # smesh.elements = np.copy(corr_faces)
        smesh.elements = np.copy(faces_to_plot)
        smesh.nelem = smesh.elements.shape[0]
        smesh.points = mesh.points[np.unique(smesh.elements),:]


        # MAP TO ORIGIN    
        unique_elements, inv = np.unique(smesh.elements,return_inverse=True)
        mapper = np.arange(unique_elements.shape[0])
        smesh.elements = mapper[inv].reshape(smesh.elements.shape)
 
        smesh.GetBoundaryEdgesQuad()
        smesh.GetEdgesQuad()
        edge_elements = smesh.GetElementsEdgeNumberingQuad()


        
        # GET EDGE ORDERING IN THE REFERENCE ELEMENT
        reference_edges = NodeArrangementQuad(CActual)[0]
        reference_edges = np.concatenate((reference_edges,reference_edges[:,1,None]),axis=1)
        reference_edges = np.delete(reference_edges,1,1)

        # GET EULERIAN GEOMETRY
        if TotalDisp.ndim == 3:
            vpoints = mesh.points + TotalDisp[:,:ndim,-1]
        elif TotalDisp.ndim == 2:
            vpoints = mesh.points + TotalDisp[:,:ndim]
        else:
            raise AssertionError("mesh points and displacment arrays are incompatible")

        # svpoints = vpoints[np.unique(mesh.faces),:]
        svpoints = vpoints[np.unique(faces_to_plot),:]
        del vpoints
        gc.collect()

        if plot_edges:
            # GET X, Y & Z OF CURVED EDGES  
            x_edges = np.zeros((C+2,smesh.all_edges.shape[0]))
            y_edges = np.zeros((C+2,smesh.all_edges.shape[0]))
            z_edges = np.zeros((C+2,smesh.all_edges.shape[0]))

            for iedge in range(smesh.all_edges.shape[0]):
                ielem = edge_elements[iedge,0]
                edge = smesh.elements[ielem,reference_edges[edge_elements[iedge,1],:]]
                coord_edge = svpoints[edge,:]
                x_edges[:,iedge], y_edges[:,iedge], z_edges[:,iedge] = np.dot(coord_edge.T,BasesOneD)

            
            # PLOT CURVED EDGES
            connections_elements = np.arange(x_edges.size).reshape(x_edges.shape[1],x_edges.shape[0])
            connections = np.zeros((x_edges.size,2),dtype=np.int64)
            for i in range(connections_elements.shape[0]):
                connections[i*(x_edges.shape[0]-1):(i+1)*(x_edges.shape[0]-1),0] = connections_elements[i,:-1]
                connections[i*(x_edges.shape[0]-1):(i+1)*(x_edges.shape[0]-1),1] = connections_elements[i,1:]
            connections = connections[:(i+1)*(x_edges.shape[0]-1),:]

        # CURVED SURFACES
        if plot_surfaces:

            nface = smesh.elements.shape[0]
            nnode = nsize*nface
            nelem = Triangles.shape[0]*nface

            Xplot = np.zeros((nnode,3),dtype=np.float64)
            Tplot = np.zeros((nelem,3),dtype=np.int64)
            Uplot = np.zeros(nnode,dtype=np.float64)

            # FOR CURVED ELEMENTS
            for ielem in range(nface):
                Xplot[ielem*nsize:(ielem+1)*nsize,:] = np.dot(BasesQuad.T, svpoints[smesh.elements[ielem,:],:])
                Tplot[ielem*TrianglesFunc.nsimplex:(ielem+1)*TrianglesFunc.nsimplex,:] = Triangles + ielem*nsize

            if QuantityToPlot is not None:
                if plot_on_faces:
                    for ielem in range(nface):
                        Uplot[ielem*nsize:(ielem+1)*nsize] = quantity_to_plot[ielem]
                else:
                    # IF QUANTITY IS DEFINED ON NODES
                    quantity = QuantityToPlot[np.unique(faces_to_plot)]
                    for ielem in range(nface):
                        Uplot[ielem*nsize:(ielem+1)*nsize] = np.dot(BasesQuad.T, quantity[smesh.elements[ielem,:]])



        # THIS IS NOT A FLORENCE MESH COMPLIANT MESH
        tmesh = Mesh()
        tmesh.element_type = "tri"
        if plot_surfaces:
            tmesh.elements = Tplot
            tmesh.points = Xplot
            tmesh.quantity = Uplot
            tmesh.nelem = nelem
            tmesh.nnode = nnode
            tmesh.nface = nface
        tmesh.nsize = nsize
        tmesh.bases_1 = BasesOneD
        tmesh.bases_2 = BasesQuad.T

        tmesh.smesh = smesh
        tmesh.faces_to_plot = faces_to_plot
        tmesh.svpoints = svpoints

        if plot_edges:
            tmesh.x_edges = x_edges
            tmesh.y_edges = y_edges
            tmesh.z_edges = z_edges
            tmesh.connections = connections
            tmesh.edge_elements = edge_elements
            tmesh.reference_edges = reference_edges

        return tmesh






















































    ###################################################################################
    ###################################################################################

    @staticmethod   
    def HighOrderPatchPlot(mesh,TotalDisp):

        import matplotlib.pyplot as plt
        
        fig = plt.figure()
        ax = fig.axes

        C = mesh.InferPolynomialDegree()

        # TotalDisp = np.zeros_like(TotalDisp)
        # MainData.ScaledJacobian = np.ones_like(MainData.ScaledJacobian)

        # print TotalDisp[:,0,-1]
        # MainData.ScaledJacobian = np.zeros_like(MainData.ScaledJacobian)+1
        vpoints = np.copy(mesh.points)
        # print TotalDisp[:,:MainData.ndim,-1]
        vpoints += TotalDisp[:,:self.ndim,-1]

        dum1=[]; dum2=[]; dum3 = []; ddum=np.array([0,1,2,0])
        for i in range(0,MainData.C):
            dum1=np.append(dum1,i+3)
            dum2 = np.append(dum2, 2*C+3 +i*C -i*(i-1)/2 )
            dum3 = np.append(dum3,C+3 +i*(C+1) -i*(i-1)/2 )

        if C>0:
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