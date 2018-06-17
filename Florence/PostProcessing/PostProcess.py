from __future__ import print_function
import os, sys, gc
from time import time
from copy import deepcopy
import numpy as np
import numpy.linalg as la
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
from Florence.FunctionSpace import Quad, QuadES
from Florence.FunctionSpace import Hex, HexES

from Florence.FiniteElements.LocalAssembly.KinematicMeasures import *
from Florence.FiniteElements.LocalAssembly._KinematicMeasures_ import _KinematicMeasures_
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

    def SetSolutionVectors(self,sol):
        self.sol = sol

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

    def GetSolutionVectors(self):
        if self.sol is None:
            warn("Solution is not yet computed")
        return self.sol

    def GetSolution(self):
        # FOR COMAPTABILITY WITH SetSolution
        return self.GetSolutionVectors()


    def TotalComponentSol(self, sol, ColumnsIn, ColumnsOut, AppliedDirichletInc, Iter, fsize):

        nvar = self.nvar
        ndim = self.ndim

        # GET TOTAL SOLUTION
        TotalSol = np.zeros((fsize,1))
        TotalSol[ColumnsIn,0] = sol
        TotalSol[ColumnsOut,0] = AppliedDirichletInc

        # RE-ORDER SOLUTION COMPONENTS
        dU = TotalSol.reshape(int(TotalSol.shape[0]/nvar),nvar)

        return dU


    def StressRecovery(self, steps=None, average_derived_quantities=True):

        """
            steps:          [list,np.1darray] for which time steps/increments the data should
                            be recovered
        """

        if self.mesh is None:
            raise ValueError("Mesh not set for post-processing")
        if self.sol is None:
            raise ValueError("Solution not set for post-processing")
        if self.formulation is None:
            raise ValueError("formulation not set for post-processing")
        if self.material is None:
            raise ValueError("material not set for post-processing")
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
        # quadrature = QuadratureRule(qtype="gauss", norder=norder, mesh_type=mesh.element_type, optimal=3)
        # Domain = FunctionSpace(mesh, quadrature, p=C+1)
        Domain = FunctionSpace(mesh, p=C+1, evaluate_at_nodes=True)

        fem_solver = self.fem_solver
        formulation = self.formulation
        material = self.material

        det = np.linalg.det
        inv = np.linalg.inv
        Jm = Domain.Jm
        AllGauss = Domain.AllGauss


        elements = mesh.elements
        points = mesh.points
        nelem = elements.shape[0]; npoint = points.shape[0]
        nodeperelem = elements.shape[1]
        # requires_geometry_update = fem_solver.requires_geometry_update
        requires_geometry_update = True # ALWAYS TRUE FOR THIS ROUTINE
        TotalDisp = self.sol[:,:]

        LoadIncrement = fem_solver.number_of_load_increments
        increments = range(LoadIncrement)
        if steps!=None:
            LoadIncrement = len(steps)
            increments = steps

        # COMPUTE THE COMMON/NEIGHBOUR NODES ONCE
        all_nodes = np.unique(elements)
        # Elss, Poss = [], []
        # for inode in all_nodes:
        #     Els, Pos = np.where(elements==inode)
        #     Elss.append(Els)
        #     Poss.append(Pos)
        Elss, Poss = mesh.GetNodeCommonality()[:2]


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


        for incr, Increment in enumerate(increments):
            if TotalDisp.ndim == 3:
                Eulerx = points + TotalDisp[:,:ndim,Increment]
            else:
                Eulerx = points + TotalDisp[:,:ndim]
            if self.formulation.fields == 'electro_mechanics':
                if TotalDisp.ndim == 3:
                    Eulerp = TotalDisp[:,ndim,Increment]
                else:
                    Eulerp = TotalDisp[:,ndim]

            # LOOP OVER ELEMENTS
            for elem in range(nelem):
                # GET THE FIELDS AT THE ELEMENT LEVEL
                LagrangeElemCoords = points[elements[elem,:],:]
                EulerELemCoords = Eulerx[elements[elem,:],:]
                if self.formulation.fields == 'electro_mechanics':
                    ElectricPotentialElem =  Eulerp[elements[elem,:]]

                if material.has_low_level_dispatcher:

                    # GET LOCAL KINEMATICS
                    SpatialGradient, F[elem,:,:,:], detJ = _KinematicMeasures_(Jm, AllGauss[:,0], LagrangeElemCoords,
                        EulerELemCoords, requires_geometry_update)

                    if self.formulation.fields == "electro_mechanics":
                        # GET ELECTRIC FIELD
                        ElectricFieldx[elem,:,:] = - np.einsum('ijk,j',SpatialGradient,ElectricPotentialElem)
                        # COMPUTE WORK-CONJUGATES AND HESSIAN AT THIS GAUSS POINT
                        _D_dum ,CauchyStressTensor[elem,:,:], _ = material.KineticMeasures(F[elem,:,:,:], ElectricFieldx[elem,:,:], elem=elem)
                        ElectricDisplacementx[elem,:,:] = _D_dum[:,:,0]
                    elif self.formulation.fields == "mechanics":
                        # COMPUTE WORK-CONJUGATES AND HESSIAN AT THIS GAUSS POINT
                        CauchyStressTensor[elem,:,:], _ = material.KineticMeasures(F[elem,:,:,:],elem=elem)

                else:
                    # GAUSS LOOP IN VECTORISED FORM
                    ParentGradientX = np.einsum('ijk,jl->kil', Jm, LagrangeElemCoords)
                    # MATERIAL GRADIENT TENSOR IN PHYSICAL ELEMENT [\nabla_0 (N)]
                    MaterialGradient = np.einsum('ijk,kli->ijl', inv(ParentGradientX), Jm)
                    # DEFORMATION GRADIENT TENSOR [\vec{x} \otimes \nabla_0 (N)]
                    F[elem,:,:,:] = np.einsum('ij,kli->kjl', EulerELemCoords, MaterialGradient)
                    # COMPUTE REMAINING KINEMATIC MEASURES
                    StrainTensors = KinematicMeasures(F[elem,:,:,:], fem_solver.analysis_nature)


                    # GEOMETRY UPDATE IS A MUST
                    # MAPPING TENSOR [\partial\vec{X}/ \partial\vec{\varepsilon} (ndim x ndim)]
                    ParentGradientx = np.einsum('ijk,jl->kil',Jm,EulerELemCoords)
                    # SPATIAL GRADIENT TENSOR IN PHYSICAL ELEMENT [\nabla (N)]
                    SpatialGradient = np.einsum('ijk,kli->ilj',inv(ParentGradientx),Jm)
                    # COMPUTE ONCE detJ (GOOD SPEEDUP COMPARED TO COMPUTING TWICE)
                    detJ = np.einsum('i,i,i->i',AllGauss[:,0],np.abs(det(ParentGradientX)),
                        np.abs(StrainTensors['J']))

                    # LOOP OVER GAUSS POINTS
                    for counter in range(AllGauss.shape[0]):

                        if self.formulation.fields == 'electro_mechanics':
                            # GET ELECTRIC FILED
                            ElectricFieldx[elem,counter,:] = - np.dot(SpatialGradient[counter,:,:].T,
                                ElectricPotentialElem)

                            # COMPUTE ELECTRIC DISPLACEMENT
                            ElectricDisplacementx[elem,counter,:] = (material.ElectricDisplacementx(StrainTensors,
                                ElectricFieldx[elem,counter,:], elem, counter))[:,0]

                        if material.energy_type == "enthalpy":
                            # COMPUTE CAUCHY STRESS TENSOR
                            CauchyStressTensor[elem,counter,:] = material.CauchyStress(StrainTensors,
                                ElectricFieldx[elem,counter,:],elem,counter)

                        elif material.energy_type == "internal_energy":
                            # COMPUTE CAUCHY STRESS TENSOR
                            CauchyStressTensor[elem,counter,:] = material.CauchyStress(StrainTensors,
                                ElectricDisplacementx[elem,counter,:],elem,counter)

            if average_derived_quantities:
                for inode in all_nodes:
                    # Els, Pos = np.where(elements==inode)
                    Els, Pos = Elss[inode], Poss[inode]
                    ncommon_nodes = Els.shape[0]
                    for uelem in range(ncommon_nodes):
                        MainDict['F'][incr,inode,:,:] += F[Els[uelem],Pos[uelem],:,:]
                        if formulation.fields == "electro_mechanics":
                            MainDict['ElectricFieldx'][incr,inode,:] += ElectricFieldx[Els[uelem],Pos[uelem],:]
                            MainDict['ElectricDisplacementx'][incr,inode,:] += ElectricDisplacementx[Els[uelem],Pos[uelem],:]
                        MainDict['CauchyStress'][incr,inode,:,:] += CauchyStressTensor[Els[uelem],Pos[uelem],:,:]

                    # AVERAGE OUT
                    MainDict['F'][incr,inode,:,:] /= ncommon_nodes
                    if formulation.fields == "electro_mechanics":
                        MainDict['ElectricFieldx'][incr,inode,:] /= ncommon_nodes
                        MainDict['ElectricDisplacementx'][incr,inode,:] /= ncommon_nodes
                    MainDict['CauchyStress'][incr,inode,:,:] /= ncommon_nodes

            else:
                for inode in all_nodes:
                    # Els, Pos = np.where(elements==inode)
                    Els, Pos = Elss[inode], Poss[inode]
                    ncommon_nodes = Els.shape[0]
                    uelem = 0
                    MainDict['F'][incr,inode,:,:] = F[Els[uelem],Pos[uelem],:,:]
                    if formulation.fields == "electro_mechanics":
                        MainDict['ElectricFieldx'][incr,inode,:] = ElectricFieldx[Els[uelem],Pos[uelem],:]
                        MainDict['ElectricDisplacementx'][incr,inode,:] = ElectricDisplacementx[Els[uelem],Pos[uelem],:]
                    MainDict['CauchyStress'][incr,inode,:,:] = CauchyStressTensor[Els[uelem],Pos[uelem],:,:]


        self.recovered_fields = MainDict
        return


    def GetAugmentedSolution(self, steps=None, average_derived_quantities=True):
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
                21          p_hyd           J               S_yy                    H_zz
                ----------------------------------------------------------------------------------------
                22                          C_xx            E_x                     J
                ----------------------------------------------------------------------------------------
                23                          C_xy            E_y                     C_xx
                ----------------------------------------------------------------------------------------
                24                          C_xz            D_x                     C_xy
                ----------------------------------------------------------------------------------------
                25                          C_yy            D_y                     C_xz
                ----------------------------------------------------------------------------------------
                26                          C_yz            p_hyd                   C_yy
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
                38                          S_yy                                    S_xz
                ----------------------------------------------------------------------------------------
                39                          S_yz                                    S_yy
                ----------------------------------------------------------------------------------------
                40                          S_zz                                    S_yz
                ----------------------------------------------------------------------------------------
                41                          p_hyd                                   S_zz
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
                48                                                                  p_hyd
                ----------------------------------------------------------------------------------------




            where S represents Cauchy stress tensor, E the electric field, D the electric
            displacements and p_hyd the hydrostatic pressure

            This function modifies self.sol to augmented_sol and returns the augmented solution
            augmented_sol


        """

        if self.sol.shape[1] > self.nvar:
            return self.sol

        print("Computing recovered quantities. This is going to take some time...")
        # GET RECOVERED VARIABLES ALL VARIABLE CHECKS ARE DONE IN STRESS RECOVERY
        self.StressRecovery(steps=steps,average_derived_quantities=average_derived_quantities)

        ndim = self.formulation.ndim
        fields = self.formulation.fields
        nnode = self.mesh.points.shape[0]
        if self.sol.ndim == 3:
            increments = self.sol.shape[2]
        else:
            increments = 1
        if steps != None:
            increments = len(steps)
        else:
            if increments != self.fem_solver.number_of_load_increments:
                raise ValueError("Incosistent number of load increments between FEMSolver and provided solution")

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
            # Assuming sigma_zz=0
            p_hyd = 1./2.*Cauchy[:,[0,2],:].sum(axis=1)
        elif ndim == 3:
            C = C[:,[0,1,2,4,5,8],:]
            G = G[:,[0,1,2,4,5,8],:]
            Cauchy = Cauchy[:,[0,1,2,4,5,8],:]
            p_hyd  = 1./3.*Cauchy[:,[0,3,5],:].sum(axis=1)


        if fields == "mechanics" and ndim == 2:

            augmented_sol = np.zeros((nnode,22,increments),dtype=np.float64)
            augmented_sol[:,:2,:]     = self.sol[:,:2,steps].reshape(augmented_sol[:,:2,:].shape)
            augmented_sol[:,2:6,:]    = F
            augmented_sol[:,6:10,:]   = H
            augmented_sol[:,10,:]     = J
            augmented_sol[:,11:14,:]  = C
            augmented_sol[:,14:17,:]  = G
            augmented_sol[:,17,:]     = detC
            augmented_sol[:,18:21,:]  = Cauchy
            augmented_sol[:,21,:]     = p_hyd

        elif fields == "mechanics" and ndim == 3:

            augmented_sol = np.zeros((nnode,42,increments),dtype=np.float64)
            augmented_sol[:,:3,:]     = self.sol[:,:3,steps].reshape(augmented_sol[:,:3,:].shape)
            augmented_sol[:,3:12,:]   = F
            augmented_sol[:,12:21,:]  = H
            augmented_sol[:,21,:]     = J
            augmented_sol[:,22:28,:]  = C
            augmented_sol[:,28:34,:]  = G
            augmented_sol[:,34,:]     = detC
            augmented_sol[:,35:41,:]  = Cauchy
            augmented_sol[:,41,:]     = p_hyd


        elif fields == "electro_mechanics" and ndim == 2:

            augmented_sol = np.zeros((nnode,27,increments),dtype=np.float64)
            augmented_sol[:,:3,:]     = self.sol[:,:3,steps].reshape(augmented_sol[:,:3,:].shape)
            augmented_sol[:,3:7,:]    = F
            augmented_sol[:,7:11,:]   = H
            augmented_sol[:,11,:]     = J
            augmented_sol[:,12:15,:]  = C
            augmented_sol[:,15:18,:]  = G
            augmented_sol[:,18,:]     = detC
            augmented_sol[:,19:22,:]  = Cauchy
            augmented_sol[:,22:24,:]  = ElectricFieldx
            augmented_sol[:,24:26,:]  = ElectricDisplacementx
            augmented_sol[:,26,:]     = p_hyd


        elif fields == "electro_mechanics" and ndim == 3:
            augmented_sol = np.zeros((nnode,49,increments),dtype=np.float64)

            augmented_sol[:,:4,:]     = self.sol[:,:4,steps].reshape(augmented_sol[:,:4,:].shape)
            augmented_sol[:,4:13,:]   = F
            augmented_sol[:,13:22,:]  = H
            augmented_sol[:,22,:]     = J
            augmented_sol[:,23:29,:]  = C
            augmented_sol[:,29:35,:]  = G
            augmented_sol[:,35,:]     = detC
            augmented_sol[:,36:42,:]  = Cauchy
            augmented_sol[:,42:45,:]  = ElectricFieldx
            augmented_sol[:,45:48,:]  = ElectricDisplacementx
            augmented_sol[:,48,:]     = p_hyd

        self.sol = augmented_sol
        return augmented_sol


    def QuantityNamer(self, num, print_name=True):
        """Returns the quantity (for augmented solution i.e. primary and recovered variables)
            name given its number (from numbering order)
        """

        namer = None
        if num > 48:
            if print_name:
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
                spl = list(filter(None, line.split(" ")))
                if spl[0] == str(num):
                    if self.nvar == 2 and self.ndim==2:
                        namer = spl[-4]
                    elif self.nvar == 3 and self.ndim==2:
                        namer = spl[-2]
                    elif self.nvar == 3 and self.ndim==3:
                        namer = spl[-3]
                    elif self.nvar == 4:
                        namer = spl[-1]
                    break

        if print_name:
            print('Quantity corresponds to ' + str(namer))

        if "ux" in namer:
            namer = "u_x"
        elif "uy" in namer:
            namer = "u_y"
        elif "uz" in namer:
            namer = "u_z"
        elif "phi" in namer:
            namer = "\phi"
        return namer


    def ConstructDifferentOrderSolution(self, mesh=None, sol=None, p=2, equally_spaced=False, filename=None):
        """Build a solution for a different polynomial degree
            This is an immutable function and does not modify self
            input:
                mesh:               [np.ndarray] actual mesh
                sol:                [Mesh] actual solution
                p:                  [int] desired polynomial degree to construct the solution for
                equally_spaced:     [bool] Construct other order solution wit equally spaced or Gauss Lobatto/Fekete points
                filename            [str] name of the file where the solution has to be stored in case it is to big to fit in memory

            output:
                ho_mesh:            [Mesh] Mesh of desired degree on which the desired solution is built
                ho_sol:             [np.ndarray] desired solution
        """

        from Florence.QuadratureRules import GaussLobattoPointsHex, GaussLobattoPointsQuad
        from Florence.QuadratureRules.FeketePointsTet import FeketePointsTet
        from Florence.QuadratureRules.FeketePointsTri import FeketePointsTri
        from Florence.QuadratureRules.EquallySpacedPoints import EquallySpacedPoints, EquallySpacedPointsTet, EquallySpacedPointsTri
        from Florence.Tensor import makezero

        if mesh is None and self.mesh is None:
            raise ValueError("Mesh not set for post-processing")
        if self.mesh is not None and mesh is None:
            mesh = self.mesh
        if sol is None and self.sol is None:
            raise ValueError("Solution not set for post-processing")
        if self.sol is not None and sol is None:
            sol = self.sol

        C = p - 1
        actual_p = mesh.InferPolynomialDegree()
        print("Constructing solution of degree p = {} from solution of degree p = {}".format(p,actual_p))
        t_sol = time()

        if p == actual_p:
            print("Finished constructing p = {} solution. Time elapsed is {} seconds".format(p,time() - t_sol))
            return mesh, sol
        if p == 1 and actual_p > 1:
            print("Finished constructing p = {} solution. Time elapsed is {} seconds".format(p,time() - t_sol))
            return mesh.GetLinearMesh(solution=sol)

        et = mesh.element_type

        if et == "hex":
            nsize = (actual_p+1)**3
            ho_nsize = (p+1)**3

            if not equally_spaced:
                eps = GaussLobattoPointsHex(C)
                # COMPUTE BASES FUNCTIONS AT ALL NODAL POINTS
                Neval = np.zeros((nsize,eps.shape[0]),dtype=np.float64)
                hpBases = Hex.LagrangeGaussLobatto
                for i in range(eps.shape[0]):
                    Neval[:,i] = hpBases(actual_p-1,eps[i,0],eps[i,1],eps[i,2])[:,0]
            else:
                eps = EquallySpacedPoints(4,C)
                # COMPUTE BASES FUNCTIONS AT ALL NODAL POINTS
                Neval = np.zeros((nsize,eps.shape[0]),dtype=np.float64)
                hpBases = HexES.Lagrange
                for i in range(eps.shape[0]):
                    Neval[:,i] = hpBases(actual_p-1,eps[i,0],eps[i,1],eps[i,2])[:,0]

        elif et == "tet":
            nsize = (actual_p+1)*(actual_p+2)*(actual_p+3) // 6
            ho_nsize = (p+1)*(p+2)*(p+3) // 6

            if not equally_spaced:
                eps = FeketePointsTet(C)
                # COMPUTE BASES FUNCTIONS AT ALL NODAL POINTS
                Neval = np.zeros((nsize,eps.shape[0]),dtype=np.float64)
                hpBases = Tet.hpNodal.hpBases
                for i in range(eps.shape[0]):
                    Neval[:,i] = hpBases(actual_p-1,eps[i,0],eps[i,1],eps[i,2],Transform=1,EvalOpt=1)[0]
            else:
                eps =  EquallySpacedPointsTet(C)
                # COMPUTE BASES FUNCTIONS AT ALL NODAL POINTS
                hpBases = Tet.hpNodal.hpBases
                Neval = np.zeros((nsize,eps.shape[0]),dtype=np.float64)
                for i in range(eps.shape[0]):
                    Neval[:,i]  = hpBases(actual_p-1,eps[i,0],eps[i,1],eps[i,2],Transform=1,EvalOpt=1,equally_spaced=True)[0]

        elif et == "tri":
            nsize = (actual_p+1)*(actual_p+2) // 2
            ho_nsize = (p+1)*(p+2) // 2

            if not equally_spaced:
                eps =  FeketePointsTri(C)
                # COMPUTE BASES FUNCTIONS AT ALL NODAL POINTS
                hpBases = Tri.hpNodal.hpBases
                Neval = np.zeros((nsize,eps.shape[0]),dtype=np.float64)
                for i in range(eps.shape[0]):
                    Neval[:,i]  = hpBases(actual_p-1,eps[i,0],eps[i,1],Transform=1,EvalOpt=1,equally_spaced=True)[0]
            else:
                eps =  EquallySpacedPointsTri(C)
                # COMPUTE BASES FUNCTIONS AT ALL NODAL POINTS
                hpBases = Tri.hpNodal.hpBases
                Neval = np.zeros((nsize,eps.shape[0]),dtype=np.float64)
                for i in range(eps.shape[0]):
                    Neval[:,i]  = hpBases(actual_p-1,eps[i,0],eps[i,1],Transform=1,EvalOpt=1,equally_spaced=True)[0]

        elif et == "quad":
            nsize = (actual_p+1)**2
            ho_nsize = (p+1)**2

            if not equally_spaced:
                eps = GaussLobattoPointsQuad(C)
                # COMPUTE BASES FUNCTIONS AT ALL NODAL POINTS
                Neval = np.zeros((nsize,eps.shape[0]),dtype=np.float64)
                for i in range(eps.shape[0]):
                    Neval[:,i] = Quad.LagrangeGaussLobatto(actual_p-1,eps[i,0],eps[i,1],arrange=1)[:,0]
            else:
                eps = EquallySpacedPoints(3,C)
                # COMPUTE BASES FUNCTIONS AT ALL NODAL POINTS
                Neval = np.zeros((nsize,eps.shape[0]),dtype=np.float64)
                for i in range(eps.shape[0]):
                    Neval[:,i] = QuadES.Lagrange(actual_p-1,eps[i,0],eps[i,1],arrange=1)[:,0]
        else:
            raise ValueError("Element type not understood")
        makezero(Neval,tol=1e-11)

        ho_mesh = deepcopy(mesh)
        sys.stdout = open(os.devnull, "w")
        ho_mesh.GetHighOrderMesh(p=p, equally_spaced=equally_spaced, check_duplicates=False)
        sys.stdout = sys.__stdout__

        if sol.ndim == 2:
            sol = sol[:,:,None]
        increments = range(sol.shape[2])

        try:
            import psutil
        except ImportError:
            has_psutil = False
            raise ImportError("No module named psutil. Please install it using 'pip install psutil'")
        # GET MEMORY INFO
        memory = psutil.virtual_memory()
        sol_size = ho_mesh.nnode*sol.shape[1]*sol.shape[2]//1024**3
        if memory.available//1024**3 > 8*sol_size:
            ho_sol = np.zeros((ho_mesh.nnode,sol.shape[1],sol.shape[2]),dtype=np.float64)
        elif memory.available//1024**3 > 4*sol_size:
            ho_sol = np.zeros((ho_mesh.nnode,sol.shape[1],sol.shape[2]),dtype=np.float32)
        elif memory.available//1024**3 < 4*sol_size:
            warn("Not enough memory to store the solution. Going to activate out of core procedure."
                " As a remedy limit the solution to specific quantity/ies")
            try:
                import h5py
            except ImportError:
                has_h5py = False
                raise ImportError('h5py is not installed. Please install it first by running "pip install h5py"')

            if filename == None:
                filename = os.path.join(os.path.expanduser('~'),"output.hdf5")

            hdf_file = h5py.File(filename,'w')
            ho_sol = hdf_file.create_dataset("Solution",(ho_mesh.nnode,sol.shape[1],sol.shape[2]),dtype=np.float32)

            for ielem in range(mesh.nelem):
                ho_sol[ho_mesh.elements[ielem,:],:,:] = np.tensordot(Neval, sol[mesh.elements[ielem,:],:,:], axes=(0,0))
            # for inc in increments:
                # for ielem in range(mesh.nelem):
                    # ho_sol[ho_mesh.elements[ielem,:],:,inc] = np.dot(Neval.T, sol[mesh.elements[ielem,:],:,inc])

            hdf_file.close()
            print("Results written in {}".format(filename))
            print("Finished constructing p = {} solution. Time elapsed is {} seconds".format(p,time() - t_sol))
            return ho_mesh, ho_sol
        else:
            ho_sol = np.zeros((ho_mesh.nnode,sol.shape[1],sol.shape[2]),dtype=np.float64)

        # DO NOT VECTORISE THE ELEMENT LOOP AS IT LEADS TO AN EXPENSIVE OPERATION
        for ielem in range(mesh.nelem):
            ho_sol[ho_mesh.elements[ielem,:],:,:] = np.tensordot(Neval, sol[mesh.elements[ielem,:],:,:], axes=(0,0))
            # ho_sol[ho_mesh.elements[ielem,:],:,:] = np.einsum("ij,ikl", Neval, sol[mesh.elements[ielem,:],:,:], optimize=True)

        # EXPENSIVE VECTORISED VERSION
        # for inc in increments:
        #     for ielem in range(mesh.nelem):
        #         ho_sol[ho_mesh.elements[ielem,:],:,inc] = np.dot(Neval.T, sol[mesh.elements[ielem,:],:,inc])

        print("Finished constructing p = {} solution. Time elapsed is {} seconds".format(p,time() - t_sol))
        return ho_mesh, ho_sol




    def MeshQualityMeasures(self, mesh, TotalDisp, plot=False, show_plot=False):
        """Computes mesh quality measures, Q_1, Q_2, Q_3 [edge distortion, face distortion, Jacobian]

            input:
                mesh:                   [Mesh] an instance of class mesh can be any mesh type

        """

        if self.is_scaledjacobian_computed is None:
            self.is_scaledjacobian_computed = False
        if self.is_material_anisotropic is None:
            self.is_material_anisotropic = False

        if self.is_scaledjacobian_computed is True:
            raise AssertionError('Scaled Jacobian seems to have been already computed. Re-computing it may return incorrect results')

        PostDomain = self.postdomain_bases
        if self.postdomain_bases is None:
            raise ValueError("Function spaces/bases not set for post-processing")

        ndim = mesh.InferSpatialDimension()
        vpoints = mesh.points
        if TotalDisp.ndim == 3:
            vpoints = vpoints + TotalDisp[:,:ndim,-1]
        elif TotalDisp.ndim == 2:
            vpoints = vpoints + TotalDisp[:,:ndim]
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
            Q1 = np.sqrt(np.abs(np.einsum('kij,lij->kl',F,F))).diagonal()
            # USING INVARIANT H:H
            Q2 = np.sqrt(np.abs(np.einsum('ijk,ijl->kl',H,H))).diagonal()

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
            AverageJacobian[elem] = np.mean(Jacobian)

            if self.is_material_anisotropic:
                ScaledFNFN[elem] = 1.0*np.min(Q4)/np.max(Q4)
                ScaledCNCN[elem] = 1.0*np.min(Q5)/np.max(Q5)

        if np.isnan(ScaledJacobian).any():
            warn("Jacobian of mapping is close to zero")

        print('Minimum scaled F:F value is', ScaledFF.min(), \
        'corresponding to element', ScaledFF.argmin())

        print('Minimum scaled H:H value is', ScaledHH.min(), \
        'corresponding to element', ScaledHH.argmin())

        print('Minimum scaled Jacobian value is', ScaledJacobian.min(), \
        'corresponding to element', ScaledJacobian.argmin())

        if self.is_material_anisotropic:
            print('Minimum scaled FN.FN value is', ScaledFNFN.min(), \
            'corresponding to element', ScaledFNFN.argmin())

            print('Minimum scaled CN.CN value is', ScaledCNCN.min(), \
            'corresponding to element', ScaledCNCN.argmin())


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
        self.ScaledJacobian = ScaledJacobian

        if not self.is_material_anisotropic:
            return self.is_scaledjacobian_computed, ScaledFF, ScaledHH, ScaledJacobian
        else:
            return self.is_scaledjacobian_computed, ScaledFF, ScaledHH, ScaledJacobian, ScaledFNFN, ScaledCNCN





    def WriteVTK(self,filename=None, quantity="all", configuration="deformed", steps=None, write_curved_mesh=True,
        interpolation_degree=10, ProjectionFlags=None, fmt="binary"):
        """Writes results to a VTK file for Paraview

            quantity = "all" means write all solution fields, otherwise specific quantities
            would be written based on augmented solution numbering order
            step - [list or np.1darray of sequentially aranged steps] which time steps/increments should be written

            inputs:
                fmt:                    [str] VTK writer format either "binary" or "xml".
                                        "xml" files do not support big vtk/vtu files
                                        typically greater than 2GB whereas "binary" files can.  Also "xml" writer is
                                        in-built whereas "binary" writer depends on evtk/pyevtk module
        """

        if fmt is "xml":
            pass
        elif fmt is "binary":
            try:
                from pyevtk.hl import pointsToVTK, linesToVTK, gridToVTK, unstructuredGridToVTK
                from pyevtk.vtk import VtkVertex, VtkLine, VtkTriangle, VtkQuad, VtkTetra, VtkPyramid, VtkHexahedron
            except ImportError:
                raise ImportError("Could not import evtk. Install it using 'pip install pyevtk'")
        else:
            raise ValueError("Writer format not understood")
        formatter = fmt

        if self.formulation is None:
            raise ValueError("formulation not set for post-processing")
        if self.sol is None:
            raise ValueError("solution not set for post-processing")
        if self.formulation.fields == "electrostatics":
            configuration = "original"
            tmp = np.copy(self.sol)
            self.sol = np.zeros((self.sol.shape[0],self.formulation.ndim+1,self.sol.shape[1]))
            # self.sol[:,:self.formulation.ndim,:] = 0.
            self.sol[:,-1,:] = tmp
            quantity = self.formulation.ndim

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
        elif isinstance(quantity,list):
            requires_augmented_solution = False
            for i in quantity:
                if i >= self.sol.shape[1]:
                    requires_augmented_solution = True
                    break
            if requires_augmented_solution:
                self.GetAugmentedSolution()
            iterator = quantity
        else:
            raise ValueError('Plotting quantity not understood')

        if filename is None:
            warn("file name not specified. I am going to write in the current directory")
            filename = PWD(__file__) + "/output.vtu"
        elif filename is not None:
            if isinstance(filename,str) is False:
                raise ValueError("file name should be a string")
        if ".vtu" in filename and fmt is "binary":
            filename  = filename.split('.')[0]

        C = self.mesh.InferPolynomialDegree() - 1
        if C == 0:
            write_curved_mesh = False


        # GET LINEAR MESH & SOLUTION
        # lmesh = self.mesh.GetLinearMesh()
        # sol = self.sol[:lmesh.nnode,...]
        lmesh, sol = self.mesh.GetLinearMesh(remap=True,solution=self.sol)


        if lmesh.element_type =='tri':
            cellflag = 5
            offset = 3
            actual_ndim = 2
        elif lmesh.element_type =='quad':
            cellflag = 9
            offset = 4
            actual_ndim = 2
        if lmesh.element_type =='tet':
            cellflag = 10
            offset = 4
            actual_ndim = 3
        elif lmesh.element_type == 'hex':
            cellflag = 12
            offset = 8
            actual_ndim = 3
        actual_ndim = lmesh.points.shape[1]

        ndim = lmesh.points.shape[1]
        if self.formulation.fields == "electrostatics":
            sol = self.sol[:lmesh.nnode,...]
            q_names = ["phi","phi","phi","phi"]
        else:
            # q_names = ["$"+self.QuantityNamer(quant, print_name=False)+"$" for quant in iterator]
            q_names = [self.QuantityNamer(quant, print_name=False) for quant in iterator]

        LoadIncrement = self.sol.shape[2]



        if write_curved_mesh:

            if lmesh.element_type =='tet':
                cellflag = 5
                tmesh = PostProcess.TessellateTets(self.mesh, np.zeros_like(self.mesh.points),
                    QuantityToPlot=self.sol[:,0,0], plot_on_faces=False, plot_points=True,
                    interpolation_degree=interpolation_degree, ProjectionFlags=ProjectionFlags)
            elif lmesh.element_type =='hex':
                cellflag = 5
                tmesh = PostProcess.TessellateHexes(self.mesh, np.zeros_like(self.mesh.points),
                    QuantityToPlot=self.sol[:,0,0], plot_on_faces=False, plot_points=True,
                    interpolation_degree=interpolation_degree, ProjectionFlags=ProjectionFlags)
            elif lmesh.element_type =='quad':
                cellflag = 5
                tmesh = PostProcess.TessellateQuads(self.mesh, np.zeros_like(self.mesh.points),
                    QuantityToPlot=self.sol[:,0,0], plot_points=True,
                    interpolation_degree=interpolation_degree, ProjectionFlags=ProjectionFlags)
            elif lmesh.element_type =='tri':
                cellflag = 5
                tmesh = PostProcess.TessellateTris(self.mesh, np.zeros_like(self.mesh.points),
                    QuantityToPlot=self.sol[:,0,0], plot_points=True,
                    interpolation_degree=interpolation_degree, ProjectionFlags=ProjectionFlags)
            else:
                raise ValueError('Element type not understood')

            nsize = tmesh.nsize
            if hasattr(tmesh,'nface'):
                # FOR 3D ELEMENTS E.G. TETS AND HEXES
                nface = tmesh.nface
            else:
                tmesh.smesh = self.mesh
                tmesh.faces_to_plot = tmesh.smesh.elements
                nface = tmesh.smesh.elements.shape[0]
                tmesh.smesh.GetEdges()

                connections_elements = np.arange(tmesh.x_edges.size).reshape(tmesh.x_edges.shape[1],tmesh.x_edges.shape[0])
                connections = np.zeros((tmesh.x_edges.size,2),dtype=np.int64)
                for i in range(connections_elements.shape[0]):
                    connections[i*(tmesh.x_edges.shape[0]-1):(i+1)*(tmesh.x_edges.shape[0]-1),0] = connections_elements[i,:-1]
                    connections[i*(tmesh.x_edges.shape[0]-1):(i+1)*(tmesh.x_edges.shape[0]-1),1] = connections_elements[i,1:]
                connections = connections[:(i+1)*(tmesh.x_edges.shape[0]-1),:]
                tmesh.connections = connections

            un_faces_to_plot = np.unique(tmesh.faces_to_plot)
            fail_flag = False
            try:
                ssol = self.sol[un_faces_to_plot,:,:]
            except:
                fail_flag = True

            if fail_flag is False:
                if tmesh.smesh.elements.max() > un_faces_to_plot.shape[0]:
                    ssol = self.sol
                    fail_flag = True
                    warn("Something went wrong with mesh tessellation for VTK writer. I will proceed anyway")

            if tmesh.smesh.all_edges.shape[0] > tmesh.edge_elements.shape[0]:
                tmesh.smesh.all_edges = tmesh.edge_elements
                fail_flag = True
                warn("Something went wrong with mesh tessellation for VTK writer. I will proceed anyway")

            increments = range(LoadIncrement)
            if steps!=None:
                increments = steps

            for Increment in increments:

                extrapolated_sol = np.zeros((tmesh.points.shape[0], self.sol.shape[1]))
                for ielem in range(nface):
                    extrapolated_sol[ielem*nsize:(ielem+1)*nsize,:] = np.dot(tmesh.bases_2,
                        ssol[tmesh.smesh.elements[ielem,:],:, Increment])

                if not fail_flag:
                    svpoints = self.mesh.points[np.unique(tmesh.faces_to_plot),:] + ssol[:,:tmesh.points.shape[1],Increment]
                else:
                    svpoints = self.mesh.points + ssol[:,:tmesh.points.shape[1],Increment]

                for iedge in range(tmesh.smesh.all_edges.shape[0]):
                    ielem = tmesh.edge_elements[iedge,0]
                    edge = tmesh.smesh.elements[ielem,tmesh.reference_edges[tmesh.edge_elements[iedge,1],:]]
                    coord_edge = svpoints[edge,:]
                    if tmesh.points.shape[1] == 3:
                        tmesh.x_edges[:,iedge], tmesh.y_edges[:,iedge], tmesh.z_edges[:,iedge] = np.dot(coord_edge.T,tmesh.bases_1)
                    elif tmesh.points.shape[1] == 2:
                        tmesh.x_edges[:,iedge], tmesh.y_edges[:,iedge] = np.dot(coord_edge.T,tmesh.bases_1)

                if tmesh.points.shape[1] == 3:
                    edge_coords = np.concatenate((tmesh.x_edges.T.copy().flatten()[:,None],
                        tmesh.y_edges.T.copy().flatten()[:,None],
                        tmesh.z_edges.T.copy().flatten()[:,None]),axis=1)
                elif tmesh.points.shape[1] == 2:
                    edge_coords = np.concatenate((tmesh.x_edges.T.copy().flatten()[:,None],
                        tmesh.y_edges.T.copy().flatten()[:,None], np.zeros_like(tmesh.y_edges.T.copy().flatten()[:,None])),axis=1)
                    svpoints = np.concatenate((svpoints, np.zeros((svpoints.shape[0],1))),axis=1)

                if formatter is "xml":
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

                elif formatter is "binary":

                    unstructuredGridToVTK(filename.split('.')[0]+'_curved_lines_increment_'+str(Increment),
                        np.ascontiguousarray(edge_coords[:,0]), np.ascontiguousarray(edge_coords[:,1]), np.ascontiguousarray(edge_coords[:,2]),
                        np.ascontiguousarray(tmesh.connections.ravel()), np.arange(0,2*tmesh.connections.shape[0],2)+2,
                        np.ones(tmesh.connections.shape[0])*3)

                    pointsToVTK(filename.split('.')[0]+'_curved_points_increment_'+str(Increment),
                        np.ascontiguousarray(svpoints[:,0]), np.ascontiguousarray(svpoints[:,1]), np.ascontiguousarray(svpoints[:,2]),
                        data=None)

                    if tmesh.points.shape[1] == 2:
                        points = np.zeros((tmesh.points.shape[0],3))
                        points[:,:2] = tmesh.points+extrapolated_sol[:,:ndim]
                    else:
                        points = tmesh.points+extrapolated_sol[:,:ndim]
                    for counter, quant in enumerate(iterator):
                        unstructuredGridToVTK(filename.split('.')[0]+'_curved_quantity_'+str(quant)+'_increment_'+str(Increment),
                            np.ascontiguousarray(points[:,0]), np.ascontiguousarray(points[:,1]), np.ascontiguousarray(points[:,2]),
                            np.ascontiguousarray(tmesh.elements.ravel()), np.arange(0,3*tmesh.elements.shape[0],3)+3,
                            np.ones(tmesh.elements.shape[0])*cellflag,
                            pointData={q_names[counter]: np.ascontiguousarray(extrapolated_sol[:,quant])})

        else:

            increments = range(LoadIncrement)
            if steps!=None:
                increments = steps

            if configuration == "original":
                for Increment in increments:
                    if formatter is "xml":
                        for quant in iterator:
                            vtk_writer.write_vtu(Verts=lmesh.points,
                                Cells={cellflag:lmesh.elements}, pdata=sol[:,quant,Increment],
                                fname=filename.split('.')[0]+'_quantity_'+str(quant)+'_increment_'+str(Increment)+'.vtu')
                    elif formatter is "binary":
                        # points = lmesh.points
                        if lmesh.InferSpatialDimension() == 2:
                            points = np.zeros((lmesh.points.shape[0],3))
                            points[:,:2] = lmesh.points
                        else:
                            points = lmesh.points
                        for counter, quant in enumerate(iterator):
                            unstructuredGridToVTK(filename.split('.')[0]+'_quantity_'+str(quant)+'_increment_'+str(Increment),
                                np.ascontiguousarray(points[:,0]), np.ascontiguousarray(points[:,1]), np.ascontiguousarray(points[:,2]),
                                np.ascontiguousarray(lmesh.elements.ravel()), np.arange(0,offset*lmesh.nelem,offset)+offset,
                                np.ones(lmesh.nelem)*cellflag,
                                pointData={q_names[counter]: np.ascontiguousarray(sol[:,quant,Increment])})

            elif configuration == "deformed":
                for Increment in increments:
                    if formatter is "xml":
                        for quant in iterator:
                            vtk_writer.write_vtu(Verts=lmesh.points+sol[:,:ndim,Increment],
                                Cells={cellflag:lmesh.elements}, pdata=sol[:,quant,Increment],
                                fname=filename.split('.')[0]+'_quantity_'+str(quant)+'_increment_'+str(Increment)+'.vtu')
                    elif formatter is "binary":
                        if lmesh.InferSpatialDimension() == 2:
                            points = np.zeros((lmesh.points.shape[0],3))
                            points[:,:2] = lmesh.points + sol[:,:ndim,Increment]
                        else:
                            points = lmesh.points + sol[:,:ndim,Increment]

                        for counter, quant in enumerate(iterator):
                            unstructuredGridToVTK(filename.split('.')[0]+'_quantity_'+str(quant)+'_increment_'+str(Increment),
                                np.ascontiguousarray(points[:,0]), np.ascontiguousarray(points[:,1]), np.ascontiguousarray(points[:,2]),
                                np.ascontiguousarray(lmesh.elements.ravel()), np.arange(0,offset*lmesh.nelem,offset)+offset,
                                np.ones(lmesh.nelem)*cellflag,
                                pointData={q_names[counter]: np.ascontiguousarray(sol[:,quant,Increment])})

        return


    def WriteHDF5(self, filename=None, compute_recovered_fields=True, dict_wise=False, do_compression=True):
        """Writes the solution data to a HDF5 file. Give the extension name while providing filename

            Input:
                dict_wise:                  saves the dictionary of recovered variables as they are
                                            computed in StressRecovery
                do_compression:             the reason this is given as input arguments is that big
                                            arrays fail due to the overflow bug python standard library
                                            for arrays with more than 2**31 elements
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
            savemat(filename,MainDict, do_compression=do_compression)
        else:
            if dict_wise:
                MainDict = self.recovered_fields
            else:
                MainDict = {}
            MainDict['Solution'] = self.sol
            savemat(filename,MainDict, do_compression=do_compression)




    def PlotNewtonRaphsonConvergence(self, increment=None, figure=None, show_plot=True, save=False, filename=None):
        """Plots convergence of Newton-Raphson for a given increment"""

        if self.fem_solver is None:
            raise ValueError("FEM solver not set for post-processing")

        if increment == None:
            increment = len(self.fem_solver.NRConvergence)-1

        import matplotlib.pyplot as plt
        if figure is None:
            figure = plt.figure()

        plt.plot(np.log10(self.fem_solver.NRConvergence['Increment_'+str(increment)]),'-ko')
        axis_font = {'size':'18'}
        plt.xlabel(r'$No\;\; of\;\; Iterations$', **axis_font)
        plt.ylabel(r'$log_{10}|Residual|$', **axis_font)
        plt.grid('on')

        if save:
            if filename is None:
                warn("No filename provided. I am going to write one in the current directory")
                filename = PWD(__file__) + '/output.eps'
            plt.savefig(filename, format='eps', dpi=500)

        if show_plot:
            plt.show()



    def Plot(self, figure=None, quantity=0, configuration="original", increment=-1, colorbar=True, axis_type=None, interpolation_degree=10,
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
            if increment==-1:
                self.GetAugmentedSolution(steps=[self.sol.shape[2]-1])
            else:
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
            from Florence.MeshGeneration.NodeArrangement import NodeArrangementTri
            from Florence.FunctionSpace import Tri
            from Florence.FunctionSpace.OneDimensional.Line import LagrangeGaussLobatto, Lagrange


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
                        interpolation_degree=interpolation_degree, show_plot=False, figure=figure,
                        save_tessellation=True, plot_points=plot_points, plot_edges=plot_edges)[-1]
                else:
                    incr = -1
                    tmesh = PostProcess.CurvilinearPlotTri(self.mesh, self.sol,
                        interpolation_degree=interpolation_degree, show_plot=False, figure=figure,
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
                        interpolation_degree=interpolation_degree, figure=figure, show_plot=show_plot, plot_on_faces=False,
                        plot_points=plot_points, point_radius=point_radius, plot_edges=plot_edges,
                        colorbar=colorbar, save=save, filename=filename)

                elif configuration=="deformed":

                    PostProcess.CurvilinearPlotTet(self.mesh, self.sol[:,:ndim,-1],
                        QuantityToPlot= self.sol[:,quantity,increment],
                        interpolation_degree=interpolation_degree,figure=figure, show_plot=show_plot, plot_on_faces=False,
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
                        interpolation_degree=interpolation_degree, show_plot=show_plot, figure=figure,
                        save_tessellation=True, plot_points=plot_points, plot_edges=plot_edges,
                        colorbar=colorbar, plot_on_faces=False)
                else:
                    PostProcess.CurvilinearPlotQuad(self.mesh, self.sol,
                        QuantityToPlot=self.sol[:,quantity,increment],
                        interpolation_degree=interpolation_degree, show_plot=show_plot, figure=figure,
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
                    interpolation_degree=interpolation_degree, show_plot=show_plot, figure=figure,
                    plot_points=plot_points, point_radius=point_radius, plot_edges=plot_edges,
                    colorbar=colorbar, plot_on_faces=False, save=save, filename=filename)
            else:
                PostProcess.CurvilinearPlotHex(self.mesh, self.sol,
                    QuantityToPlot=self.sol[:,quantity,increment],
                    interpolation_degree=interpolation_degree, show_plot=show_plot, figure=figure,
                    plot_points=plot_points, point_radius=point_radius, plot_edges=plot_edges,
                    colorbar=colorbar, plot_on_faces=False, save=save, filename=filename)

            return



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
                TotalDisp = np.zeros_like(mesh.points)
            else:
                TotalDisp = self.sol

        if mesh.element_type == "line":
            return self.CurvilinearPlotLine(mesh,TotalDisp,**kwargs)
        elif mesh.element_type == "tri":
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
    def CurvilinearPlotLine(mesh, TotalDisp=None, QuantityToPlot=None, plot_on_faces=True,
        ProjectionFlags=None, interpolation_degree=20, EquallySpacedPoints=False, PlotActualCurve=False,
        plot_points=False, plot_edges=True, plot_surfaces=True, point_radius=0.02, colorbar=False, color=None, figure=None,
        show_plot=True, save=False, filename=None, save_tessellation=False):

        """High order curved line mesh plots, based on high order nodal FEM.
        """

        if not isinstance(mesh,Mesh):
            raise TypeError("mesh has to be an instance of type {}".format(Mesh))
        if mesh.element_type != "line":
            raise RuntimeError("Calling line plotting function with element type {}".format(mesh.element_type))
        if TotalDisp is None:
            TotalDisp = np.zeros_like(mesh.points)


        tmesh = PostProcess.TessellateLines(mesh, TotalDisp, QuantityToPlot=QuantityToPlot,
            ProjectionFlags=ProjectionFlags, interpolation_degree=interpolation_degree,
            EquallySpacedPoints=EquallySpacedPoints, plot_points=plot_points,
            plot_edges=plot_edges, plot_on_faces=plot_on_faces)

        # UNPACK
        x_edges = tmesh.x_edges
        y_edges = tmesh.y_edges
        z_edges = tmesh.z_edges
        nnode = tmesh.nnode
        nelem = tmesh.nelem
        nsize = tmesh.nsize

        # Xplot = tmesh.points
        # Tplot = tmesh.elements
        vpoints = tmesh.vpoints
        connections = tmesh.elements


        import os
        os.environ['ETS_TOOLKIT'] = 'qt4'
        from mayavi import mlab

        if figure is None:
            figure = mlab.figure(bgcolor=(1,1,1),fgcolor=(1,1,1),size=(1000,800))

        # PLOT LINES
        if plot_points:
            h_points = mlab.points3d(vpoints[:,0],vpoints[:,1],vpoints[:,2],color=(0,0,0),mode='sphere',scale_factor=point_radius)

        # PLOT CURVED EDGES
        if plot_edges:
            src = mlab.pipeline.scalar_scatter(x_edges.T.copy().flatten(), y_edges.T.copy().flatten(), z_edges.T.copy().flatten())
            src.mlab_source.dataset.lines = connections
            lines = mlab.pipeline.stripper(src)
            h_edges = mlab.pipeline.surface(lines, color = (0,0,0), line_width=2)


        mlab.view(azimuth=0, roll=0)
        mlab.show()
        return




    @staticmethod
    def CurvilinearPlotTri(mesh, TotalDisp=None, QuantityToPlot=None,
        ProjectionFlags=None, interpolation_degree=30, EquallySpacedPoints=False,
        TriSurf=False, colorbar=False, PlotActualCurve=False, point_radius = 3, color="#C5F1C5",
        plot_points=False, plot_edges=True, save=False, filename=None, figure=None, show_plot=True,
        save_tessellation=False, plot_surfaces=True):

        """High order curved triangular mesh plots, based on high order nodal FEM.
            The equally spaced FEM points do not work as good as the Fekete points
        """

        if not isinstance(mesh,Mesh):
            raise TypeError("mesh has to be an instance of type {}".format(Mesh))
        if mesh.element_type != "tri":
            raise RuntimeError("Calling triangular plotting function with element type {}".format(mesh.element_type))
        if TotalDisp is None:
            TotalDisp = np.zeros_like(mesh.points)


        from Florence.QuadratureRules.FeketePointsTri import FeketePointsTri
        from Florence.QuadratureRules.EquallySpacedPoints import EquallySpacedPointsTri
        from Florence.QuadratureRules.NumericIntegrator import GaussLobattoQuadrature
        from Florence.MeshGeneration.NodeArrangement import NodeArrangementTri
        from Florence.FunctionSpace import Tri
        from Florence.FunctionSpace.OneDimensional.Line import LagrangeGaussLobatto, Lagrange

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
                EvalOpt=1,equally_spaced=EquallySpacedPoints,Transform=1)[0]

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

        pdim = mesh.points.shape[1]
        # GET EULERIAN GEOMETRY
        if TotalDisp.ndim==3:
            vpoints = mesh.points + TotalDisp[:,:pdim,-1]
        else:
            vpoints = mesh.points + TotalDisp[:,:pdim]

        # GET X & Y OF CURVED EDGES
        if plot_edges:
            x_edges = np.zeros((C+2,smesh.all_edges.shape[0]))
            y_edges = np.zeros((C+2,smesh.all_edges.shape[0]))
            z_edges = np.zeros((C+2,smesh.all_edges.shape[0]))

            for iedge in range(smesh.all_edges.shape[0]):
                ielem = edge_elements[iedge,0]
                edge = mesh.elements[ielem,reference_edges[edge_elements[iedge,1],:]]
                coord_edge = vpoints[edge,:]
                if pdim == 2:
                    x_edges[:,iedge], y_edges[:,iedge] = np.dot(coord_edge.T,BasesOneD)
                elif pdim == 3:
                    x_edges[:,iedge], y_edges[:,iedge], z_edges[:,iedge] = np.dot(coord_edge.T,BasesOneD)


        # MAKE FIGURE
        if figure is None:
            fig = plt.figure()
        else:
            fig = figure

        if color is not None:
            if isinstance(color,tuple):
                if len(color) != 3:
                    raise ValueError("Color should be given in a rgb/RGB tuple format with 3 values i.e. (x,y,z)")
                if color[0] > 1.0 or color[1] > 1.0 or color[2] > 1.0:
                    color = (color[0]/255.,color[1]/255.,color[2]/255.)
                color = mpl.colors.rgb2hex(color)
            elif isinstance(color,str):
                pass

        h_surfaces, h_edges, h_points = None, None, None
        # ls = LightSource(azdeg=315, altdeg=45)
        if TriSurf is True:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

        if plot_edges:
            # PLOT CURVED EDGES
            h_edges = ax.plot(x_edges,y_edges,'k')

        if plot_surfaces:
            mesh.nelem = int(mesh.nelem)
            nnode = int(nsize*mesh.nelem)
            nelem = int(Triangles.shape[0]*mesh.nelem)

            Xplot = np.zeros((nnode,pdim),dtype=np.float64)
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
                triang = mtri.Triangulation(Xplot[:,0], Xplot[:,1],Tplot)
                ax.plot_trisurf(triang,Xplot[:,0]*0, edgecolor="none",facecolor="#ffddbb")
                ax.view_init(90,-90)
                ax.dist = 7
            else:
                # plt.tricontourf(Xplot[:,0], Xplot[:,1], Tplot[:4,:], np.ones(Xplot.shape[0]),alpha=0.8,origin='lower')
                if QuantityToPlot is None:
                    h_surfaces = plt.tricontourf(Xplot[:,0], Xplot[:,1], Tplot, Uplot, colors=color)
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

        if plot_surfaces:
            if colorbar is True:
                ax_cbar = mpl.colorbar.make_axes(plt.gca(), shrink=1.0)[0]
                cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=cm.viridis,
                                   norm=mpl.colors.Normalize(vmin=-0, vmax=1))
                cbar.set_clim(0, 1)
                divider = make_axes_locatable(ax_cbar)
                cax = divider.append_axes("right", size="1%", pad=0.005)

        if PlotActualCurve is True:
            if ActualCurve is not None:
                for i in range(len(ActualCurve)):
                    actual_curve_points = ActualCurve[i]
                    plt.plot(actual_curve_points[:,0],actual_curve_points[:,1],'-r',linewidth=3)
            else:
                raise KeyError("You have not computed the CAD curve points")

        plt.axis('equal')
        # plt.xlim([-11,11.])
        # plt.ylim([-11,11.])
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
                tmesh.z_edges = z_edges
                tmesh.edge_elements = edge_elements
                tmesh.reference_edges = reference_edges

            return h_surfaces, h_edges, h_points, tmesh


        return h_surfaces, h_edges, h_points







    @staticmethod
    def CurvilinearPlotTet(mesh, TotalDisp=None, QuantityToPlot=None, plot_on_faces=True,
        ProjectionFlags=None, interpolation_degree=20, EquallySpacedPoints=False, PlotActualCurve=False,
        plot_points=False, plot_edges=True, plot_surfaces=True, point_radius=0.02, colorbar=False, color=None, figure=None,
        show_plot=True, save=False, filename=None, save_tessellation=False):

        """High order curved tetrahedral surfaces mesh plots, based on high order nodal FEM.
            The equally spaced FEM points do not work as good as the Fekete points
        """

        if not isinstance(mesh,Mesh):
            raise TypeError("mesh has to be an instance of type {}".format(Mesh))
        if mesh.element_type != "tet":
            raise RuntimeError("Calling tetrahedral plotting function with element type {}".format(mesh.element_type))
        if TotalDisp is None:
            TotalDisp = np.zeros_like(mesh.points)


        from Florence.QuadratureRules.FeketePointsTri import FeketePointsTri
        from Florence.QuadratureRules.EquallySpacedPoints import EquallySpacedPointsTri
        from Florence.QuadratureRules.NumericIntegrator import GaussLobattoQuadrature
        from Florence.MeshGeneration.NodeArrangement import NodeArrangementTri
        from Florence.FunctionSpace import Tri
        from Florence.FunctionSpace.OneDimensional.Line import LagrangeGaussLobatto, Lagrange

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
                EvalOpt=1,equally_spaced=EquallySpacedPoints,Transform=1)[0]

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

        if color is not None:
            if isinstance(color,tuple):
                if len(color) != 3:
                    raise ValueError("Color should be given in a rgb/RGB tuple format with 3 values i.e. (x,y,z)")
                if color[0] > 1.0 or color[1] > 1.0 or color[2] > 1.0:
                    color = (color[0]/255.,color[1]/255.,color[2]/255.)
            elif isinstance(color,str):
                color = mpl.colors.hex2color(color)

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

            # REDIRECT FILTER NOISE TO /dev/null
            devnull = open('/dev/null', 'w')
            oldstdout_fno = os.dup(sys.stdout.fileno())
            os.dup2(devnull.fileno(), 1)

            src = mlab.pipeline.scalar_scatter(x_edges.T.copy().flatten(), y_edges.T.copy().flatten(), z_edges.T.copy().flatten())
            src.mlab_source.dataset.lines = connections
            lines = mlab.pipeline.stripper(src)
            h_edges = mlab.pipeline.surface(lines, color = (0,0,0), line_width=2)
            # h_edges = mlab.pipeline.surface(lines, color = (0,0,0), line_width=1)
            # mlab.pipeline.surface(lines, color = (0.72,0.72,0.72), line_width=2)
            os.dup2(oldstdout_fno, 1)

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
    def CurvilinearPlotQuad(mesh, TotalDisp=None, QuantityToPlot=None,
        ProjectionFlags=None, interpolation_degree=30, EquallySpacedPoints=False,
        TriSurf=False, colorbar=False, PlotActualCurve=False, point_radius = 3, color="#C5F1C5",
        plot_points=False, plot_edges=True, save=False, filename=None, figure=None, show_plot=True,
        save_tessellation=False, plot_on_faces=True, plot_surfaces=True):

        """High order curved quad mesh plots, based on high order nodal FEM.
        """

        if not isinstance(mesh,Mesh):
            raise TypeError("mesh has to be an instance of type {}".format(Mesh))
        if mesh.element_type != "quad":
            raise RuntimeError("Calling quadrilateral plotting function with element type {}".format(mesh.element_type))
        if TotalDisp is None:
            TotalDisp = np.zeros_like(mesh.points)

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
        if plot_edges:
            x_edges = tmesh.x_edges
            y_edges = tmesh.y_edges
            z_edges = tmesh.z_edges
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

        if color is not None:
            if isinstance(color,tuple):
                if len(color) != 3:
                    raise ValueError("Color should be given in a rgb/RGB tuple format with 3 values i.e. (x,y,z)")
                if color[0] > 1.0 or color[1] > 1.0 or color[2] > 1.0:
                    color = (color[0]/255.,color[1]/255.,color[2]/255.)
                color = mpl.colors.rgb2hex(color)
            elif isinstance(color,str):
                pass

        h_surfaces, h_edges, h_points = None, None, None
        # ls = LightSource(azdeg=315, altdeg=45)
        if TriSurf is True:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

        if plot_edges:
            # PLOT CURVED EDGES
            h_edges = ax.plot(x_edges,y_edges,'k')

        if plot_surfaces:
            # PLOT CURVED ELEMENTS
            if QuantityToPlot is None:
                h_surfaces = plt.tricontourf(Xplot[:,0], Xplot[:,1], Tplot, Uplot, colors=color)
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
            if ActualCurve is not None:
                for i in range(len(ActualCurve)):
                    actual_curve_points = ActualCurve[i]
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
    def CurvilinearPlotHex(mesh, TotalDisp=None, QuantityToPlot=None, plot_on_faces=True,
        ProjectionFlags=None, interpolation_degree=20, EquallySpacedPoints=False, PlotActualCurve=False,
        plot_points=False, plot_edges=True, plot_surfaces=True, point_radius=0.02, colorbar=False, color=None, figure=None,
        show_plot=True, save=False, filename=None, save_tessellation=False):

        """High order curved hexahedral surfaces mesh plots, based on high order nodal FEM.
            The equally spaced FEM points do not work as good as the Fekete points
        """

        if not isinstance(mesh,Mesh):
            raise TypeError("mesh has to be an instance of type {}".format(Mesh))
        if mesh.element_type != "hex":
            raise RuntimeError("Calling hexahedral plotting function with element type {}".format(mesh.element_type))
        if TotalDisp is None:
            TotalDisp = np.zeros_like(mesh.points)

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

        # MAKE A FIGURE
        if figure is None:
            # figure = mlab.figure(bgcolor=(1,1,1),fgcolor=(1,1,1),size=(800,600))
            figure = mlab.figure(bgcolor=(1,1,1),fgcolor=(0,0,0),size=(800,600))

        if color is not None:
            if isinstance(color,tuple):
                if len(color) != 3:
                    raise ValueError("Color should be given in a rgb/RGB tuple format with 3 values i.e. (x,y,z)")
                if color[0] > 1.0 or color[1] > 1.0 or color[2] > 1.0:
                    color = (color[0]/255.,color[1]/255.,color[2]/255.)
            elif isinstance(color,str):
                color = mpl.colors.hex2color(color)

        figure.scene.disable_render = True

        h_points, h_edges, trimesh_h = None, None, None

        if plot_edges:
            x_edges = tmesh.x_edges
            y_edges = tmesh.y_edges
            z_edges = tmesh.z_edges
            connections = tmesh.connections

            # REDIRECT FILTER NOISE TO /dev/null
            devnull = open('/dev/null', 'w')
            # oldstdout_fno = os.dup(sys.stdout.fileno())
            oldstdout_fno = os.dup(sys.stderr.fileno())
            # os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)

            src = mlab.pipeline.scalar_scatter(x_edges.T.copy().flatten(), y_edges.T.copy().flatten(), z_edges.T.copy().flatten())
            src.mlab_source.dataset.lines = connections
            lines = mlab.pipeline.stripper(src)
            h_edges = mlab.pipeline.surface(lines, color = (0,0,0), line_width=2)

            os.dup2(oldstdout_fno, 1)


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
    def Tessellate(self,*args,**kwargs):
        """Tesselate meshes"""
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
                TotalDisp = np.zeros_like(mesh.points)
            else:
                TotalDisp = self.sol

        if mesh.element_type == "line":
            return self.TessellateLines(mesh,TotalDisp,**kwargs)
        elif mesh.element_type == "tri":
            return self.TessellateTris(mesh,TotalDisp,**kwargs)
        elif mesh.element_type == "quad":
            return self.TessellateQuads(mesh,TotalDisp,**kwargs)
        elif mesh.element_type == "tet":
            return self.TessellateTets(mesh,TotalDisp,**kwargs)
        elif mesh.element_type == "hex":
            return self.TessellateHexes(mesh,TotalDisp,**kwargs)
        else:
            raise ValueError("Unknown mesh type")



    @staticmethod
    def TessellateLines(mesh, TotalDisp, QuantityToPlot=None,
        ProjectionFlags=None, interpolation_degree=10, EquallySpacedPoints=False,
        plot_points=False, plot_edges=True, plot_on_faces=None):

        """High order curved line tessellation, based on high order nodal FEM.
        """


        from Florence.QuadratureRules.EquallySpacedPoints import EquallySpacedPoints as ESPoints
        from Florence.QuadratureRules.NumericIntegrator import GaussLobattoQuadrature
        from Florence.FunctionSpace.OneDimensional.Line import LagrangeGaussLobatto, Lagrange

        # SINCE THIS IS A 1D PLOT
        ndim = 1

        C = interpolation_degree
        p = C+1
        nsize = int((C+2)**ndim)
        CActual = mesh.InferPolynomialDegree() - 1
        nsize_2 = int((CActual+2)**ndim)

        # GET EQUALLY-SPACED/GAUSS-LOBATTO POINTS FOR THE EDGES
        GaussLobattoPointsOneD = GaussLobattoQuadrature(C+2)[0].flatten()
        if EquallySpacedPoints is True:
            GaussLobattoPointsOneD = ESPoints(2,C).flatten()
            # GaussLobattoPointsOneD = np.linspace(-1,1,C+2)

        BasesOneD = np.zeros((CActual+2,GaussLobattoPointsOneD.shape[0]),dtype=np.float64)
        if EquallySpacedPoints is False:
            for i in range(GaussLobattoPointsOneD.shape[0]):
                BasesOneD[:,i] = LagrangeGaussLobatto(CActual,GaussLobattoPointsOneD[i])[0]
        else:
            for i in range(GaussLobattoPointsOneD.shape[0]):
                BasesOneD[:,i] = Lagrange(CActual,GaussLobattoPointsOneD[i])[0]

        pdim = mesh.points.shape[1]
        # GET EULERIAN GEOMETRY
        if TotalDisp.ndim==3:
            vpoints = mesh.points + TotalDisp[:,:pdim,-1]
        else:
            vpoints = mesh.points + TotalDisp[:,:pdim]


        # GET X & Y OF CURVED EDGES
        if plot_edges:
            x_edges = np.zeros((C+2,mesh.nelem))
            y_edges = np.zeros((C+2,mesh.nelem))
            z_edges = np.zeros((C+2,mesh.nelem))

            for iedge in range(mesh.nelem):
                edge = mesh.elements[iedge,:]
                coord_edge = vpoints[edge,:]
                if pdim == 1:
                    x_edges[:,iedge] = np.dot(coord_edge.T,BasesOneD)
                elif pdim == 2:
                    x_edges[:,iedge], y_edges[:,iedge] = np.dot(coord_edge.T,BasesOneD)
                else:
                    x_edges[:,iedge], y_edges[:,iedge], z_edges[:,iedge] = np.dot(coord_edge.T,BasesOneD)

        # PLOT CURVED EDGES
        if plot_edges:
            connections_elements = np.arange(x_edges.size).reshape(x_edges.shape[1],x_edges.shape[0])
            connections = np.zeros((x_edges.size,2),dtype=np.int64)
            for i in range(connections_elements.shape[0]):
                connections[i*(x_edges.shape[0]-1):(i+1)*(x_edges.shape[0]-1),0] = connections_elements[i,:-1]
                connections[i*(x_edges.shape[0]-1):(i+1)*(x_edges.shape[0]-1),1] = connections_elements[i,1:]
            connections = connections[:(i+1)*(x_edges.shape[0]-1),:]


        # SAVE TESSELLATION
        tmesh = Mesh()
        tmesh.element_type = "line"
        tmesh.elements = connections
        tmesh.points = np.concatenate((x_edges.T.ravel()[:,None],y_edges.T.ravel()[:,None],z_edges.T.ravel()[:,None]),1)
        tmesh.vpoints = vpoints
        tmesh.nelem = connections.shape[0]
        tmesh.nnode = tmesh.points.shape[0]
        tmesh.nsize = nsize
        tmesh.bases_1 = BasesOneD
        tmesh.bases_2 = None

        if plot_edges:
            tmesh.x_edges = x_edges
            tmesh.y_edges = y_edges
            tmesh.z_edges = z_edges
            tmesh.edge_elements = mesh.elements
            tmesh.reference_edges = np.array([[0,1]])

            return tmesh







    @staticmethod
    def TessellateTris(mesh, TotalDisp, QuantityToPlot=None,
        ProjectionFlags=None, interpolation_degree=30, EquallySpacedPoints=False,
        plot_points=False, plot_edges=True, plot_on_faces=False):

        """High order curved triangular mesh plots, based on high order nodal FEM.
            The equally spaced FEM points do not work as good as the Fekete points
        """


        from Florence.QuadratureRules.FeketePointsTri import FeketePointsTri
        from Florence.QuadratureRules.EquallySpacedPoints import EquallySpacedPointsTri
        from Florence.QuadratureRules.NumericIntegrator import GaussLobattoQuadrature
        from Florence.MeshGeneration.NodeArrangement import NodeArrangementTri
        from Florence.FunctionSpace import Tri
        from Florence.FunctionSpace.OneDimensional.Line import LagrangeGaussLobatto, Lagrange

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
                EvalOpt=1,equally_spaced=EquallySpacedPoints,Transform=1)[0]

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

        pdim = mesh.points.shape[1]
        # GET EULERIAN GEOMETRY
        if TotalDisp.ndim==3:
            vpoints = mesh.points + TotalDisp[:,:pdim,-1]
        else:
            vpoints = mesh.points + TotalDisp[:,:pdim]

        # GET X & Y OF CURVED EDGES
        if plot_edges:
            x_edges = np.zeros((C+2,smesh.all_edges.shape[0]))
            y_edges = np.zeros((C+2,smesh.all_edges.shape[0]))
            z_edges = np.zeros((C+2,smesh.all_edges.shape[0]))

            for iedge in range(smesh.all_edges.shape[0]):
                ielem = edge_elements[iedge,0]
                edge = mesh.elements[ielem,reference_edges[edge_elements[iedge,1],:]]
                coord_edge = vpoints[edge,:]
                if pdim == 2:
                    x_edges[:,iedge], y_edges[:,iedge] = np.dot(coord_edge.T,BasesOneD)
                elif pdim == 3:
                    x_edges[:,iedge], y_edges[:,iedge], z_edges[:,iedge] = np.dot(coord_edge.T,BasesOneD)


        nnode = nsize*mesh.nelem
        nelem = Triangles.shape[0]*mesh.nelem

        Xplot = np.zeros((nnode,pdim),dtype=np.float64)
        Tplot = np.zeros((nelem,3),dtype=np.int64)
        Uplot = np.zeros(nnode,dtype=np.float64)
        # if plot_on_faces and QuantityToPlot is not None:
        #     Uplot = np.zeros(nelem,dtype=np.float64)

        if QuantityToPlot is None:
            if plot_on_faces:
                quantity_to_plot = np.zeros(mesh.nelem)
            else:
                quantity_to_plot = np.zeros(mesh.points.shape[0])
        else:
            quantity_to_plot = QuantityToPlot

        # FOR CURVED ELEMENTS
        for ielem in range(mesh.nelem):
            Xplot[ielem*nsize:(ielem+1)*nsize,:] = np.dot(BasesTri.T, vpoints[mesh.elements[ielem,:],:])
            Tplot[ielem*TrianglesFunc.nsimplex:(ielem+1)*TrianglesFunc.nsimplex,:] = Triangles + ielem*nsize
            if plot_on_faces:
                # Uplot[ielem*nsize:(ielem+1)*nsize] = quantity_to_plot[ielem]
                Uplot[ielem*TrianglesFunc.nsimplex:(ielem+1)*TrianglesFunc.nsimplex] = quantity_to_plot[ielem]
            else:
                # IF QUANTITY IS DEFINED ON NODES
                Uplot[ielem*nsize:(ielem+1)*nsize] = np.dot(BasesTri.T, quantity_to_plot[mesh.elements[ielem,:]]).flatten()


        tmesh = Mesh()
        tmesh.element_type = "tri"
        tmesh.elements = Tplot
        tmesh.points = Xplot
        tmesh.quantity = Uplot
        tmesh.nelem = nelem
        tmesh.nnode = nnode
        tmesh.nsize = nsize
        tmesh.bases_1 = BasesOneD
        tmesh.bases_2 = BasesTri.T

        if plot_edges:
            tmesh.x_edges = x_edges
            tmesh.y_edges = y_edges
            tmesh.z_edges = z_edges
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
        from Florence.MeshGeneration.NodeArrangement import NodeArrangementQuad
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
        nmax = int(np.max(smesh.elements)+1)
        smesh.points = mesh.points[:nmax,:]
        smesh.GetEdgesQuad()
        edge_elements = smesh.GetElementsEdgeNumberingQuad()


        # GET EDGE ORDERING IN THE REFERENCE ELEMENT
        reference_edges = NodeArrangementQuad(CActual)[0]
        reference_edges = np.concatenate((reference_edges,reference_edges[:,1,None]),axis=1)
        reference_edges = np.delete(reference_edges,1,1)

        pdim = mesh.points.shape[1]
        # GET EULERIAN GEOMETRY
        if TotalDisp.ndim==3:
            vpoints = mesh.points + TotalDisp[:,:pdim,-1]
        else:
            vpoints = mesh.points + TotalDisp[:,:pdim]


        # GET X & Y OF CURVED EDGES
        if plot_edges:
            x_edges = np.zeros((C+2,smesh.all_edges.shape[0]))
            y_edges = np.zeros((C+2,smesh.all_edges.shape[0]))
            z_edges = np.zeros((C+2,smesh.all_edges.shape[0]))

            for iedge in range(smesh.all_edges.shape[0]):
                ielem = edge_elements[iedge,0]
                edge = mesh.elements[ielem,reference_edges[edge_elements[iedge,1],:]]
                coord_edge = vpoints[edge,:]
                if pdim == 2:
                    x_edges[:,iedge], y_edges[:,iedge] = np.dot(coord_edge.T,BasesOneD)
                else:
                    x_edges[:,iedge], y_edges[:,iedge], z_edges[:,iedge] = np.dot(coord_edge.T,BasesOneD)


        nnode = nsize*mesh.nelem
        nelem = Triangles.shape[0]*mesh.nelem

        Xplot = np.zeros((nnode,pdim),dtype=np.float64)
        Tplot = np.zeros((nelem,3),dtype=np.int64)
        Uplot = np.zeros(nnode,dtype=np.float64)
        # if plot_on_faces and QuantityToPlot is not None:
            # Uplot = np.zeros(nelem,dtype=np.float64)

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
            if plot_on_faces:
                # Uplot[ielem*nsize:(ielem+1)*nsize] = quantity_to_plot[ielem]
                Uplot[ielem*TrianglesFunc.nsimplex:(ielem+1)*TrianglesFunc.nsimplex] = quantity_to_plot[ielem]
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
            tmesh.z_edges = z_edges
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
        from Florence.MeshGeneration.NodeArrangement import NodeArrangementTri
        from Florence.FunctionSpace import Tri
        from Florence.FunctionSpace.OneDimensional.Line import LagrangeGaussLobatto, Lagrange

        from scipy.spatial import Delaunay

        assert mesh.element_type == "tet"

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
                EvalOpt=1,equally_spaced=EquallySpacedPoints,Transform=1)[0]

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
                    # for ielem in range(nface):
                    #     Uplot[ielem*nsize:(ielem+1)*nsize] = quantity_to_plot[ielem]
                    # NEW APPROACH FOR CELL DATA - CHECK
                    Uplot = np.zeros(nelem,dtype=np.float64)
                    for ielem in range(nface):
                        Uplot[ielem*TrianglesFunc.nsimplex:(ielem+1)*TrianglesFunc.nsimplex] = quantity_to_plot[ielem]
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

        if QuantityToPlot is not None:
            tmesh.quantity = Uplot

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
        from Florence.MeshGeneration.NodeArrangement import NodeArrangementQuad
        from Florence.FunctionSpace import Quad
        from Florence.FunctionSpace.OneDimensional.Line import LagrangeGaussLobatto, Lagrange

        from copy import deepcopy
        from scipy.spatial import Delaunay

        assert mesh.element_type == "hex"

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
                    # for ielem in range(nface):
                    #     Uplot[ielem*nsize:(ielem+1)*nsize] = quantity_to_plot[ielem]
                    # NEW APPROACH FOR CELL DATA - CHECK
                    Uplot = np.zeros(nelem,dtype=np.float64)
                    for ielem in range(nface):
                        Uplot[ielem*TrianglesFunc.nsimplex:(ielem+1)*TrianglesFunc.nsimplex] = quantity_to_plot[ielem]
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
        vpoints = np.copy(mesh.points)
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