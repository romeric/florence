import numpy as np 
import numpy.linalg as la
from Florence.FiniteElements.LocalAssembly.KinematicMeasures import *
from .PostProcess import *
from copy import copy




class ErrorNorms(PostProcess):

    def __init__(self,post_process=None,functors=None):
        
        # COPY THE WHOLE OBJECT / SHALLOW COPY
        self.ndim = post_process.ndim
        self.formulation = post_process.formulation
        self.domain_bases = post_process.domain_bases
        self.postdomain_bases = post_process.postdomain_bases
        self.boundary_bases = post_process.boundary_bases
        self.analysis_type = post_process.analysis_type
        self.analysis_nature = post_process.analysis_nature
        self.material_type = post_process.material_type
        self.functors = functors

    def SetExactSolution(self,functors):
        self.functors = functor

    def ExactSolution(self,func,*args,**kwargs):
        """Provide analytical/exact/closed-form solution through func"""
        return func(*args,**kwargs)


    def InterpolationBasedNormNonlinear(self,mesh,solution):
        """Compute error norms in convex multi-variable extended kinematic set {F,H,J,D0,d} 
            and the associated work conjugates - Based on work-conjugates of model 106"""

        if self.functors is None:
            raise ValueError("An exact solution functor/lambda function not provided")

        func = self.functors
        TotalDisp = solution

        Domain = self.domain_bases
        nodeperelem = mesh.elements.shape[1] 
        ndim = mesh.points.shape[1]

        if self.material is not None:
            mu1 = self.material.mu1
            mu2 = self.material.mu2
            lamb = self.material.lamb
            eps_1 = self.material.eps_1
            eps_2 = self.material.eps_2
        else:
            mu1, mu2, lamb = 1., 0.5, 1 
            eps_1, eps_2 = 4., 4.

        I = np.zeros((Domain.AllGauss.shape[0],ndim,ndim))
        for i in range(Domain.AllGauss.shape[0]):
            I[i,:,:] = np.eye(ndim,ndim)    

        # L2_normX = 0; L2_denormX = 0
        # L2_normE = 0.; L2_denormE = 0.
        L2_normx = 0.; L2_denormx = 0.
        L2_normF = 0.; L2_denormF = 0.
        L2_normH = 0.; L2_denormH = 0.
        L2_normJ = 0.; L2_denormJ = 0.
        L2_normPhi = 0.; L2_denormPhi = 0.
        L2_normD = 0.; L2_denormD = 0.
        L2_normd = 0.; L2_denormd = 0.
        L2_normSF = 0.; L2_denormSF = 0.
        L2_normSH = 0.; L2_denormSH = 0.
        L2_normSJ = 0.; L2_denormSJ = 0.
        L2_normSD = 0.; L2_denormSD = 0.
        L2_normSd = 0.; L2_denormSd = 0.

        for elem in range(mesh.nelem):

            # GET ELEMENTAL COORDINATES
            LagrangeElemCoords = mesh.points[mesh.elements[elem,:],:]
            vpoints = mesh.points + TotalDisp[:,:,-1]
            EulerElemCoords = vpoints[mesh.elements[elem,:],:]

            # GET PHYSICAL ELEMENTAL GAUSS POINTS - USING FE BASES INTERPLOLATE COORDINATES AT GAUSS POINTS
            LagrangeElemGaussCoords = np.einsum('ij,ik',Domain.Bases,LagrangeElemCoords)
            EulerElemGaussCoords    = np.einsum('ij,ik',Domain.Bases,EulerElemCoords)

            # EVALAUTE STRAINS AT ALL GAUSS POINTS
            # ParentGradientX = np.einsum('ijk,jl->kil',Domain.Jm,LagrangeElemCoords)
            # ParentGradientx = np.einsum('ijk,jl->kil',Domain.Jm,EulerElemCoords)
            # MaterialGradient = np.einsum('ijk,kli->ijl',la.inv(ParentGradientX),Domain.Jm)
            # F = np.einsum('ij,kli->kjl', EulerElemCoords, MaterialGradient)
            # StrainTensors = KinematicMeasures(F, self.analysis_nature)


            # GEOMETRY ERROR
            GeomNodes = func.Exact_x(LagrangeElemCoords)
            ExactGeom = func.Exact_x(LagrangeElemGaussCoords)
            NumericalGeom = np.einsum('ij,ik->kj',GeomNodes,Domain.Bases)

            # DEFORMATION GRADIENT ERROR
            ExactEulerxNodes = GeomNodes + LagrangeElemCoords
            ParentGradientX = np.einsum('ijk,jl->kil',Domain.Jm,LagrangeElemCoords)
            MaterialGradient = np.einsum('ijk,kli->ijl',la.inv(ParentGradientX),Domain.Jm)
            F = np.einsum('ij,kli->kjl', ExactEulerxNodes, MaterialGradient)
            F_exact = func.Exact_F(LagrangeElemGaussCoords)
 
            # JACOBIAN ERROR
            J = np.linalg.det(F)
            J_exact = np.linalg.det(F_exact)

            # COFACTOR ERROR
            H = np.einsum('i,ijk->ijk',J,np.einsum('ikj',np.linalg.inv(F)))
            H_exact = np.einsum('i,ijk->ijk',J_exact,np.einsum('ikj',np.linalg.inv(F_exact)))

            # ELECTRIC POTENTIAL
            PotenialNodes = func.Exact_Phi(LagrangeElemCoords)
            ExactPotential = func.Exact_Phi(LagrangeElemGaussCoords)
            NumericalPotential = np.einsum('i,ij',PotenialNodes,Domain.Bases)

            # ELECTRIC FIELD ERROR
            ENodes = func.Exact_E(LagrangeElemCoords)
            E_exact = func.Exact_E(LagrangeElemGaussCoords)
            E = np.einsum('ij,ik->kj',ENodes,Domain.Bases)

            # ELECTRIC DISPLACEMENT ERROR
            C = np.einsum('ikj,ikl->ijl',F,F)
            C_exact = np.einsum('ikj,ikl->ijl',F,F)
            D = np.einsum('ijk,ik->ij',np.linalg.inv(eps_1*I+eps_2*C),E)
            D_exact = np.einsum('ijk,ik->ij',np.linalg.inv(eps_1*I+eps_2*C_exact),E_exact)

            # d = FD0 ERROR
            d = np.einsum('ijk,ik->ij',F,D)
            d_exact = np.einsum('ijk,ik->ij',F_exact,D_exact)


            # KINETICS - WORK CONJUGATES
            SF = 2.*mu1*F
            SF_exact = 2.*mu1*F_exact

            SH = 2.*mu2*H
            SH_exact = 2.*mu2*H_exact

            SJ = -2.*(mu1+2*mu2)*1./J + lamb*(J-1) - 1/eps_1/J**2*np.einsum('ij,ij->i',d,d)
            SJ_exact = -2.*(mu1+2*mu2)*1./J_exact + lamb*(J_exact-1) - 1/eps_1/J_exact**2*np.einsum('ij,ij->i',d_exact,d_exact)

            SD = 1./eps_1*D
            SD_exact = 1./eps_1*D_exact

            Sd = 1./eps_1*d
            Sd_exact = 1./eps_1*d_exact



            for counter in range(0,Domain.AllGauss.shape[0]):
            # for counter in range(1):    

                # L2_normx += (ExactGeom - NumericalGeom)**2*Domain.AllGauss[counter,0]
                # L2_denormx += (ExactGeom)**2*Domain.AllGauss[counter,0]

                # L2_normF += (F_exact - F)**2*Domain.AllGauss[counter,0]
                # L2_denormF += (F_exact)**2*Domain.AllGauss[counter,0]

                # L2_normH += (H_exact - H)**2*Domain.AllGauss[counter,0]
                # L2_denormH += (H_exact)**2*Domain.AllGauss[counter,0]

                # L2_normJ += (J_exact - J)**2*Domain.AllGauss[counter,0]
                # L2_denormJ += (J_exact)**2*Domain.AllGauss[counter,0]

                # L2_normPhi += (ExactPotential - NumericalPotential)**2*Domain.AllGauss[counter,0]
                # L2_denormPhi += (ExactPotential)**2*Domain.AllGauss[counter,0]

                # L2_normD += (D_exact - D)**2*Domain.AllGauss[counter,0]
                # L2_denormD += (D_exact)**2*Domain.AllGauss[counter,0]

                # L2_normd += (d_exact - d)**2*Domain.AllGauss[counter,0]
                # L2_denormd += (d_exact)**2*Domain.AllGauss[counter,0]

                # L2_normSF += (SF_exact - SF)**2*Domain.AllGauss[counter,0]
                # L2_denormSF += (SF_exact)**2*Domain.AllGauss[counter,0]

                # L2_normSH += (SH_exact - SH)**2*Domain.AllGauss[counter,0]
                # L2_denormSH += (SH_exact)**2*Domain.AllGauss[counter,0]

                # L2_normSJ += (SJ_exact - SJ)**2*Domain.AllGauss[counter,0]
                # L2_denormSJ += (SJ_exact)**2*Domain.AllGauss[counter,0]

                # L2_normSD += (SD_exact - SD)**2*Domain.AllGauss[counter,0]
                # L2_denormSD += SD_exact**2*Domain.AllGauss[counter,0]

                # L2_normSd += (Sd_exact - Sd)**2*Domain.AllGauss[counter,0]
                # L2_denormSd += Sd_exact**2*Domain.AllGauss[counter,0]

                ###########################################################

                L2_normx += (ExactGeom[counter,:] - NumericalGeom[counter,:])**2*Domain.AllGauss[counter,0]
                L2_denormx += (ExactGeom[counter,:])**2*Domain.AllGauss[counter,0]

                L2_normF += (F_exact[counter,:,:] - F[counter,:,:])**2*Domain.AllGauss[counter,0]
                L2_denormF += (F_exact[counter,:,:])**2*Domain.AllGauss[counter,0]

                L2_normH += (H_exact[counter,:,:] - H[counter,:,:])**2*Domain.AllGauss[counter,0]
                L2_denormH += (H_exact[counter,:,:])**2*Domain.AllGauss[counter,0]

                L2_normJ += (J_exact[counter] - J[counter])**2*Domain.AllGauss[counter,0]
                L2_denormJ += (J_exact[counter])**2*Domain.AllGauss[counter,0]

                L2_normPhi += (ExactPotential[counter] - NumericalPotential[counter])**2*Domain.AllGauss[counter,0]
                L2_denormPhi += (ExactPotential[counter])**2*Domain.AllGauss[counter,0]

                L2_normD += (D_exact[counter,:] - D[counter,:])**2*Domain.AllGauss[counter,0]
                L2_denormD += (D_exact[counter,:])**2*Domain.AllGauss[counter,0]

                L2_normd += (d_exact[counter,:] - d[counter,:])**2*Domain.AllGauss[counter,0]
                L2_denormd += (d_exact[counter,:])**2*Domain.AllGauss[counter,0]

                L2_normSF += (SF_exact[counter,:,:] - SF[counter,:,:])**2*Domain.AllGauss[counter,0]
                L2_denormSF += (SF_exact[counter,:,:])**2*Domain.AllGauss[counter,0]

                L2_normSH += (SH_exact[counter,:,:] - SH[counter,:,:])**2*Domain.AllGauss[counter,0]
                L2_denormSH += (SH_exact[counter,:,:])**2*Domain.AllGauss[counter,0]

                L2_normSJ += (SJ_exact[counter] - SJ[counter])**2*Domain.AllGauss[counter,0]
                L2_denormSJ += (SJ_exact[counter])**2*Domain.AllGauss[counter,0]

                L2_normSD += (SD_exact[counter,:] - SD[counter,:])**2*Domain.AllGauss[counter,0]
                L2_denormSD += SD_exact[counter,:]**2*Domain.AllGauss[counter,0]

                L2_normSd += (Sd_exact[counter,:] - Sd[counter,:])**2*Domain.AllGauss[counter,0]
                L2_denormSd += Sd_exact[counter,:]**2*Domain.AllGauss[counter,0]

                ###########################################################

            # L2_normSH = np.einsum('ijk,i',(SH_exact - SH)**2,Domain.AllGauss[:,0])
            # L2_denormSH = np.einsum('ijk,i',SH_exact**2,Domain.AllGauss[counter,0])

            # L2_normSJ = np.einsum('ijk,i',(SF_exact - SF)**2,Domain.AllGauss[:,0])
            # L2_denormSJ = np.einsum('ijk,i',SF_exact**2,Domain.AllGauss[:,0])

            # L2_normSD = np.einsum('ij,i',(SD_exact - SD)**2,Domain.AllGauss[:,0])
            # L2_denormSD = np.einsum('ij,i',SD_exact**2,Domain.AllGauss[:,0])

            # L2_normSd = np.sum(np.einsum('ij,k',(Sd_exact - Sd)**2,Domain.AllGauss[:,0]),axis=2)
            # L2_denormSd = np.sum(np.einsum('ij,k',Sd_exact**2,Domain.AllGauss[:,0]),axis=2)


        # L2NormX = np.sqrt(np.sum(L2_normX))/np.sqrt(np.sum(L2_denormX))
        # L2Normx = np.sqrt(np.sum(L2_normx))/np.sqrt(np.sum(L2_denormx))

        L2Normx = np.sum(L2_normx)/np.sum(L2_denormx)
        L2NormF = np.sum(L2_normF)/np.sum(L2_denormF)
        L2NormH = np.sum(L2_normH)/np.sum(L2_denormH)
        L2NormJ = np.sum(L2_normJ)/np.sum(L2_denormJ)
        L2NormPhi = np.sum(L2_normPhi)/np.sum(L2_denormPhi)
        L2NormD = np.sum(L2_normD)/np.sum(L2_denormD)
        L2Normd = np.sum(L2_normd)/np.sum(L2_denormd)
        L2NormSF = np.sum(L2_normSF)/np.sum(L2_denormSF)
        L2NormSH = np.sum(L2_normSH)/np.sum(L2_denormSH)
        L2NormSJ = np.sum(L2_normSJ)/np.sum(L2_denormSJ)
        L2NormSD = np.sum(L2_normSD)/np.sum(L2_denormSD)
        L2NormSd = np.sum(L2_normSd)/np.sum(L2_denormSd)


        return L2Normx, L2NormF, L2NormH, L2NormJ, L2NormPhi, L2NormD, L2Normd, L2NormSF, L2NormSH, L2NormSJ, L2NormSD, L2NormSd





    def InterpolationBasedNormNonlinearObjective(self,mesh,solution):
        """Compute error norms in convex multi-variable extended kinematic set {C,G,C,D0} 
            and the associated work conjugates - Based on work-conjugates of model 106"""

        if self.functors is None:
            raise ValueError("An exact solution functor/lambda function not provided")

        func = self.functors
        TotalDisp = solution

        Domain = self.domain_bases
        nodeperelem = mesh.elements.shape[1] 
        ndim = mesh.points.shape[1]

        if self.material is not None:
            mu1 = self.material.mu1
            mu2 = self.material.mu2
            lamb = self.material.lamb
            eps_1 = self.material.eps_1
            eps_2 = self.material.eps_2
        else:
            mu1, mu2, lamb = 1., 0.5, 1 
            eps_1, eps_2 = 4., 4.

        alpha, beta = 0.2, 0.2

        I = np.zeros((Domain.AllGauss.shape[0],ndim,ndim))
        for i in range(Domain.AllGauss.shape[0]):
            I[i,:,:] = np.eye(ndim,ndim)    


        L2_normx = 0.; L2_denormx = 0.
        L2_normC = 0.; L2_denormC = 0.
        L2_normG = 0.; L2_denormG = 0.
        L2_normdetC = 0.; L2_denormdetC = 0.
        L2_normPhi = 0.; L2_denormPhi = 0.
        L2_normD0 = 0.; L2_denormD0 = 0.
        L2_normSC = 0.; L2_denormSC = 0.
        L2_normSG = 0.; L2_denormSG = 0.
        L2_normSdetC = 0.; L2_denormSdetC = 0.
        L2_normSD0 = 0.; L2_denormSD0 = 0.

        for elem in range(mesh.nelem):

            # GET ELEMENTAL COORDINATES
            LagrangeElemCoords = mesh.points[mesh.elements[elem,:],:]
            vpoints = mesh.points + TotalDisp[:,:,-1]
            EulerElemCoords = vpoints[mesh.elements[elem,:],:]

            # GET PHYSICAL ELEMENTAL GAUSS POINTS - USING FE BASES INTERPLOLATE COORDINATES AT GAUSS POINTS
            LagrangeElemGaussCoords = np.einsum('ij,ik',Domain.Bases,LagrangeElemCoords)
            EulerElemGaussCoords    = np.einsum('ij,ik',Domain.Bases,EulerElemCoords)

            # GEOMETRY ERROR
            GeomNodes = func.Exact_x(LagrangeElemCoords)
            ExactGeom = func.Exact_x(LagrangeElemGaussCoords)
            NumericalGeom = np.einsum('ij,ik->kj',GeomNodes,Domain.Bases)

            # DEFORMATION GRADIENT ERROR
            ExactEulerxNodes = GeomNodes + LagrangeElemCoords
            ParentGradientX = np.einsum('ijk,jl->kil',Domain.Jm,LagrangeElemCoords)
            MaterialGradient = np.einsum('ijk,kli->ijl',la.inv(ParentGradientX),Domain.Jm)
            F = np.einsum('ij,kli->kjl', ExactEulerxNodes, MaterialGradient)
            F_exact = func.Exact_F(LagrangeElemGaussCoords)
 
            # RIGHT CAUCHY GREEN
            C = np.einsum('ikj,ikl->ijl',F,F)
            C_exact = np.einsum('ikj,ikl->ijl',F_exact,F_exact)
            
            # JACOBIAN ERROR
            detC = np.abs(np.linalg.det(C))
            detC_exact = np.abs(np.linalg.det(C_exact))

            # COFACTOR ERROR
            G = np.einsum('i,ijk->ijk',detC,np.einsum('ikj',np.linalg.inv(C)))
            G_exact = np.einsum('i,ijk->ijk',detC_exact,np.einsum('ikj',np.linalg.inv(C_exact)))

            # ELECTRIC POTENTIAL ERROR
            PotenialNodes = func.Exact_Phi(LagrangeElemCoords)
            ExactPotential = func.Exact_Phi(LagrangeElemGaussCoords)
            NumericalPotential = np.einsum('i,ij',PotenialNodes,Domain.Bases)

            # ELECTRIC FIELD ERROR
            ENodes = func.Exact_E(LagrangeElemCoords)
            E_exact = func.Exact_E(LagrangeElemGaussCoords)
            E = np.einsum('ij,ik->kj',ENodes,Domain.Bases)

            # ELECTRIC DISPLACEMENT ERROR
            D = np.einsum('ijk,ik->ij',np.linalg.inv(eps_1*I+eps_2*C),E)
            D_exact = np.einsum('ijk,ik->ij',np.linalg.inv(eps_1*I+eps_2*C_exact),E_exact)

            # d = FD0 ERROR
            d = np.einsum('ijk,ik->ij',F,D)
            d_exact = np.einsum('ijk,ik->ij',F_exact,D_exact)


            # KINETICS - WORK CONJUGATES
            SC = 2.*mu1*I + 4.*mu1*alpha*np.einsum('ijk,ijk',C,I)*I
            SC_exact = 2.*mu1*I + 4.*mu1*alpha*np.einsum('ijk,ijk',C_exact,I)*I

            SG = 2.*mu2*I + 4.*mu2*beta*np.einsum('ijk,ijk',G,I)*I
            SG_exact = 2.*mu2*I + 4.*mu2*beta*np.einsum('ijk,ijk',G_exact,I)*I

            SdetC = -2.*(mu1+2*mu2)*1./detC + lamb*(1.-1./detC) - \
                1/2./eps_1/detC/np.sqrt(detC)*np.einsum('ij,ij->i',d,d)
            SdetC_exact = -2.*(mu1+2*mu2)*1./detC_exact + lamb*(1.-1./detC_exact) - \
                1/2./eps_1/detC_exact/np.sqrt(detC_exact)*np.einsum('ij,ij->i',d_exact,d_exact)

            SD = 1./eps_1*D + np.einsum('i,ij->ij',1./eps_2/np.sqrt(detC),np.einsum('ijk,ik->ij',C,D))
            SD_exact = 1./eps_1*D_exact + np.einsum('i,ij->ij',1./eps_2/np.sqrt(detC_exact),np.einsum('ijk,ik->ij',C_exact,D_exact))


            for counter in range(0,Domain.AllGauss.shape[0]):
            # for counter in range(1):
  
                ###########################################################

                L2_normx += (ExactGeom[counter,:] - NumericalGeom[counter,:])**2*Domain.AllGauss[counter,0]
                L2_denormx += (ExactGeom[counter,:])**2*Domain.AllGauss[counter,0]

                L2_normC += (C_exact[counter,:,:] - C[counter,:,:])**2*Domain.AllGauss[counter,0]
                L2_denormC += (C_exact[counter,:,:])**2*Domain.AllGauss[counter,0]

                L2_normG += (G_exact[counter,:,:] - G[counter,:,:])**2*Domain.AllGauss[counter,0]
                L2_denormG += (G_exact[counter,:,:])**2*Domain.AllGauss[counter,0]

                L2_normdetC += (detC_exact[counter] - detC[counter])**2*Domain.AllGauss[counter,0]
                L2_denormdetC += (detC_exact[counter])**2*Domain.AllGauss[counter,0]

                L2_normPhi += (ExactPotential[counter] - NumericalPotential[counter])**2*Domain.AllGauss[counter,0]
                L2_denormPhi += (ExactPotential[counter])**2*Domain.AllGauss[counter,0]

                L2_normD0 += (D_exact[counter,:] - D[counter,:])**2*Domain.AllGauss[counter,0]
                L2_denormD0 += (D_exact[counter,:])**2*Domain.AllGauss[counter,0]

                L2_normSC += (SC_exact[counter,:,:] - SC[counter,:,:])**2*Domain.AllGauss[counter,0]
                L2_denormSC += (SC_exact[counter,:,:])**2*Domain.AllGauss[counter,0]

                L2_normSG += (SG_exact[counter,:,:] - SG[counter,:,:])**2*Domain.AllGauss[counter,0]
                L2_denormSG += (SG_exact[counter,:,:])**2*Domain.AllGauss[counter,0]

                L2_normSdetC += (SdetC_exact[counter] - SdetC[counter])**2*Domain.AllGauss[counter,0]
                L2_denormSdetC += (SdetC_exact[counter])**2*Domain.AllGauss[counter,0]

                L2_normSD0 += (SD_exact[counter,:] - SD[counter,:])**2*Domain.AllGauss[counter,0]
                L2_denormSD0 += SD_exact[counter,:]**2*Domain.AllGauss[counter,0]

                ###########################################################

            # L2_normSd = np.sum(np.einsum('ij,k',(Sd_exact - Sd)**2,Domain.AllGauss[:,0]),axis=2)
            # L2_denormSd = np.sum(np.einsum('ij,k',Sd_exact**2,Domain.AllGauss[:,0]),axis=2)


        L2Normx = np.sum(L2_normx)/np.sum(L2_denormx)
        L2NormC = np.sum(L2_normC)/np.sum(L2_denormC)
        L2NormG = np.sum(L2_normG)/np.sum(L2_denormG)
        L2NormdetC = np.sum(L2_normdetC)/np.sum(L2_denormdetC)
        L2NormPhi = np.sum(L2_normPhi)/np.sum(L2_denormPhi)
        L2NormD0 = np.sum(L2_normD0)/np.sum(L2_denormD0)
        L2NormSC = np.sum(L2_normSC)/np.sum(L2_denormSC)
        L2NormSG = np.sum(L2_normSG)/np.sum(L2_denormSG)
        L2NormSdetC = np.sum(L2_normSdetC)/np.sum(L2_denormSdetC)
        L2NormSD0 = np.sum(L2_normSD0)/np.sum(L2_denormSD0)


        return L2Normx, L2NormC, L2NormG, L2NormdetC, L2NormPhi, L2NormD0, L2NormSC, L2NormSG, L2NormSdetC, L2NormSD0









## CHEAP NORM FOR CURVED MESH GENERATOR
def InterpolationBasedNorm(MainData,mesh,TotalDisp):

    if MainData.AssemblyParameters.FailedToConverge is True:
        MainData.L2NormX = np.NAN
        MainData.L2Normx = np.NAN
        MainData.DoF = mesh.points.shape[0]*MainData.nvar
        MainData.NELEM = mesh.elements.shape[0]

        return


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
    print(L2NormX, L2Normx)
    # print np.linalg.norm(L2NormX)

    MainData.L2NormX = L2NormX
    MainData.L2Normx = L2Normx
    MainData.DoF = mesh.points.shape[0]*MainData.nvar
    MainData.NELEM = mesh.elements.shape[0]






##
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
    print(L2Norm, EnergyNorm)
    return L2Norm, EnergyNorm