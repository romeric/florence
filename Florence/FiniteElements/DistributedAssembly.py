#!/usr/bin/env python
import os, imp, sys, gc
from sys import exit
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from scipy.io import savemat
from ElementalMatrices.GetElementalMatrices import *
import Florence.Formulations.DisplacementApproach as DB
from Florence.ParallelProcessing.mpi4py_map import map as parmap
from mpi4py import MPI
comm = MPI.COMM_WORLD


#-------------- ASSEMBLY ROUTINE FOR RELATIVELY LARGER MATRICES ( 1e06 < NELEM < 1e07 3D)------------------------#
#----------------------------------------------------------------------------------------------------------------#

def DistributedAssembly(tmp_dir):

    from Florence.Utils import par_pickle

    MainData, mesh, material, Eulerx, TotalPot = par_pickle(tmp_dir)

    # BROADCAST FROM RANK 0 TO EVERYBODY
    # comm.Bcast([MainData])  
    C = MainData.C
    nvar = MainData.nvar
    ndim = MainData.ndim

    nelem = mesh.nelem
    nodeperelem = mesh.elements.shape[1]

    # THE I & J VECTORS OF LOCAL STIFFNESS MATRIX DO NOT CHANGE, HENCE COMPUTE THEM ONCE
    I_stiff_elem = np.repeat(np.arange(0,nvar*nodeperelem),nvar*nodeperelem,axis=0)
    J_stiff_elem = np.tile(np.arange(0,nvar*nodeperelem),nvar*nodeperelem)

    I_mass=[];J_mass=[];V_mass=[]; I_mass_elem = []; J_mass_elem = []
    if MainData.Analysis !='Static':
        # ALLOCATE VECTORS FOR SPARSE ASSEMBLY OF MASS MATRIX
        I_mass=np.zeros((nvar*nodeperelem)**2*mesh.elements.shape[0],dtype=np.int64)
        J_mass=np.zeros((nvar*nodeperelem)**2*mesh.elements.shape[0],dtype=np.int64)
        V_mass=np.zeros((nvar*nodeperelem)**2*mesh.elements.shape[0],dtype=np.float64)

        # THE I & J VECTORS OF LOCAL MASS MATRIX DO NOT CHANGE, HENCE COMPUTE THEM ONCE
        I_mass_elem = np.repeat(np.arange(0,nvar*nodeperelem),nvar*nodeperelem,axis=0)
        J_mass_elem = np.tile(np.arange(0,nvar*nodeperelem),nvar*nodeperelem)

    # ASSIGN OTHER NECESSARY MATRICES
    full_current_row_stiff = []; full_current_column_stiff = []; coeff_stiff = [] 
    full_current_row_mass = []; full_current_column_mass = []; coeff_mass = []
    mass = []


    MainData.GeometryUpdate = False
    MainData.ConstitutiveStiffnessIntegrand = DB.ConstitutiveStiffnessIntegrand
    MainData.GeometricStiffnessIntegrand = DB.GeometricStiffnessIntegrand
    MainData.MassIntegrand =  DB.MassIntegrand
    MainData.Prestress = False
    MainData.__NO_DEBUG__ = True

    # print parmap(lambda x, y: x**y, [1,2,3,4], 2) # square the sequence
    ParallelTuple = parmap(DistributedMatrices,np.arange(0,nelem),MainData,mesh,material,Eulerx,I_stiff_elem,J_stiff_elem,I_mass_elem,J_mass_elem)
    
    # print ParallelTuple[9][0].shape
    # print len(ParallelTuple[0])

    # COMPUATE ALL LOCAL ELEMENTAL MATRICES (STIFFNESS, MASS, INTERNAL & EXTERNAL TRACTION FORCES )
    # ParallelTuple = parmap(GetElementalMatrices,np.arange(0,nelem),MainData,mesh.elements,mesh.points,nodeperelem,
        # Eulerx,TotalPot,I_stiff_elem,J_stiff_elem,I_mass_elem,J_mass_elem)

    # exit()
    if comm.rank == 0:

         # ALLOCATE VECTORS FOR SPARSE ASSEMBLY OF STIFFNESS MATRIX
        I_stiffness=np.zeros((nvar*nodeperelem)**2*nelem,dtype=np.int32)
        J_stiffness=np.zeros((nvar*nodeperelem)**2*nelem,dtype=np.int32)
        V_stiffness=np.zeros((nvar*nodeperelem)**2*nelem,dtype=np.float32)


        # ALLOCATE RHS VECTORS
        F = np.zeros((mesh.points.shape[0]*nvar,1)) 
        T =  np.zeros((mesh.points.shape[0]*nvar,1)) 

        for elem in range(nelem):

            # UNPACK PARALLEL TUPLE VALUES
            full_current_row_stiff = ParallelTuple[elem][0]; full_current_column_stiff = ParallelTuple[elem][1]
            coeff_stiff = ParallelTuple[elem][2]; t = ParallelTuple[elem][3]; f = ParallelTuple[elem][4]
            full_current_row_mass = ParallelTuple[elem][5]; full_current_column_mass = ParallelTuple[elem][6]; coeff_mass = ParallelTuple[elem][6]

            I_stiffness[(nvar*nodeperelem)**2*elem:(nvar*nodeperelem)**2*(elem+1)] = full_current_row_stiff
            J_stiffness[(nvar*nodeperelem)**2*elem:(nvar*nodeperelem)**2*(elem+1)] = full_current_column_stiff
            V_stiffness[(nvar*nodeperelem)**2*elem:(nvar*nodeperelem)**2*(elem+1)] = coeff_stiff

            if MainData.Analysis != 'Static':
                # SPARSE ASSEMBLY - MASS MATRIX
                I_mass, J_mass, V_mass = SparseAssembly_Step_2(I_mass,J_mass,V_mass,full_current_row_mass,full_current_column_mass,coeff_mass,
                    nvar,nodeperelem,elem)

            # INTERNAL TRACTION FORCE ASSEMBLY
            for iterator in range(0,nvar):
                    T[mesh.elements[elem,:]*nvar+iterator,0]+=t[iterator::nvar,0]

        # CALL BUILT-IN SPARSE ASSEMBLER 
        stiffness = coo_matrix((V_stiffness,(I_stiffness,J_stiffness)),
            shape=((nvar*mesh.points.shape[0],nvar*mesh.points.shape[0])),dtype=np.float64).tocsc()

        # GET STORAGE/MEMORY DETAILS
        MainData.spmat = stiffness.data.nbytes/1024./1024.
        MainData.ijv = (I_stiffness.nbytes + J_stiffness.nbytes + V_stiffness.nbytes)/1024./1024.
        del I_stiffness, J_stiffness, V_stiffness
        gc.collect()

        if MainData.Analysis != 'Static':
            # CALL BUILT-IN SPARSE ASSEMBLER
            mass = coo_matrix((V_mass,(I_mass,J_mass)),shape=((nvar*mesh.points.shape[0],nvar*mesh.points.shape[0]))).tocsc()


    # return stiffness, T, F, mass
        if not mass:
            mass = 0

    # print os.path.join(tmp_dir,"results.mat")
    # exit()
        savemat(os.path.join(tmp_dir,"results.mat"),{'stiffness':stiffness,'T':T,'F':F,'mass':mass},do_compression=True)


if __name__ == "__main__":

    # prints DistributedAssembly.py
    # print sys.argv[0] 
    tmp_dir = sys.argv[1]
    
    if comm.rank==0:
        print("Launching MPI processes")

    DistributedAssembly(tmp_dir)

