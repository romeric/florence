from __future__ import print_function
import gc, os, sys
from copy import deepcopy
from warnings import warn
from time import time
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix

from .SparseAssembly import SparseAssembly_Step_2
from .SparseAssemblySmall import SparseAssemblySmall
from ._LowLevelAssembly_ import _LowLevelAssembly_, _LowLevelAssemblyExplicit_

import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
from .SparseAssemblyNative import SparseAssemblyNative
from .RHSAssemblyNative import RHSAssemblyNative

# PARALLEL PROCESSING ROUTINES
import multiprocessing
import Florence.ParallelProcessing.parmap as parmap


__all__ = ['Assemble', 'AssembleForces', 'AssembleExplicit']



def Assemble(fem_solver, function_space, formulation, mesh, material, Eulerx, Eulerp):

    if fem_solver.memory_model == "shared" or fem_solver.memory_model is None:
        if not fem_solver.has_low_level_dispatcher:
            if mesh.nelem <= 600000:
                return AssemblySmall(fem_solver, function_space, formulation, mesh, material, Eulerx, Eulerp)
            elif mesh.nelem > 600000:
                print("Larger than memory system. Dask on disk parallel assembly is turned on")
                return OutofCoreAssembly(fem_solver, function_space, formulation, mesh, material, Eulerx, Eulerp)
        else:
            return LowLevelAssembly(fem_solver, function_space, formulation, mesh, material, Eulerx, Eulerp)

    elif fem_solver.memory_model == "distributed":
        # RUN THIS PROGRAM FROM SHELL WITH python RunSession.py INSTEAD
        if not __PARALLEL__:
            warn("parallelisation is going to be turned on")

        import subprocess, os, shutil
        from time import time
        from Florence.Utils import par_unpickle
        from scipy.io import loadmat

        tmp_dir = par_unpickle(function_space,mesh,material,Eulerx,Eulerp)
        pwd = os.path.dirname(os.path.realpath(__file__))
        distributed_caller = os.path.join(pwd,"DistributedAssembly.py")

        t_dassembly = time()
        p = subprocess.Popen("time mpirun -np "+str(MP.cpu_count())+" Florence/FiniteElements/DistributedAssembly.py"+" /home/roman/tmp/",
            cwd="/home/roman/Dropbox/florence/", stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        # p = subprocess.Popen("./mpi_runner.sh", cwd="/home/roman/Dropbox/florence/", shell=True)
        p.wait()
        print('MPI took', time() - t_dassembly, 'seconds for distributed assembly')
        Dict = loadmat(os.path.join(tmp_dir,"results.mat"))

        try:
            shutil.rmtree(tmp_dir)
        except IOError:
            raise IOError("Could not delete the directory")

        return Dict['stiffness'], Dict['T'], Dict['F'], []


def LowLevelAssembly(fem_solver,function_space, formulation, mesh, material, Eulerx, Eulerp):

    if not material.has_low_level_dispatcher:
        raise RuntimeError("Cannot dispatch to low level module, since material {} does not support it".format(type(material).__name__))

    stiffness, T, F, mass = _LowLevelAssembly_(fem_solver, function_space, formulation, mesh, material, Eulerx, Eulerp)
    if isinstance(F,np.ndarray):
        F = F[:,None]
    if mass is not None:
        fem_solver.is_mass_computed = True

    return stiffness, T[:,None], F, mass


def AssemblySmall(fem_solver, function_space, formulation, mesh, material, Eulerx, Eulerp):

    # GET MESH DETAILS
    C = mesh.InferPolynomialDegree() - 1
    nvar = formulation.nvar
    ndim = formulation.ndim
    nelem = mesh.nelem
    nodeperelem = mesh.elements.shape[1]

    # ALLOCATE VECTORS FOR SPARSE ASSEMBLY OF STIFFNESS MATRIX - CHANGE TYPES TO INT64 FOR DoF > 1e09
    I_stiffness=np.zeros(int((nvar*nodeperelem)**2*nelem),dtype=np.int32)
    J_stiffness=np.zeros(int((nvar*nodeperelem)**2*nelem),dtype=np.int32)
    # V_stiffness=np.zeros((nvar*nodeperelem)**2*nelem,dtype=np.float32)
    V_stiffness=np.zeros(int((nvar*nodeperelem)**2*nelem),dtype=np.float64)

    I_mass=[]; J_mass=[]; V_mass=[]
    if fem_solver.analysis_type !='static':
        # ALLOCATE VECTORS FOR SPARSE ASSEMBLY OF MASS MATRIX - CHANGE TYPES TO INT64 FOR DoF > 1e09
        I_mass=np.zeros(int((nvar*nodeperelem)**2*nelem),dtype=np.int32)
        J_mass=np.zeros(int((nvar*nodeperelem)**2*nelem),dtype=np.int32)
        V_mass=np.zeros(int((nvar*nodeperelem)**2*nelem),dtype=np.float64)

    T = np.zeros((mesh.points.shape[0]*nvar,1),np.float64)
    # T = np.zeros((mesh.points.shape[0]*nvar,1),np.float32)

    mass, F = [], []
    if fem_solver.has_moving_boundary:
        F = np.zeros((mesh.points.shape[0]*nvar,1),np.float64)


    if fem_solver.parallel:
        # COMPUATE ALL LOCAL ELEMENTAL MATRICES (STIFFNESS, MASS, INTERNAL & EXTERNAL TRACTION FORCES )
        # ParallelTuple = parmap.map(formulation.GetElementalMatrices,np.arange(0,nelem,dtype=np.int32),
            # function_space, mesh, material, fem_solver, Eulerx, Eulerp)

        ParallelTuple = parmap.map(formulation,np.arange(0,nelem,dtype=np.int32),
            function_space, mesh, material, fem_solver, Eulerx, Eulerp, processes= int(multiprocessing.cpu_count()/2))

    for elem in range(nelem):

        if fem_solver.parallel:
            # UNPACK PARALLEL TUPLE VALUES
            I_stiff_elem = ParallelTuple[elem][0]; J_stiff_elem = ParallelTuple[elem][1]; V_stiff_elem = ParallelTuple[elem][2]
            t = ParallelTuple[elem][3]; f = ParallelTuple[elem][4]
            I_mass_elem = ParallelTuple[elem][5]; J_mass_elem = ParallelTuple[elem][6]; V_mass_elem = ParallelTuple[elem][6]

        else:
            # COMPUATE ALL LOCAL ELEMENTAL MATRICES (STIFFNESS, MASS, INTERNAL & EXTERNAL TRACTION FORCES )
            I_stiff_elem, J_stiff_elem, V_stiff_elem, t, f, \
            I_mass_elem, J_mass_elem, V_mass_elem = formulation.GetElementalMatrices(elem,
                function_space, mesh, material, fem_solver, Eulerx, Eulerp)

        # SPARSE ASSEMBLY - STIFFNESS MATRIX
        SparseAssemblyNative(I_stiff_elem,J_stiff_elem,V_stiff_elem,I_stiffness,J_stiffness,V_stiffness,
            elem,nvar,nodeperelem,mesh.elements)

        if fem_solver.analysis_type != 'static' and fem_solver.is_mass_computed==False:
            # SPARSE ASSEMBLY - MASS MATRIX
            SparseAssemblyNative(I_mass_elem,J_mass_elem,V_mass_elem,I_mass,J_mass,V_mass,
                elem,nvar,nodeperelem,mesh.elements)

        if fem_solver.has_moving_boundary:
            # RHS ASSEMBLY
            # for iterator in range(0,nvar):
            #     F[mesh.elements[elem,:]*nvar+iterator,0]+=f[iterator::nvar,0]
            RHSAssemblyNative(F,f,elem,nvar,nodeperelem,mesh.elements)

        # INTERNAL TRACTION FORCE ASSEMBLY
        # for iterator in range(0,nvar):
            # T[mesh.elements[elem,:]*nvar+iterator,0]+=t[iterator::nvar,0]
        RHSAssemblyNative(T,t,elem,nvar,nodeperelem,mesh.elements)


    if fem_solver.parallel:
        del ParallelTuple
        gc.collect()

    # REALLY DANGEROUS FOR MULTIPHYSICS PROBLEMS - NOTE THAT SCIPY RUNS A PRUNE ANYWAY
    # V_stiffness[np.isclose(V_stiffness,0.)] = 0.

    stiffness = coo_matrix((V_stiffness,(I_stiffness,J_stiffness)),
        shape=((nvar*mesh.points.shape[0],nvar*mesh.points.shape[0])),dtype=np.float64).tocsr()

    # stiffness = csc_matrix((V_stiffness,(I_stiffness,J_stiffness)),
        # shape=((nvar*mesh.points.shape[0],nvar*mesh.points.shape[0])),dtype=np.float32)
    # stiffness = csc_matrix((V_stiffness,(I_stiffness,J_stiffness)),
        # shape=((nvar*mesh.points.shape[0],nvar*mesh.points.shape[0])),dtype=np.float64)
    # stiffness = csr_matrix((V_stiffness,(I_stiffness,J_stiffness)),
        # shape=((nvar*mesh.points.shape[0],nvar*mesh.points.shape[0])),dtype=np.float64)

    # GET STORAGE/MEMORY DETAILS
    fem_solver.spmat = stiffness.data.nbytes/1024./1024.
    fem_solver.ijv = (I_stiffness.nbytes + J_stiffness.nbytes + V_stiffness.nbytes)/1024./1024.

    del I_stiffness, J_stiffness, V_stiffness
    gc.collect()

    if fem_solver.analysis_type != 'static' and fem_solver.is_mass_computed==False:
        mass = csr_matrix((V_mass,(I_mass,J_mass)),shape=((nvar*mesh.points.shape[0],
            nvar*mesh.points.shape[0])),dtype=np.float64)

        fem_solver.is_mass_computed = True

    return stiffness, T, F, mass



#-------------- ASSEMBLY ROUTINE FOR RELATIVELY LARGER MATRICES ( 1e06 < NELEM < 1e07 3D)------------------------#
#----------------------------------------------------------------------------------------------------------------#

def AssemblyLarge(MainData,mesh,material,Eulerx,TotalPot):

    # GET MESH DETAILS
    C = MainData.C
    nvar = MainData.nvar
    ndim = MainData.ndim

    # nelem = mesh.elements.shape[0]
    nelem = mesh.nelem
    nodeperelem = mesh.elements.shape[1]

    from tempfile import mkdtemp

    # WRITE TRIPLETS ON DESK
    pwd = os.path.dirname(os.path.realpath(__file__))
    tmp_dir = mkdtemp()
    I_filename = os.path.join(tmp_dir, 'I_stiffness.dat')
    J_filename = os.path.join(tmp_dir, 'J_stiffness.dat')
    V_filename = os.path.join(tmp_dir, 'V_stiffness.dat')

    # ALLOCATE VECTORS FOR SPARSE ASSEMBLY OF STIFFNESS MATRIX
    I_stiffness = np.memmap(I_filename,dtype=np.int32,mode='w+',shape=((nvar*nodeperelem)**2*nelem,))
    J_stiffness = np.memmap(J_filename,dtype=np.int32,mode='w+',shape=((nvar*nodeperelem)**2*nelem,))
    V_stiffness = np.memmap(V_filename,dtype=np.float32,mode='w+',shape=((nvar*nodeperelem)**2*nelem,))

    # I_stiffness=np.zeros((nvar*nodeperelem)**2*nelem,dtype=np.int64)
    # J_stiffness=np.zeros((nvar*nodeperelem)**2*nelem,dtype=np.int64)
    # V_stiffness=np.zeros((nvar*nodeperelem)**2*nelem,dtype=np.float64)

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


    # ALLOCATE RHS VECTORS
    F = np.zeros((mesh.points.shape[0]*nvar,1))
    T =  np.zeros((mesh.points.shape[0]*nvar,1))
    # ASSIGN OTHER NECESSARY MATRICES
    full_current_row_stiff = []; full_current_column_stiff = []; coeff_stiff = []
    full_current_row_mass = []; full_current_column_mass = []; coeff_mass = []
    mass = []

    if MainData.Parallel:
        # COMPUATE ALL LOCAL ELEMENTAL MATRICES (STIFFNESS, MASS, INTERNAL & EXTERNAL TRACTION FORCES )
        ParallelTuple = parmap.map(GetElementalMatrices,np.arange(0,nelem),MainData,mesh.elements,mesh.points,nodeperelem,
            Eulerx,TotalPot,I_stiff_elem,J_stiff_elem,I_mass_elem,J_mass_elem,pool=MP.Pool(processes=MainData.nCPU))

    for elem in range(nelem):

        if MainData.Parallel:
            # UNPACK PARALLEL TUPLE VALUES
            full_current_row_stiff = ParallelTuple[elem][0]; full_current_column_stiff = ParallelTuple[elem][1]
            coeff_stiff = ParallelTuple[elem][2]; t = ParallelTuple[elem][3]; f = ParallelTuple[elem][4]
            full_current_row_mass = ParallelTuple[elem][5]; full_current_column_mass = ParallelTuple[elem][6]; coeff_mass = ParallelTuple[elem][6]

        else:
            # COMPUATE ALL LOCAL ELEMENTAL MATRICES (STIFFNESS, MASS, INTERNAL & EXTERNAL TRACTION FORCES )
            full_current_row_stiff, full_current_column_stiff, coeff_stiff, t, f, \
            full_current_row_mass, full_current_column_mass, coeff_mass = GetElementalMatrices(elem,
                MainData,mesh.elements,mesh.points,nodeperelem,Eulerx,TotalPot,I_stiff_elem,J_stiff_elem,I_mass_elem,J_mass_elem)

        # SPARSE ASSEMBLY - STIFFNESS MATRIX
        # I_stiffness, J_stiffness, V_stiffness = SparseAssembly_Step_2(I_stiffness,J_stiffness,V_stiffness,
            # full_current_row_stiff,full_current_column_stiff,coeff_stiff,
        #   nvar,nodeperelem,elem)

        I_stiffness[(nvar*nodeperelem)**2*elem:(nvar*nodeperelem)**2*(elem+1)] = full_current_row_stiff
        J_stiffness[(nvar*nodeperelem)**2*elem:(nvar*nodeperelem)**2*(elem+1)] = full_current_column_stiff
        V_stiffness[(nvar*nodeperelem)**2*elem:(nvar*nodeperelem)**2*(elem+1)] = coeff_stiff

        if MainData.Analysis != 'Static':
            # SPARSE ASSEMBLY - MASS MATRIX
            I_mass, J_mass, V_mass = SparseAssembly_Step_2(I_mass,J_mass,V_mass,full_current_row_mass,full_current_column_mass,coeff_mass,
                nvar,nodeperelem,elem)

        if fem_solver.has_moving_boundary:
            # RHS ASSEMBLY
            for iterator in range(0,nvar):
                F[mesh.elements[elem,:]*nvar+iterator,0]+=f[iterator::nvar]
        # INTERNAL TRACTION FORCE ASSEMBLY
        for iterator in range(0,nvar):
                T[mesh.elements[elem,:]*nvar+iterator,0]+=t[iterator::nvar,0]

    # CALL BUILT-IN SPARSE ASSEMBLER
    stiffness = coo_matrix((V_stiffness,(I_stiffness,J_stiffness)),
        shape=((nvar*mesh.points.shape[0],nvar*mesh.points.shape[0])),dtype=np.float64).tocsc()

    # GET STORAGE/MEMORY DETAILS
    fem_solver.spmat = stiffness.data.nbytes/1024./1024.
    fem_solver.ijv = (I_stiffness.nbytes + J_stiffness.nbytes + V_stiffness.nbytes)/1024./1024.
    del I_stiffness, J_stiffness, V_stiffness
    gc.collect()

    if fem_solver.analysis_type != 'static':
        # CALL BUILT-IN SPARSE ASSEMBLER
        mass = coo_matrix((V_mass,(I_mass,J_mass)),shape=((nvar*mesh.points.shape[0],nvar*mesh.points.shape[0]))).tocsc()


    return stiffness, T, F, mass


def OutofCoreAssembly(fem_solver, function_space, formulation, mesh, material, Eulerx, Eulerp, calculate_rhs=True, filename=None, chunk_size=None):
# def OutofCoreAssembly(MainData, mesh, material, Eulerx, TotalPot, calculate_rhs=True, filename=None, chunk_size=None):
    """Assembly routine for larger than memory system of equations.
        Usage of h5py and dask allow us to store the triplets and build a sparse matrix out of
        them on disk.

        Note: The sparse matrix itfem_solver is created on the memory.
    """

    import sys, os
    from warnings import warn
    from time import time
    try:
        import psutil
    except IOError:
        has_psutil = False
        raise ImportError("No module named psutil. Please install it using 'pip install psutil'")
    # from Core.Supplementary.dsparse.sparse import dok_matrix


    if fem_solver.parallel:
        warn("Parallel assembly cannot performed on large arrays. \n"
            "Out of core 'i.e. Dask' parallelisation is turned on instead. "
            "This is an innocuous warning")

    try:
        import h5py
    except ImportError:
        has_h5py = False
        raise ImportError('h5py is not installed. Please install it first by running "pip install h5py"')

    try:
        import dask.array as da
    except ImportError:
        has_dask = False
        raise ImportError('dask is not installed. Please install it first by running "pip install toolz && pip install dask"')

    if filename is None:
        warn("filename not given. I am going to write the output in the current directory")
        pwd = os.path.dirname(os.path.realpath(__file__))
        filename = os.path.join(pwd,"output.hdf5")


    # GET MESH DETAILS
    C = mesh.InferPolynomialDegree() - 1
    nvar = formulation.nvar
    ndim = formulation.ndim

    nelem = mesh.nelem
    nodeperelem = mesh.elements.shape[1]

    # GET MEMORY INFO
    memory = psutil.virtual_memory()
    size_of_triplets_gbytes = (mesh.points.shape[0]*nvar)**2*nelem*(4)*(3)//1024**3
    if memory.available//1024**3 > 2*size_of_triplets_gbytes:
        warn("Out of core assembly is only efficient for larger than memory "
            "system of equations. Using it on smaller matrices can be very inefficient")

    hdf_file = h5py.File(filename,'w')
    IJV_triplets = hdf_file.create_dataset("IJV_triplets",((nvar*nodeperelem)**2*nelem,3),dtype=np.float32)


    # THE I & J VECTORS OF LOCAL STIFFNESS MATRIX DO NOT CHANGE, HENCE COMPUTE THEM ONCE
    I_stiff_elem = np.repeat(np.arange(0,nvar*nodeperelem),nvar*nodeperelem,axis=0)
    J_stiff_elem = np.tile(np.arange(0,nvar*nodeperelem),nvar*nodeperelem)

    I_mass=[];J_mass=[];V_mass=[]; I_mass_elem = []; J_mass_elem = []

    if calculate_rhs is False:
        F = []
        T = []
    else:
        F = np.zeros((mesh.points.shape[0]*nvar,1),np.float64)
        T = np.zeros((mesh.points.shape[0]*nvar,1),np.float64)

    # ASSIGN OTHER NECESSARY MATRICES
    full_current_row_stiff = []; full_current_column_stiff = []; coeff_stiff = []
    full_current_row_mass = []; full_current_column_mass = []; coeff_mass = []
    mass = []

    gc.collect()

    print('Writing the triplets to disk')
    t_hdf5 = time()
    for elem in range(nelem):

        full_current_row_stiff, full_current_column_stiff, coeff_stiff, t, f, \
            full_current_row_mass, full_current_column_mass, coeff_mass = GetElementalMatrices(elem,
                MainData,mesh.elements,mesh.points,nodeperelem,Eulerx,TotalPot,I_stiff_elem,J_stiff_elem,I_mass_elem,J_mass_elem)

        IJV_triplets[(nvar*nodeperelem)**2*elem:(nvar*nodeperelem)**2*(elem+1),0] = full_current_row_stiff.flatten()
        IJV_triplets[(nvar*nodeperelem)**2*elem:(nvar*nodeperelem)**2*(elem+1),1] = full_current_column_stiff.flatten()
        IJV_triplets[(nvar*nodeperelem)**2*elem:(nvar*nodeperelem)**2*(elem+1),2] = coeff_stiff.flatten()

        if calculate_rhs is True:

            if MainData.Analysis != 'Static':
                # SPARSE ASSEMBLY - MASS MATRIX
                I_mass, J_mass, V_mass = SparseAssembly_Step_2(I_mass,J_mass,V_mass,full_current_row_mass,full_current_column_mass,coeff_mass,
                    nvar,nodeperelem,elem)

            if MainData.AssemblyParameters.ExternalLoadNature == 'Nonlinear':
                # RHS ASSEMBLY
                for iterator in range(0,nvar):
                    F[mesh.elements[elem,:]*nvar+iterator,0]+=f[iterator::nvar]
            # INTERNAL TRACTION FORCE ASSEMBLY
            for iterator in range(0,nvar):
                    T[mesh.elements[elem,:]*nvar+iterator,0]+=t[iterator::nvar,0]

        if elem % 10000 == 0:
            print("Processed ", elem, " elements")

    hdf_file.close()
    print('Finished writing the triplets to disk. Time taken', time() - t_hdf5, 'seconds')


    print('Reading the triplets back from disk')
    hdf_file = h5py.File(filename,'r')
    if chunk_size is None:
        chunk_size = mesh.points.shape[0]*nvar // 300

    print('Creating dask array from triplets')
    IJV_triplets = da.from_array(hdf_file['IJV_triplets'],chunks=(chunk_size,3))


    print('Creating the sparse matrix')
    t_sparse = time()

    stiffness = csr_matrix((IJV_triplets[:,2].astype(np.float32),
        (IJV_triplets[:,0].astype(np.int32),IJV_triplets[:,1].astype(np.int32))),
        shape=((mesh.points.shape[0]*nvar,mesh.points.shape[0]*nvar)),dtype=np.float32)

    print('Done creating the sparse matrix, time taken', time() - t_sparse)

    hdf_file.close()

    return stiffness, T, F, mass




#--------------------- ASSEMBLY ROUTINE FOR MASS MATRIX ONLY - FOR MODIFIED NEWTON RAPHSON ----------------------#
#----------------------------------------------------------------------------------------------------------------#


def AssembleMass(fem_solver, function_space, formulation, mesh, material, Eulerx):

    # GET MESH DETAILS
    C = mesh.InferPolynomialDegree() - 1
    nvar = formulation.nvar
    ndim = formulation.ndim
    nelem = mesh.nelem
    nodeperelem = mesh.elements.shape[1]

    # ALLOCATE VECTORS FOR SPARSE ASSEMBLY OF MASS MATRIX - CHANGE TYPES TO INT64 FOR DoF > 1e09
    I_mass=np.zeros(int((nvar*nodeperelem)**2*nelem),dtype=np.int32)
    J_mass=np.zeros(int((nvar*nodeperelem)**2*nelem),dtype=np.int32)
    V_mass=np.zeros(int((nvar*nodeperelem)**2*nelem),dtype=np.float64)

    # COMPUATE ALL LOCAL ELEMENTAL MATRICES (STIFFNESS, MASS, INTERNAL & EXTERNAL TRACTION FORCES )
    ParallelTuple = parmap.map(formulation,np.arange(0,nelem,dtype=np.int32),
        function_space, mesh, material, fem_solver, Eulerx, Eulerp, processes= int(multiprocessing.cpu_count()/2))

    for elem in range(nelem):
        # COMPUATE LOCAL ELEMENTAL MASS MATRIX
        I_mass_elem, J_mass_elem, V_mass_elem = formulation.GetMassMatrix(elem,
            function_space, mesh, material, fem_solver, Eulerx, Eulerp)
        # SPARSE ASSEMBLY - MASS MATRIX
        SparseAssemblyNative(I_mass_elem,J_mass_elem,V_mass_elem,I_mass,J_mass,V_mass,
            elem,nvar,nodeperelem,mesh.elements)

    mass = csr_matrix((V_mass,(I_mass,J_mass)),shape=((nvar*mesh.points.shape[0],
        nvar*mesh.points.shape[0])),dtype=np.float64)

    return mass




#----------------- ASSEMBLY ROUTINE FOR TRACTION FORCES ONLY - FOR MODIFIED NEWTON RAPHSON ----------------------#
#----------------------------------------------------------------------------------------------------------------#


def AssembleInternalTractionForces(fem_solver, function_space, formulation, mesh, material, Eulerx, Eulerp):

    # GET MESH DETAILS
    C = mesh.InferPolynomialDegree() - 1
    nvar = formulation.nvar
    ndim = formulation.ndim
    nelem = mesh.nelem
    nodeperelem = mesh.elements.shape[1]

    T = np.zeros((mesh.points.shape[0]*nvar,1),np.float64)
    # T = np.zeros((mesh.points.shape[0]*nvar,1),np.float32)

    for elem in range(nelem):

        t = formulation.GetElementalMatrices(elem,
            function_space, mesh, material, fem_solver, Eulerx, Eulerp)[3]

        # INTERNAL TRACTION FORCE ASSEMBLY
        # for iterator in range(0,nvar):
            # T[mesh.elements[elem,:]*nvar+iterator,0]+=t[iterator::nvar,0]
        RHSAssemblyNative(T,t,elem,nvar,nodeperelem,mesh.elements)


    return T







#------------------------------- ASSEMBLY ROUTINE FOR EXTERNAL TRACTION FORCES ----------------------------------#
#----------------------------------------------------------------------------------------------------------------#


def AssembleForces(boundary_condition, mesh, material, function_spaces, compute_traction_forces=True, compute_body_forces=False):

    Ft = np.zeros((mesh.points.shape[0]*material.nvar,1))
    Fb = np.zeros((mesh.points.shape[0]*material.nvar,1))

    if compute_traction_forces:
        Ft = AssembleExternalTractionForces(boundary_condition, mesh, material, function_spaces[2])
    if compute_body_forces:
        Fb = AssembleBodyForces(boundary_condition, mesh, material, function_spaces[0])

    return Ft + Fb



def AssembleExternalTractionForces(boundary_condition, mesh, material, function_space):


    nvar = material.nvar
    ndim = material.ndim
    ngauss = function_space.AllGauss.shape[0]

    if ndim == 2:
        faces = mesh.edges
        nodeperelem = mesh.edges.shape[1]
    else:
        faces = mesh.faces
        nodeperelem = mesh.faces.shape[1]

    if boundary_condition.is_applied_neumann_shape_functions_computed is False:
        N = np.zeros((nodeperelem*nvar,nvar,ngauss))
        for i in range(nvar):
            N[i::nvar,i,:] = function_space.Bases
        boundary_condition.__Nt__ = N
        boundary_condition.is_applied_neumann_shape_functions_computed = True
    else:
        N = boundary_condition.__Nt__


    F = np.zeros((mesh.points.shape[0]*nvar,1))
    for face in range(faces.shape[0]):
        if boundary_condition.neumann_flags[face] == True:
            ElemTraction = boundary_condition.applied_neumann[face,:]
            external_traction = np.einsum("ijk,j,k->ik",N,ElemTraction,function_space.AllGauss[:,0]).sum(axis=1)
            RHSAssemblyNative(F,np.ascontiguousarray(external_traction[:,None]),face,nvar,nodeperelem,faces)


    # nvar = material.nvar
    # ndim = material.ndim

    # if ndim == 2:
    #     faces = np.copy(mesh.edges)
    #     nodeperelem = mesh.edges.shape[1]
    # else:
    #     faces = np.copy(mesh.faces)
    #     nodeperelem = mesh.faces.shape[1]

    # F = np.zeros((mesh.points.shape[0]*nvar,1))
    # for face in range(faces.shape[0]):
    #     if boundary_condition.neumann_flags[face] == True:
    #         ElemTraction = boundary_condition.applied_neumann[face,:]
    #         # LagrangeFaceCoords = mesh.points[faces[face,:],:]
    #         # ParentGradientX = np.einsum('ijk,jl->kil', function_space.Jm, LagrangeFaceCoords)
    #         # detJ = np.einsum('i,i->i',function_space.AllGauss[:,0],np.abs(np.linalg.det(ParentGradientX)))

    #         external_traction = np.zeros((nodeperelem*nvar))
    #         N = np.zeros((nodeperelem*nvar,nvar))
    #         for counter in range(function_space.AllGauss.shape[0]):
    #             for i in range(nvar):
    #                 N[i::nvar,i] = function_space.Bases[:,counter]

    #             external_traction += np.dot(N,ElemTraction)*function_space.AllGauss[counter,0]

    #        # RHS ASSEMBLY
    #         # for iterator in range(0,nvar):
    #             # F[faces[face,:]*nvar+iterator,0]+=external_traction[iterator::nvar]
    #         RHSAssemblyNative(F,np.ascontiguousarray(external_traction[:,None]),face,nvar,nodeperelem,faces)

    return F


def AssembleBodyForces(boundary_condition, mesh, material, function_space):


    nvar = material.nvar
    ndim = material.ndim
    nodeperelem = mesh.elements.shape[1]
    ngauss = function_space.AllGauss.shape[0]

    if boundary_condition.is_body_force_shape_functions_computed is False:
        N = np.zeros((nodeperelem*nvar,nvar,ngauss))
        for i in range(nvar):
            N[i::nvar,i,:] = function_space.Bases
        boundary_condition.__Nb__ = N
        boundary_condition.is_body_force_shape_functions_computed = True
    else:
        N = boundary_condition.__Nb__

    F = np.zeros((mesh.points.shape[0]*nvar,1))
    # BODY FORCE IS APPLIED IN THE Z-DIRECTION
    ElemTraction = np.zeros(nvar); ElemTraction[ndim-1] = -material.rho
    for elem in range(mesh.nelem):
        body_force = np.einsum("ijk,j,k->ik",N,ElemTraction,function_space.AllGauss[:,0]).sum(axis=1)
        RHSAssemblyNative(F,np.ascontiguousarray(body_force[:,None]),elem,nvar,nodeperelem,mesh.elements)


    # nvar = material.nvar
    # ndim = material.ndim
    # nodeperelem = mesh.elements.shape[1]

    # F = np.zeros((mesh.points.shape[0]*nvar,1))
    # # BODY FORCE IS APPLIED IN THE Z-DIRECTION
    # ElemTraction = np.zeros(nvar); ElemTraction[ndim-1] = -material.rho
    # for elem in range(mesh.nelem):
    #     # LagrangeElemCoords = mesh.points[mesh.elements[elem,:],:]

    #     body_force = np.zeros((nodeperelem*nvar))
    #     N = np.zeros((nodeperelem*nvar,nvar))
    #     for counter in range(function_space.AllGauss.shape[0]):
    #         for i in range(nvar):
    #             N[i::nvar,i] = function_space.Bases[:,counter]

    #         body_force += np.dot(N,ElemTraction)*function_space.AllGauss[counter,0]

    #    # RHS ASSEMBLY
    #     # for iterator in range(0,nvar):
    #         # F[faces[elem,:]*nvar+iterator,0]+=body_force[iterator::nvar]
    #     RHSAssemblyNative(F,np.ascontiguousarray(body_force[:,None]),elem,nvar,nodeperelem,mesh.elements)


    return F










# def AssembleExplicit(fem_solver, function_space, formulation, mesh, material, Eulerx, Eulerp):

#     # GET MESH DETAILS
#     C = mesh.InferPolynomialDegree() - 1
#     nvar = formulation.nvar
#     ndim = formulation.ndim
#     nelem = mesh.nelem
#     nodeperelem = mesh.elements.shape[1]

#     T = np.zeros((mesh.points.shape[0]*nvar,1),np.float64)
#     M = np.zeros((mesh.points.shape[0]*nvar,1),np.float64)

#     mass, F = [], []
#     if fem_solver.has_moving_boundary:
#         F = np.zeros((mesh.points.shape[0]*nvar,1),np.float64)


#     for elem in range(nelem):

#         t, f, mass = formulation.GetElementalMatricesInVectorForm(elem,
#                 function_space, mesh, material, fem_solver, Eulerx, Eulerp)

#         if fem_solver.has_moving_boundary:
#             # RHS ASSEMBLY
#             RHSAssemblyNative(F,f,elem,nvar,nodeperelem,mesh.elements)

#         # LUMPED MASS ASSEMBLY
#         if fem_solver.analysis_type != 'static' and fem_solver.is_mass_computed==False:
#             RHSAssemblyNative(M,mass,elem,nvar,nodeperelem,mesh.elements)

#         # INTERNAL TRACTION FORCE ASSEMBLY
#         RHSAssemblyNative(T,t,elem,nvar,nodeperelem,mesh.elements)

#     if fem_solver.analysis_type != 'static' and fem_solver.is_mass_computed==False:
#         fem_solver.is_mass_computed = True

#     return T, F, M






def AssembleExplicit(fem_solver, function_space, formulation, mesh, material, Eulerx, Eulerp):

    if fem_solver.has_low_level_dispatcher and fem_solver.is_mass_computed is True:

        if not material.has_low_level_dispatcher:
            raise RuntimeError("Cannot dispatch to low level module, since material {} does not support it".format(type(material).__name__))

        T, F, M = _LowLevelAssemblyExplicit_(fem_solver, function_space, formulation, mesh, material, Eulerx, Eulerp)
        if isinstance(F,np.ndarray):
            F = F[:,None]
        if M is not None:
            fem_solver.is_mass_computed = True

        return T[:,None], F, M


    # GET MESH DETAILS
    nvar = formulation.nvar
    ndim = formulation.ndim
    nelem = mesh.nelem
    nodeperelem = mesh.elements.shape[1]

    T = np.zeros((mesh.points.shape[0]*nvar,1),np.float64)

    I_mass=[]; J_mass=[]; V_mass=[]
    if fem_solver.analysis_type !='static' and fem_solver.mass_type == "consistent":
        # ALLOCATE VECTORS FOR SPARSE ASSEMBLY OF MASS MATRIX - CHANGE TYPES TO INT64 FOR DoF > 1e09
        I_mass=np.zeros(int((nvar*nodeperelem)**2*nelem),dtype=np.int32)
        J_mass=np.zeros(int((nvar*nodeperelem)**2*nelem),dtype=np.int32)
        V_mass=np.zeros(int((nvar*nodeperelem)**2*nelem),dtype=np.float64)
        M = []
    else:
        M = np.zeros((mesh.points.shape[0]*nvar,1),np.float64)

    F = []
    if fem_solver.has_moving_boundary:
        F = np.zeros((mesh.points.shape[0]*nvar,1),np.float64)

    if fem_solver.parallel:
        # from joblib import Parallel, delayed
        # Parallel(n_jobs=2)(delayed(funcer)(elem, nvar,
        #     nodeperelem, T, F, M, formulation, function_space, mesh, material,
        #     fem_solver, Eulerx, Eulerp) for elem in range(0,nelem))

        parmap.map(AssembleExplicitFunctor,np.arange(0,nelem,dtype=np.int32),
            nvar, nodeperelem, T, F, I_mass, J_mass, V_mass, M, formulation, function_space, mesh, material,
            fem_solver, Eulerx, Eulerp, processes= int(multiprocessing.cpu_count()))
    else:
        for elem in range(nelem):
            AssembleExplicitFunctor(elem, nvar, nodeperelem, T, F, I_mass, J_mass, V_mass, M, formulation,
                function_space, mesh, material, fem_solver, Eulerx, Eulerp)

    # SET MASS FLAG HERE
    if fem_solver.analysis_type != 'static' and fem_solver.is_mass_computed==False:
        if fem_solver.mass_type == "consistent":
            M = csr_matrix((V_mass,(I_mass,J_mass)),shape=((nvar*mesh.points.shape[0],
            nvar*mesh.points.shape[0])),dtype=np.float64)
        fem_solver.is_mass_computed = True


    return T, F, M


def AssembleExplicitFunctor(elem, nvar, nodeperelem, T, F, I_mass, J_mass, V_mass, M,
    formulation, function_space, mesh, material, fem_solver, Eulerx, Eulerp):

    t, f, mass = formulation.GetElementalMatricesInVectorForm(elem,
            function_space, mesh, material, fem_solver, Eulerx, Eulerp)


    if fem_solver.has_moving_boundary:
        # RHS ASSEMBLY
        RHSAssemblyNative(F,f,elem,nvar,nodeperelem,mesh.elements)
    # INTERNAL TRACTION FORCE ASSEMBLY
    RHSAssemblyNative(T,t,elem,nvar,nodeperelem,mesh.elements)

    # LUMPED MASS ASSEMBLY
    if fem_solver.analysis_type != 'static' and fem_solver.is_mass_computed==False:
        # MASS ASSEMBLY
        if fem_solver.mass_type == "lumped":
            RHSAssemblyNative(M,mass,elem,nvar,nodeperelem,mesh.elements)
        else:
            # SPARSE ASSEMBLY - MASS MATRIX
            I_mass_elem, J_mass_elem, V_mass_elem = formulation.FindIndices(mass)
            SparseAssemblyNative(I_mass_elem,J_mass_elem,V_mass_elem,I_mass,J_mass,V_mass,
                elem,nvar,nodeperelem,mesh.elements)