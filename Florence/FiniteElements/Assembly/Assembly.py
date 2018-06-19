from __future__ import print_function
import gc, os, sys
from copy import deepcopy
from warnings import warn
from time import time
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix

from ._LowLevelAssembly_ import _LowLevelAssembly_, _LowLevelAssemblyExplicit_, _LowLevelAssemblyLaplacian_
from ._LowLevelAssembly_ import _LowLevelAssembly_Par_, _LowLevelAssemblyExplicit_Par_

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


def LowLevelAssembly(fem_solver, function_space, formulation, mesh, material, Eulerx, Eulerp):

    t_assembly = time()

    if not material.has_low_level_dispatcher:
        raise RuntimeError("Cannot dispatch to low level module since material {} does not support it".format(type(material).__name__))

    if formulation.fields == "electrostatics":
        stiffness, T = _LowLevelAssemblyLaplacian_(fem_solver, function_space, formulation, mesh, material, Eulerx, Eulerp)
        fem_solver.assembly_time = time() - t_assembly
        return stiffness, T[:,None], None, None

    if fem_solver.parallel:
        stiffness, T, F, mass = ImplicitParallelLauncher(fem_solver, function_space, formulation, mesh, material, Eulerx, Eulerp)
    else:
        stiffness, T, F, mass = _LowLevelAssembly_(fem_solver, function_space, formulation, mesh, material, Eulerx, Eulerp)

    if isinstance(F,np.ndarray):
        F = F[:,None]
    if mass is not None:
        fem_solver.is_mass_computed = True

    fem_solver.assembly_time = time() - t_assembly

    return stiffness, T[:,None], F, mass



def AssemblySmall(fem_solver, function_space, formulation, mesh, material, Eulerx, Eulerp):

    t_assembly = time()

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
    if fem_solver.analysis_type !='static' and fem_solver.is_mass_computed is False:
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

    fem_solver.assembly_time = time() - t_assembly

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
    except ImportError:
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
        Ft = AssembleExternalTractionForces(boundary_condition, mesh, material, function_spaces[-1])
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






#---------------------------------------- EXPLICIT ASSEMBLY ROUTINES --------------------------------------------#
#----------------------------------------------------------------------------------------------------------------#


def AssembleExplicit_NoLLD(fem_solver, function_space, formulation, mesh, material, Eulerx, Eulerp):

    # GET MESH DETAILS
    C = mesh.InferPolynomialDegree() - 1
    nvar = formulation.nvar
    ndim = formulation.ndim
    nelem = mesh.nelem
    nodeperelem = mesh.elements.shape[1]

    T = np.zeros((mesh.points.shape[0]*nvar,1),np.float64)
    M = np.zeros((mesh.points.shape[0]*nvar,1),np.float64)

    mass, F = [], []
    if fem_solver.has_moving_boundary:
        F = np.zeros((mesh.points.shape[0]*nvar,1),np.float64)


    for elem in range(nelem):

        t, f, mass = formulation.GetElementalMatricesInVectorForm(elem,
                function_space, mesh, material, fem_solver, Eulerx, Eulerp)

        if fem_solver.has_moving_boundary:
            # RHS ASSEMBLY
            RHSAssemblyNative(F,f,elem,nvar,nodeperelem,mesh.elements)

        # LUMPED MASS ASSEMBLY
        if fem_solver.analysis_type != 'static' and fem_solver.is_mass_computed==False:
            RHSAssemblyNative(M,mass,elem,nvar,nodeperelem,mesh.elements)

        # INTERNAL TRACTION FORCE ASSEMBLY
        RHSAssemblyNative(T,t,elem,nvar,nodeperelem,mesh.elements)

    if fem_solver.analysis_type != 'static' and fem_solver.is_mass_computed==False:
        fem_solver.is_mass_computed = True

    return T, F, M



def AssembleExplicit(fem_solver, function_space, formulation, mesh, material, Eulerx, Eulerp):

    if fem_solver.has_low_level_dispatcher and fem_solver.is_mass_computed is True:
        if not material.has_low_level_dispatcher:
            raise RuntimeError("Cannot dispatch to low level module, since material {} does not support it".format(type(material).__name__))

        if fem_solver.parallel:
            T = ExplicitParallelLauncher(fem_solver, function_space, formulation, mesh, material, Eulerx, Eulerp)
            return T[:,None], [], []
        else:
            T, F, M = _LowLevelAssemblyExplicit_(fem_solver, function_space, formulation, mesh, material, Eulerx, Eulerp)
            return T[:,None], F, M

    else:

        if fem_solver.has_low_level_dispatcher:
            if fem_solver.parallel:
                T = ExplicitParallelLauncher(fem_solver, function_space, formulation, mesh, material, Eulerx, Eulerp)
            else:
                T = _LowLevelAssemblyExplicit_(fem_solver, function_space, formulation, mesh, material, Eulerx, Eulerp)[0]
        else:
            return AssembleExplicit_NoLLD(fem_solver, function_space, formulation, mesh, material, Eulerx, Eulerp)


        # GET MESH DETAILS
        nvar = formulation.nvar
        ndim = formulation.ndim
        nelem = mesh.nelem
        nodeperelem = mesh.elements.shape[1]

        F = []
        I_mass=[]; J_mass=[]; V_mass=[]
        if fem_solver.mass_type == "lumped":
            M = np.zeros((mesh.points.shape[0]*nvar,1),np.float64)
        else:
            # ALLOCATE VECTORS FOR SPARSE ASSEMBLY OF MASS MATRIX - CHANGE TYPES TO INT64 FOR DoF > 1e09
            I_mass=np.zeros(int((nvar*nodeperelem)**2*nelem),dtype=np.int32)
            J_mass=np.zeros(int((nvar*nodeperelem)**2*nelem),dtype=np.int32)
            V_mass=np.zeros(int((nvar*nodeperelem)**2*nelem),dtype=np.float64)
            M = []

        for elem in range(nelem):

            LagrangeElemCoords = mesh.points[mesh.elements[elem,:],:]
            EulerElemCoords = Eulerx[mesh.elements[elem,:],:]
            if formulation.fields == "electro_mechanics":
                ElectricPotentialElem = Eulerp[mesh.elements[elem,:]]
            else:
                ElectricPotentialElem = []

            # COMPUTE THE MASS MATRIX
            if material.has_low_level_dispatcher:
                mass = formulation.__GetLocalMass_Efficient__(function_space,material,LagrangeElemCoords,EulerElemCoords,fem_solver,elem)
            else:
                mass = formulation.GetLocalMass_Efficient(function_space,material,LagrangeElemCoords,EulerElemCoords,fem_solver,elem)

            if fem_solver.mass_type == "lumped":
                mass = formulation.GetLumpedMass(mass)
                RHSAssemblyNative(M,mass,elem,nvar,nodeperelem,mesh.elements)
            else:
                # SPARSE ASSEMBLY - MASS MATRIX
                I_mass_elem, J_mass_elem, V_mass_elem = formulation.FindIndices(mass)
                SparseAssemblyNative(I_mass_elem,J_mass_elem,V_mass_elem,I_mass,J_mass,V_mass,
                    elem,nvar,nodeperelem,mesh.elements)

        # SET MASS FLAG HERE
        if fem_solver.is_mass_computed is False:
            if fem_solver.mass_type == "consistent":
                M = csr_matrix((V_mass,(I_mass,J_mass)),shape=((nvar*mesh.points.shape[0],
                nvar*mesh.points.shape[0])),dtype=np.float64)
            fem_solver.is_mass_computed = True


    return T[:,None], F, M





#---------------------------------------- PARALLEL ASSEMBLY ROUTINES --------------------------------------------#
#----------------------------------------------------------------------------------------------------------------#

class ImplicitParallelZipper(object):

    def __init__(self, fem_solver, function_space, formulation, mesh, material, Eulerx, Eulerp):
        self.fem_solver = fem_solver.__class__(analysis_type=fem_solver.analysis_type,
                                                analysis_nature=fem_solver.analysis_nature)
        self.function_space = function_space
        self.formulation = formulation
        self.mesh = mesh
        self.material = material
        self.Eulerx = Eulerx
        self.Eulerp = Eulerp

def ImplicitParallelExecuter_PoolBased(functor):
    return _LowLevelAssembly_Par_(functor.fem_solver, functor.function_space,
        functor.formulation, functor.mesh, functor.material, functor.Eulerx, functor.Eulerp)

def ImplicitParallelExecuter_ProcessBased(functor, proc, tups):
    tup = _LowLevelAssembly_Par_(functor.fem_solver, functor.function_space,
        functor.formulation, functor.mesh, functor.material, functor.Eulerx, functor.Eulerp)
    tups[proc] = tup
    # tups.append(tup) # FOR SERIAL CHECKS

def ImplicitParallelExecuter_ProcessQueueBased(functor, queue):
    tups = _LowLevelAssembly_Par_(functor.fem_solver, functor.function_space,
        functor.formulation, functor.mesh, functor.material, functor.Eulerx, functor.Eulerp)
    queue.put(tups)


def ImplicitParallelLauncher(fem_solver, function_space, formulation, mesh, material, Eulerx, Eulerp):

    from multiprocessing import Process, Pool, Manager, Queue
    from contextlib import closing

    # GET MESH DETAILS
    nvar = formulation.nvar
    ndim = formulation.ndim
    nelem = mesh.nelem
    nnode = mesh.points.shape[0]
    nodeperelem = mesh.elements.shape[1]
    local_capacity = int((nvar*nodeperelem)**2)

    pmesh, pelement_indices, pnode_indices, partitioned_maps = fem_solver.pmesh, \
        fem_solver.pelement_indices, fem_solver.pnode_indices, fem_solver.partitioned_maps


    # ALLOCATE VECTORS FOR SPARSE ASSEMBLY OF STIFFNESS MATRIX - CHANGE TYPES TO INT64 FOR DoF > 1e09
    I_stiffness=np.zeros((nelem,local_capacity),dtype=np.int32)
    J_stiffness=np.zeros((nelem,local_capacity),dtype=np.int32)
    V_stiffness=np.zeros((nelem,local_capacity),dtype=np.float64)

    I_mass=[]; J_mass=[]; V_mass=[]
    if fem_solver.analysis_type !='static' and fem_solver.is_mass_computed is False:
        # ALLOCATE VECTORS FOR SPARSE ASSEMBLY OF MASS MATRIX - CHANGE TYPES TO INT64 FOR DoF > 1e09
        I_mass=np.zeros((nelem,local_capacity),dtype=np.int32)
        J_mass=np.zeros((nelem,local_capacity),dtype=np.int32)
        V_mass=np.zeros((nelem,local_capacity),dtype=np.float64)

    T = np.zeros((mesh.points.shape[0],nvar),np.float64)

    funcs = []
    for proc in range(fem_solver.no_of_cpu_cores):
        pnodes = pnode_indices[proc]
        Eulerx_current = Eulerx[pnodes,:]
        Eulerp_current = Eulerp[pnodes]
        funcs.append(ImplicitParallelZipper(fem_solver, function_space, formulation,
            pmesh[proc], material, Eulerx_current, Eulerp_current))

    # # SERIAL
    # tups = []
    # for i in range(fem_solver.no_of_cpu_cores):
    #     ImplicitParallelExecuter_ProcessBased(funcs[i], i, tups)
    # for i in range(fem_solver.no_of_cpu_cores):
    #     pnodes = pnode_indices[i]
    #     pelements = pelement_indices[i]
    #     I_stiffness[pelements,:] = partitioned_maps[i][tups[i][0]].reshape(pmesh[i].nelem,local_capacity)
    #     J_stiffness[pelements,:] = partitioned_maps[i][tups[i][1]].reshape(pmesh[i].nelem,local_capacity)
    #     V_stiffness[pelements,:] = tups[i][2].reshape(pmesh[i].nelem,local_capacity)
    #     T[pnodes,:] += tups[i][-1].reshape(pnodes.shape[0],nvar)

    #     if fem_solver.analysis_type != "static" and fem_solver.is_mass_computed is False:
    #         I_stiffness[pelements,:] = partitioned_maps[i][tups[i][3]].reshape(pmesh[i].nelem,local_capacity)
    #         J_stiffness[pelements,:] = partitioned_maps[i][tups[i][4]].reshape(pmesh[i].nelem,local_capacity)
    #         V_stiffness[pelements,:] = tups[i][5].reshape(pmesh[i].nelem,local_capacity)


    # POOL BASED
    if fem_solver.parallel_model == "pool":
        with closing(Pool(processes=fem_solver.no_of_cpu_cores)) as pool:
            tups = pool.map(ImplicitParallelExecuter_PoolBased,funcs)
            pool.terminate()

    # JOBLIB BASED
    elif fem_solver.parallel_model == "joblib":
        try:
            from joblib import Parallel, delayed
        except ImportError:
            raise ImportError("Joblib is not installed. Install it 'using pip install joblib'")
        tups = Parallel(n_jobs=fem_solver.no_of_cpu_cores)(delayed(ImplicitParallelExecuter_PoolBased)(func) for func in funcs)
        # tups = Parallel(n_jobs=10, backend="threading")(delayed(ImplicitParallelExecuter_PoolBased)(func) for func in funcs)

    # SCOOP BASED
    elif fem_solver.parallel_model == "scoop":
        try:
            from scoop import futures
        except ImportError:
            raise ImportError("Scoop is not installed. Install it 'using pip install scoop'")
        # tups = futures.map(ImplicitParallelExecuter_PoolBased, funcs)
        tups = list(futures.map(ImplicitParallelExecuter_PoolBased, funcs))

    # PROCESS AND MANAGER BASED
    elif fem_solver.parallel_model == "context_manager":
        procs = []
        manager = Manager(); tups = manager.dict() # SPAWNS A NEW PROCESS
        for i, func in enumerate(funcs):
            proc = Process(target=ImplicitParallelExecuter_ProcessBased, args=(func,i,tups))
            procs.append(proc)
            proc.start()
        for proc in procs:
            proc.join()

    # PROCESS AND QUEUE BASED
    elif fem_solver.parallel_model == "queue":
        procs = []
        for i, func in enumerate(funcs):
            queue = Queue()
            proc = Process(target=ImplicitParallelExecuter_ProcessQueueBased, args=(func,queue))
            proc.daemon = True
            procs.append(proc)
            proc.start()
            tups = queue.get()
            pnodes = pnode_indices[i]
            pelements = pelement_indices[i]
            I_stiffness[pelements,:] = partitioned_maps[i][tups[0]].reshape(pmesh[i].nelem,local_capacity)
            J_stiffness[pelements,:] = partitioned_maps[i][tups[1]].reshape(pmesh[i].nelem,local_capacity)
            V_stiffness[pelements,:] = tups[2].reshape(pmesh[i].nelem,local_capacity)
            T[pnodes,:] += tups[-1].reshape(pnodes.shape[0],nvar)

            if fem_solver.analysis_type != "static" and fem_solver.is_mass_computed is False:
                I_mass[pelements,:] = partitioned_maps[i][tups[3]].reshape(pmesh[i].nelem,local_capacity)
                J_mass[pelements,:] = partitioned_maps[i][tups[4]].reshape(pmesh[i].nelem,local_capacity)
                V_mass[pelements,:] = tups[5].reshape(pmesh[i].nelem,local_capacity)
            proc.join()

    if fem_solver.parallel_model == "pool" or fem_solver.parallel_model == "context_manager" \
        or fem_solver.parallel_model == "joblib" or fem_solver.parallel_model == "scoop":
        for i in range(fem_solver.no_of_cpu_cores):
            pnodes = pnode_indices[i]
            pelements = pelement_indices[i]
            I_stiffness[pelements,:] = partitioned_maps[i][tups[i][0]].reshape(pmesh[i].nelem,local_capacity)
            J_stiffness[pelements,:] = partitioned_maps[i][tups[i][1]].reshape(pmesh[i].nelem,local_capacity)
            V_stiffness[pelements,:] = tups[i][2].reshape(pmesh[i].nelem,local_capacity)
            T[pnodes,:] += tups[i][-1].reshape(pnodes.shape[0],nvar)

            if fem_solver.analysis_type != "static" and fem_solver.is_mass_computed is False:
                I_mass[pelements,:] = partitioned_maps[i][tups[i][3]].reshape(pmesh[i].nelem,local_capacity)
                J_mass[pelements,:] = partitioned_maps[i][tups[i][4]].reshape(pmesh[i].nelem,local_capacity)
                V_mass[pelements,:] = tups[i][5].reshape(pmesh[i].nelem,local_capacity)


    stiffness = csr_matrix((V_stiffness.ravel(),(I_stiffness.ravel(),J_stiffness.ravel())),
        shape=((nvar*mesh.points.shape[0],nvar*mesh.points.shape[0])),dtype=np.float64)

    F, mass = [], []

    if fem_solver.analysis_type != "static" and fem_solver.is_mass_computed is False:
        mass = csr_matrix((V_mass.ravel(),(I_mass.ravel(),J_mass.ravel())),
            shape=((nvar*mesh.points.shape[0],nvar*mesh.points.shape[0])),dtype=np.float64)


    return stiffness, T.ravel(), F, mass

















class ExplicitParallelZipper(object):
    def __init__(self, function_space, formulation, mesh, material, Eulerx, Eulerp):
        self.function_space = function_space
        self.formulation = formulation
        self.mesh = mesh
        self.material = material
        self.Eulerx = Eulerx
        self.Eulerp = Eulerp

class ExplicitParallelZipperMPI(object):
    def __init__(self, formulation, mesh, material, pnodes):
        self.formulation = formulation
        self.material = material
        self.mesh = mesh
        self.pnodes = pnodes

class ExplicitParallelZipperHDF5(object):
    def __init__(self, formulation, mesh, material):
        self.formulation = formulation
        self.material = material
        self.mesh = mesh


def ExplicitParallelExecuter_PoolBased(functor):
    return _LowLevelAssemblyExplicit_Par_(functor.function_space,
        functor.formulation, functor.mesh, functor.material, functor.Eulerx, functor.Eulerp)

def ExplicitParallelExecuter_ProcessBased(functor, proc, Ts):
    T = _LowLevelAssemblyExplicit_Par_(functor.function_space,
        functor.formulation, functor.mesh, functor.material, functor.Eulerx, functor.Eulerp)
    Ts[proc] = T

def ExplicitParallelExecuter_ProcessQueueBased(functor, queue):
    T = _LowLevelAssemblyExplicit_Par_(functor.function_space,
        functor.formulation, functor.mesh, functor.material, functor.Eulerx, functor.Eulerp)
    queue.put(T)

def ExplicitParallelExecuter_HDF5Based(functor, proc, fname_in, fname_out):

    import h5py

    h5f_out = h5py.File(fname_out+str(proc)+'.h5','r')
    Eulerx = h5f_out['Geometry']['Eulerx'][:]
    Eulerp = h5f_out['Geometry']['Eulerp'][:]
    functor.mesh.points = h5f_out['Geometry']['points'][:]
    functor.mesh.elements = h5f_out['Geometry']['elements'][:]

    T = _LowLevelAssemblyExplicit_Par_(functor.formulation.function_spaces[0],
        functor.formulation, functor.mesh, functor.material, Eulerx, Eulerp)

    # T = _LowLevelAssemblyExplicit_Par_(functor.formulation.function_spaces[0],
    #     functor.formulation, functor.mesh, functor.material, functor.Eulerx, functor.Eulerp)

    h5f = h5py.File(fname_in+str(proc)+'.h5','w')
    h5f.create_dataset('T', data=T)
    h5f.close()


def ExplicitParallelLauncher(fem_solver, function_space, formulation, mesh, material, Eulerx, Eulerp):

    from multiprocessing import Process, Pool, Manager, Queue
    from contextlib import closing

    pmesh, pelement_indices, pnode_indices = fem_solver.pmesh, fem_solver.pelement_indices, fem_solver.pnode_indices
    T_all = np.zeros((mesh.points.shape[0],formulation.nvar),np.float64)

    # MPI BASED
    if fem_solver.parallel_model == "mpi":
        try:
            from mpi4py import MPI
        except ImportError:
            raise ImportError("mpi4py is not installed. Install it using 'pip install mpi4py'")
        from Florence import PWD
        comm = MPI.COMM_SELF.Spawn(sys.executable,
                                   args=[PWD(__file__)+'/MPIParallelExplicitAssembler.py'],
                                   maxprocs=fem_solver.no_of_cpu_cores)

        funcs = []
        for proc in range(fem_solver.no_of_cpu_cores):
            obj = ExplicitParallelZipperMPI(formulation, pmesh[proc], material, pnode_indices[proc])
            funcs.append(obj)

        T_all_size = np.array([mesh.points.shape[0],formulation.ndim, formulation.nvar],dtype="i")
        comm.Bcast([T_all_size, MPI.INT], root=MPI.ROOT)
        comm.bcast(funcs, root=MPI.ROOT)
        comm.Bcast([Eulerx, MPI.DOUBLE], root=MPI.ROOT)
        comm.Bcast([Eulerp, MPI.DOUBLE], root=MPI.ROOT)

        # for proc in range(fem_solver.no_of_cpu_cores):
        #     globals()['points%s' % proc] = pmesh[proc].points
        #     globals()['elements%s' % proc] = pmesh[proc].elements
        #     globals()['nelems%s' % proc] = pmesh[proc].elements.nelem
        #     globals()['nnodes%s' % proc] = pmesh[proc].points.nnode

        # Main T_all TO BE FILLED
        comm.Reduce(None, [T_all, MPI.DOUBLE], root=MPI.ROOT)

        comm.Disconnect()

        return T_all.ravel()


    # PROCESS AND HDF5 BASED
    elif fem_solver.parallel_model == "hdf5":

        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is not installed. Install it using 'pip install h5py'")
        import shutil
        from Florence import Mesh

        home = os.path.expanduser("~")
        tmp_folder = os.path.join(home,".florence_tmp000")
        if not os.path.exists(tmp_folder):
            os.makedirs(tmp_folder)

        fname_in = os.path.join(tmp_folder,"results_explicit")
        fname_out = os.path.join(tmp_folder,"geometry_explicit")

        # funcs = []
        # for proc in range(fem_solver.no_of_cpu_cores):
        #     pnodes = pnode_indices[proc]
        #     Eulerx_current = Eulerx[pnodes,:]
        #     Eulerp_current = Eulerp[pnodes]
        #     obj = ExplicitParallelZipper(function_space, formulation,
        #         pmesh[proc], material, Eulerx_current, Eulerp_current)
        #     funcs.append(obj)

        funcs = []
        for proc in range(fem_solver.no_of_cpu_cores):
            pnodes = pnode_indices[proc]
            Eulerx_current = Eulerx[pnodes,:]
            Eulerp_current = Eulerp[pnodes]


            h5f_out = h5py.File(fname_out+str(proc)+'.h5','w')
            grp = h5f_out.create_group('Geometry')

            grp.create_dataset('elements', data=pmesh[proc].elements)
            grp.create_dataset('points', data=pmesh[proc].points)
            grp.create_dataset('Eulerx', data=Eulerx_current)
            grp.create_dataset('Eulerp', data=Eulerp_current)

            h5f_out.close()

            imesh = Mesh()
            imesh.nnode, imesh.nelem, imesh.element_type = pmesh[proc].nnode, pmesh[proc].nelem, pmesh[proc].element_type

            obj = ExplicitParallelZipperHDF5(formulation, imesh, material)
            funcs.append(obj)

        procs = []
        for i, func in enumerate(funcs):
            proc = Process(target=ExplicitParallelExecuter_HDF5Based, args=(func, i, fname_in, fname_out))
            procs.append(proc)
            proc.start()
        for proc in procs:
            proc.join()

        for proc in range(fem_solver.no_of_cpu_cores):
            h5f = h5py.File(fname_in+str(proc)+'.h5','r')
            T = h5f['T'][:]
            pnodes = pnode_indices[proc]
            T_all[pnodes,:] += T.reshape(pnodes.shape[0],formulation.nvar)

        shutil.rmtree(tmp_folder)

        return T_all.ravel()




    funcs = []
    for proc in range(fem_solver.no_of_cpu_cores):
        pnodes = pnode_indices[proc]
        Eulerx_current = Eulerx[pnodes,:]
        Eulerp_current = Eulerp[pnodes]
        obj = ExplicitParallelZipper(function_space, formulation,
            pmesh[proc], material, Eulerx_current, Eulerp_current)
        funcs.append(obj)

    # SERIAL
    # for proc in range(fem_solver.no_of_cpu_cores):
    #     T = ExplicitParallelExecuter_PoolBased(funcs[proc])
    #     pnodes = pnode_indices[proc]
    #     T_all[pnodes,:] += T.reshape(pnodes.shape[0],formulation.nvar)


    # PROCESS AND MANAGER BASED
    if fem_solver.parallel_model == "context_manager":
        procs = []
        manager = Manager(); Ts = manager.dict() # SPAWNS A NEW PROCESS
        for i, func in enumerate(funcs):
            proc = Process(target=ExplicitParallelExecuter_ProcessBased, args=(func,i,Ts))
            procs.append(proc)
            proc.start()
        for proc in procs:
            proc.join()

    # POOL BASED
    elif fem_solver.parallel_model == "pool":
        with closing(Pool(processes=fem_solver.no_of_cpu_cores)) as pool:
            Ts = pool.map(ExplicitParallelExecuter_PoolBased,funcs)
            pool.terminate()
            # DOESN'T SCALE WELL
            # Ts = pool.map_async(ExplicitParallelExecuter_PoolBased,funcs)
            # Ts.wait()
            # Ts = Ts.get()

    # JOBLIB BASED
    elif fem_solver.parallel_model == "joblib":
        try:
            from joblib import Parallel, delayed
        except ImportError:
            raise ImportError("Joblib is not installed. Install it using 'pip install joblib'")
        Ts = Parallel(n_jobs=fem_solver.no_of_cpu_cores)(delayed(ExplicitParallelExecuter_PoolBased)(func) for func in funcs)
        # Ts = Parallel(n_jobs=10, backend="threading")(delayed(ImplicitParallelExecuter_PoolBased)(func) for func in funcs)

    # SCOOP BASED
    elif fem_solver.parallel_model == "scoop":
        try:
            from scoop import futures
        except ImportError:
            raise ImportError("Scoop is not installed. Install it using 'pip install scoop'")
        Ts = list(futures.map(ExplicitParallelExecuter_PoolBased, funcs))

    # PROCESS AND QUEUE BASED
    elif fem_solver.parallel_model == "queue":
        procs = []
        for i, func in enumerate(funcs):
            queue = Queue()
            proc = Process(target=ExplicitParallelExecuter_ProcessQueueBased, args=(func,queue))
            proc.daemon = True
            procs.append(proc)
            proc.start()
            pnodes = pnode_indices[i]
            T = queue.get()
            T_all[pnodes,:] += T.reshape(pnodes.shape[0],formulation.nvar)
            proc.join()



    if fem_solver.parallel_model == "pool" or fem_solver.parallel_model == "context_manager" \
        or fem_solver.parallel_model == "joblib" or fem_solver.parallel_model == "scoop":

        for proc in range(fem_solver.no_of_cpu_cores):
            pnodes = pnode_indices[proc]
            T_all[pnodes,:] += Ts[proc].reshape(pnodes.shape[0],formulation.nvar)



    return T_all.ravel()
