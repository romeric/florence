from copy import deepcopy
from time import time
import numpy as np
from .FEMSolver import FEMSolver
from Florence import BoundaryCondition


__all__ = ["DetachedParallelFEMSolver"]


class DetachedParallelFEMSolver(FEMSolver):

    def __init__(self, **kwargs):
        if 'number_of_partitions' in kwargs.keys():
            self.number_of_partitions = kwargs['number_of_partitions']
            del kwargs['number_of_partitions']
        else:
            self.number_of_partitions = 1

        if 'fix_interface' in kwargs.keys():
            self.fix_interface = kwargs['fix_interface']
            del kwargs['fix_interface']
        else:
            self.fix_interface = False

        if 'interface_fixity' in kwargs.keys():
            self.interface_fixity = kwargs['interface_fixity']
            del kwargs['interface_fixity']
        else:
            self.interface_fixity = [0,1,2]

        if 'force_solution' in kwargs.keys():
            self.force_solution = kwargs['force_solution']
            del kwargs['force_solution']
        else:
            self.force_solution = False


        if 'do_not_sync' in kwargs.keys():
            self.do_not_sync = kwargs['do_not_sync']
            del kwargs['do_not_sync']
        else:
            self.do_not_sync = False

        super(DetachedParallelFEMSolver, self).__init__(**kwargs)



    def Solve(self, formulation=None, mesh=None,
        material=None, boundary_condition=None,
        function_spaces=None, solver=None,
        contact_formulation=None,
        Eulerx=None, Eulerp=None):

        from multiprocessing import Process, Pool, Manager, Queue
        from contextlib import closing
        from Florence.Tensor import in2d

        # CHECK DATA CONSISTENCY
        #---------------------------------------------------------------------------#
        self.parallel = True
        function_spaces, solver = self.__checkdata__(material, boundary_condition,
            formulation, mesh, function_spaces, solver, contact_formulation=contact_formulation)

        # MORE CHECKES
        if boundary_condition.neumann_flags is not None:
            raise NotImplementedError("Problems with Neumann BC are not supported yet by detached solver")
        if boundary_condition.applied_neumann is not None:
            raise NotImplementedError("Problems with Neumann BC are not supported yet by detached solver")
        #---------------------------------------------------------------------------#

        self.PartitionMeshForParallelFEM(mesh,self.no_of_cpu_cores,formulation.nvar)
        pmesh, pelement_indices, pnode_indices, partitioned_maps = self.pmesh, self.pelement_indices, \
            self.pnode_indices, self.partitioned_maps


        ndim = mesh.InferSpatialDimension()
        if ndim == 3:
            boundary = mesh.faces
        elif ndim == 2:
            boundary = mesh.edges

        pboundary_conditions = []
        for proc in range(self.no_of_cpu_cores):
            imesh = pmesh[proc]
            if ndim==3:
                imesh.GetBoundaryFaces()
                boundary_normals = imesh.FaceNormals()
            else:
                imesh.GetBoundaryEdges()
            unit_outward_normals = imesh.Normals()

            pnodes = pnode_indices[proc]

            # APPLY BOUNDARY CONDITION COMING FROM BIG PROBLEM
            pboundary_condition = BoundaryCondition()
            pboundary_condition.dirichlet_flags = boundary_condition.dirichlet_flags[pnodes,:]
            # CHECK IF THERE ARE REGIONS WHERE BOUNDARY CONDITITION IS NOT APPLIED AT ALL
            bc_not_applied = np.isnan(pboundary_condition.dirichlet_flags).all()
            if bc_not_applied:
                if self.force_solution:
                    warn("There are regions where BC will not be applied properly. Detached solution can be incorrect")
                else:
                    raise RuntimeError("There are regions where BC will not be applied properly. Detached solution can be incorrect")


            # FIND PARTITIONED INTERFACES
            if ndim == 3:
                pboundary = imesh.faces
            elif ndim == 2:
                pboundary = imesh.edges
            pboundary_mapped = pnodes[pboundary]
            boundaries_not_in_big_mesh = ~in2d(pboundary_mapped, boundary, consider_sort=True)
            normals_of_boundaries_not_in_big_mesh = unit_outward_normals[boundaries_not_in_big_mesh,:]
            # IF NORMALS ARE NOT ORIENTED WITH X/Y/Z WE NEED CONTACT FORMULATION
            if self.force_solution is False:
                for i in range(ndim):
                    if not np.any(np.logical_or(np.isclose(unit_outward_normals[:,i],0.),np.isclose(unit_outward_normals[:,i],1.))):
                        raise RuntimeError("Cannot run detached parallel solver as a contact formulation is needed")
                        return

            local_interface_boundary = pboundary[boundaries_not_in_big_mesh]
            interface_nodes = np.unique(local_interface_boundary)
            if self.fix_interface:
                # FIXED BC
                self.interface_fixity = np.array(self.interface_fixity).ravel()
                for i in self.interface_fixity:
                    pboundary_condition.dirichlet_flags[interface_nodes,i] = 0.
            else:
                # SYMMETRY BC
                symmetry_direction_to_fix_boundaries = np.nonzero(normals_of_boundaries_not_in_big_mesh)[1]
                symmetry_nodes_to_fix = local_interface_boundary.ravel()
                symmetry_direction_to_fix_nodes = np.repeat(symmetry_direction_to_fix_boundaries,local_interface_boundary.shape[1])
                pboundary_condition.dirichlet_flags[symmetry_nodes_to_fix,symmetry_direction_to_fix_nodes] = 0.
                # # LOOP APPROACH
                # for i in range(local_interface_boundary.shape[0]):
                #     pboundary_condition.dirichlet_flags[local_interface_boundary[i,:],symmetry_direction_to_fix_boundaries[i]] = 0.


            pboundary_conditions.append(pboundary_condition)



        # TURN OFF PARALLELISATION
        self.parallel = False

        if self.save_incremental_solution is True:
            fname = deepcopy(self.incremental_solution_filename)
            fnames = []
            for proc in range(self.no_of_cpu_cores):
                fnames.append(fname.split(".")[0]+"_proc"+str(proc))

        self.parallel_model = "context_manager"

        if self.parallel_model == "context_manager":
            procs = []
            manager = Manager(); solutions = manager.dict() # SPAWNS A NEW PROCESS
            for proc in range(self.no_of_cpu_cores):
                self.incremental_solution_filename = fnames[proc]
                proc = Process(target=self.__DetachedFEMRunner_ContextManager__,
                    args=(formulation, pmesh[proc],
                    material, pboundary_conditions[proc],
                    function_spaces, solver,
                    contact_formulation,
                    Eulerx, Eulerp, proc, solutions))
                procs.append(proc)
                proc.start()
            for proc in procs:
                proc.join()

        elif self.parallel_model == "pool":
            # with closing(Pool(processes=fem_solver.no_of_cpu_cores)) as pool:
            #     tups = pool.map(super(DetachedParallelFEMSolver, self).Solve,funcs)
            #     pool.terminate()
            raise RuntimeError("Pool based detached parallelism not implemented yet")

        elif self.parallel_model == "mpi":
            raise RuntimeError("MPI based detached parallelism not implemented yet")


        if not self.do_not_sync:
            # FIND COMMON AVAILABLE SOLUTION ACROSS ALL PARTITIONS
            min_nincr = 1e20
            for proc in range(self.no_of_cpu_cores):
                incr = solutions[proc].sol.shape[2]
                if incr < min_nincr:
                    min_nincr = incr

            TotalDisp = np.zeros((mesh.points.shape[0], formulation.nvar, min_nincr))
            for proc in range(self.no_of_cpu_cores):
                pnodes = pnode_indices[proc]
                TotalDisp[pnodes,:,:] = solutions[proc].sol[:,:,:min_nincr]

            return self.__makeoutput__(mesh, TotalDisp, formulation, function_spaces, material)
        else:
            return self.__makeoutput__(mesh, np.zeros_like(mesh.points), formulation, function_spaces, material)



    def __DetachedFEMRunner_ContextManager__(self, formulation, mesh,
                material, boundary_condition,
                function_spaces, solver,
                contact_formulation,
                Eulerx, Eulerp, proc, solutions):
        solution = super(DetachedParallelFEMSolver, self).Solve(formulation=formulation, mesh=mesh,
            material=material, boundary_condition=boundary_condition,
            function_spaces=function_spaces, solver=solver,
            contact_formulation=contact_formulation,
            Eulerx=Eulerx, Eulerp=Eulerp)

        solutions[proc] = solution


    def __DetachedFEMRunner_Pool__(self, formulation, mesh,
                material, boundary_condition,
                function_spaces, solver,
                contact_formulation,
                Eulerx, Eulerp):
        solution = super(DetachedParallelFEMSolver, self).Solve(formulation=formulation, mesh=mesh,
            material=material, boundary_condition=boundary_condition,
            function_spaces=function_spaces, solver=solver,
            contact_formulation=contact_formulation,
            Eulerx=Eulerx, Eulerp=Eulerp)


