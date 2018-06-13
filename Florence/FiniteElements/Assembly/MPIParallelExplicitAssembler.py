#!/usr/bin/env python
from mpi4py import MPI
import numpy as np
from time import time


def runParallel():

    from Florence.FiniteElements.Assembly._LowLevelAssembly_ import _LowLevelAssemblyExplicit_Par_
    from Florence import DisplacementFormulation, DisplacementPotentialFormulation

    comm = MPI.Comm.Get_parent()
    size = comm.Get_size()
    rank = comm.Get_rank()

    T_all_size = np.empty(3,'i')
    comm.Bcast(T_all_size, root=0)
    funcs = None
    funcs = comm.bcast(funcs, root=0)

    nnode = T_all_size[0]
    ndim  = T_all_size[1]
    nvar  = T_all_size[2]
    T_all = np.zeros((nnode,nvar),np.float64)

    Eulerx = np.zeros((nnode,ndim),np.float64)
    comm.Bcast([Eulerx, MPI.DOUBLE], root=0)

    Eulerp = np.zeros((nnode),np.float64)
    comm.Bcast([Eulerp, MPI.DOUBLE], root=0)

    for proc in range(size):
        if proc == rank:
            functor = funcs[proc]
            pnodes = funcs[proc].pnodes
            # tt = time()
            T = _LowLevelAssemblyExplicit_Par_(functor.formulation.function_spaces[0],
                functor.formulation, functor.mesh, functor.material, Eulerx[pnodes,:], Eulerp[pnodes])
            T_all[pnodes,:] += T.reshape(pnodes.shape[0],nvar)
            # print(time()-tt)


    comm.Reduce([T_all, MPI.DOUBLE], None, root=0)
    comm.Disconnect()



if __name__ == "__main__":
    runParallel()