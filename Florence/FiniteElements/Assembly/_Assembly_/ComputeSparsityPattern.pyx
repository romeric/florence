#cython: profile=False
#cython: infer_types=True
#cython: boundscheck=False
#cython: nonecheck=False

import numpy as np
cimport numpy as np
from libc.stdint cimport int64_t, uint64_t

ctypedef int64_t Integer
ctypedef uint64_t UInteger
ctypedef double Real


cdef extern from "ComputeSparsityPattern.h":

    void _ComputeSparsityPattern_  (
            const int *elements,
            const int *idx_start,
            const int *elem_container,
            int nvar,
            int nnode,
            int nelem,
            int nodeperelem,
            int idx_start_size,
            int *counts,
            int *indices,
            int &nnz
            ) nogil

    void _ComputeDataIndices_  (
            const int *indices,
            const int *indptr,
            int nelem,
            int nvar,
            int nodeperelem,
            const int *elements,
            const long *sorter,
            int *data_local_indices,
            int *data_global_indices
            ) nogil


def ComputeSparsityPattern(mesh, int nvar, squeeze_sparsity_pattern=False):

    cdef int nnode = mesh.points.shape[0]
    cdef int nelem = mesh.nelem
    cdef int nodeperelem = mesh.InferNumberOfNodesPerElement()

    cdef np.ndarray[int, ndim=2, mode='c'] to_c_elements = np.copy(mesh.elements.astype(np.int32))
    cdef np.ndarray[long, ndim=2, mode='c'] sorter = np.argsort(mesh.elements,axis=1)
    to_c_elements = to_c_elements[np.arange(nelem)[:,None], sorter]
    # to_c_elements.sort(axis=1)
    cdef np.ndarray[int, ndim=1, mode='c'] elements = to_c_elements.ravel()
    idx_sort = np.argsort(elements).astype(np.int32)
    sorted_elements = elements[idx_sort]
    cdef np.ndarray[int, ndim=1, mode='c'] elem_container = idx_sort // nodeperelem
    cdef np.ndarray[int, ndim=1, mode='c'] idx_start = np.zeros(nnode+1,dtype=np.int32)
    idx_start[:-1] = np.unique(sorted_elements, return_index=True)[1].astype(np.int32)
    idx_start[-1] = elem_container.shape[0]
    cdef int idx_start_size = idx_start.shape[0]
    cdef np.ndarray[int, ndim=1, mode='c'] counts = np.zeros(idx_start_size-1,dtype=np.int32)

    cdef np.ndarray[int, ndim=1, mode='c'] indices = np.zeros(int((nvar*nodeperelem)**2*nelem),dtype=np.int32)
    cdef int nnz = 0

    _ComputeSparsityPattern_(
        &elements[0],
        &idx_start[0],
        &elem_container[0],
        nvar,
        nnode,
        nelem,
        nodeperelem,
        idx_start_size,
        &counts[0],
        &indices[0],
        nnz
        )
    counts = np.repeat(counts,nvar)

    cdef int ndof = nodeperelem*nvar
    cdef int local_capacity = ndof*ndof
    cdef int all_ndof = nnode*nvar
    cdef int i, max_
    cdef np.ndarray[int, ndim=1, mode='c'] indptr = np.zeros((nnode*nvar+1),dtype=np.int32)
    for i in range(1,indptr.shape[0]):
        max_ = counts[i-1]*nvar if counts[i-1]*nvar <= all_ndof else all_ndof
        indptr[i]   = indptr[i-1]+max_

    indices = indices[:nnz]

    if squeeze_sparsity_pattern:
        return indices, indptr

    # NOW COMPUTE INDICES INTO DATA WHERE THESE INDICES HAVE TO GO
    cdef np.ndarray[int, ndim=1, mode='c'] data_local_indices  = np.zeros(int((local_capacity)*nelem),dtype=np.int32)
    cdef np.ndarray[int, ndim=1, mode='c'] data_global_indices = np.zeros(int((local_capacity)*nelem),dtype=np.int32)

    _ComputeDataIndices_  (
            &indices[0],
            &indptr[0],
            nelem,
            nvar,
            nodeperelem,
            &to_c_elements[0,0],
            &sorter[0,0],
            &data_local_indices[0],
            &data_global_indices[0]
            )

    return indices, indptr, data_local_indices, data_global_indices

