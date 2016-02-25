# 
# Out of Core Dictionary
# 
# 
#

from __future__ import division, print_function, absolute_import

__docformat__ = "restructuredtext en"

__all__ = ['dok_matrix', 'isspmatrix_dok']

import functools
import operator
import sqlite3, dill, io, time

import numpy as np

from scipy.lib.six import zip as izip, xrange
from scipy.lib.six import iteritems

from scipy.sparse import spmatrix, isspmatrix
from scipy.sparse.sputils import (isdense, getdtype,
                                  isshape, isintlike, isscalarlike,
                                  upcast, upcast_scalar,
                                  IndexMixin, get_index_dtype)

def def_serial(obj):
    en = io.BytesIO()
    dill.dump(obj, en)
    en.seek(0)
    return en.read()

def def_unserial(obj):
    en = io.BytesIO()
    en.write(obj)
    en.seek(0)
    return dill.load(en)

def int_tuple_ser(width, height):
    return lambda obj: width*obj[0] + obj[1]

def int_tuple_unser(width, height):
    return lambda obj: (int(obj/width), obj % width)


class ddict:
    # 
    # Commit Frequency is in hertz
    # 
    def __init__(self, filename, tablename="dict", clear=True, commit_freq=1.0,
                 key_types = ("BLOB", def_serial, def_unserial),
                 val_types = ("BLOB", def_serial, def_unserial)):
        self.conn = sqlite3.connect(filename)
        self.conn.text_factory = str
        self.cur = self.conn.cursor()
        self.freq = commit_freq
        self.last_commit = time.time()

        self.key_ser, self.key_unser = key_types[1:]
        self.val_ser, self.val_unser = val_types[1:]
        
        if clear:
            self.cur.execute("DROP TABLE IF EXISTS " + tablename)
        self.cur.execute("CREATE TABLE IF NOT EXISTS " + tablename +
                         " (key %s PRIMARY KEY, value %s);" % (key_types[0],
                                                               val_types[0]))
        self.commit()
        self.T = tablename

    def force_commit(self):
        self.conn.commit()
        self.last_commit = time.time()

    def commit(self):
        if self.freq <= 0 or (time.time() - self.last_commit)*self.freq >= 1:
            self.conn.commit()
            self.last_commit = time.time()
        
    def get(self, key, default=None):
        self.cur.execute("SELECT value FROM %s WHERE key=?;" % self.T,
                         (self.key_ser(key),));

        res = self.cur.fetchone()
        if res == None or len(res) <= 0:
            if default == None:
                raise KeyError("The key does not exist: " + str(key))
            else:
                return default

        return self.val_unser(res[0])


    def __getitem__(self, key, default=None):
        return self.get(key, default)

    def __setitem__(self, key, val):
        self.cur.execute(("INSERT OR REPLACE " +
                          "INTO %s VALUES (?, ?)") % self.T,
                         (self.key_ser(key), self.val_ser(val)))
        self.commit()
        
    def __delitem__(self, key):
        self.cur.execute("DELETE FROM %s WHERE key=?;" %
                         self.T, (self.key_ser(key),))
        self.commit()

    def iteritems(self):
        self.cur.execute("SELECT key, value FROM %s" % self.T)
        while True:
            row = self.cur.fetchone()
            if not row:
                break
            else:
                key = self.key_unser(row[0])
                val = self.val_unser(row[1])
                yield (key, val)

    def iterkeys(self):
        for key, val in ddict.iteritems(self):
            yield key

    def itervalues(self):
        for key, val in ddict.iteritems(self):
            yield val

    def keys(self):
        return [k for k in ddict.iterkeys(self)]

    def values(self):
        return [v for v in ddict.itervalues(self)]

    def __iter__(self):
        for key in ddict.iterkeys(self):
            yield key

    def update(self, d):
        if isinstance(d, dict) or isinstance(d, ddict):
            for key, val in d.iteritems():
                # For disambiguation
                ddict.__setitem__(self, key, val)
        else: # Assume stream of pairs
            for k, v in d:
                ddict.__setitem__(self, k, v)
    
    # Can be made more efficient
    def __len__(self):
        return len(ddict.keys(self))

#
# The below code was taken from Scipy
# with minor changes
# 
# 

            
"""Dictionary Of Keys based matrix"""


try:
    from operator import isSequenceType as _is_sequence
except ImportError:
    def _is_sequence(x):
        return (hasattr(x, '__len__') or hasattr(x, '__next__')
                or hasattr(x, 'next'))


class dok_matrix(ddict, spmatrix, IndexMixin):
    """
    Dictionary Of Keys based sparse matrix.
    This is an efficient structure for constructing sparse
    matrices incrementally.
    This can be instantiated in several ways:
        dok_matrix(D)
            with a dense matrix, D
        dok_matrix(S)
            with a sparse matrix, S
        dok_matrix((M,N), [dtype])
            create the matrix with initial shape (M,N)
            dtype is optional, defaulting to dtype='d'
    Attributes
    ----------
    dtype : dtype
        Data type of the matrix
    shape : 2-tuple
        Shape of the matrix
    ndim : int
        Number of dimensions (this is always 2)
    nnz
        Number of nonzero elements
    Notes
    -----
    Sparse matrices can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.
    Allows for efficient O(1) access of individual elements.
    Duplicates are not allowed.
    Can be efficiently converted to a coo_matrix once constructed.
    Examples
    --------
    >>> from scipy.sparse import *
    >>> from scipy import *
    >>> S = dok_matrix((5,5), dtype=float32)
    >>> for i in range(5):
    >>>     for j in range(5):
    >>>         S[i,j] = i+j # Update element
    """

    def __init__(self, arg1, shape=None, filename="sparse.spy",
                 tablename="dok_matrix", dtype=None, copy=False,
                 commit_freq=1.0):
        spmatrix.__init__(self)

        self.dtype = getdtype(dtype, default=float)
        if isinstance(arg1, tuple) and isshape(arg1):  # (M,N)
            M, N = arg1
            self.shape = (M, N)
        elif isspmatrix(arg1):  # Sparse ctor
            if isspmatrix_dok(arg1) and copy:
                arg1 = arg1.copy()
            else:
                arg1 = arg1.todok()

            if dtype is not None:
                arg1 = arg1.astype(dtype)

            self.shape = arg1.shape
            self.dtype = arg1.dtype
        else:  # Dense ctor
            try:
                arg1 = np.asarray(arg1)
            except:
                raise TypeError('invalid input format')

            if len(arg1.shape) != 2:
                raise TypeError('expected rank <=2 dense array or matrix')

            from scipy.sparse.coo import coo_matrix
            d = coo_matrix(arg1, dtype=dtype).todok()
            self.shape = arg1.shape
            self.dtype = d.dtype

        ddict.__init__(self, filename, tablename=tablename,
                       commit_freq=commit_freq,
                       key_types=("UNSIGNED INTEGER", 
                                  int_tuple_ser(*self.shape), 
                                  int_tuple_unser(*self.shape)),
                       val_types=("REAL", float, float))
        
        if isspmatrix(arg1):  # Sparse ctor
            ddict.update(self, arg1)
        elif not (isinstance(arg1, tuple) and isshape(arg1)):
            ddict.update(self, d)
            

    def getnnz(self):
        return ddict.__len__(self)
    nnz = property(fget=getnnz)

    def __len__(self):
        return ddict.__len__(self)

    def get(self, key, default=0.):
        """This overrides the dict.get method, providing type checking
        but otherwise equivalent functionality.
        """
        try:
            i, j = key
            assert isintlike(i) and isintlike(j)
        except (AssertionError, TypeError, ValueError):
            raise IndexError('index must be a pair of integers')
        if (i < 0 or i >= self.shape[0] or j < 0 or j >= self.shape[1]):
            raise IndexError('index out of bounds')
        return ddict.get(self, key, default)

    def __getitem__(self, index):
        """If key=(i,j) is a pair of integers, return the corresponding
        element.  If either i or j is a slice or sequence, return a new sparse
        matrix with just these elements.
        """
        i, j = self._unpack_index(index)

        i_intlike = isintlike(i)
        j_intlike = isintlike(j)

        if i_intlike and j_intlike:
            # Scalar index case
            i = int(i)
            j = int(j)
            if i < 0:
                i += self.shape[0]
            if i < 0 or i >= self.shape[0]:
                raise IndexError('index out of bounds')
            if j < 0:
                j += self.shape[1]
            if j < 0 or j >= self.shape[1]:
                raise IndexError('index out of bounds')
            return ddict.get(self, (i,j), 0.)
        elif ((i_intlike or isinstance(i, slice)) and
              (j_intlike or isinstance(j, slice))):
            # Fast path for slicing very sparse matrices
            i_slice = slice(i, i+1) if i_intlike else i
            j_slice = slice(j, j+1) if j_intlike else j
            i_indices = i_slice.indices(self.shape[0])
            j_indices = j_slice.indices(self.shape[1])
            i_seq = xrange(*i_indices)
            j_seq = xrange(*j_indices)
            newshape = (len(i_seq), len(j_seq))
            newsize = _prod(newshape)

            if len(self) < 2*newsize and newsize != 0:
                # Switch to the fast path only when advantageous
                # (count the iterations in the loops, adjust for complexity)
                #
                # We also don't handle newsize == 0 here (if
                # i/j_intlike, it can mean index i or j was out of
                # bounds)
                return self._getitem_ranges(i_indices, j_indices, newshape)

        i, j = self._index_to_arrays(i, j)

        if i.size == 0:
            return dok_matrix(i.shape, dtype=self.dtype)

        min_i = i.min()
        if min_i < -self.shape[0] or i.max() >= self.shape[0]:
            raise IndexError('index (%d) out of range -%d to %d)' %
                             (i.min(), self.shape[0], self.shape[0]-1))
        if min_i < 0:
            i = i.copy()
            i[i < 0] += self.shape[0]

        min_j = j.min()
        if min_j < -self.shape[0] or j.max() >= self.shape[1]:
            raise IndexError('index (%d) out of range -%d to %d)' %
                             (j.min(), self.shape[1], self.shape[1]-1))
        if min_j < 0:
            j = j.copy()
            j[j < 0] += self.shape[1]

        newdok = dok_matrix(i.shape, dtype=self.dtype)

        for a in xrange(i.shape[0]):
            for b in xrange(i.shape[1]):
                v = ddict.get(self, (i[a,b], j[a,b]), 0.)
                if v != 0:
                    ddict.__setitem__(newdok, (a, b), v)

        return newdok

    def _getitem_ranges(self, i_indices, j_indices, shape):
        # performance golf: we don't want Numpy scalars here, they are slow
        i_start, i_stop, i_stride = map(int, i_indices)
        j_start, j_stop, j_stride = map(int, j_indices)

        newdok = dok_matrix(shape, dtype=self.dtype)

        for (ii, jj) in self.keys():
            # ditto for numpy scalars
            ii = int(ii)
            jj = int(jj)
            a, ra = divmod(ii - i_start, i_stride)
            if a < 0 or a >= shape[0] or ra != 0:
                continue
            b, rb = divmod(jj - j_start, j_stride)
            if b < 0 or b >= shape[1] or rb != 0:
                continue
            ddict.__setitem__(newdok, (a, b),
                             ddict.__getitem__(self, (ii, jj)))

        return newdok

    def __setitem__(self, index, x):
        if isinstance(index, tuple) and len(index) == 2:
            # Integer index fast path
            i, j = index
            if (isintlike(i) and isintlike(j) and 
                0 <= i < self.shape[0] and 0 <= j < self.shape[1]):
                v = np.asarray(x, dtype=self.dtype)
                if v.ndim == 0 and v != 0:
                    ddict.__setitem__(self, (int(i), int(j)), v[()])
                    return

        i, j = self._unpack_index(index)
        i, j = self._index_to_arrays(i, j)

        if isspmatrix(x):
            x = x.toarray()

        # Make x and i into the same shape
        x = np.asarray(x, dtype=self.dtype)
        x, _ = np.broadcast_arrays(x, i)

        if x.shape != i.shape:
            raise ValueError("shape mismatch in assignment")

        if np.size(x) == 0:
            return

        min_i = i.min()
        if min_i < -self.shape[0] or i.max() >= self.shape[0]:
            raise IndexError('index (%d) out of range -%d to %d)' %
                             (i.min(), self.shape[0], self.shape[0]-1))
        if min_i < 0:
            i = i.copy()
            i[i < 0] += self.shape[0]

        min_j = j.min()
        if min_j < -self.shape[0] or j.max() >= self.shape[1]:
            raise IndexError('index (%d) out of range -%d to %d)' %
                             (j.min(), self.shape[1], self.shape[1]-1))
        if min_j < 0:
            j = j.copy()
            j[j < 0] += self.shape[1]

        ddict.update(self, izip(izip(i.flat, j.flat), x.flat))

        if 0 in x:
            zeroes = x == 0
            for key in izip(i[zeroes].flat, j[zeroes].flat):
                if ddict.__getitem__(self, key) == 0:
                    # may have been superseded by later update
                    del self[key]

    def __add__(self, other):
        # First check if argument is a scalar
        if isscalarlike(other):
            res_dtype = upcast_scalar(self.dtype, other)
            new = dok_matrix(self.shape, dtype=res_dtype)
            # Add this scalar to every element.
            M, N = self.shape
            for i in xrange(M):
                for j in xrange(N):
                    aij = self.get((i, j), 0) + other
                    if aij != 0:
                        new[i, j] = aij
            # new.dtype.char = self.dtype.char
        elif isinstance(other, dok_matrix):
            if other.shape != self.shape:
                raise ValueError("matrix dimensions are not equal")
            # We could alternatively set the dimensions to the largest of
            # the two matrices to be summed.  Would this be a good idea?
            res_dtype = upcast(self.dtype, other.dtype)
            new = dok_matrix(self.shape, dtype=res_dtype)
            new.update(self)
            for key in other.keys():
                new[key] += other[key]
        elif isspmatrix(other):
            csc = self.tocsc()
            new = csc + other
        elif isdense(other):
            new = self.todense() + other
        else:
            raise TypeError("data type not understood")
        return new

    def __radd__(self, other):
        # First check if argument is a scalar
        if isscalarlike(other):
            new = dok_matrix(self.shape, dtype=self.dtype)
            # Add this scalar to every element.
            M, N = self.shape
            for i in xrange(M):
                for j in xrange(N):
                    aij = self.get((i, j), 0) + other
                    if aij != 0:
                        new[i, j] = aij
        elif isinstance(other, dok_matrix):
            if other.shape != self.shape:
                raise ValueError("matrix dimensions are not equal")
            new = dok_matrix(self.shape, dtype=self.dtype)
            new.update(self)
            for key in other:
                new[key] += other[key]
        elif isspmatrix(other):
            csc = self.tocsc()
            new = csc + other
        elif isdense(other):
            new = other + self.todense()
        else:
            raise TypeError("data type not understood")
        return new

    def __neg__(self):
        new = dok_matrix(self.shape, dtype=self.dtype)
        for key in self.keys():
            new[key] = -self[key]
        return new

    def _mul_scalar(self, other):
        res_dtype = upcast_scalar(self.dtype, other)
        # Multiply this scalar by every element.
        new = dok_matrix(self.shape, dtype=res_dtype)
        for (key, val) in iteritems(self):
            new[key] = val * other
        return new

    def _mul_vector(self, other):
        # matrix * vector
        result = np.zeros(self.shape[0], dtype=upcast(self.dtype,other.dtype))
        for (i,j),v in iteritems(self):
            result[i] += v * other[j]
        return result

    def _mul_multivector(self, other):
        # matrix * multivector
        M,N = self.shape
        n_vecs = other.shape[1]  # number of column vectors
        result = np.zeros((M,n_vecs), dtype=upcast(self.dtype,other.dtype))
        for (i,j),v in iteritems(self):
            result[i,:] += v * other[j,:]
        return result

    def __imul__(self, other):
        if isscalarlike(other):
            # Multiply this scalar by every element.
            for (key, val) in iteritems(self):
                self[key] = val * other
            # new.dtype.char = self.dtype.char
            return self
        else:
            raise NotImplementedError

    def __truediv__(self, other):
        if isscalarlike(other):
            res_dtype = upcast_scalar(self.dtype, other)
            new = dok_matrix(self.shape, dtype=res_dtype)
            # Multiply this scalar by every element.
            for (key, val) in iteritems(self):
                new[key] = val / other
            # new.dtype.char = self.dtype.char
            return new
        else:
            return self.tocsr() / other

    def __itruediv__(self, other):
        if isscalarlike(other):
            # Multiply this scalar by every element.
            for (key, val) in iteritems(self):
                self[key] = val / other
            return self
        else:
            raise NotImplementedError

    # What should len(sparse) return? For consistency with dense matrices,
    # perhaps it should be the number of rows?  For now it returns the number
    # of non-zeros.

    def transpose(self):
        """ Return the transpose
        """
        M, N = self.shape
        new = dok_matrix((N, M), dtype=self.dtype)
        for key, value in iteritems(self):
            new[key[1], key[0]] = value
        return new

    def conjtransp(self):
        """ Return the conjugate transpose
        """
        M, N = self.shape
        new = dok_matrix((N, M), dtype=self.dtype)
        for key, value in iteritems(self):
            new[key[1], key[0]] = np.conj(value)
        return new

    def copy(self):
        new = dok_matrix(self.shape, dtype=self.dtype)
        new.update(self)
        return new

    def getrow(self, i):
        """Returns a copy of row i of the matrix as a (1 x n)
        DOK matrix.
        """
        out = self.__class__((1, self.shape[1]), dtype=self.dtype)
        for j in range(self.shape[1]):
            out[0, j] = self[i, j]
        return out

    def getcol(self, j):
        """Returns a copy of column j of the matrix as a (m x 1)
        DOK matrix.
        """
        out = self.__class__((self.shape[0], 1), dtype=self.dtype)
        for i in range(self.shape[0]):
            out[i, 0] = self[i, j]
        return out

    def tocoo(self):
        """ Return a copy of this matrix in COOrdinate format"""
        from scipy.sparse.coo import coo_matrix
        if self.nnz == 0:
            return coo_matrix(self.shape, dtype=self.dtype)
        else:
            idx_dtype = get_index_dtype(maxval=max(self.shape[0],
                                                   self.shape[1]))
            data    = np.asarray(_list(self.values()),
                                 dtype=self.dtype)
            indices = np.asarray(_list(self.keys()),
                                 dtype=idx_dtype).T
            return coo_matrix((data,indices), shape=self.shape,
                              dtype=self.dtype)

    def todok(self,copy=False):
        if copy:
            return self.copy()
        else:
            return self

    def tocsr(self):
        """ Return a copy of this matrix in Compressed Sparse Row format"""
        return self.tocoo().tocsr()

    def tocsc(self):
        """ Return a copy of this matrix in Compressed Sparse Column format"""
        return self.tocoo().tocsc()

    def toarray(self, order=None, out=None):
        """See the docstring for `spmatrix.toarray`."""
        return self.tocoo().toarray(order=order, out=out)

    def resize(self, shape):
        """ Resize the matrix in-place to dimensions given by 'shape'.
        Any non-zero elements that lie outside the new shape are removed.
        """
        if not isshape(shape):
            raise TypeError("dimensions must be a 2-tuple of positive"
                             " integers")
        newM, newN = shape
        M, N = self.shape
        if newM < M or newN < N:
            # Remove all elements outside new dimensions
            for (i, j) in list(self.keys()):
                if i >= newM or j >= newN:
                    del self[i, j]
        self._shape = shape


def _list(x):
    """Force x to a list."""
    if not isinstance(x, list):
        x = list(x)
    return x


def isspmatrix_dok(x):
    return isinstance(x, dok_matrix)


def _prod(x):
    """Product of a list of numbers; ~40x faster vs np.prod for Python tuples"""
    if len(x) == 0:
        return 1
    return functools.reduce(operator.mul, x)