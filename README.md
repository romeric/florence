[![Build Status](https://travis-ci.com/romeric/florence.svg?token=HFW6d19YsYpKDNwvtqDr&branch=master)](https://travis-ci.com/romeric/florence)
[![Coverage Status](https://coveralls.io/repos/github/romeric/florence/badge.svg?branch=master&service=github)](https://coveralls.io/github/romeric/florence?branch=master)

**Florence** is a Python-based computational framework for multi-physics simulation of electro-magneto-mechanical systems using the finite element and boundary element methods.

# Features
A non-exhaustive list of core features:
- High order planar and curved finite and boundary elements (line, tri, quad, tet, hex)
- In-built CAD-conformal curvilinear mesh generator
- Powerful in-built pre and post processor
- Poisson, electrostatic and heat transfer solvers
- Linear, geometrically linearised and fully nonlinear solid/structural mechanics solvers
- Linear, geometrically linearised and fully nonlinear electromechanics solvers
- Implicit and explicit dynamic solver with contact formulation
- Generic monolithic, staggered and multigrid solvers for coupled multiphysics driven problems
- Strain gradient and micropolar elasticity and electro-elasticty solvers
- A suite of advanced hyperelastic, electrostatic and electro-hyperelastic material models
- Ability to read/write mesh/simulation data to/from gmsh, Salome, GID, Tetgen, obj, FRO, VTK and HDF5
- Support for heterogeneous computing using SIMD, shared parallelism, cloud-based parallelism and cluster-based parallelism
- Interface to a suite of sparse direct and iterative solvers including MUMPS, Pardiso & AMG

In addition, the framework also provides Python interfaces to many low-level numerical subroutines written in C, C++ and Cython.

# Platform support
Florence supports Linux, macOS and Windows (under Cygwin) under
- Python 2.7
- Python >= 3.5
- PyPy >= v5.7.0


# Dependencies
The following packages are hard dependencies
- [Fastor](https://github.com/romeric/Fastor):          Data parallel (SIMD) FEM assembler
- cython
- numpy
- scipy

The following packages are optional (but recommended) dependencies
- [PostMesh](https://github.com/romeric/PostMesh):      High order curvilinear mesh generator
- pyevtk
- matplotlib
- mayavi
- scikit-umfpack
- pyamg
- psutil
- h5py

In addition, it is recommended to have an optimised BLAS library such as OpenBLAS or MKL installed and configured on your machine.

# Installation
## The easy way
using pip

```
pip install Florence
```

For pip installation to work you need to have `Fastor` installed. You can achieve this by

```
cd ~
git clone https://github.com/romeric/Fastor
mv Fastor/ /usr/local/include/Fastor/
```

It is also a good practice to set your compilers before pip installing florence

```
export CC=/path/to/c/compiler
export CXX=/path/to/c++/compiler
```

## Building from source
Have a look at `travis.yml` file for directions on installing florence's core library. First install `cython`, `numpy` and `scipy`. Install `Fastor` using

```
cd ~
git clone https://github.com/romeric/Fastor
mv Fastor/ /usr/local/include/Fastor/
```

Then installation of the core library is as easy as

```
git clone https://github.com/romeric/florence
cd florence
python setup.py build
export PYTHONPATH="/path/to/florence:$PYTHONPATH"
```

This builds many low-level cython modules, ahead of time. Options can be given to `setup.py` for instance

```
python setup.py build BLAS=mkl CXX=/usr/local/bin/g++ CC=~/LLVM/clang
```

By default, florence builds in parallel using all the machine's CPU cores. To limit the build process to a specific number of cores, use the `np` flag for instance, for serial build one can trigger the build process as

```
python setup.py build np=1
```

## Configuring MUMPS direct sparse solver
Florence can automatically switch to `MUMPS` sparse direct solver if available. To install `MUMPS`, the easiest way is to use `homebrew` on macOS and `linuxbrew` on linux:

```
brew install mumps --without-mpi --with-openblas
git clone https://github.com/romeric/MUMPS.py
cd MUMPS.py
python setup.py build
python setup.py install
```

And whenever `MUMPS` solver is needed, just open a new terminal window/tab and do (this is the default setting for linuxbrew)
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/linuxbrew/.linuxbrew/lib
```

## Configuring Pardiso direct sparse solver
The direct sparse solver shipped with `MKL`, `Pardiso` can be used if `MKL` is available. Both Anaconda and Intel distribution for python ship these.
If `MKL` is installed, the low-level FEM assembler in florence is also automatically linked to it during compilation, as long as "`BLAS=mkl`" flag is issued to `setup.py`.

```shell
conda install -c haasad pypardiso
```
We typically do not recommed adding `anaconda/bin` to your path. Hence, whenever `MKL` features or `Pardiso` solver is needed, just open a new terminal window/tab and type

```
export PATH="/path/to/anaconda2/bin:$PATH"
```

# Documentation
A series of well explained examples are provided in the example folder that cover most of the functionality of florence. As an example, setting up and solving the Laplace equation using fourth order hexahedral Lagrange shape functions over a cube is as simple as

~~~python
import numpy as np
from Florence import *


def simple_laplace():
    """An example of solving the Laplace equation using
        fourth order hexahedral elements on a cube
    """

    # generate a linear hexahedral mesh on a cube
    mesh = Mesh()
    mesh.Cube(element_type="hex", nx=6, ny=6, nz=6)
    # generate the corresponding fourth order mesh
    mesh.GetHighOrderMesh(p=4)

    # set up boundary conditions
    def dirichlet_function(mesh):
        # create boundary flags - nan values would be treated as free boundary
        boundary_data = np.zeros(mesh.nnode)+np.NAN
        # potential at left (Y=0)
        Y_0 = np.isclose(mesh.points[:,1],0)
        boundary_data[Y_0] = 0.
        # potential at right (Y=1)
        Y_1 = np.isclose(mesh.points[:,1],mesh.points[:,1].max())
        boundary_data[Y_1] = 10.

        return boundary_data

    boundary_condition = BoundaryCondition()
    boundary_condition.SetDirichletCriteria(dirichlet_function, mesh)

    # set up material
    material = IdealDielectric(mesh.InferSpatialDimension(), eps=2.35)
    # set up variational form
    formulation = LaplacianFormulation(mesh)
    # set up solver
    fem_solver = FEMSolver(optimise=True)
    # solve
    results = fem_solver.Solve( boundary_condition=boundary_condition,
                                material=material,
                                formulation=formulation,
                                mesh=mesh)

    # write results to vtk file
    results.WriteVTK("laplacian_results")


if __name__ == "__main__":
    simple_laplace()
~~~