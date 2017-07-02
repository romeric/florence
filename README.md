[![Build Status](https://travis-ci.com/romeric/florence.svg?token=HFW6d19YsYpKDNwvtqDr&branch=master)](https://travis-ci.com/romeric/florence)

**Florence** is a Python-based computational framework for multi-physics simulation of electro-magneto-mechanical systems. The framework also includes Python interfaces to many low-level numerical sub-routines written in C, C++ and Cython.


# Dependencies
The followings are hard dependencies
- [PostMesh](https://github.com/romeric/PostMesh):      High order curvilinear mesh generator
- [Fastor](https://github.com/romeric/Fastor):          Data parallel (SIMD) FEM assembler
- Cython
- NumPy
- SciPy

# Installation
Have a look at `travis.yml` file directions on installation of core library. Installation of the core library (not external dependencies) is as easy as

```
git clone https://github.com/romeric/florence
cd florence
python setup.py build
export PYTHONPATH="/path/to/florence:$PYTHONPATH"
```

This builds many low-level cython modules ahead of time. Options can be given to `setup.py` for instance

```
python setup.py build BLAS=mkl CXX=/usr/local/bin/g++ CC=~/LLVM/clang
```

Optional external dependies such as `MUMPS` solver, `Pardiso` solver and `mayavi` typically need special care for installation.

To install `MUMPS`, use `homebrew` on macOS and `linuxbrew` on linux:

```
brew install mumps --without-mpi --with-openblas
git clone https://github.com/romeric/MUMPS.py
cd MUMPS.py
python setup.py build
python setup.py install
```

And whenever `MUMPS` solver is needed, just open a new terminal window/tab and do
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/linuxbrew/.linuxbrew/lib
```

The direct sparse solver shipped with `MKL`, `Pardiso` can be used if `MKL` is available. Both Anaconda and Intel distribution for python ship these.
If `MKL` is installed, the low-level FEM assembler in florence is also automatically linked to it during compilation, as long as "`BLAS=mkl`" flag is issued to `setup.py`.

```shell
conda install -c haasad pypardiso
```
We typically do not recommed adding `anaconda/bin` to your path. Hence, whenever `MKL` features or `Pardiso` solver is needed, just open a new terminal window/tab and do

```
export PATH="/Users/romanpoya/anaconda2/bin:$PATH"
```

