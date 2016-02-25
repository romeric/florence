from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import os
# Get the current directory
_pwd_ = os.path.dirname(os.path.realpath('__file__'))

# sourcefiles = ["JacobiPoly.pyx","jacobi.c"]
sourcefiles = ["JacobiPolynomials.pyx"]
extensions = [
		Extension("JacobiPolynomials",  sourcefiles,
		extra_compile_args = ["-ftree-vectorize"],
		libraries=[":jacobi.so"], 
		library_dirs = [_pwd_],
			),
		]

setup( ext_modules = cythonize(extensions))

