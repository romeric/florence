from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

# sourcefiles = ["JacobiPoly.pyx","jacobi.c"]
sourcefiles = ["JacobiPolynomials.pyx"]
extensions = [
		Extension("JacobiPolynomials",  sourcefiles,
		extra_compile_args = ["-ftree-vectorize"],
		libraries=[":jacobi.so"], 
		library_dirs = ["/home/roman/Dropbox/zDumps/DumpStudies/jacobi"],
			),
		]

setup( ext_modules = cythonize(extensions))

