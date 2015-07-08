from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

sourcefiles = ['numpy_eigen.pyx', 'convert_to_eigen.cpp']

extensions = [
    Extension("numpy_eigen",  sourcefiles,
    	language="c++",
        include_dirs = ["/home/roman/Desktop/EigenTest/NumpyEigen4/"],
        libraries=["stdc++",":convert_to_eigen.so"], 
        library_dirs = ["/home/roman/Desktop/EigenTest/NumpyEigen4/"]
        ),
]

setup(
    ext_modules = cythonize(extensions)
)
