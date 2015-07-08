from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

sourcefiles = ['numpy_eigen.pyx', 'cpp_backend.cpp']

extensions = [
    Extension("numpy_eigen",  sourcefiles,
    	language="c++",
        include_dirs = ["/home/roman/Desktop/EigenTest/NumpyEigen/"],
        libraries=["stdc++",":libcpp_backend.so"], 
        library_dirs = ["/home/roman/Desktop/EigenTest/NumpyEigen/"]
        ),
]

setup(
    ext_modules = cythonize(extensions)
)

# setup(
#     ext_modules = cythonize("pytest_addtwo.pyx")
# )

# python setup.py build_ext --inplace
