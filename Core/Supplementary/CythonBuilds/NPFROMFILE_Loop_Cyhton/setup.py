from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("NPFROMFILE_Loop_Cython.pyx")
)