from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("whereEQ_1HitCount.pyx")
)