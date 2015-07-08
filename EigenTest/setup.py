from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize


extensions = [
    Extension("pytest_addtwo", ["pytest_addtwo.pyx"],
        include_dirs = ['/home/roman/Desktop/EigenTest/'],
        # libraries = [...],
        # library_dirs = [...]
        ),
]

setup(
    ext_modules = cythonize(extensions)
)

# setup(
#     ext_modules = cythonize("pytest_addtwo.pyx")
# )

# python setup.py build_ext --inplace
