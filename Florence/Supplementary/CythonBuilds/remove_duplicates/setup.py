from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

# Create extension module
extensions = [
    Extension(
        name = "remove_duplicates",  
        sources = ["remove_duplicates.pyx"],
        language="c++",
        libraries= ["stdc++"], 
        ),
]

setup(
    ext_modules = cythonize(extensions)
)
