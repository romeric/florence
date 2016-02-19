from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

# Create extension module
extensions = [
    Extension(
        name = "get_duplicates",  
        sources = ["get_duplicates.pyx","findin.cpp"],
        language="c++",
        libraries= ["stdc++"], 
        extra_compile_args = ["-std=c++11","-fopenmp"],
        extra_link_args=['-fopenmp'],
        ),
]

setup(
    ext_modules = cythonize(extensions)
)