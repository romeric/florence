from distutils.core import setup
from distutils.command.clean import clean
from distutils.extension import Extension
from distutils.sysconfig import get_config_vars 
from Cython.Build import cythonize
import os, platform
import sys

# Get the current directory
_pwd_ = os.path.dirname(os.path.realpath('__file__'))

# Remove the "-Wstrict-prototypes" compiler option, which isn't valid for C++.
cfg_vars = get_config_vars()
for key, value in cfg_vars.items():
    if isinstance(value,str):
        cfg_vars[key] = value.replace("-Wstrict-prototypes", "")

# Suppress numpy deprecation warnings
no_deprecated = ("NPY_NO_DEPRECATED_API",None)

# Compiler arguments
compiler_args = ["-std=c++11","-march=native","-mtune=native",
                "-mfpmath=sse","-ffast-math","-ftree-vectorize",
                "-funroll-loops","-finline-functions","-Wno-unused-function",
                "-flto","-DNPY_NO_DEPRECATED_API","-Wno-cpp"]

# Source files
# sourcefiles = ["PostMeshPy.pyx",_pwd_+"/src/PyInterfaceEmulator.cpp",
        # _pwd_+"/src/PostMeshBase.cpp",_pwd_+"/src/PostMeshCurve.cpp",_pwd_+"/src/PostMeshSurface.cpp"]

sourcefiles = ["PostMeshPy.pyx",
                _pwd_+"/src/PostMeshBase.cpp",
                _pwd_+"/src/PostMeshCurve.cpp",
                _pwd_+"/src/PostMeshSurface.cpp"]

# OpenCascade runtime libraries
occ_libs = [":libTKIGES.so.9",":libTKSTEP.so.9",":libTKXSBase.so.9",":libTKBRep.so.9",
        ":libTKernel.so.9",":libTKTopAlgo.so.9",":libTKGeomBase.so.9",":libTKMath.so.9",":libTKHLR.so.9",
        ":libTKHLR.so.9", ":libTKG3d.so.9", ":libTKBool.so.9", ":libTKG3d.so.9", ":libTKOffset.so.9", ":libTKG2d.so.9",
        ":libTKXMesh.so.9", ":libTKMesh.so.9", ":libTKMeshVS.so.9",":libTKGeomAlgo.so.9", ":libTKShHealing.so.9", ":libTKFeat.so.9", 
        ":libTKFillet.so.9", ":libTKBO.so.9", ":libTKPrim.so.9"]

# Create extension module
extensions = [
    Extension(
        name = "PostMeshPy",  
        sources = sourcefiles,
    	language="c++",
        include_dirs = [_pwd_,_pwd_+"/include/",
                        "/home/roman/Dropbox/eigen-devel/",
                        "/usr/local/include/oce/"],
        libraries= ["stdc++"] + occ_libs, 
        library_dirs = [_pwd_,_pwd_+"/include","/usr/local/lib/"],
        extra_compile_args = compiler_args,
        define_macros=[no_deprecated],
        ),
]

setup(
    ext_modules = cythonize(extensions),
    description = "A Python wrapper for PostMesh - a high order curvilinear mesh generator based on OpenCascade",
    author="Roman Poya",
    author_email = "r.poya@swansea.ac.uk",
    url = "https://github.com/romeric/PostMesh",
    version = "0.1",
)
