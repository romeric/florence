from distutils.core import setup
from distutils.command.clean import clean
from distutils.extension import Extension
from Cython.Build import cythonize
import os, platform
import sys

# Get the current directory
_pwd_ = os.path.dirname(os.path.realpath('__file__'))

# Compiler arguments
compiler_args = ["-std=c++11"]

# Determine OS 
if os.name is "posix":
    # Determine intrinsics
    for line in open("/proc/cpuinfo"):    
        item  = line.rsplit()
        if isinstance(item,list) and len(item)>0 and item[0]=='flags':
            if 'sse' in item and '-msse' not in compiler_args:
                compiler_args.append("-msse")
            if 'sse2' in item and '-msse2' not in compiler_args:
                compiler_args.append("-msse2")
            if 'ssse3' in item and '-mssse3' not in compiler_args:
                compiler_args.append("-mssse3")
            if 'sse4' in item and '-msse4' not in compiler_args:
                compiler_args.append("-msse4")
            if 'avx' in item and '-mavx' not in compiler_args:
                compiler_args.append("-mavx")
            if 'avx2' in item and '-mavx2' not in compiler_args:
                compiler_args.append("-mavx2")
            if 'fma' in item and '-mfma' not in compiler_args:
                compiler_args.append("-mfma")

elif os.name is "windows":
    pass

# Source files
sourcefiles = ["PostMeshPy.pyx",_pwd_+"/src/PyInterfaceEmulator.cpp",
        _pwd_+"/src/PostMeshBase.cpp",_pwd_+"/src/PostMeshCurve.cpp",_pwd_+"/src/PostMeshSurface.cpp"]

# OpenCascade runtime libraries
occ_libs = [":libTKIGES.so.9",":libTKSTEP.so.9",":libTKXSBase.so.9",":libTKBRep.so.9",
        ":libTKernel.so.9",":libTKTopAlgo.so.9",":libTKGeomBase.so.9",":libTKMath.so.9",":libTKHLR.so.9",
        ":libTKHLR.so.9", ":libTKG2d.so.9", ":libTKBool.so.9", ":libTKG3d.so.9", ":libTKOffset.so.9", ":libTKG2d.so.9",
        ":libTKXMesh.so.9", ":libTKGeomAlgo.so.9", ":libTKShHealing.so.9", ":libTKFeat.so.9", ":libTKFillet.so.9",
        ":libTKBO.so.9", ":libTKPrim.so.9"]

# Create extension module
extensions = [
    Extension(
        name = "PostMeshPy",  
        sources = sourcefiles,
    	language="c++",
        include_dirs = [_pwd_,_pwd_+"/include/","/home/roman/Dropbox/eigen-devel/",
        "/usr/local/include/oce/"],
        libraries= ["stdc++"] + occ_libs, 
        library_dirs = [_pwd_,_pwd_+"/include","/usr/local/lib/"],
        extra_compile_args = compiler_args + ["-O3"]
        ),
]

setup(
    ext_modules = cythonize(extensions)
)

# extra_compile_args = compiler_args + ["-O3", "-fopenmp"]