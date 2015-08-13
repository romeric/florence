from distutils.core import setup
from distutils.command.clean import clean
from distutils.extension import Extension
from Cython.Build import cythonize
import os

# Get the current directory
_pwd_ = os.path.dirname(os.path.realpath('__file__'))

# Source files
sourcefiles = ["OCCPluginPy.pyx",_pwd_+"/src/OCCPluginInterface.cpp",_pwd_+"/src/OCCPlugin.cpp"]
# OpenCascade runtime libraries
occ_libs = [":libTKIGES.so.9",":libTKXSBase.so.9",":libTKBRep.so.9",
        ":libTKernel.so.9",":libTKTopAlgo.so.9",":libTKGeomBase.so.9",":libTKMath.so.9",":libTKHLR.so.9",
        ":libTKHLR.so.9", ":libTKG2d.so.9", ":libTKBool.so.9", ":libTKG3d.so.9", ":libTKOffset.so.9", ":libTKG2d.so.9",
        ":libTKXMesh.so.9", ":libTKGeomAlgo.so.9", ":libTKShHealing.so.9", ":libTKFeat.so.9", ":libTKFillet.so.9",
        ":libTKBO.so.9", ":libTKPrim.so.9"]

# Create extension module
extensions = [
    Extension(
        name = "OCCPluginPy",  
        sources = sourcefiles,
    	language="c++",
        include_dirs = [_pwd_+"/include/","/home/roman/Dropbox/eigen-devel/",
        "/usr/local/include/oce/"],
        libraries= ["stdc++"] + occ_libs, 
        library_dirs = [_pwd_+"/include","/usr/local/lib/"],
        extra_compile_args = ["-std=c++11","-msse","-msse2","-msse4","-mavx"]
        ),
]

setup(
    ext_modules = cythonize(extensions)
)

