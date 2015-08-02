from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

sourcefiles = ["OCCPluginPy.pyx","OCCPluginInterface.cpp","OCCPlugin.cpp"]

extensions = [
    Extension("OCCPluginPy",  sourcefiles,
    	language="c++",
        include_dirs = ["/home/roman/Dropbox/Python/Core/Supplementary/cpp_src/occ_backend/","/home/roman/Dropbox/eigen-devel/",
        "/usr/local/include/oce/"],
        libraries=["stdc++",":OCCPluginInterface.so",":OCCPlugin.so",":libTKIGES.so.9",":libTKXSBase.so.9",":libTKBRep.so.9",
        ":libTKernel.so.9",":libTKTopAlgo.so.9",":libTKGeomBase.so.9",":libTKMath.so.9",":libTKHLR.so.9",
        ":libTKHLR.so.9", ":libTKG2d.so.9", ":libTKBool.so.9", ":libTKG3d.so.9", ":libTKOffset.so.9", ":libTKG2d.so.9",
        ":libTKXMesh.so.9", ":libTKGeomAlgo.so.9", ":libTKShHealing.so.9", ":libTKFeat.so.9", ":libTKFillet.so.9",
        ":libTKBO.so.9", ":libTKPrim.so.9"], 
        library_dirs = ["/home/roman/Dropbox/Python/Core/Supplementary/cpp_src/occ_backend/","/usr/local/lib/"],
        extra_compile_args = ['-std=c++11']
        ),
]

setup(
    ext_modules = cythonize(extensions)
)
