from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

sourcefiles = ['main_interface.pyx', 'py_to_occ_frontend.cpp','occ_frontend.cpp']
# sourcefiles = ['main_interface.pyx']

extensions = [
    Extension("main_interface",  sourcefiles,
    	language="c++",
        include_dirs = ["/home/roman/Dropbox/Python/Core/Supplementary/cpp_src/occ_backend/","/usr/local/include/eigen_3_2_0/",
        "/usr/local/include/oce/","/usr/include/python2.7"],
        libraries=["stdc++",":py_to_occ_frontend.so",":occ_frontend.so",":libTKIGES.so.9",":libTKXSBase.so.9",":libTKBRep.so.9",
        ":libTKernel.so.9",":libTKTopAlgo.so.9",":libTKGeomBase.so.9",":libTKMath.so.9",":libTKHLR.so.9",
        ":libTKHLR.so.9", ":libTKG2d.so.9", ":libTKBool.so.9", ":libTKG3d.so.9", ":libTKOffset.so.9", ":libTKG2d.so.9",
        ":libTKXMesh.so.9", ":libTKGeomAlgo.so.9", ":libTKShHealing.so.9", ":libTKFeat.so.9", ":libTKFillet.so.9",
        ":libTKBO.so.9", ":libTKPrim.so.9","python2.7"], 
        library_dirs = ["/home/roman/Dropbox/Python/Core/Supplementary/cpp_src/occ_backend/","/usr/local/lib/","/usr/lib/python2.7"]
        ),
]

setup(
    ext_modules = cythonize(extensions)
)
