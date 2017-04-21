from __future__ import print_function
import os, platform, sys, subprocess
from distutils.core import setup
from distutils.command.clean import clean
from distutils.extension import Extension
from distutils.sysconfig import get_config_vars
try:
    from Cython.Build import cythonize
except ImportError:
    raise ImportError("Could not import cython. Please install cython first")
try:
    import numpy as np
except ImportError:
    raise ImportError("Could not import numpy. Please install numpy first")



class FlorenceSetup(object):

    os = sys.platform

    python_version = None
    python_interpreter = None
    python_include_path = None
    python_ld_path = None

    numpy_version = None
    numpy_include_path = None

    fc_compiler = None
    cc_compiler = None
    cxx_compiler = None

    compiler_args = None
    fc_compiler_args = None
    cc_compiler_args = None
    cxx_compiler_args = None

    extension_paths = None

    def __init__(self, _fc_compiler=None, _cc_compiler=None, _cxx_compiler=None):

        # GET THE CURRENT WORKING DIRECTORY
        self._pwd_ = os.path.dirname(os.path.realpath('__file__'))
        # Get python version and paths to header
        self.GetPythonPath()
        # Get numpy version and paths to header
        self.GetNumPyPath()
        # Set C/Fortran/C++ compiler
        self.SetCompiler(_fc_compiler,_cc_compiler,_cxx_compiler)
        # Set up compiler arguments for all extension modules
        self.SetCompilerArgs()
        # Collect all extension module paths
        self.CollectExtensionModulePaths()


    def GetPythonPath(self):

        if self._pwd_ is None:
            self._pwd_ = os.path.dirname(os.path.realpath('__file__'))

        # Get python version
        python_version = sys.version_info

        if python_version[:2] < (2, 7) or (3, 0) <= python_version[:2] < (3, 5):
            raise RuntimeError("Python version 2.7 or >= 3.5 required.")

        py_major = python_version.major
        py_minor = python_version.minor
        py_micro = python_version.micro
        self.python_interpreter  = 'python' + str(py_major) +'.' + str(py_minor)

        # Get python version
        actual_path = None
        for path in sys.path:
            lpath = path.split("/")
            if lpath[-1] == self.python_interpreter:
                actual_path = path
                break

        self.python_ld_path = actual_path
        self.python_include_path = self.python_ld_path.replace("lib","include")


    def GetNumPyPath(self):
        numpy_version = np.__version__
        if numpy_version.split('.')[0]==1 and numpy_version.split('.')[1] < 8:
            raise RuntimeError("Numpy version >= 1.8 required")
        elif numpy_version.split('.')[0] < 1:
            raise RuntimeError("Numpy version >= 1.8 required")
        self.numpy_include_path = np.get_include()


    def SetCompiler(self, _fc_compiler=None, _cc_compiler=None, _cxx_compiler=None):

        if not "darwin" in self.os and not "linux" in self.os:
            raise RuntimeError("Florence is not yet tested on any other platform apart from Linux & macOS")

        self.fc_compiler = _fc_compiler
        self.cc_compiler = _cc_compiler
        self.cxx_compiler = _cxx_compiler

        if self.fc_compiler is None:
            self.fc_compiler = "gfortran"

        if self.cc_compiler is None:
            if "darwin" in self.os:
                self.cc_compiler = "clang"
            elif "linux" in self.os:
                self.cc_compiler = "gcc"

        if self.cxx_compiler is None:
            if "darwin" in self.os:
                self.cxx_compiler = "clang++"
            elif "linux" in self.os:
                self.cxx_compiler = "g++"

    def SetCompilerArgs(self):
        # Generic compiler arguments
        self.compiler_args = "PYTHON_VERSION=" + self.python_interpreter + " PYTHON_INCLUDE_PATH=" + \
            self.python_include_path + " PYTHON_LD_PATH=" + self.python_ld_path + \
            " NUMPY_INCLUDE_PATH=" + self.numpy_include_path

        self.fc_compiler_args = "FC=" + self.fc_compiler + " " + self.compiler_args
        self.cc_compiler_args = "CC=" + self.cc_compiler + " " + self.compiler_args
        self.cxx_compiler_args = "CXX=" + self.cxx_compiler + " " + self.compiler_args

        self.compiler_args = "FC=" + self.fc_compiler + " " + "CC=" + self.cc_compiler + " " +\
            "CXX=" + self.cxx_compiler + " " + self.compiler_args


    def CollectExtensionModulePaths(self):
        # All modules paths should be specified in the following list
        _pwd_ = os.path.join(self._pwd_,"Florence")

        tensor_path = os.path.join(_pwd_,"Tensor")
        mesh_path = os.path.join(_pwd_,"MeshGeneration")
        jacobi_path = os.path.join(_pwd_,"FunctionSpace","JacobiPolynomials")
        bp_path = os.path.join(_pwd_,"FunctionSpace","OneDimensional","_OneD")
        km_path = os.path.join(_pwd_,"FiniteElements","ElementalMatrices","_KinematicMeasures_")
        gm_path = os.path.join(_pwd_,"VariationalPrinciple","_GeometricStiffness_")
        material_path = os.path.join(_pwd_,"MaterialLibrary","LLDispatch")
        occ_path = os.path.join(_pwd_,"BoundaryCondition","CurvilinearMeshing","PostMesh")

        self.extension_paths = [tensor_path,mesh_path,jacobi_path,bp_path,km_path,gm_path,material_path,occ_path]

        # self.extension_paths = [material_path]

    def SourceClean(self):

        assert self.extension_paths != None

        for _path in self.extension_paths:
            if "PostMesh" not in _path and "LLDispatch" not in _path:
                execute('cd '+_path+' && make source_clean')
            elif "LLDispatch" in _path:
                execute('cd '+_path+' && echo rm -rf *.cpp CythonSource/*.cpp && rm -rf *.cpp CythonSource/*.cpp')
            elif "PostMesh" in _path:
                execute('cd '+_path+' && echo rm -rf PostMeshPy.cpp build/ && rm -rf m PostMeshPy.cpp build')


    def Clean(self):

        assert self.extension_paths != None

        self.SourceClean()
        # You need to run both make clean and rm -rf as some modules relocate the shared libraries
        for _path in self.extension_paths:
            if "PostMesh" not in _path and "LLDispatch" not in _path:
                execute('cd '+_path+' && echo rm -rf *.so && make clean && rm -rf *.so')
            elif "LLDispatch" in _path:
                execute('cd '+_path+' && echo rm -rf *.so CythonSource/*.so && rm -rf *.so CythonSource/*.so')
            else:
                execute('cd '+_path+' && echo rm -rf *.so && rm -rf *.so')


    def Build(self):


        low_level_material_list = [ "_NeoHookean_2_", 
                                    "_MooneyRivlin_0_", 
                                    "_NearlyIncompressibleMooneyRivlin_",
                                    "_AnisotropicMooneyRivlin_1_", 
                                    "_IsotropicElectroMechanics_0_", 
                                    "_IsotropicElectroMechanics_3_", 
                                    "_SteinmannModel_",
                                    "_IsotropicElectroMechanics_105_", 
                                    "_IsotropicElectroMechanics_106_", 
                                    "_IsotropicElectroMechanics_107_"
                                ]

        # low_level_material_list = ["_IsotropicElectroMechanics_107_"]

        assert self.extension_paths != None

        for _path in self.extension_paths:
            if "PostMesh" not in _path and "LLDispatch" not in _path:
                execute('cd '+_path+' && make ' + self.compiler_args)
            elif "LLDispatch" in _path:
                for material in low_level_material_list:
                    material = material.lstrip('_').rstrip('_')
                    execute('cd '+_path+' && make ' + self.compiler_args + " MATERIAL=" + material)
            elif "PostMesh" in _path:
                execute('cd '+_path+' && python setup.py build_ext -ifq')

        # Get rid off cython sources
        sys.stdout = open(os.devnull, 'w')
        self.SourceClean()
        sys.stdout = sys.__stdout__


        sys.path.insert(1,self._pwd_)
        from Florence import Mesh, MaterialLibrary, FEMSolver
        from Florence.VariationalPrinciple import VariationalPrinciple


    def Install(self):
        pass


    # def PerformBuild(self):

    #     # BUILD TENSOR MODULE (NUMERIC AND LINALG)
    #     tensor_path = os.path.join(self._pwd_,"Florence/Tensor/")
    #     print("Building tensor module")
    #     p = subprocess.Popen('cd '+tensor_path+' && make clean && make ' + self.cxx_compiler_args, shell=True)
    #     p.wait()

    #     # BUILD SALOME MESH READER
    #     print("Building mesh reader for Salome")
    #     mesh_path = os.path.join(self._pwd_,"Florence/MeshGeneration/")
    #     p = subprocess.Popen('cd '+mesh_path+' && make clean && make ' + self.cc_compiler_args, shell=True)
    #     p.wait()

    #     # BUILD JACOBI MODULE
    #     print("Building Jacobi polynomials module")
    #     jacobi_path = os.path.join(self._pwd_,"Florence/FunctionSpace/JacobiPolynomials/")
    #     p = subprocess.Popen('cd '+jacobi_path+' && make clean && make ' + self.cc_compiler_args, shell=True)
    #     p.wait()

    #     # BUILD BJORCK PEREYRA BASES
    #     print("Building Bjorck Pereyra bases module")
    #     bp_path = os.path.join(self._pwd_,"Florence/FunctionSpace/OneDimensional/_OneD")
    #     p = subprocess.Popen('cd '+bp_path+' && make clean && make ' + self.cxx_compiler_args, shell=True)
    #     p.wait()

    #     # BUILD KINEMATICS MEASURE
    #     print("Building low level dispatcher for kinematic measures")
    #     km_path = os.path.join(self._pwd_,"Florence/FiniteElements/ElementalMatrices/_KinematicMeasures_")
    #     p = subprocess.Popen('cd '+km_path+' && make clean && make ' + self.cc_compiler_args, shell=True)
    #     p.wait()

    #     # BUILD GEOMETRIC STIFFNESS 
    #     print("Building low level dispatcher for nonlinear geometric stiffnesses")
    #     gm_path = os.path.join(self._pwd_,"Florence/VariationalPrinciple/_GeometricStiffness_")
    #     p = subprocess.Popen('cd '+gm_path+' && make clean && make ' + self.cc_compiler_args, shell=True)
    #     p.wait()

    #     # BUILD MATERIAL MODELS
    #     print("Building low level dispatcher for material models")
    #     material_path = os.path.join(self._pwd_,"Florence/MaterialLibrary/LLDispatch/")
    #     # p = subprocess.Popen('cd '+material_path+' && ./Makefile.sh', shell=True)
    #     low_level_material_list = [ "_NeoHookean_2_", 
    #                                 "_MooneyRivlin_0_", 
    #                                 "_NearlyIncompressibleMooneyRivlin_",
    #                                 "_AnisotropicMooneyRivlin_1_", 
    #                                 "_IsotropicElectroMechanics_0_", 
    #                                 "_IsotropicElectroMechanics_3_", 
    #                                 "_SteinmannModel_",
    #                                 "_IsotropicElectroMechanics_105_", 
    #                                 "_IsotropicElectroMechanics_106_", 
    #                                 "_IsotropicElectroMechanics_107_"
    #                             ]
    #     for material in low_level_material_list:
    #         material = material.lstrip('_').rstrip('_')
    #         p = subprocess.Popen('cd '+material_path+' && make clean && make ' + self.cxx_compiler_args + " MATERIAL=" + material, shell=True)
    #         p.wait()

    #     # BUILD OPENCASCADE FRONT-END
    #     print("Building OpenCascade curvilinear mesh generation front-end")
    #     occ_path = os.path.join(self._pwd_,"Florence/BoundaryCondition/CurvilinearMeshing/PostMesh")
    #     p = subprocess.Popen('cd '+occ_path+' && rm PostMeshPy.cpp PostMeshPy.so', shell=True)
    #     p = subprocess.Popen('cd '+occ_path+' && python setup.py build_ext -ifq', shell=True)
    #     p = subprocess.Popen('cd '+occ_path+' && rm -rf build', shell=True)
    #     p.wait()




# helper functions
def execute(_cmd):
    _process = subprocess.Popen(_cmd, shell=True)
    _process.wait()










if __name__ == "__main__":

    _fc_compiler = None
    _cc_compiler = None
    _cxx_compiler = None

    args = sys.argv

    # should be either source_clean, clean or build
    _op = None

    if len(args) > 1:
        for arg in args:
            if arg == "source_clean" or arg == "clean" or arg == "build":
                if _op is not None:
                    raise RuntimeError("Multiple conflicting arguments passed to setup")
                _op = arg

            if "FC" in arg:
                _fc_compiler = arg.split("=")[-1]
            elif "CC" in arg:
                _cc_compiler = arg.split("=")[-1]
            elif "CXX" in arg:
                _cxx_compiler = arg.split("=")[-1]

    setup_instance = FlorenceSetup(_fc_compiler=_fc_compiler, 
        _cc_compiler=_cc_compiler, _cxx_compiler=_cxx_compiler)

    if _op == "source_clean":
        setup_instance.SourceClean()
    elif _op == "clean":
        setup_instance.Clean()
    elif _op == "install":
        setup_instance.Install()
    else:
        setup_instance.Build()


