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

    blas_version = None
    blas_include_path = None
    blas_ld_path = None

    extension_paths = None

    def __init__(self, _fc_compiler=None, _cc_compiler=None, _cxx_compiler=None,
        _blas=None):

        # GET THE CURRENT WORKING DIRECTORY
        self._pwd_ = os.path.dirname(os.path.realpath('__file__'))
        # Get python version and paths to header
        self.GetPythonPath()
        # Get numpy version and paths to header
        self.GetNumPyPath()
        # Get BLAS version and paths
        self.GetBLAS(_blas)
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


    def GetBLAS(self, _blas=None):

        if _blas is not None:
            self.blas_version = _blas
        else:
            self.blas_version = "openblas"

        dirs = ["/opt/OpenBLAS/lib","/usr/local/Cellar/openblas/","/usr/lib/","/usr/local/lib/"]
        aux_path = "/usr/local/Cellar/openblas"
        if os.path.isdir(aux_path):
            files = os.listdir(aux_path)
            if self.blas_version+".dylib" not in files:
                for d in files:
                    if os.path.isdir(os.path.join(aux_path,d)):
                        files2 = os.listdir(os.path.join(aux_path,d))
                        if self.blas_version+".dylib" not in files:
                            if os.path.isdir(os.path.join(aux_path,d,"lib")):
                                dirs.append(os.path.join(aux_path,d,"lib"))

        found_blas = False
        for d in dirs:
            if os.path.isdir(d):
                libs = os.listdir(d)
                for blas in libs:
                    if self.blas_version+".so" in blas or self.blas_version+".dylib" in blas:
                        self.blas_ld_path = d
                        self.blas_include_path = d.replace("lib","include")
                        found_blas = True
                        break
                if found_blas:
                    break


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
            " NUMPY_INCLUDE_PATH=" + self.numpy_include_path + \
            " BLAS_VERSION=" + self.blas_version + " BLAS_INCLUDE_PATH="+ self.blas_include_path + \
            " BLAS_LD_PATH=" + self.blas_ld_path

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
        cm_path = os.path.join(_pwd_,"VariationalPrinciple","_ConstitutiveStiffness_")
        material_path = os.path.join(_pwd_,"MaterialLibrary","LLDispatch")
        occ_path = os.path.join(_pwd_,"BoundaryCondition","CurvilinearMeshing","PostMesh")

        self.extension_paths = [tensor_path,mesh_path,jacobi_path,bp_path,km_path,gm_path,cm_path,material_path,occ_path]

        # self.extension_paths = [cm_path]

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
                                    "_IsotropicElectroMechanics_101_", 
                                    "_IsotropicElectroMechanics_105_", 
                                    "_IsotropicElectroMechanics_106_", 
                                    "_IsotropicElectroMechanics_107_",
                                    "_IsotropicElectroMechanics_108_",
                                    "_Piezoelectric_100_"
                                ]

        # low_level_material_list = ["_IsotropicElectroMechanics_108_"]

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
        var = raw_input("This includes florence in your python path. Do you agree (y/n): ")
        if var=="n" or "no" in var:
            return
        execute('export PYTHONPATH="$HOME/florence:$PYTHONPATH" >> ~/.profile && source ~/.profile')
        execute('export PYTHONPATH="' + self._pwd_ + ':$PYTHONPATH" >> ~/.profile && source ~/.profile')


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
            if arg == "source_clean" or arg == "clean" or arg == "build" or arg=="install":
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



