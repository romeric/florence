from __future__ import print_function
from setuptools import setup
from setuptools import find_packages
import os, platform, sys, subprocess, imp
from distutils.command.clean import clean
from distutils.extension import Extension
from distutils.sysconfig import get_config_var, get_config_vars, get_python_inc, get_python_lib
import fnmatch
try:
    from Cython.Build import cythonize
except ImportError:
    raise ImportError("Could not import cython. Please install cython first")
try:
    import numpy as np
except ImportError:
    raise ImportError("Could not import numpy. Please install numpy first")

from multiprocessing import Pool, cpu_count


try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False
except ImportError:
    bdist_wheel = None



class FlorenceSetup(object):

    _os = sys.platform

    python_version = None
    python_interpreter = None
    python_implementation = None
    python_include_path = None
    python_ld_path = None

    numpy_version = None
    numpy_include_path = None

    fastor_include_path = None

    blas_version = "openblas"
    blas_include_path = None
    blas_ld_path = None

    fc_compiler = None
    cc_compiler = None
    cxx_compiler = None

    compiler_args = None
    fc_compiler_args = None
    cc_compiler_args = None
    cxx_compiler_args = None

    fc_version = None
    cc_version = None
    cxx_version = None

    extension_paths = None
    extension_postfix = None

    parallel = True
    num_cpu = cpu_count()

    kinematics_version = "-DUSE_AVX_VERSION"

    def __init__(self, _fc_compiler=None, _cc_compiler=None, _cxx_compiler=None,
        _blas=None, _kinematics_version=None, _fastor_path=None, _ncpu=1):

        # Get current working working directory
        self._pwd_ = os.path.dirname(os.path.realpath('__file__'))
        # Get python version and paths to header
        self.GetPythonPath()
        # Get numpy version and paths to header
        self.GetNumPyPath()
        # Get Fastor path
        self.GetFastorPath(_fastor_path)
        # Get BLAS version and paths
        self.GetBLAS(_blas)
        # Get kinematics version
        self.GetKinematicsVersion(_kinematics_version)
        # Set C/Fortran/C++ compiler
        self.SetCompiler(_fc_compiler,_cc_compiler,_cxx_compiler)
        # Set up compiler arguments for all extension modules
        self.SetCompilerArgs()
        # Collect all extension module paths
        self.CollectExtensionModulePaths()
        # Determine parallel builds
        self.SetParallelism(_ncpu)


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

        with_pymalloc = ""
        if get_config_var("ABIFLAGS") is not None:
            with_pymalloc = get_config_var("ABIFLAGS")
        self.python_interpreter += with_pymalloc

        # Get python implementation e.g. CPython, PyPy etc
        self.python_implementation = platform.python_implementation()

        # Get python include path
        self.python_include_path = get_python_inc()

        # Get python lib path
        # Note that we need the actual ld path where libpython.so/dylib/dll resides
        # and this seems like the only portable way at the moment
        lib_postfix = ".so"
        if "darwin" in self._os:
            lib_postfix = ".dylib"

        # The following search is split into two searches for /usr/lib and /usr/local
        # for speed purposes
        libpython = "lib"+self.python_interpreter+lib_postfix
        for root, _, filenames in os.walk('/usr/lib/'):
            for filename in fnmatch.filter(filenames, libpython):
                self.python_ld_path = os.path.join(root, filename).rsplit(libpython)[0]
                break
        if self.python_ld_path is None:
            for root, _, filenames in os.walk('/usr/local/'):
                for filename in fnmatch.filter(filenames, libpython):
                    self.python_ld_path = os.path.join(root, filename).rsplit(libpython)[0]
                    break

        # For conda envs change python_ld_path
        if "conda" in sys.version or "Continuum" in sys.version or "Intel" in sys.version:
            self.python_ld_path = get_config_var("LIBDIR")

        # Sanity check
        if self.python_ld_path is None:
            try:
                self.python_ld_path = get_config_var("LIBDIR")
            except:
                raise RuntimeError("Could not find libpython")


        # Get postfix for extensions
        self.extension_postfix = get_config_vars()['SO'][1:]
        if self.extension_postfix is None:
            if "darwin" in self._os:
                self.extension_postfix = "dylib"
            else:
                self.extension_postfix = "so"


    def GetNumPyPath(self):
        numpy_version = np.__version__
        if int(numpy_version.split('.')[0])==1 and int(numpy_version.split('.')[1]) < 8:
            raise RuntimeError("Numpy version >= 1.8 required")
        elif int(numpy_version.split('.')[0]) < 1:
            raise RuntimeError("Numpy version >= 1.8 required")
        self.numpy_include_path = np.get_include()


    def GetFastorPath(self, _fastor_path=None):
        if _fastor_path is not None:
            if os.path.isdir(_fastor_path):
                self.fastor_include_path = _fastor_path
                return

        if os.path.isdir("/usr/local/include/Fastor"):
            self.fastor_include_path = "/usr/local/include/Fastor"
        else:
            raise RuntimeError("Florence expects Fastor headers to be under '/usr/local/include/Fastor/' directory")
            return


    def GetBLAS(self, _blas=None):

        if _blas is not None:
            self.blas_version = _blas

        lib_postfix = ".so"
        if "darwin" in self._os:
            lib_postfix = ".dylib"

        if self.blas_version == "openblas":

            dirs = ["/opt/OpenBLAS/lib","/usr/local/Cellar/openblas/","/usr/lib/","/usr/local/lib/"]
            aux_path = "/usr/local/Cellar/openblas"
            if os.path.isdir(aux_path):
                files = os.listdir(aux_path)
                if self.blas_version+lib_postfix not in files:
                    for d in files:
                        if os.path.isdir(os.path.join(aux_path,d)):
                            files2 = os.listdir(os.path.join(aux_path,d))
                            if self.blas_version+lib_postfix not in files:
                                if os.path.isdir(os.path.join(aux_path,d,"lib")):
                                    dirs.append(os.path.join(aux_path,d,"lib"))

            found_blas = False
            for d in dirs:
                if os.path.isdir(d):
                    libs = os.listdir(d)
                    for blas in libs:
                        if self.blas_version+lib_postfix in blas:
                            self.blas_ld_path = d
                            self.blas_include_path = d.replace("lib","include")
                            found_blas = True
                            break
                    if found_blas:
                        break

        elif self.blas_version == "mkl":
            self.blas_ld_path = get_config_vars()['LIBDIR']
            self.blas_include_path = get_config_vars()['INCLUDEDIR']
            self.blas_version = "mkl_rt"


        # Fall back to reference BLAS if OpenBLAS is not found
        if self.blas_ld_path == None:
            self.blas_version = "blas"
            self.blas_ld_path = "/usr/lib"
            self.blas_include_path = "/usr/include/"


    def GetKinematicsVersion(self,_kinematics_version=None):
        if _kinematics_version == "scalar" or _kinematics_version == "None":
            self.kinematics_version = ""
        else:
            self.kinematics_version = "-DUSE_AVX_VERSION"


    def SetCompiler(self, _fc_compiler=None, _cc_compiler=None, _cxx_compiler=None):

        if not "darwin" in self._os and not "linux" in self._os and not "cygwin" in self._os:
            raise RuntimeError("Florence is not yet tested on any other platform apart from Linux & macOS")

        self.fc_compiler = _fc_compiler
        self.cc_compiler = _cc_compiler
        self.cxx_compiler = _cxx_compiler

        if self.fc_compiler is None:
            self.fc_compiler = "gfortran"

        if self.cc_compiler is None:
            self.cc_compiler = get_config_vars()['CC'].split(' ')[0]

        if self.cxx_compiler is None:
            self.cxx_compiler = get_config_vars()['CXX'].split(' ')[0]

        if self.cc_compiler == "cc" and "linux" in self._os:
            self.cc_compiler = "gcc"
        if self.cxx_compiler == "c++" and "linux" in self._os:
            self.cxx_compiler = "g++"

        # Sanity check
        if self.cc_compiler is None:
            self.cc_compiler = "gcc"
        if self.cxx_compiler is None:
            self.cxx_compiler = "g++"


    def SetCompilerArgs(self):

        if "darwin" in self._os:
            self.cxx_version = "-std=c++14"
        else:
            self.cxx_version = "-std=c++11"

        # Generic compiler arguments
        self.compiler_args = "PYTHON_VERSION=" + self.python_interpreter + " PYTHON_INCLUDE_PATH=" + \
            self.python_include_path + " PYTHON_LD_PATH=" + self.python_ld_path + \
            " NUMPY_INCLUDE_PATH=" + self.numpy_include_path + \
            " FASTOR_INCLUDE_PATH=" + self.fastor_include_path +\
            " BLAS_VERSION=" + self.blas_version + " BLAS_INCLUDE_PATH="+ self.blas_include_path + \
            " BLAS_LD_PATH=" + self.blas_ld_path + " EXT_POSTFIX=" + self.extension_postfix +\
            " CXXSTD=" + self.cxx_version + " KINEMATICS=" + self.kinematics_version

        self.compiler_args = self.compiler_args + ' REMOVE="rm -rf" MOVE=mv '

        self.fc_compiler_args = "FC=" + self.fc_compiler + " " + self.compiler_args
        self.cc_compiler_args = "CC=" + self.cc_compiler + " " + self.compiler_args
        self.cxx_compiler_args = "CXX=" + self.cxx_compiler + " " + self.compiler_args

        self.compiler_args = "FC=" + self.fc_compiler + " " + "CC=" + self.cc_compiler + " " +\
            "CXX=" + self.cxx_compiler + " " + self.compiler_args



    def CollectExtensionModulePaths(self):
        # All modules paths should be specified in the following list
        _pwd_ = os.path.join(self._pwd_,"Florence")

        tensor_path = os.path.join(_pwd_,"Tensor")
        jacobi_path = os.path.join(_pwd_,"FunctionSpace","JacobiPolynomials")
        bp_path = os.path.join(_pwd_,"FunctionSpace","OneDimensional","_OneD")
        km_path = os.path.join(_pwd_,"FiniteElements","LocalAssembly","_KinematicMeasures_")
        gm_path = os.path.join(_pwd_,"VariationalPrinciple","_GeometricStiffness_")
        cm_path = os.path.join(_pwd_,"VariationalPrinciple","_ConstitutiveStiffness_")
        tm_path = os.path.join(_pwd_,"VariationalPrinciple","_Traction_")
        mm_path = os.path.join(_pwd_,"VariationalPrinciple","_Mass_")
        material_path = os.path.join(_pwd_,"MaterialLibrary","LLDispatch")
        assemble_path = os.path.join(_pwd_,"FiniteElements","Assembly","_Assembly_")

        self.extension_paths = [tensor_path,jacobi_path,bp_path,
            km_path,gm_path,cm_path,tm_path,mm_path,material_path,assemble_path]
        # self.extension_paths = [tensor_path]
        # self.extension_paths = [assemble_path]

    def SetParallelism(self,_ncpu=1):
        if _ncpu > 1:
            self.parallel = True
            self.num_cpu = int(_ncpu)
        elif _ncpu == 1:
            self.parallel = False
            self.num_cpu = 1

    def SourceClean(self):

        assert self.extension_paths != None

        source_clean_cmd = 'make source_clean ' + self.compiler_args
        for _path in self.extension_paths:
            if "_Assembly_" not in _path and "LLDispatch" not in _path:
                execute('cd '+_path+' && '+source_clean_cmd)
            elif "LLDispatch" in _path:
                execute('cd '+_path+' && echo rm -rf *.cpp CythonSource/*.cpp && rm -rf *.cpp CythonSource/*.cpp')
            elif "Assembly" in _path:
                execute('cd '+_path+' && echo rm -rf *.cpp && rm -rf *.cpp')

    def Clean(self):

        assert self.extension_paths != None

        self.SourceClean()
        # You need to run both make clean and rm -rf as some modules relocate the shared libraries
        clean_cmd = 'make clean ' + self.compiler_args
        for _path in self.extension_paths:
            if "LLDispatch" not in _path:
                execute('cd '+_path+' && echo rm -rf *.'+self.extension_postfix+' && '+
                    clean_cmd+' && rm -rf *.'+self.extension_postfix)
            elif "LLDispatch" in _path:
                execute('cd '+_path+
                    ' && echo rm -rf *.'+self.extension_postfix+' CythonSource/*.'+self.extension_postfix
                    +' && rm -rf *.'+self.extension_postfix+' CythonSource/*.'+
                    self.extension_postfix)
            else:
                execute('cd '+_path+' && echo rm -rf *.'+self.extension_postfix+' && rm -rf *.'+self.extension_postfix)

        # # clean all crude and ext libs if any - this is dangerous if setup.py is ever invoked from outside the directory
        # # which is certainly never the case
        # execute("find . -name \*.so -delete")
        # execute("find . -name \*.pyc -delete")


    def Build(self):


        low_level_material_list = [ "_LinearElastic_",
                                    "_NeoHookean_",
                                    "_MooneyRivlin_",
                                    "_NearlyIncompressibleMooneyRivlin_",
                                    "_AnisotropicMooneyRivlin_1_",
                                    "_ExplicitMooneyRivlin_",
                                    "_IsotropicElectroMechanics_0_",
                                    "_IsotropicElectroMechanics_3_",
                                    "_SteinmannModel_",
                                    "_IsotropicElectroMechanics_101_",
                                    "_IsotropicElectroMechanics_105_",
                                    "_IsotropicElectroMechanics_106_",
                                    "_IsotropicElectroMechanics_107_",
                                    "_IsotropicElectroMechanics_108_",
                                    "_IsotropicElectroMechanics_109_",
                                    "_Piezoelectric_100_",
                                    "_ExplicitIsotropicElectroMechanics_108_",
                                ]

        # low_level_material_list = ["_IsotropicElectroMechanics_109_"]
        # low_level_material_list = ["_NeoHookean_"]
        # low_level_material_list = ["_LinearElastic_"]
        # low_level_material_list = ["_ExplicitMooneyRivlin_"]


        assert self.extension_paths != None

        if self.parallel == False:
            for _path in self.extension_paths:
                if "LLDispatch" not in _path and "_Assembly_" not in _path:
                    execute('cd '+_path+' && make ' + self.compiler_args)
                if "LLDispatch" in _path:
                    _cmds = []
                    for material in low_level_material_list:
                        material = material.lstrip('_').rstrip('_')
                        execute('cd '+_path+' && make ' + self.compiler_args + " MATERIAL=" + material)

                elif "_Assembly_" in _path:

                    # Sparse and RHS assembler
                    execute('cd '+_path+' && make cython_assembler_build ' + self.compiler_args)

                    ll_material_mech = low_level_material_list[:6]
                    ll_material_electro_mech = low_level_material_list[6:]

                    ll_material_mech.remove("_ExplicitMooneyRivlin_")
                    ll_material_electro_mech.remove("_IsotropicElectroMechanics_109_")
                    ll_material_electro_mech.remove("_ExplicitIsotropicElectroMechanics_108_")

                    # ll_material_mech = []
                    # ll_material_electro_mech = low_level_material_list
                    # ll_material_mech = low_level_material_list
                    # ll_material_electro_mech = []

                    execute('cd '+_path+' && python AOT_Assembler.py clean')
                    execute('cd '+_path+' && python AOT_Assembler.py configure')

                    for material in ll_material_mech:
                        execute('cd '+_path+' && make ' + self.compiler_args + " ASSEMBLY_NAME=_LowLevelAssemblyDF_"  + material)
                    for material in ll_material_electro_mech:
                        execute('cd '+_path+' && make ' + self.compiler_args + " ASSEMBLY_NAME=_LowLevelAssemblyDPF_" + material)

                    execute('cd '+_path+' && python AOT_Assembler.py clean')

                    # Explicit assembler
                    execute('cd '+_path+' && make ' + self.compiler_args +\
                        " ASSEMBLY_NAME=_LowLevelAssemblyExplicit_DF_DPF_ CONDF_INC=../../../VariationalPrinciple/_Traction_/\
                        CONDF_INC=../../../VariationalPrinciple/_Traction_/")

                    # Perfect Laplacian assembler
                    execute('cd '+_path+' && make ' + self.compiler_args +\
                        " ASSEMBLY_NAME=_LowLevelAssemblyPerfectLaplacian_ ")

        else:
            # Parallel build
            pool = Pool(self.num_cpu)

            # All modules apart from LLDispatch and _Assembly_ modules
            _cmds = []
            for _path in self.extension_paths:
                if "LLDispatch" not in _path and "_Assembly_" not in _path:
                    _cmds.append('cd '+_path+' && make ' + self.compiler_args)

            pool.map(execute,_cmds)


            # LLDispatch modules
            _cmds = []
            for _path in self.extension_paths:
                if "LLDispatch" in _path:
                    for material in low_level_material_list:
                        material = material.lstrip('_').rstrip('_')
                        _cmds.append('cd '+_path+' && make ' + self.compiler_args + " MATERIAL=" + material)

            pool.map(execute,_cmds)


            # _Assembly_ modules
            _cmds = []
            for _path in self.extension_paths:
                if "_Assembly_" in _path:

                    ll_material_mech = low_level_material_list[:6]
                    ll_material_electro_mech = low_level_material_list[6:]

                    ll_material_mech.remove("_ExplicitMooneyRivlin_")
                    ll_material_electro_mech.remove("_IsotropicElectroMechanics_109_")
                    ll_material_electro_mech.remove("_ExplicitIsotropicElectroMechanics_108_")

                    # ll_material_mech = []
                    # ll_material_electro_mech = low_level_material_list
                    # ll_material_mech = low_level_material_list
                    # ll_material_electro_mech = []

                    execute('cd '+_path+' && python AOT_Assembler.py clean')
                    execute('cd '+_path+' && python AOT_Assembler.py configure')

                    for material in ll_material_mech:
                        _cmds.append('cd '+_path+' && make ' + self.compiler_args + " ASSEMBLY_NAME=_LowLevelAssemblyDF_"  + material)
                    for material in ll_material_electro_mech:
                        _cmds.append('cd '+_path+' && make ' + self.compiler_args + " ASSEMBLY_NAME=_LowLevelAssemblyDPF_" + material)

            pool.map(execute,_cmds)
            execute('cd '+_path+' && python AOT_Assembler.py clean')


            # Modules built sequentially
            for _path in self.extension_paths:
                if "_Assembly_" in _path:
                    # Sparse and RHS assembler
                    execute('cd '+_path+' && make cython_assembler_build ' + self.compiler_args)

                    # Explicit assembler
                    execute('cd '+_path+' && make ' + self.compiler_args +\
                        " ASSEMBLY_NAME=_LowLevelAssemblyExplicit_DF_DPF_ CONDF_INC=../../../VariationalPrinciple/_Traction_/\
                        CONDF_INC=../../../VariationalPrinciple/_Traction_/")

                    # Perfect Laplacian assembler
                    execute('cd '+_path+' && make ' + self.compiler_args +\
                        " ASSEMBLY_NAME=_LowLevelAssemblyPerfectLaplacian_ ")

            pool.close()


        # Get rid of cython sources
        sys.stdout = open(os.devnull, 'w')
        self.SourceClean()
        sys.stdout = sys.__stdout__


    def Install(self):
        var = raw_input("This includes florence in your python path. Do you agree (y/n): ")
        if var=="n" or "no" in var:
            return
        execute('export PYTHONPATH="$HOME/florence:$PYTHONPATH" >> ~/.profile && source ~/.profile')
        execute('export PYTHONPATH="' + self._pwd_ + ':$PYTHONPATH" >> ~/.profile && source ~/.profile')



def execute(_cmd):
    _process = subprocess.Popen(_cmd, shell=True)
    _process.wait()



def setup_package():

    _fc_compiler = None
    _cc_compiler = None
    _cxx_compiler = None
    _blas = None
    _kinematics_version = None
    _fastor_path = None
    _ncpu = cpu_count()

    args = sys.argv

    # should be either source_clean, clean or build
    _op = None

    if len(args) > 1:
        for arg in args:
            if arg == "source_clean" or arg == "clean" or arg == "build" or arg=="manual_install" or "dist" in arg:
                _op = arg
            if "FC" in arg:
                _fc_compiler = arg.split("=")[-1]
            elif "CC" in arg:
                _cc_compiler = arg.split("=")[-1]
            elif "CXX" in arg:
                _cxx_compiler = arg.split("=")[-1]
            if "BLAS" in arg:
                _blas = arg.split("=")[-1]
            if "KINEMATICS" in arg:
                _kinematics_version = arg.split("=")[-1]
            if "FASTORPATH=" in arg:
                _fastor_path = int(arg.split("=")[-1])
            if "np=" in arg:
                _ncpu = int(arg.split("=")[-1])

    setup_instance = FlorenceSetup(_fc_compiler=_fc_compiler,
        _cc_compiler=_cc_compiler, _cxx_compiler=_cxx_compiler, _blas=_blas,
        _kinematics_version=_kinematics_version, _fastor_path=_fastor_path,_ncpu=_ncpu)

    if _op is None:
        _op = ""

    if _op == "source_clean":
        setup_instance.SourceClean()
    elif _op == "clean":
        setup_instance.Clean()
    elif _op == "manual_install":
        setup_instance.Install()
    else:

        with open("README.md", "r") as fh:
            long_description = fh.read()

        metadata =  dict(
                        name = "Florence",
                        version = "0.1.4",
                        description = """A Python based computational framework for integrated computer aided design,
                            curvilinear mesh generation and finite and boundary element methods for linear and nonlinear
                            analysis of solids and coupled multiphysics problems""",
                        long_description_content_type="text/markdown",
                        long_description=long_description,
                        author="Roman Poya",
                        author_email = "roman_poya@yahoo.com",
                        url = "https://github.com/romeric/florence",
                        license="MIT",
                        platforms=["Linux", "macOS", "Unix", "Windows"],
                        cmdclass={'bdist_wheel': bdist_wheel},
                        python_requires='>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*',
                        install_requires=[
                          'cython>=0.23',
                          'numpy>=1.9',
                          'scipy>=0.14',
                          'scikit-umfpack>=0.2'],
                        packages=find_packages(),
                        include_package_data=True,
                        package_data={'': ['*.pyx', '*.pxd', '*.h', '*.hpp', '*.dll', '*.dylib', '*.so',
                            '*.py', '*.txt', '*.md', 'Makefile']},
                    )


        if "build" in _op or "dist" in _op:
            metadata['ext_modules'] = setup_instance.Build()

        setup(**metadata)


if __name__ == "__main__":
    setup_package()
