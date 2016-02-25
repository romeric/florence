import os, platform, sys, subprocess
from distutils.core import setup
from distutils.command.clean import clean
from distutils.extension import Extension
from distutils.sysconfig import get_config_vars 
from Cython.Build import cythonize

# GET THE CURRENT WORKING DIRECTORY
_pwd_ = os.path.dirname(os.path.realpath('__file__'))
# BUILD TENSOR MODULE (NUMERIC AND LINALG)
tensor_path = os.path.join(_pwd_,"Florence/Tensor/")
print("Building Tensor module")
p = subprocess.Popen('cd '+tensor_path+' && ./Makefile.sh', shell=True)
p.wait()
print("Building OpenCascade front-end")
occ_path = os.path.join(_pwd_,"Florence/BoundaryCondition/CurvilinearMeshing/PostMesh")
p = subprocess.Popen('cd '+occ_path+' && ./Makefile.sh', shell=True)
p.wait()
