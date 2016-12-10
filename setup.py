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

# BUILD SALOME MESH READER
print("Building mesh reader for salome")
mesh_path = os.path.join(_pwd_,"Florence/MeshGeneration/")
p = subprocess.Popen('cd '+mesh_path+' && ./Makefile.sh', shell=True)
p.wait()

# BUILD JACOBI MODULE
print("Building Jacobi module")
jacobi_path = os.path.join(_pwd_,"Florence/FunctionSpace/JacobiPolynomials/")
p = subprocess.Popen('cd '+jacobi_path+' && ./Makefile.sh', shell=True)
p.wait()

# BUILD MATERIAL MODELS
print("Building material models")
material_path = os.path.join(_pwd_,"Florence/MaterialLibrary/LLDispatch/")
p = subprocess.Popen('cd '+material_path+' && ./Makefile.sh', shell=True)
p.wait()

# BUILD OPENCASCADE FRONT-END
print("Building OpenCascade front-end")
occ_path = os.path.join(_pwd_,"Florence/BoundaryCondition/CurvilinearMeshing/PostMesh")
p = subprocess.Popen('cd '+occ_path+' && ./Makefile.sh', shell=True)
p.wait()
