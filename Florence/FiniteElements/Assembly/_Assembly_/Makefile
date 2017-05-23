PYTHON_VERSION = python2.7
PYTHON_INCLUDE_PATH = /usr/include/python2.7/
PYTHON_LD_PATH = /usr/lib/
EXT_POSTFIX= so
CYTHONFLAGS = -DNPY_NO_DEPRECATED_API

CYTHON = cython --cplus
REMOVE = rm -rf
MOVE = mv
ENTER = cd

# FASTOR_INCLUDE_PATH=
# UNAME_S := $(shell uname -s)
# ifeq ($(UNAME_S),Linux)
#     FASTOR_INCLUDE_PATH +=/home/roman/Dropbox/Fastor
# endif
# ifeq ($(UNAME_S),Darwin)
#     FASTOR_INCLUDE_PATH +=/Users/romanpoya/Dropbox/Fastor
# endif

# ***NOTE***: For python compatibility do NOT supply FASTOR_USE_BREAKING_PERMUTATION
FASTOR_OPTFLAGS = -DFASTOR_NO_ALIAS -DFASTOR_USE_VECTORISED_EXPR_ASSIGN -DCONTRACT_OPT=2 -DFASTOR_NO_STATIC_WARNING

INLINE_THRESHOLD = 100000
CXXSTD=-std=c++14
OPTFLAGS= -O3 -fno-strict-aliasing -DNDEBUG
CXXFLAGS= -fPIC -shared -pthread -Wall $(CYTHONFLAGS) $(OPTFLAGS)

# NUMPY_INCLUDE_PATH=python -c "import numpy; print numpy.get_include()"
# NUMPY_INCLUDE_PATH = /usr/local/lib/python2.7/dist-packages/numpy/core/include
NUMPY_INCLUDE_PATH =  /Users/romanpoya/VE/env1/lib/python2.7/site-packages/numpy/core/include/

# BLAS_VERSION = openblas
# BLAS_INCLUDE_PATH = /opt/OpenBLAS/include/
# BLAS_LD_PATH = /opt/OpenBLAS/lib/

BLAS_VERSION = openblas
BLAS_INCLUDE_PATH = /usr/local/Cellar/openblas/0.2.18_2/include/
BLAS_LD_PATH = /usr/local/Cellar/openblas/0.2.18_2/lib/

KIN_INC = ../../LocalAssembly/_KinematicMeasures_/
MAT_INC = ../../../MaterialLibrary/LLDispatch/CythonSource/
CONDF_INC = ../../../VariationalPrinciple/_ConstitutiveStiffness_/
CONDPF_INC = ../../../VariationalPrinciple/_ConstitutiveStiffness_/
GEO_INC = ../../../VariationalPrinciple/_GeometricStiffness_/ 

ifeq ($(findstring g++,$(CXX)),g++)
ifneq ($(findstring clang++,$(CXX)),clang++)
CXXFLAGS += -Wno-cpp -Wno-unused-function -Wno-maybe-uninitialized
CXXFLAGS += -march=native -ffast-math -fwrapv -funroll-loops -finline-functions -finline-limit=$(INLINE_THRESHOLD) -ffp-contract=fast 
else
CXXFLAGS += -Wno-uninitialized -Wno-missing-braces -Wno-#warnings
CXXFLAGS += -march=native -ffast-math -mllvm -inline-threshold=$(INLINE_THRESHOLD) -ffp-contract=fast
endif
endif
ifeq ($(findstring c++,$(CXX)),c++)
CXXFLAGS += -Wno-uninitialized -Wno-missing-braces -Wno-#NPY_NO_DEPRECATED_API
CXXFLAGS += -march=native -ffast-math -mllvm -inline-threshold=$(INLINE_THRESHOLD) -ffp-contract=fast
endif
ifeq ($(findstring icpc,$(CXX)),icpc)
CXXFLAGS += -march=native -no-ansi-alias -xHost -fp-model fast=1 -inline-forceinline -inline-factor=$(INLINE_THRESHOLD)
endif

CXXFLAGS += $(FASTOR_OPTFLAGS) -I$(FASTOR_INCLUDE_PATH)

all: build_module

build_module: 
	@echo "Building florence FEM assembly module: " $(ASSEMBLY_NAME)
	$(CYTHON) $(ASSEMBLY_NAME).pyx
	$(CXX) $(CXXSTD) $(CXXFLAGS) $(ASSEMBLY_NAME).cpp -o $(ASSEMBLY_NAME).$(EXT_POSTFIX) -I. \
	-I$(PYTHON_INCLUDE_PATH) -I$(NUMPY_INCLUDE_PATH) -L$(PYTHON_LD_PATH) -l$(PYTHON_VERSION) -I$(KIN_INC) -I../../../Tensor/ -I$(MAT_INC) \
	-I$(GEO_INC) -I$(CONDF_INC) -I$(CONDPF_INC) -I$(BLAS_INCLUDE_PATH) -L$(BLAS_LD_PATH) -l$(BLAS_VERSION)
	$(MOVE) $(ASSEMBLY_NAME).$(EXT_POSTFIX) ../

clean:
	$(REMOVE) $(ASSEMBLY_NAME).cpp $(ASSEMBLY_NAME).$(EXT_POSTFIX)

source_clean:
	$(REMOVE) $(ASSEMBLY_NAME).cpp
