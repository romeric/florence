PYTHON_VERSION = python2.7
PYTHON_INCLUDE_PATH = /usr/include/python2.7/
PYTHON_LD_PATH = /usr/lib/
EXT_POSTFIX = so
CYTHONFLAGS = -DNPY_NO_DEPRECATED_API

CYTHON = cython --cplus

ifeq ($(OS),Windows_NT)
REMOVE = del
MOVE = move
else
REMOVE = rm -rf
MOVE = mv
endif
ENTER = cd


# ***NOTE***: For python compatibility do NOT supply FASTOR_USE_BREAKING_PERMUTATION
FASTOR_OPTFLAGS = -DFASTOR_NO_ALIAS -DFASTOR_USE_VECTORISED_EXPR_ASSIGN -DCONTRACT_OPT=2 -DFASTOR_NO_STATIC_WARNING
KINEMATICS =

INLINE_THRESHOLD = 100000
CXXSTD=-std=c++14
OPTFLAGS= -O3 -fno-strict-aliasing -DNDEBUG
CXXFLAGS= -fPIC -shared -pthread -Wall $(CYTHONFLAGS) $(OPTFLAGS)

NUMPY_INCLUDE_PATH = /usr/local/lib/python2.7/dist-packages/numpy/core/include

BLAS_VERSION = openblas
BLAS_INCLUDE_PATH = /opt/OpenBLAS/include/
BLAS_LD_PATH = /opt/OpenBLAS/lib/

ifeq ($(findstring mkl,$(BLAS_VERSION)),mkl)
CXXFLAGS += -DHAS_MKL
endif

KIN_INC = ../../LocalAssembly/_KinematicMeasures_/
MAT_INC = ../../../MaterialLibrary/LLDispatch/CythonSource/
CONDF_INC = ../../../VariationalPrinciple/_ConstitutiveStiffness_/
CONDPF_INC = ../../../VariationalPrinciple/_ConstitutiveStiffness_/
GEO_INC = ../../../VariationalPrinciple/_GeometricStiffness_/
MASS_INC = ../../../VariationalPrinciple/_Mass_/
TRAC_INC = ../../../VariationalPrinciple/_Traction_/

ifeq ($(findstring g++,$(CXX)),g++)
ifneq ($(findstring clang++,$(CXX)),clang++)
CXXFLAGS += -Wno-cpp -Wno-unused-function -Wno-maybe-uninitialized -fabi-version=6
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

# On some architectures -march=native does not define -mfma
HAS_FMA := $(shell $(CXX) -march=native -dM -E - < /dev/null | egrep "AVX2" | sort)
ifeq ($(HAS_FMA),)
else
CXXFLAGS += -mfma
endif

CXXFLAGS += $(FASTOR_OPTFLAGS) -I$(FASTOR_INCLUDE_PATH)
CXXFLAGS += $(KINEMATICS)

ifeq ($(OS),Windows_NT)
CXXFLAGS += -D_hypot=hypot -m64 -Wno-format-extra-args -Wno-format
endif

all: build_module

build_module:
	@echo "Building florence FEM assembly module: " $(ASSEMBLY_NAME)
	$(CYTHON) $(ASSEMBLY_NAME).pyx
	$(CXX) $(CXXSTD) $(CXXFLAGS) $(ASSEMBLY_NAME).cpp -o $(ASSEMBLY_NAME).$(EXT_POSTFIX) -I. \
	-I$(PYTHON_INCLUDE_PATH) -I$(NUMPY_INCLUDE_PATH) -L$(PYTHON_LD_PATH) -l$(PYTHON_VERSION) -I$(KIN_INC) -I../../../Tensor/ -I$(MAT_INC) \
	-I$(GEO_INC) -I$(CONDF_INC) -I$(CONDPF_INC) -I$(MASS_INC) -I$(BLAS_INCLUDE_PATH) -L$(BLAS_LD_PATH) -l$(BLAS_VERSION)
	$(MOVE) $(ASSEMBLY_NAME).$(EXT_POSTFIX) ../

clean:
	$(REMOVE) $(ASSEMBLY_NAME).cpp $(ASSEMBLY_NAME).$(EXT_POSTFIX) ../$(ASSEMBLY_NAME).$(EXT_POSTFIX)

source_clean:
	$(REMOVE) $(ASSEMBLY_NAME).cpp



cython_assembler_build:
	@echo "Building florence FEM sparse assembler module"
	$(CYTHON) SparseAssemblyNative.pyx
	$(CXX) $(CXXSTD) $(CXXFLAGS) SparseAssemblyNative.cpp -o SparseAssemblyNative.$(EXT_POSTFIX) -I. \
	-I$(PYTHON_INCLUDE_PATH) -I$(NUMPY_INCLUDE_PATH) -L$(PYTHON_LD_PATH) -l$(PYTHON_VERSION)
	$(MOVE) SparseAssemblyNative.$(EXT_POSTFIX) ../

	@echo "Building florence FEM RHS assembler module"
	$(CYTHON) RHSAssemblyNative.pyx
	$(CXX) $(CXXSTD) $(CXXFLAGS) RHSAssemblyNative.cpp -o RHSAssemblyNative.$(EXT_POSTFIX) -I. \
	-I$(PYTHON_INCLUDE_PATH) -I$(NUMPY_INCLUDE_PATH) -L$(PYTHON_LD_PATH) -l$(PYTHON_VERSION)
	$(MOVE) RHSAssemblyNative.$(EXT_POSTFIX) ../

cython_assembler_clean:
	$(REMOVE) SparseAssemblyNative.cpp SparseAssemblyNative.$(EXT_POSTFIX) ../SparseAssemblyNative.$(EXT_POSTFIX)
	$(REMOVE) RHSAssemblyNative.cpp RHSAssemblyNative.$(EXT_POSTFIX) ../RHSAssemblyNative.$(EXT_POSTFIX)

cython_assembler_source_clean:
	$(REMOVE) SparseAssemblyNative.cpp
	$(REMOVE) RHSAssemblyNative.cpp
