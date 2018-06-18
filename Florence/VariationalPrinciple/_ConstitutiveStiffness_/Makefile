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

INLINE_THRESHOLD = 100000
CXXSTD=-std=c++11
OPTFLAGS= -O3 -fno-strict-aliasing -DNDEBUG
CXXFLAGS= -fPIC -shared -pthread -Wall $(CYTHONFLAGS) $(OPTFLAGS)

BLAS_VERSION = openblas
BLAS_INCLUDE_PATH = /opt/OpenBLAS/include/
BLAS_LD_PATH = /opt/OpenBLAS/lib/

ifeq ($(findstring mkl,$(BLAS_VERSION)),mkl)
CXXFLAGS += -DHAS_MKL
endif

NUMPY_INCLUDE_PATH = /usr/local/lib/python2.7/dist-packages/numpy/core/include

ifeq ($(findstring g++,$(CXX)),g++)
ifneq ($(findstring clang++,$(CXX)),clang++)
CXXFLAGS += -Wno-cpp -Wno-unused-function -Wno-maybe-uninitialized
CXXFLAGS += -march=native -ffast-math -fwrapv -mfpmath=sse -funroll-loops -finline-functions -finline-limit=$(INLINE_THRESHOLD) -ffp-contract=fast
else
CXXFLAGS += -Wno-uninitialized -Wno-#warnings
CXXFLAGS += -march=native -ffast-math -mllvm -inline-threshold=$(INLINE_THRESHOLD) -ffp-contract=fast
endif
endif
ifeq ($(findstring icpc,$(CXX)),icpc)
CXXFLAGS += -march=native -no-ansi-alias -xHost -fp-model fast=1 -inline-forceinline -inline-factor=$(INLINE_THRESHOLD)
endif

ifeq ($(OS),Windows_NT)
CXXFLAGS += -D_hypot=hypot -m64 -Wno-format-extra-args -Wno-format
endif

all: build_module

build_module:
	@echo "Building low level mechanical constitutive stiffness integrands"
	$(CYTHON) DisplacementApproachIndices.pyx
	$(CXX) $(CXXSTD) $(CXXFLAGS) DisplacementApproachIndices.cpp -o DisplacementApproachIndices.$(EXT_POSTFIX) -I. -I$(PYTHON_INCLUDE_PATH) \
	     -I$(NUMPY_INCLUDE_PATH) -I$(BLAS_INCLUDE_PATH) -L$(PYTHON_LD_PATH) -l$(PYTHON_VERSION) -L$(BLAS_LD_PATH) -l$(BLAS_VERSION)
	@$(MOVE) DisplacementApproachIndices.$(EXT_POSTFIX) ../

	@echo "Building low level mechanical constitutive stiffness dispatcher modules"
	$(CYTHON) _ConstitutiveStiffnessDF_.pyx
	$(CXX) $(CXXSTD) $(CXXFLAGS) _ConstitutiveStiffnessDF_.cpp -o _ConstitutiveStiffnessDF_.$(EXT_POSTFIX) -I. -I$(PYTHON_INCLUDE_PATH) \
	     -I$(NUMPY_INCLUDE_PATH) -I$(BLAS_INCLUDE_PATH) -L$(PYTHON_LD_PATH) -l$(PYTHON_VERSION) -L$(BLAS_LD_PATH) -l$(BLAS_VERSION)
	@$(MOVE) _ConstitutiveStiffnessDF_.$(EXT_POSTFIX) ../

	@echo "Building low level electromechanical constitutive stiffness integrands"
	$(CYTHON) DisplacementPotentialApproachIndices.pyx
	$(CXX) $(CXXSTD) $(CXXFLAGS) DisplacementPotentialApproachIndices.cpp -o DisplacementPotentialApproachIndices.$(EXT_POSTFIX) -I. -I$(PYTHON_INCLUDE_PATH) \
	     -I$(NUMPY_INCLUDE_PATH) -I$(BLAS_INCLUDE_PATH) -L$(PYTHON_LD_PATH) -l$(PYTHON_VERSION) -L$(BLAS_LD_PATH) -l$(BLAS_VERSION)
	@$(MOVE) DisplacementPotentialApproachIndices.$(EXT_POSTFIX) ../

	@echo "Building low level electromechanical constitutive stiffness dispatcher module"
	$(CYTHON) _ConstitutiveStiffnessDPF_.pyx
	$(CXX) $(CXXSTD) $(CXXFLAGS) _ConstitutiveStiffnessDPF_.cpp -o _ConstitutiveStiffnessDPF_.$(EXT_POSTFIX) -I. -I$(PYTHON_INCLUDE_PATH) \
	     -I$(NUMPY_INCLUDE_PATH) -I$(BLAS_INCLUDE_PATH) -L$(PYTHON_LD_PATH) -l$(PYTHON_VERSION) -L$(BLAS_LD_PATH) -l$(BLAS_VERSION)
	@$(MOVE) _ConstitutiveStiffnessDPF_.$(EXT_POSTFIX) ../

	@echo "Building low level electrostatic constitutive stiffness dispatcher module"
	$(CYTHON) _ConstitutiveStiffnessLaplacian_.pyx
	$(CXX) $(CXXSTD) $(CXXFLAGS) _ConstitutiveStiffnessLaplacian_.cpp -o _ConstitutiveStiffnessLaplacian_.$(EXT_POSTFIX) -I. -I$(PYTHON_INCLUDE_PATH) \
	     -I$(NUMPY_INCLUDE_PATH) -I$(BLAS_INCLUDE_PATH) -L$(PYTHON_LD_PATH) -l$(PYTHON_VERSION) -L$(BLAS_LD_PATH) -l$(BLAS_VERSION)
	@$(MOVE) _ConstitutiveStiffnessLaplacian_.$(EXT_POSTFIX) ../

clean:
	$(REMOVE) DisplacementApproachIndices.cpp DisplacementApproachIndices.$(EXT_POSTFIX) ../DisplacementApproachIndices.$(EXT_POSTFIX)
	$(REMOVE) _ConstitutiveStiffnessDF_.cpp _ConstitutiveStiffnessDF_.$(EXT_POSTFIX) ../_ConstitutiveStiffnessDF_.$(EXT_POSTFIX)
	$(REMOVE) DisplacementPotentialApproachIndices.cpp DisplacementPotentialApproachIndices.$(EXT_POSTFIX) ../DisplacementPotentialApproachIndices.$(EXT_POSTFIX)
	$(REMOVE) _ConstitutiveStiffnessDPF_.cpp _ConstitutiveStiffnessDPF_.$(EXT_POSTFIX) ../_ConstitutiveStiffnessDPF_.$(EXT_POSTFIX)
	$(REMOVE)  _ConstitutiveStiffnessLaplacian_.cpp _ConstitutiveStiffnessLaplacian_.$(EXT_POSTFIX) ../_ConstitutiveStiffnessLaplacian_.$(EXT_POSTFIX)

source_clean:
	$(REMOVE) DisplacementApproachIndices.cpp DisplacementPotentialApproachIndices.cpp
	$(REMOVE) _ConstitutiveStiffnessDF_.cpp _ConstitutiveStiffnessDPF_.cpp _ConstitutiveStiffnessLaplacian_.cpp