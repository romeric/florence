PYTHON_VERSION = python2.7
PYTHON_INCLUDE_PATH = /usr/include/python2.7/
PYTHON_LD_PATH = /usr/lib/
EXT_POSTFIX= so
CYTHONFLAGS = -DNPY_NO_DEPRECATED_API

CYTHON = cython --cplus
REMOVE = rm -rf
MOVE = mv

INLINE_THRESHOLD = 100000
CXXSTD=-std=c++11
OPTFLAGS= -O3 -fno-strict-aliasing -DNDEBUG
CXXFLAGS= -fPIC -shared -pthread -Wall $(CYTHONFLAGS) $(OPTFLAGS)

BLAS_VERSION = openblas
BLAS_INCLUDE_PATH = /opt/OpenBLAS/include/
BLAS_LD_PATH = /opt/OpenBLAS/lib/

# NUMPY_INCLUDE_PATH=python -c "import numpy; print numpy.get_include()"
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

all: build_module

build_module:
	@echo "Building low level mechanical constitutive stiffness dispatcher module"
	$(CYTHON) _ConstitutiveStiffnessDF_.pyx
	$(CXX) $(CXXSTD) $(CXXFLAGS) _ConstitutiveStiffnessDF_.cpp -o _ConstitutiveStiffnessDF_.$(EXT_POSTFIX) -I. -I$(PYTHON_INCLUDE_PATH) \
	     -I$(NUMPY_INCLUDE_PATH) -I$(BLAS_INCLUDE_PATH) -L$(PYTHON_LD_PATH) -l$(PYTHON_VERSION) -L$(BLAS_LD_PATH) -l$(BLAS_VERSION)
	@$(MOVE) _ConstitutiveStiffnessDF_.$(EXT_POSTFIX) ../

	@echo "Building low level electromechanical constitutive stiffness dispatcher module"
	$(CYTHON) _ConstitutiveStiffnessDPF_.pyx
	$(CXX) $(CXXSTD) $(CXXFLAGS) _ConstitutiveStiffnessDPF_.cpp -o _ConstitutiveStiffnessDPF_.$(EXT_POSTFIX) -I. -I$(PYTHON_INCLUDE_PATH) \
	     -I$(NUMPY_INCLUDE_PATH) -I$(BLAS_INCLUDE_PATH) -L$(PYTHON_LD_PATH) -l$(PYTHON_VERSION) -L$(BLAS_LD_PATH) -l$(BLAS_VERSION)
	@$(MOVE) _ConstitutiveStiffnessDPF_.$(EXT_POSTFIX) ../

clean:
	$(REMOVE) _ConstitutiveStiffnessDF_.cpp _ConstitutiveStiffnessDF_.$(EXT_POSTFIX) ../_ConstitutiveStiffnessDF_.$(EXT_POSTFIX)
	$(REMOVE) _ConstitutiveStiffnessDPF_.cpp _ConstitutiveStiffnessDPF_.$(EXT_POSTFIX) ../_ConstitutiveStiffnessDPF_.$(EXT_POSTFIX)

source_clean:
	$(REMOVE) _ConstitutiveStiffnessDF_.cpp _ConstitutiveStiffnessDPF_.cpp