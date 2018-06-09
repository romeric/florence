PYTHON_VERSION = python2.7
PYTHON_INCLUDE_PATH = /usr/include/python2.7/
PYTHON_LD_PATH = /usr/lib/
EXT_POSTFIX= so
CYTHONFLAGS = -DNPY_NO_DEPRECATED_API

CYTHON = cython --cplus

ifeq ($(OS),Windows_NT)
REMOVE = del
MOVE = move
else
REMOVE = rm -rf
MOVE = mv
endif

INLINE_THRESHOLD = 100000
CXXSTD=-std=c++11
OPTFLAGS= -O3 -fno-strict-aliasing -DNDEBUG
CXXFLAGS= -fPIC -shared -pthread -Wall $(CYTHONFLAGS) $(OPTFLAGS)

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

ifeq ($(OS),Windows_NT)
CXXFLAGS += -D_hypot=hypot -m64 -Wno-format-extra-args -Wno-format
endif

all: build_module

build_module:
	@echo "Building low level mechanical traction dispatcher modules"
	$(CYTHON) _TractionDF_.pyx
	$(CXX) $(CXXSTD) $(CXXFLAGS) _TractionDF_.cpp -o _TractionDF_.$(EXT_POSTFIX) -I. -I$(PYTHON_INCLUDE_PATH) \
	     -I$(NUMPY_INCLUDE_PATH) -L$(PYTHON_LD_PATH) -l$(PYTHON_VERSION)
	@$(MOVE) _TractionDF_.$(EXT_POSTFIX) ../

	@echo "Building low level electromechanical traction dispatcher module"
	$(CYTHON) _TractionDPF_.pyx
	$(CXX) $(CXXSTD) $(CXXFLAGS) _TractionDPF_.cpp -o _TractionDPF_.$(EXT_POSTFIX) -I. -I$(PYTHON_INCLUDE_PATH) \
	     -I$(NUMPY_INCLUDE_PATH) -L$(PYTHON_LD_PATH) -l$(PYTHON_VERSION)
	@$(MOVE) _TractionDPF_.$(EXT_POSTFIX) ../

clean:
	$(REMOVE) _TractionDF_.cpp _TractionDF_.$(EXT_POSTFIX) ../_TractionDF_.$(EXT_POSTFIX)
	$(REMOVE) _TractionDPF_.cpp _TractionDPF_.$(EXT_POSTFIX) ../_TractionDPF_.$(EXT_POSTFIX)

source_clean:
	$(REMOVE) _TractionDF_.cpp _TractionDPF_.cpp