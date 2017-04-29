PYTHON_VERSION = python2.7
PYTHON_INCLUDE_PATH = /usr/include/
PYTHON_LD_PATH = /usr/lib/
EXT_POSTFIX= so
CYTHONFLAGS = -DNPY_NO_DEPRECATED_API

CYTHON = cython
REMOVE = rm -rf
MOVE = mv

INLINE_THRESHOLD = 100000
CCSTD=-std=c99
OPTFLAGS= -O3 -fno-strict-aliasing -DNDEBUG
CCFLAGS= -fPIC -shared -pthread -Wall $(CYTHONFLAGS) $(OPTFLAGS) -lm

# NUMPY_INCLUDE_PATH=python -c "import numpy; print numpy.get_include()"
NUMPY_INCLUDE_PATH = /usr/local/lib/python2.7/dist-packages/numpy/core/include

ifeq ($(findstring gcc,$(CC)),gcc)
CCFLAGS += -Wno-cpp -Wno-unused-function -Wno-maybe-uninitialized
CCFLAGS += -march=native -ffast-math -fwrapv -mfpmath=sse -funroll-loops -finline-functions -finline-limit=$(INLINE_THRESHOLD) -ffp-contract=fast
endif
ifeq ($(findstring clang,$(CC)),clang)
CCFLAGS += -Wno-uninitialized -Wno-#warnings
CCFLAGS += -march=native -ffast-math -mllvm -inline-threshold=$(INLINE_THRESHOLD) -ffp-contract=fast
endif
ifeq ($(findstring icc,$(CC)),icc)
CCFLAGS += -march=native -no-ansi-alias -xHost -fp-model fast=1 -inline-forceinline -inline-factor=$(INLINE_THRESHOLD)
endif

all: build_module

build_module: 
	@echo "Building SIMD optimised kinematics module"
	$(CYTHON) _KinematicMeasures_.pyx
	$(CC) $(CCSTD) $(CCFLAGS) _KinematicMeasures_.c -o _KinematicMeasures_.$(EXT_POSTFIX) -I../../../Tensor/ -I. -I$(PYTHON_INCLUDE_PATH) \
	-I$(NUMPY_INCLUDE_PATH) -L$(PYTHON_LD_PATH) -l$(PYTHON_VERSION)
	@$(MOVE) _KinematicMeasures_.$(EXT_POSTFIX) ../_KinematicMeasures_.$(EXT_POSTFIX)

clean:
	$(REMOVE) _KinematicMeasures_.c _KinematicMeasures_.$(EXT_POSTFIX) ../_KinematicMeasures_.$(EXT_POSTFIX)

source_clean:
	$(REMOVE) _KinematicMeasures_.c