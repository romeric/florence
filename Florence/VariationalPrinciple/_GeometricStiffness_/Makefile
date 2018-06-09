PYTHON_VERSION = python2.7
PYTHON_INCLUDE_PATH = /usr/include/
PYTHON_LD_PATH = /usr/lib/
EXT_POSTFIX= so
CYTHONFLAGS = -DNPY_NO_DEPRECATED_API

CYTHON = cython

ifeq ($(OS),Windows_NT)
REMOVE = del
MOVE = move
else
REMOVE = rm -rf
MOVE = mv
endif

INLINE_THRESHOLD = 100000
CCSTD=-std=c99
OPTFLAGS= -O3 -fno-strict-aliasing -DNDEBUG
CCFLAGS= -fPIC -shared -pthread -Wall $(CYTHONFLAGS) $(OPTFLAGS)

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

ifeq ($(OS),Windows_NT)
CCFLAGS += -D_hypot=hypot -m64 -Wno-format-extra-args -Wno-format
endif

all: build_module

build_module: 
	@echo "Building low level geometric stiffness dispatcher module"
	$(CYTHON) _GeometricStiffness_.pyx
	$(CC) $(CCSTD) $(CCFLAGS) _GeometricStiffness_.c -o _GeometricStiffness_.$(EXT_POSTFIX) -I. -I$(PYTHON_INCLUDE_PATH) -I$(NUMPY_INCLUDE_PATH) -L$(PYTHON_LD_PATH) -l$(PYTHON_VERSION)
	@$(MOVE) _GeometricStiffness_.$(EXT_POSTFIX) ../

clean:
	$(REMOVE) _GeometricStiffness_.c _GeometricStiffness_.$(EXT_POSTFIX) ../_GeometricStiffness_.$(EXT_POSTFIX)

source_clean:
	$(REMOVE) _GeometricStiffness_.c