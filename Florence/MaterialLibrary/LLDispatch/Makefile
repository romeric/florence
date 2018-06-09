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
NUMPY_INCLUDE_PATH = /usr/local/lib/python2.7/dist-packages/numpy/core/include

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

ifeq ($(OS),Windows_NT)
CXXFLAGS += -D_hypot=hypot -m64 -Wno-format-extra-args -Wno-format
endif

all: build_module

build_module: 
	@echo "Building Fastor based SIMD optimised module for material model:" $(MATERIAL)
	$(CYTHON) CythonSource/_$(MATERIAL)_.pyx
	$(CXX) $(CXXSTD) $(CXXFLAGS) CythonSource/_$(MATERIAL)_.cpp -o _$(MATERIAL)_.$(EXT_POSTFIX) -I. -I$(PYTHON_INCLUDE_PATH) -I$(NUMPY_INCLUDE_PATH) -L$(PYTHON_LD_PATH) -l$(PYTHON_VERSION)

clean:
	$(REMOVE) CythonSource/_$(MATERIAL)_.cpp _$(MATERIAL)_.$(EXT_POSTFIX)

source_clean:
	clean:
	$(REMOVE) CythonSource/_$(MATERIAL)_.cpp