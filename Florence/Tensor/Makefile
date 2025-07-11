PYTHON_VERSION =
PYTHON_LANG_LEVEL =
PYTHON_INCLUDE_PATH =
PYTHON_LD_PATH =
EXT_POSTFIX=
CYTHONFLAGS = -DNPY_NO_DEPRECATED_API

CYTHON = cython --cplus -$(PYTHON_LANG_LEVEL)

ifeq ($(OS),Windows_NT)
REMOVE = del
MOVE = move
else
REMOVE = rm -rf
MOVE = mv
endif

ADDITIONAL_FLAGS =

INLINE_THRESHOLD = 100000
CXXSTD=-std=c++14
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

# On some architectures -march=native does not define -mfma
HAS_FMA := $(shell $(CXX) -march=native -dM -E - < /dev/null | egrep "AVX2" | sort)
ifeq ($(HAS_FMA),)
else
CXXFLAGS += -mfma
endif

CXXFLAGS += $(ADDITIONAL_FLAGS)

all: build_module

build_module:
	@echo "Building Florence supplementary tensor numeric module"
	$(CYTHON) Numeric.pyx
	$(CXX) $(CXXSTD) $(CXXFLAGS) Numeric.cpp _Numeric.cpp -o Numeric.$(EXT_POSTFIX) -I. -I$(PYTHON_INCLUDE_PATH) -I$(NUMPY_INCLUDE_PATH) -L$(PYTHON_LD_PATH) -l$(PYTHON_VERSION)

	@echo "Building Florence supplementary tensor linear algebra module"
	$(CYTHON) LinAlg.pyx
	$(CXX) $(CXXSTD) $(CXXFLAGS) _Numeric.cpp _LinAlg.cpp LinAlg.cpp -o LinAlg.$(EXT_POSTFIX) -I. -I$(PYTHON_INCLUDE_PATH) -I$(NUMPY_INCLUDE_PATH) -L$(PYTHON_LD_PATH) -l$(PYTHON_VERSION)

clean:
	$(REMOVE) Numeric.cpp LinAlg.cpp Numeric.$(EXT_POSTFIX) LinAlg.$(EXT_POSTFIX)

source_clean:
	$(REMOVE) Numeric.cpp LinAlg.cpp