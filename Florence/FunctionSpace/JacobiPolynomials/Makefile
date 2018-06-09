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
CCFLAGS= -fPIC -shared -pthread -Wall $(CYTHONFLAGS) $(OPTFLAGS) -lm -flto

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
	@echo "Building Jacobi polynomials module"
	$(CYTHON) JacobiPolynomials.pyx
	$(CC) $(CCSTD) $(CCFLAGS) JacobiPolynomials.c -o JacobiPolynomials.$(EXT_POSTFIX) -I. -I$(PYTHON_INCLUDE_PATH) \
	-I$(NUMPY_INCLUDE_PATH) -L$(PYTHON_LD_PATH) -l$(PYTHON_VERSION) 

clean:
	$(REMOVE) JacobiPolynomials.c JacobiPolynomials.$(EXT_POSTFIX)

source_clean:
	$(REMOVE) JacobiPolynomials.c







# PYTHON_VERSION = python2.7
# PYTHON_INCLUDE_PATH = /usr/include/
# PYTHON_LD_PATH = /usr/lib/
# CYTHONFLAGS = -DNPY_NO_DEPRECATED_API

# CYTHON = cython
# REMOVE = rm -rf

# INLINE_THRESHOLD = 100000
# CCSTD=-std=c99
# OPTFLAGS= -O3 -fno-strict-aliasing -DNDEBUG
# CCFLAGS= -fPIC -shared -pthread -Wall $(CYTHONFLAGS) $(OPTFLAGS) -lm -flto

# # NUMPY_INCLUDE_PATH=python -c "import numpy; print numpy.get_include()"
# NUMPY_INCLUDE_PATH = /usr/local/lib/python2.7/dist-packages/numpy/core/include

# ifeq ($(findstring gcc,$(CC)),gcc)
# CCFLAGS += -Wno-cpp -Wno-unused-function -Wno-maybe-uninitialized
# CCFLAGS += -march=native -ffast-math -fwrapv -mfpmath=sse -funroll-loops -finline-functions -finline-limit=$(INLINE_THRESHOLD) -ffp-contract=fast
# endif
# ifeq ($(findstring clang,$(CC)),clang)
# CCFLAGS += -Wno-uninitialized
# CCFLAGS += -march=native -ffast-math -mllvm -inline-threshold=$(INLINE_THRESHOLD) -ffp-contract=fast
# endif
# ifeq ($(findstring icc,$(CC)),icc)
# CCFLAGS += -march=native -no-ansi-alias -xHost -fp-model fast=1 -inline-forceinline -inline-factor=$(INLINE_THRESHOLD)
# endif


# UNAME_S := $(shell uname -s)
# ifeq ($(UNAME_S),Linux)
# JACOBI_LD = -L. -l:jacobi.so
# endif
# ifeq ($(UNAME_S),Darwin)
# JACOBI_LD = -L. -ljacobi
# endif

# all: build_module

# build_module: 
# 	@echo echo "Building Jacobi polynomials module"
# 	$(CC) $(CCSTD) $(CCFLAGS) jacobi.c -o libjacobi.so 
# 	$(CYTHON) JacobiPolynomials.pyx
# 	$(CC) $(CCSTD) $(CCFLAGS) JacobiPolynomials.c -o JacobiPolynomials.so -I. -I$(PYTHON_INCLUDE_PATH) \
# 	-I$(NUMPY_INCLUDE_PATH) -L$(PYTHON_LD_PATH) -l$(PYTHON_VERSION) -L. -ljacobi

# clean:
# 	$(REMOVE) rm JacobiPolynomials.c JacobiPolynomials.so libjacobi.so