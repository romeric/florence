#!/bin/bash

set -x

rm JacobiPolynomials.c JacobiPolynomials.so

NUMPY=$(python -c "import numpy; print numpy.get_include()")

echo "Building Jacobi module" 
cython JacobiPolynomials.pyx
gcc -shared -fPIC -Wall -finline-functions -ffast-math -mfpmath=sse -funroll-loops \
-O3 jacobi.c -o jacobi.so -lm
gcc -fPIC -shared -pthread -O3 -fwrapv -fno-strict-aliasing -Wall -finline-functions \
-ffast-math -mfpmath=sse -funroll-loops -DNPY_NO_DEPRECATED_API -Wno-cpp -Wno-unused-function \
JacobiPolynomials.c -o JacobiPolynomials.so -I/usr/include/python2.7 -lpython2.7 -lm -I$NUMPY -L. -l:jacobi.so


