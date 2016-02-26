#!/bin/bash

set -x

rm _fromfile_reader.so

NUMPY=$(python -c "import numpy; print numpy.get_include()")

echo "Building Salome mesh reader"
cython _fromfile_reader.pyx
gcc -fPIC -shared _fromfile_reader.c -o _fromfile_reader.so -I. -I/usr/include/python2.7 -lpython2.7 \
-DNPY_NO_DEPRECATED_API -Wno-cpp -Wno-unused-function \
-O3 -pthread -Wall -fwrapv -ffast-math -mfpmath=sse -funroll-loops -finline-functions -I$NUMPY