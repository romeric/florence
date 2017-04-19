#!/bin/bash
PYTHON_INCLUDE_PATH=$1
PYTHON_LD_PATH=$2
OPTFLAGS= -O3 -march=native

rm Numeric.cpp LinAlg.cpp Numeric.so LinAlg.so

NUMPY=$(python -c "import numpy; print numpy.get_include()")

echo "Building Numeric module" 
cython --cplus Numeric.pyx
$CXX -std=c++11 -fPIC -shared -pthread -O3 -fwrapv -fno-strict-aliasing -Wall -finline-functions \
-ffast-math -mfpmath=sse -funroll-loops -DNPY_NO_DEPRECATED_API -Wno-cpp -Wno-unused-function \
Numeric.cpp _Numeric.cpp -o Numeric.so -I/usr/include/python2.7 -lpython2.7 -lm -I$NUMPY

echo "Building LinAlg module"
cython --cplus LinAlg.pyx
$CXX -std=c++11 -fPIC -shared _Numeric.cpp LinAlg.cpp -o LinAlg.so -I. -I/usr/include/python2.7 -lpython2.7 \
-DNPY_NO_DEPRECATED_API -Wno-cpp -Wno-unused-function \
-O3 -pthread -Wall -fwrapv -ffast-math -mfpmath=sse -funroll-loops -finline-functions -I$NUMPY