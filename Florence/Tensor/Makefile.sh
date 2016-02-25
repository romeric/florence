#!/bin/bash

set -x

rm Numeric.cpp LinAlg.cpp Numeric.so LinAlg.so

echo "Building Numeric module" 
cython --cplus Numeric.pyx
g++ -std=c++11 -fPIC -shared -pthread -O3 -fwrapv -fno-strict-aliasing -Wall -finline-functions \
-ffast-math -mfpmath=sse -funroll-loops -DNPY_NO_DEPRECATED_API -Wno-cpp -Wno-unused-function \
Numeric.cpp _Numeric.cpp -o Numeric.so -I/usr/include/python2.7 -lpython2.7 -lm

echo "Building LinAlg module"
cython --cplus LinAlg.pyx
g++ -std=c++11 -fPIC -shared _Numeric.cpp LinAlg.cpp -o LinAlg.so -I. -I/usr/include/python2.7 -lpython2.7 \
-DNPY_NO_DEPRECATED_API -Wno-cpp -Wno-unused-function \
-O3 -pthread -Wall -fwrapv -ffast-math -mfpmath=sse -funroll-loops -finline-functions