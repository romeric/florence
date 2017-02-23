#!/bin/bash

set -x

clear

# CXX=clang++-3.8
# CXX=clang++-3.6
CXX=clang++


NUMPY=$(python -c "import numpy; print numpy.get_include()")
FASTOR=/home/roman/Dropbox/Fastor/

# Python libs/includes
PYLIB=python2.7
PYINC=/usr/include/python2.7


####################################################################################
function make_bases() {
    rm $1.so $1.cpp
    cython --cplus $1.pyx

    $CXX -std=c++11 -fPIC -shared -pthread -O3 -fwrapv -fno-strict-aliasing -finline-functions \
    -ffast-math -mfpmath=sse -funroll-loops -mavx -DNPY_NO_DEPRECATED_API \
    $1.cpp -o $1.so -I$PYINC -l$PYLIB -lm -I$NUMPY -I../../../Tensor/ -Wno-everything

    mv $1.so ../$1.so
    cd ..
}
#####################################################################################
printf "Building Bjorck Pereyra bases"
make_bases LineBP