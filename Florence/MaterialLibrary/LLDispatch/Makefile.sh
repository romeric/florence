#!/bin/bash

set -x

clear

# CXX=clang++-3.8
# CXX=clang++-3.6
CXX=clang++


echo "Building Fastor based material models" 
NUMPY=$(python -c "import numpy; print numpy.get_include()")
FASTOR=/home/roman/Dropbox/Fastor/

# Python libs/includes
PYLIB=python2.7
PYINC=/usr/include/python2.7


####################################################################################
function make_material() {
    rm $1.so 
    cd CythonSource
    rm $1.cpp

    cython --cplus $1.pyx

    $CXX -std=c++11 -fPIC -shared -pthread -O3 -fwrapv -fno-strict-aliasing -finline-functions \
    -ffast-math -mfpmath=sse -funroll-loops -mavx -DNPY_NO_DEPRECATED_API -DNBOUNDSCHECK \
    $1.cpp -o $1.so -I$PYINC -l$PYLIB -lm -I$NUMPY -I$FASTOR -Wno-everything

    mv $1.so ../$1.so
    cd ..
    printf "\n"
}
#####################################################################################
printf "Building low level dispatcher for material models"
make_material _NeoHookean_2_
make_material _MooneyRivlin_0_
make_material _NearlyIncompressibleMooneyRivlin_
make_material _AnisotropicMooneyRivlin_1_
make_material _IsotropicElectroMechanics_0_
make_material _IsotropicElectroMechanics_3_
make_material _SteinmannModel_
make_material _IsotropicElectroMechanics_105_
make_material _IsotropicElectroMechanics_106_
make_material _IsotropicElectroMechanics_107_