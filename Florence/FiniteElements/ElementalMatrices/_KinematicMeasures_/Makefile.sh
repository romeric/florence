
clear
set -x

C=gcc
CXX=g++

echo "Building Fastor based material models" 
NUMPY=$(python -c "import numpy; print numpy.get_include()")

rm ../_KinematicMeasures_.so _KinematicMeasures_.c
cython _KinematicMeasures_.pyx

gcc -std=c99 -shared -fPIC _KinematicMeasures_.c -o _KinematicMeasures_.so -O3 -fwrapv -fno-strict-aliasing -finline-functions \
-ffast-math -mfpmath=sse -funroll-loops -mavx -DNPY_NO_DEPRECATED_API -I/usr/include/python2.7 -lpython2.7 -lm -I$NUMPY -I../../../Tensor/

mv _KinematicMeasures_.so ../
