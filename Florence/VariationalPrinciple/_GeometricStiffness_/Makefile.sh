
clear
set -x

C=gcc 
CXX=g++

echo "Building florences constitutive and geometric integrands" 
NUMPY=$(python -c "import numpy; print numpy.get_include()")

##################################################################
rm ../_GeometricStiffness_.so _GeometricStiffness_.c
cython _GeometricStiffness_.pyx

gcc -std=c99 -shared -fPIC _GeometricStiffness_.c -o _GeometricStiffness_.so -O3 -fwrapv -fno-strict-aliasing -finline-functions \
-ffast-math -mfpmath=sse -funroll-loops -mavx -DNPY_NO_DEPRECATED_API -I/usr/include/python2.7 -lpython2.7 -lm -I$NUMPY -I../../../Tensor/ #-fopenmp

mv _GeometricStiffness_.so ../
##################################################################