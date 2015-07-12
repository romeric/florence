#!/bin/sh

rm py_to_occ_frontend.o py_to_occ_frontend.so
rm occ_frontend.o occ_frontend.so
rm PyInterface_OCC_FrontEnd.so PyInterface_OCC_FrontEnd.cpp 
echo cleaned the build directory

g++ -c -fPIC py_to_occ_frontend.cpp -I. -I/usr/local/include/eigen_3_2_0/ -I/usr/local/include/oce/ -I/usr/include/python2.7/ #-O2
g++ -shared -o py_to_occ_frontend.so py_to_occ_frontend.o
echo successfully build shared library 1/2

g++ -c -fPIC occ_frontend.cpp -I. -I/usr/local/include/eigen_3_2_0 -I/usr/local/include/oce -I/usr/include/python2.7/  #-O2
g++ -shared -o occ_frontend.so occ_frontend.o
echo successfully build shared library 2/2
echo DONE

python setup.py build_ext -i
