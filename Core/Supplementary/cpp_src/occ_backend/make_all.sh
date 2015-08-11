#!/bin/sh

clear
python setup.py build_ext -if


# make build directory if it does not exist
# if [ ! -d "build" ]; then
#   mkdir build 
# fi

# echo Cleaning the build directory
# rm OCCPluginInterface.o OCCPluginInterface.so
# rm OCCPlugin.o OCCPlugin.so
# rm OCCPluginPy.so OCCPluginPy.cpp 

# g++ -c -fPIC src/OCCPlugin.cpp -I./include/ -I/home/roman/Dropbox/eigen-devel/ \
# -I/usr/local/include/oce -I/usr/include/python2.7/  -std=c++11  #-O2
# g++ -shared -o OCCPlugin.so OCCPlugin.o
# echo Successfully built OCCPlugin shared library 1/2

# echo Building shared libraries
# g++ -c -fPIC src/OCCPluginInterface.cpp -I./include/ -I/home/roman/Dropbox/eigen-devel/ \
# 	-I/usr/local/include/oce/ -I/usr/include/python2.7/ -std=c++11 #-O2 -funroll-loops -finline-functions
# g++ -shared -o OCCPluginInterface.so OCCPluginInterface.o
# echo Successfully built OCCPluginInterface shared library 2/2


# echo DONE

# clear

# echo Cleaning the build directory
# rm OCCPluginInterface.o OCCPluginInterface.so
# rm OCCPlugin.o OCCPlugin.so
# rm OCCPluginPy.so OCCPluginPy.cpp 


# g++ -c -fPIC OCCPlugin.cpp -I. -I/home/roman/Dropbox/eigen-devel/ -I/usr/local/include/oce -I/usr/include/python2.7/  -std=c++11  #-O2
# g++ -shared -o OCCPlugin.so OCCPlugin.o
# echo Successfully built OCCPlugin shared library 1/2

# echo Building shared libraries
# g++ -c -fPIC OCCPluginInterface.cpp -I. -I/home/roman/Dropbox/eigen-devel/ -I/usr/local/include/oce/ -I/usr/include/python2.7/ -std=c++11 #-O2 -funroll-loops -finline-functions
# g++ -shared -o OCCPluginInterface.so OCCPluginInterface.o
# echo Successfully built OCCPluginInterface shared library 2/2


# echo DONE

# python setup.py build_ext -i



# -pthread -fopenmp -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC 