#!/bin/sh

clear
rm PostMeshPy.cpp PostMeshPy.so
python setup.py build_ext -ifq
rm -rf build 
# clear
