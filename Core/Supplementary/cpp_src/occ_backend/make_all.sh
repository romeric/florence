#!/bin/sh

clear
rm PostMeshPy.cpp PostMeshPy.so
python setup.py build_ext -if
rm -rf build 
# clear
