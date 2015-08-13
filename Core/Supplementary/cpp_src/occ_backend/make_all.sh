#!/bin/sh

clear
rm OCCPluginPy.cpp OCCPluginPy.so
python setup.py build_ext -if
rm -rf build 
