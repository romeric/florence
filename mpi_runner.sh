#!/bin/bash

cd ~/Dropbox/florence

# start=$SECONDS
time mpirun -np 16 Florence/FiniteElements/DistributedAssembly.py "/home/roman/tmp/"
# duration=$(( SECONDS - start ))
# echo $duration
