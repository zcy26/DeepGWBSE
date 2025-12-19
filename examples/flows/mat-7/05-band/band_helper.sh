#!/bin/bash


MPIRUN='srun -n 512'

cp ./wfn.head.in wfn.in
cat kpath.txt >> wfn.in

