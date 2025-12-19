#!/bin/bash


MPIRUN='srun -n 512'
PW='/pscratch/sd/b/bwhou/software/qe/q-e-qe-6.8/bin/pw.x'
PWFLAGS='-nk 16'

$MPIRUN $PW $PWFLAGS -in scf.in &> scf.out

