#!/bin/bash


MPIRUN='srun -n 16'
PW='/pscratch/sd/b/bwhou/software/qe/q-e-qe-6.8/bin/pw.x'
PWFLAGS=' '
PW2BGW='/pscratch/sd/b/bwhou/software/qe/q-e-qe-6.8/bin/pw2bgw.x'

$MPIRUN $PW2BGW $PWFLAGS -in wfn.pp.in &> wfn.pp.out

