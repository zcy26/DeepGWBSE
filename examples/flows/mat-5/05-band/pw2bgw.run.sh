#!/bin/bash


MPIRUN='srun -n 512'
PW='/pscratch/sd/b/bwhou/software/qe/q-e-qe-6.8/bin/pw.x'
PWFLAGS='-nk 16'
PW2BGW='/pscratch/sd/b/bwhou/software/qe/q-e-qe-6.8/bin/pw2bgw.x'

$MPIRUN $PW2BGW $PWFLAGS -in wfn.pp.in &> wfn.pp.out

BANDS='/pscratch/sd/b/bwhou/software/qe/q-e-qe-6.8/bin/bands.x'
$MPIRUN $BANDS $PWFLAGS -in bands.in &> bands.out

