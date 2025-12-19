#!/bin/bash


MPIRUN='srun -n 512'
INTEQP='inteqp.cplx.x'

ln -nfs ../13-sigma/eqp1.dat eqp_co.dat
ln -nfs ../02-wfn/wfn.cplx WFN_co
ln -nfs ../05-band/wfn.cplx WFN_fi

$MPIRUN $INTEQP &> inteqp.log
python plot.py

