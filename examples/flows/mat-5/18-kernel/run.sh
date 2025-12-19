#!/bin/bash


MPIRUN='srun -n 512'
KERNEL='kernel.cplx.x'

ln -nfs ../02-wfn/wfn.cplx WFN_co
ln -nfs ../11-epsilon/eps0mat.h5 eps0mat.h5
ln -nfs ../11-epsilon/epsmat.h5 epsmat.h5

$MPIRUN $KERNEL &> kernel.out

