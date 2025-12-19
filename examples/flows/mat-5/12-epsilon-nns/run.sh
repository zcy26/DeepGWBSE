#!/bin/bash


MPIRUN='srun -n 512'
EPSILON='/global/homes/b/bwhou/software/BerkeleyGW-master/bin/epsilon.cplx.x'

ln -nfs ../02-wfn/wfn.cplx WFN
ln -nfs ../06-wfnq-nns/wfn.cplx WFNq

$MPIRUN $EPSILON &> epsilon.out

