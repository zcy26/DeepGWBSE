#!/bin/bash


MPIRUN='srun -n 512'
ABSORPTION='absorption.cplx.x'

ln -nfs ../02-wfn/wfn.cplx WFN_co
ln -nfs ../17-wfn_fi/wfn.cplx WFN_fi
ln -nfs ../17-wfn_fi/wfn.cplx WFNq_fi
ln -nfs ../11-epsilon/eps0mat.h5 eps0mat.h5
ln -nfs ../11-epsilon/epsmat.h5 epsmat.h5
ln -nfs ../18-kernel/bsemat.h5 bsemat.h5
ln -nfs ../13-sigma/eqp1.dat eqp_co.dat

$MPIRUN $ABSORPTION &> absorption.out

