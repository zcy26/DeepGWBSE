#!/bin/bash


MPIRUN='srun -n 512'
SIGMA='/global/homes/b/bwhou/software/BerkeleyGW-master/bin/sigma.cplx.x'

ln -nfs ../02-wfn/wfn.cplx WFN_inner
ln -nfs ../02-wfn/rho.real RHO
ln -nfs ../02-wfn/VXC VXC
ln -nfs ../12-epsilon-nns/eps0mat.h5 eps0mat.h5
ln -nfs ../11-epsilon/epsmat.h5 epsmat.h5
ln -nfs ../06-wfnq-nns/subweights.dat subweights.dat

$MPIRUN $SIGMA &> sigma.out

