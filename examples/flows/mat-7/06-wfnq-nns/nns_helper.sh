#!/bin/bash


MPIRUN='srun -n 512'

cp ./wfn.head.in wfn.in
setup_subsampling_nns.x BIN ../02-wfn/wfn.cplx &> nns_kpt.out
cat kpoints_all.dat >> wfn.in

