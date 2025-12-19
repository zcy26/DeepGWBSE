#!/bin/bash


MPIRUN='srun -n 512'

srun -n 1 wfn2hdf.x BIN wfn.cplx wfn.h5 &> wfn2hdf.out
srun -n 1 python pseudobands.py --fname_in ../02-wfn/wfn.h5 --fname_in_q wfn.h5 --fname_out wfn_k.h5 --fname_out_q wfn_q.h5 --N_P_cond 10 --N_S_cond 40 --N_xi_cond 5  &> pseudo.out
srun -n 1 hdf2wfn.x BIN wfn_k.h5 ../02-wfn/wfn.cplx &> wfn2hdf.out

