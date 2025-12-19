#!/bin/bash


MPIRUN='srun -n 512'

cp epsilon.head.in epsilon.inp
cat ../06-wfnq-nns/epsilon_q0s.inp >> epsilon.inp

