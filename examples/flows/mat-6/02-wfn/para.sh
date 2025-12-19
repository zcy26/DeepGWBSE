#!/bin/bash


MPIRUN='srun -n 512'

srun -n 512 parabands.cplx.x &> parabands.out

