#!/bin/bash


MPIRUN='srun -n 16'

bash wfn.run.sh
bash pw2bgw.run.sh

