#!/bin/bash


MPIRUN='srun -n 512'

bash wfn.run.sh
bash pw2bgw.run.sh

