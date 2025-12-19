#!/bin/bash


MPIRUN='srun -n 512'

bash nns_helper.sh
bash wfn.run.sh
bash pw2bgw.run.sh

