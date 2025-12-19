#!/bin/bash


MPIRUN='srun -n 512'

bash band_helper.sh
bash wfn.run.sh
bash pw2bgw.run.sh

