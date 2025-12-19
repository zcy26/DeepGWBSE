#!/bin/bash


MPIRUN='srun -n 512'
PW='/pscratch/sd/b/bwhou/software/qe/q-e-qe-6.8/bin/pw.x'
PWFLAGS='-nk 16'

ln -nfs ../../01-density/mat-5.save/charge-density.dat mat-5.save/charge-density.dat
ln -nfs ../../01-density/mat-5.save/spin-polarization.dat mat-5.save/spin-polarization.dat

cp -f ../pp/B.upf mat-5.save/B.upf
cp -f ../pp/N.upf mat-5.save/N.upf
cp -f ../01-density/mat-5.save/data-file-schema.xml mat-5.save/data-file-schema.xml

$MPIRUN $PW $PWFLAGS -in wfn.in &> wfn.out

