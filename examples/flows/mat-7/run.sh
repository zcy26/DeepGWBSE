#!/bin/bash



  
cd 01-density
echo "01-density"
bash run.sh
cd ..
  
cd 01-density
echo "01-density"
bash pw2bgw.run.sh
cd ..
  
cd 02-wfn
echo "02-wfn"
bash run.sh
cd ..
  
cd 02-wfn
echo "02-wfn"
bash para.sh
cd ..
  
cd 03-wfnq
echo "03-wfnq"
bash run.sh
cd ..
  
cd 03-wfnq
echo "03-wfnq"
bash pseudo.sh
cd ..
  
cd 06-wfnq-nns
echo "06-wfnq-nns"
bash run.sh
cd ..
  
cd 05-band
echo "05-band"
bash run.sh
cd ..
  
cd 07-aobasis
echo "07-aobasis"
bash aobasis.run
cd ..
  
cd 11-epsilon
echo "11-epsilon"
bash run.sh
cd ..
  
cd 12-epsilon-nns
echo "12-epsilon-nns"
bash nns_helper.sh
cd ..
  
cd 12-epsilon-nns
echo "12-epsilon-nns"
bash run.sh
cd ..
  
cd 13-sigma
echo "13-sigma"
bash run.sh
cd ..
  
cd 14-inteqp
echo "14-inteqp"
bash run.sh
cd ..
  
cd 16-reconstruction
echo "16-reconstruction"
bash hpro.run
cd ..
  
cd 17-wfn_fi
echo "17-wfn_fi"
bash wfn.run.sh
cd ..
  
cd 17-wfn_fi
echo "17-wfn_fi"
bash pw2bgw.run.sh
cd ..
  
cd 18-kernel
echo "18-kernel"
bash run.sh
cd ..
  
cd 19-absorption
echo "19-absorption"
bash run.sh
cd ..

