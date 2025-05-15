#!/bin/bash

dx=$1
nparticles=$2

for no_smc in "" "--no_smc";
do
  for tweedie in "" "--tweedie";
  do
    for offset in 0. 5. 10.;
    do
      python runs_gms/wu.py --id_l=0 --id_u=99 --dx=$dx --dy=1 --offset=$offset --nparticles=$nparticles $no_smc $tweedie
    done
  done
done
