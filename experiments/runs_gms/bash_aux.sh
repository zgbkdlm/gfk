#!/bin/bash

dx=$1
nparticles=$2

for offset in 0. 5. 10.;
do
  python runs_gms/aux.py --id_l=0 --id_u=99 --dx=$dx --dy=1 --offset=$offset --nparticles=$nparticles
done
