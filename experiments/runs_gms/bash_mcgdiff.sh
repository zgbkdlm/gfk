#!/bin/bash

dx=$1
nparticles=$2

python runs_gms/mcgdiff.py --id_l=0 --id_u=99 --dx=$dx --dy=1 --offset=0. --nparticles=$nparticles
