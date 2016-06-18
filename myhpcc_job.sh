#! /bin/bash

qsub -q isi -l walltime=24:00:00,mem=16gb,nodes=1:ppn=1 -A lc_dmm -m be  ./mylocaljob.sh

