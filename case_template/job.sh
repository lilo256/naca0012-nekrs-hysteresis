#!/bin/bash
#$ -N NACA0012
#$ -pe openmpi 1
#$ -hard -l gpu=1
#$ -cwd
#$ -S /bin/bash
#$ -o $JOB_NAME.$JOB_ID.out
#$ -e $JOB_NAME.$JOB_ID.err

module load nekRS-26
mpirun --mca osc pt2pt -np $NSLOTS nekrs --setup airfoil.par --backend CUDA --device-id 0
echo "finished"
