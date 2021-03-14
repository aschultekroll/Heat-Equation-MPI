#!/bin/bash
#SBATCH --job-name=project
#SBATCH --partition=s_hadoop
#SBATCH --nodes=1
#SBATCH --no-kill
#SBATCH --ntasks-per-node=36
#SBATCH --time=45:00

module purge
module load tools/python/3.7 mpi/intel/2019-Update5
srun python3 mpi.py