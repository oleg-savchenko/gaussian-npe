#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=rome
#SBATCH --time=04:00:00
#SBATCH --job-name=regen_zarr
#SBATCH --output=./job_outputs/%x-%j-%N_slurm.out
#SBATCH --error=./job_outputs/R-%x.%j.err

module load 2024
module load Python/3.12.3-GCCcore-13.3.0
export OMP_NUM_THREADS=18

cd /home/osavchenko/gaussian_npe

python3 data_scripts/quijote_lh.py
