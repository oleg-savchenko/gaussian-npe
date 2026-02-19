#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=0:10:00
#SBATCH --output=%x-%j-%N_slurm.out
#SBATCH --error=R-%x.%j.err
## Activate right env

module load 2024
module load Python/3.12.3-GCCcore-13.3.0
# module load OpenMPI/4.1.4-GCC-11.3.0
# module load FFTW.MPI/3.3.10-gompi-2022a
# module load GSL/2.7-GCC-11.3.0
# module load HDF5/1.12.2-gompi-2022a
# module load UCX-CUDA/1.12.1-GCCcore-11.3.0-CUDA-11.7.0
# module load UCX/1.12.1-GCCcore-11.3.0
export OMP_NUM_THREADS=18
export PYTHONNOUSERSITE=0

cd /home/osavchenko/gaussian_npe
srun python3 scripts/train.py --max_epochs 1 --num_samples 100
