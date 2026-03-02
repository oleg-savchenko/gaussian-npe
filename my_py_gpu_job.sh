#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=04:00:00
#SBATCH --output=./job_outputs/%x-%j-%N_slurm.out
#SBATCH --error=./job_outputs/R-%x.%j.err
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

# srun python data_scripts/discodj_store.py
# srun python scripts/train.py --plot_only --output_dir ./runs/20260301_193604_WienerIsotropicD_test
# srun python3 scripts/train.py --network UNet_Only --max_epochs 60 --num_samples 100 --run_name UNet_Only_run_optimized

# python scripts/train.py --network default_IsotropicD --max_epochs 70 --run_name defaultIsotropicD

python scripts/train.py --network LH --store_path /gpfs/scratch1/shared/osavchenko/zarr_stores/Quijote_LH_res128_deconv_MAK --n_train 1990 --sigma_noise 1 --max_epochs 70 --run_name LH_sigma_noise_1

# srun python scripts/train.py --network WienerIsotropicD --max_epochs 120 \
#         --ckpt_path ./runs/20260224_192240_WienerIsotropicD/logs/tb_logs/version_0/checkpoints/epoch=57-step=11600.ckpt

# srun python scripts/train.py --run_name Poisson_noise --network Poisson --max_epochs 60 \

# srun python3 scripts/infer.py --model_dir runs/20260220_152350_WienerNet_resumed

# srun python scripts/train.py --run_name test_customunet_4ch --network CustomUNet --max_epochs 50 --num_samples 100 #--n_train 100

# srun python3 scripts/train.py \
#     --run_name WienerNet_resumed \
#     --network WienerNet \
#     --max_epochs 100 \
#     --ckpt_path ./runs/20260220_020149_WienerNet/logs/tb_logs/version_0/checkpoints/epoch=48-step=9800.ckpt

# srun python3scripts/train.py --max_epochs 50 --num_samples 100 --run_name WienerNet --network WienerNet --max_epochs 50 #--n_train 100

# srun python3 scripts/wiener_filter.py \
#     --target_path /home/osavchenko/Quijote/Quijote_target/Quijote_sample0_wout_MAK.pt \
#     --sigma_noise 0.1 \
#     --num_samples 100

# srun python scripts/fit_D_spectrum.py \
#     --model_dir runs/20260220_152350_WienerNet_resumed \
#     --n_bins 40 \

# python paper_test_runs/sweep_networks.py --max_epochs 70 --time 5:00:00
