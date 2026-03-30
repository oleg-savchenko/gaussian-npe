#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=1:00:00
#SBATCH --job-name=noise_0.25
#SBATCH --output=paper_test_runs/logs/noise_0.25_%j.out
#SBATCH --error=paper_test_runs/logs/noise_0.25_%j.err

module load 2024
module load Python/3.12.3-GCCcore-13.3.0
export OMP_NUM_THREADS=18

cd /home/osavchenko/gaussian_npe
python3 scripts/train.py \
    --run_name noise_0.25 \
    --output_dir paper_test_runs/runs/260329_211805_sweep_noise \
    --network default \
    --max_epochs 30 \
    --sigma_noise 0.25 \
    --learning_rate 0.01 \
    --early_stopping_patience 5 \
    --lr_scheduler_patience 3 \
    --batch_size 8 \
    --num_samples 100 \
    --noise_seed 42
