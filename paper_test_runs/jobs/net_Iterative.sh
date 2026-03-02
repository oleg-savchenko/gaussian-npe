#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=4:00:00
#SBATCH --job-name=net_Iterative
#SBATCH --output=paper_test_runs/logs/net_Iterative_%j.out
#SBATCH --error=paper_test_runs/logs/net_Iterative_%j.err

module load 2024
module load Python/3.12.3-GCCcore-13.3.0
export OMP_NUM_THREADS=18

cd /home/osavchenko/gaussian_npe
srun python3 scripts/train.py \
    --run_name net_Iterative \
    --output_dir paper_test_runs/runs \
    --network Iterative \
    --max_epochs 50 \
    --sigma_noise 0.1 \
    --learning_rate 0.01 \
    --early_stopping_patience 5 \
    --lr_scheduler_patience 3 \
    --batch_size 8 \
    --num_samples 100 \
    --noise_seed 42 \
    --target_path /home/osavchenko/Quijote/Quijote_target/Quijote_sample0_wout_MAK.pt
