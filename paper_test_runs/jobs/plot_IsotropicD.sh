#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=1:00:00
#SBATCH --job-name=plot_IsotropicD
#SBATCH --output=paper_test_runs/logs/plot_IsotropicD_%j.out
#SBATCH --error=paper_test_runs/logs/plot_IsotropicD_%j.err

module load 2024
module load Python/3.12.3-GCCcore-13.3.0
export OMP_NUM_THREADS=18

cd /home/osavchenko/gaussian_npe

RUN_DIR=paper_test_runs/runs/20260301_215801_net_IsotropicD

python3 -u paper_plots_scripts/generate_and_plot.py \
    --model_dir ${RUN_DIR} \
    --num_samples 1000 \
    --use_latex

echo "=== Done ==="
