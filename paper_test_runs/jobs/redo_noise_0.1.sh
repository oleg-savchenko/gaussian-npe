#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=5:00:00
#SBATCH --job-name=redo_noise_0.1
#SBATCH --output=paper_test_runs/logs/redo_noise_0.1_%j.out
#SBATCH --error=paper_test_runs/logs/redo_noise_0.1_%j.err

module load 2024
module load Python/3.12.3-GCCcore-13.3.0
export OMP_NUM_THREADS=18

cd /home/osavchenko/gaussian_npe

TARGET_DIR="paper_test_runs/runs/260304_233941_sweep_noise/260304_234004_noise_0.1"
SWEEP_DIR="paper_test_runs/runs/260304_233941_sweep_noise"

python3 scripts/train.py \
    --run_name noise_0.1 \
    --output_dir "$SWEEP_DIR" \
    --network IsotropicD \
    --max_epochs 70 \
    --sigma_noise 0.1 \
    --learning_rate 0.01 \
    --early_stopping_patience 5 \
    --lr_scheduler_patience 3 \
    --batch_size 8 \
    --num_samples 100 \
    --noise_seed 42

# Find the newly created run dir (newest match for *noise_0.1 in sweep dir)
NEW_DIR=$(ls -td "$SWEEP_DIR"/*noise_0.1 2>/dev/null | head -1)

if [ -z "$NEW_DIR" ] || [ "$NEW_DIR" = "$TARGET_DIR" ]; then
    echo "ERROR: could not identify new run directory"
    exit 1
fi

echo "New run dir: $NEW_DIR"
echo "Replacing: $TARGET_DIR"

rsync -av --delete "$NEW_DIR/" "$TARGET_DIR/"
rm -rf "$NEW_DIR"
echo "Done — results are in $TARGET_DIR"
