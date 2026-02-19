"""
Sweep over 5 noise levels (sigma_noise) for Gaussian NPE training.

Generates and submits one SLURM job per noise level.
All run outputs are saved under ./paper_test_runs/runs/.

Usage:
    python paper_test_runs/sweep_noise.py             # submit all jobs
    python paper_test_runs/sweep_noise.py --dry_run    # print commands without submitting
"""

import os
import subprocess
import argparse

# ── Noise levels to sweep ────────────────────────────────────────────
SIGMA_NOISE_VALUES = [0.0, 0.01, 0.05, 0.1, 0.5]

# ── SLURM template (based on my_py_gpu_job.sh) ──────────────────────
SLURM_TEMPLATE = """\
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=1:00:00
#SBATCH --job-name={run_name}
#SBATCH --output=paper_test_runs/logs/{run_name}_%j.out
#SBATCH --error=paper_test_runs/logs/{run_name}_%j.err

module load 2024
module load Python/3.12.3-GCCcore-13.3.0
export OMP_NUM_THREADS=18

cd /home/osavchenko/gaussian_npe
srun python3 scripts/train.py \
    --run_name {run_name} \
    --output_dir paper_test_runs/runs \
    --sigma_noise {sigma_noise} \
    --max_epochs 30
"""

# ── Main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without submitting")
    args = parser.parse_args()

    # Resolve paths relative to the repo root
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    jobs_dir = os.path.join(repo_root, "paper_test_runs", "jobs")
    logs_dir = os.path.join(repo_root, "paper_test_runs", "logs")
    os.makedirs(jobs_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    for sigma in SIGMA_NOISE_VALUES:
        run_name = f"noise_{sigma}"

        script = SLURM_TEMPLATE.format(
            run_name=run_name,
            sigma_noise=sigma,
        )

        script_path = os.path.join(jobs_dir, f"{run_name}.sh")
        with open(script_path, "w") as f:
            f.write(script)

        if args.dry_run:
            print(f"[dry run] would submit: {script_path}")
            print(f"          run_name={run_name}, sigma_noise={sigma}")
        else:
            print(f"Submitting: {run_name} (sigma_noise={sigma})")
            subprocess.run(["sbatch", script_path], check=True)

    print(f"\nDone. {len(SIGMA_NOISE_VALUES)} jobs {'would be' if args.dry_run else ''} submitted.")
    print(f"Run outputs will be saved to: paper_test_runs/runs/")


if __name__ == "__main__":
    main()
