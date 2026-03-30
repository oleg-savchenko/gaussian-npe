"""
Sweep over training set sizes for Gaussian NPE training.

Generates and submits one SLURM job per n_train value. All training hyperparameters
can be overridden via CLI flags; defaults match train.py.
All run outputs are saved under ./paper_test_runs/runs/{TIMESTAMP}_sweep_train/.

Usage:
    python paper_test_runs/sweep_train.py --dry_run

    python paper_test_runs/sweep_train.py \
        --n_train_values 100 250 500 1000 2000 \
        --output_dir paper_test_runs/runs \
        --network IsotropicD \
        --max_epochs 70 \
        --sigma_noise 0.1 \
        --learning_rate 0.01 \
        --early_stopping_patience 5 \
        --lr_scheduler_patience 3 \
        --batch_size 8 \
        --num_samples 100 \
        --MAS PCS \
        --noise_seed 42 \
        --store_path /gpfs/scratch1/shared/osavchenko/zarr_stores/Quijote_fiducial_res128_deconv_MAK \
        --target_path ./Quijote_target/Quijote_sample0.pt \
        --cpus_per_task 18 \
        --time 1:00:00
"""

import os
import subprocess
import argparse
from datetime import datetime

# ── Default training set sizes to sweep ──────────────────────────────
DEFAULT_N_TRAIN_VALUES = [250, 500, 1000, 1500, 2000]

# ── SLURM header template ────────────────────────────────────────────
SLURM_HEADER = """\
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time={time}
#SBATCH --job-name={run_name}
#SBATCH --output=paper_test_runs/logs/{run_name}_%j.out
#SBATCH --error=paper_test_runs/logs/{run_name}_%j.err

module load 2024
module load Python/3.12.3-GCCcore-13.3.0
export OMP_NUM_THREADS={cpus_per_task}

cd /home/osavchenko/gaussian_npe
"""


def build_train_cmd(run_name, n_train, args):
    """Build the train.py srun command for a given training set size."""
    parts = [
        f'    --run_name {run_name}',
        f'    --output_dir {args.output_dir}',
        f'    --network {args.network}',
        f'    --max_epochs {args.max_epochs}',
        f'    --sigma_noise {args.sigma_noise}',
        f'    --n_train {n_train}',
        f'    --learning_rate {args.learning_rate}',
        f'    --early_stopping_patience {args.early_stopping_patience}',
        f'    --lr_scheduler_patience {args.lr_scheduler_patience}',
        f'    --batch_size {args.batch_size}',
        f'    --num_samples {args.num_samples}',
        f'    --noise_seed {args.noise_seed}',
    ]
    if args.MAS is not None:
        parts.append(f'    --MAS {args.MAS}')
    if args.store_path is not None:
        parts.append(f'    --store_path {args.store_path}')
    if args.target_path is not None:
        parts.append(f'    --target_path {args.target_path}')
    if args.network == 'Poisson':
        parts.append(f'    --n_bar {args.n_bar}')
        parts.append(f'    --galaxy_bias {args.galaxy_bias}')
    return 'python3 scripts/train.py \\\n' + ' \\\n'.join(parts)


# ── Main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Sweep over training set sizes',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Sweep control ────────────────────────────────────────────────
    parser.add_argument('--dry_run', action='store_true',
                        help='Print commands without submitting')
    parser.add_argument('--n_train_values', type=int, nargs='+',
                        default=DEFAULT_N_TRAIN_VALUES,
                        help='List of n_train values to sweep over')
    parser.add_argument('--output_dir', type=str, default='paper_test_runs/runs',
                        help='Base output directory; runs go to {output_dir}/{timestamp}_sweep_train/')

    # ── SLURM resources ──────────────────────────────────────────────
    parser.add_argument('--cpus_per_task', type=int, default=18,
                        help='SLURM --cpus-per-task')
    parser.add_argument('--time', type=str, default='4:00:00',
                        help='SLURM --time wall-clock limit')

    # ── Training hyperparameters (same defaults as train.py) ─────────
    parser.add_argument('--network', type=str, default='IsotropicD')
    parser.add_argument('--max_epochs', type=int, default=70)
    parser.add_argument('--sigma_noise', type=float, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--early_stopping_patience', type=int, default=5)
    parser.add_argument('--lr_scheduler_patience', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--MAS', type=str, default=None)
    parser.add_argument('--noise_seed', type=int, default=42)
    parser.add_argument('--store_path', type=str, default=None,
                        help='Override the default ZarrStore path')
    parser.add_argument('--target_path', type=str, default=None,
                        help='Override the default target .pt file path')
    parser.add_argument('--n_bar', type=float, default=5e-4,
                        help='Galaxy number density [h^3/Mpc^3] for Poisson network')
    parser.add_argument('--galaxy_bias', type=float, default=1.5,
                        help='Linear galaxy bias for Poisson network')

    args = parser.parse_args()

    # Resolve paths relative to the repo root
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    jobs_dir = os.path.join(repo_root, 'paper_test_runs', 'jobs')
    logs_dir = os.path.join(repo_root, 'paper_test_runs', 'logs')
    os.makedirs(jobs_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
    sweep_output_dir = os.path.join(args.output_dir, f'{timestamp}_sweep_train')
    print(f'Sweep output dir: {sweep_output_dir}/')

    for n_train in args.n_train_values:
        run_name = f'ntrain_{n_train}'

        header = SLURM_HEADER.format(
            run_name=run_name,
            cpus_per_task=args.cpus_per_task,
            time=args.time,
        )
        # Pass sweep_output_dir so each run lands in the timestamped sweep folder
        args_copy = argparse.Namespace(**vars(args))
        args_copy.output_dir = sweep_output_dir
        script = header + build_train_cmd(run_name, n_train, args_copy) + '\n'

        script_path = os.path.join(jobs_dir, f'{run_name}.sh')
        with open(script_path, 'w') as f:
            f.write(script)

        if args.dry_run:
            print(f'[dry run] would submit: {script_path}')
            print(f'          n_train={n_train}')
        else:
            print(f'Submitting: {run_name} (n_train={n_train})')
            subprocess.run(['sbatch', script_path], check=True)

    print(f'\nDone. {len(args.n_train_values)} jobs {"would be " if args.dry_run else ""}submitted.')
    print(f'Run outputs will be saved to: {sweep_output_dir}/')


if __name__ == '__main__':
    main()
