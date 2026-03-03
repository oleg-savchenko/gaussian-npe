"""
Linear Wiener Filter Baseline for Cosmological IC Reconstruction

Applies the analytical optimal linear Wiener filter to a single target
observation and produces the same diagnostic plots as infer.py, allowing
direct comparison with trained neural networks.

No model training is required. The filter is fully determined by:
  - The fiducial linear matter power spectrum P_lin(k, z=0) from CLASS
  - The noise level sigma_noise added to the observed field

Usage:
    python scripts/wiener_filter.py \
        --target_path ./Quijote_target/Quijote_sample0.pt \
        --sigma_noise 0.1 \
        --num_samples 100

    python scripts/wiener_filter.py \
        --target_path ./Quijote_target/Quijote_sample0.pt \
        --sigma_noise 0.1 \
        --num_samples 100 \
        --run_name my_wf_run \
        --output_dir ./runs/wiener_filter \
        --MAS PCS \
        --noise_seed 42 \
        --use_latex
"""

import os

import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime

from gaussian_npe import utils, LinearWienerFilter


# ── Quijote fiducial cosmology (Planck 2018) ────────────────────────────────
BOX_PARAMS = {
    'box_size': 1000.,   # Mpc/h
    'grid_res': 128,
    'h': 0.6711,
}

COSMO_PARAMS = {
    'h': 0.6711,
    'Omega_b': 0.049,
    'Omega_cdm': 0.2685,
    'n_s': 0.9624,
    'non linear': 'halofit',
    'sigma8': 0.834,
}

Z_IC = 127  # Quijote initial redshift


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the analytical linear Wiener filter baseline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Target ───────────────────────────────────────────────────────────
    parser.add_argument(
        '--target_path', type=str, required=True,
        help='Path to the target .pt file (must contain delta_z0 and delta_z127 keys)',
    )

    # ── Filter parameters ────────────────────────────────────────────────
    parser.add_argument(
        '--sigma_noise', type=float, required=True,
        help='RMS noise level added to the observed field delta_z0',
    )
    parser.add_argument(
        '--rescaling_factor', type=float, default=None,
        help='D(z=127)/D(z=0). If not set, computed from fiducial cosmology (~0.0099)',
    )

    # ── Sampling ─────────────────────────────────────────────────────────
    parser.add_argument(
        '--num_samples', type=int, default=100,
        help='Number of posterior samples to draw',
    )
    parser.add_argument(
        '--noise_seed', type=int, default=42,
        help='Random seed for the observational noise added to the target field',
    )
    parser.add_argument(
        '--MAS', type=str, default=None,
        help='Mass assignment scheme for Pylians (e.g. PCS). None = no correction',
    )

    # ── Output ───────────────────────────────────────────────────────────
    parser.add_argument(
        '--run_name', type=str, default='',
        help='Name suffix for this run (appended to timestamp in output folder)',
    )
    parser.add_argument(
        '--output_dir', type=str, default='./runs/wiener_filter',
        help='Base output directory',
    )
    parser.add_argument(
        '--use_latex', action='store_true', default=False,
        help='Use LaTeX rendering and scienceplots style for all plots',
    )

    return parser.parse_args()


def main():
    args = parse_args()
    utils.configure_matplotlib_style(use_latex=args.use_latex)

    # ── Timestamp & output directory ─────────────────────────────────────
    timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
    suffix = f'LinearWF_{args.run_name}' if args.run_name else 'LinearWF'
    run_label = f'{timestamp}_{suffix}'
    output_dir = os.path.join(args.output_dir, run_label)
    os.makedirs(output_dir, exist_ok=True)

    print(f'Run: {run_label}')
    print(f'Output directory: {output_dir}')

    # ── Device ───────────────────────────────────────────────────────────
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # ── Box & cosmology ──────────────────────────────────────────────────
    box = utils.Power_Spectrum_Sampler(BOX_PARAMS, device=device)
    print(f'k_Nq = {box.k_Nq:.4f} h/Mpc, k_F = {box.k_F:.6f} h/Mpc')

    if args.rescaling_factor is None:
        rescaling_factor = (
            utils.growth_D_approx(COSMO_PARAMS, Z_IC)
            / utils.growth_D_approx(COSMO_PARAMS, 0)
        )
    else:
        rescaling_factor = args.rescaling_factor
    print(f'Rescaling factor: {rescaling_factor:.6f}')

    prior = box.get_prior_Q_factors(
        lambda k: torch.tensor(
            utils.get_pk_class(COSMO_PARAMS, 0, np.array(k)), device=device,
        )
    )

    # ── Build Wiener filter ──────────────────────────────────────────────
    wf = LinearWienerFilter(box, prior, args.sigma_noise, rescaling_factor)
    print(f'LinearWienerFilter: sigma_noise={args.sigma_noise}, '
          f'D_like={wf.D_like:.4f}')

    # ── Load target observation ──────────────────────────────────────────
    print(f'Loading target from {args.target_path}...')
    sample0 = torch.load(args.target_path, weights_only=False)
    delta_z0   = sample0['delta_z0'].astype('f')
    delta_z127 = sample0['delta_z127'].astype('f')

    # ── Add observational noise ──────────────────────────────────────────
    rng = np.random.default_rng(args.noise_seed)
    delta_obs = delta_z0 + rng.standard_normal(delta_z0.shape).astype('f') * args.sigma_noise
    print(f'Added observational noise (sigma={args.sigma_noise}, seed={args.noise_seed})')

    # ── Save config ──────────────────────────────────────────────────────
    config = {
        'timestamp': timestamp,
        'run_name': run_label,
        'target_path': args.target_path,
        'sigma_noise': args.sigma_noise,
        'rescaling_factor': rescaling_factor,
        'num_samples': args.num_samples,
        'noise_seed': args.noise_seed,
        'MAS': args.MAS,
        'device': device,
    }
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    print(f'Config:\n{json.dumps(config, indent=2)}')

    # ── MAP estimate & posterior samples ─────────────────────────────────
    print(f'Computing MAP estimate...')
    z_MAP = wf.get_z_MAP(torch.from_numpy(delta_obs).to(device).float())
    z_MAP_np = z_MAP.cpu().numpy()

    print(f'Drawing {args.num_samples} posterior samples...')
    samples = wf.sample(args.num_samples, z_MAP=z_MAP)
    samples_np = np.array(samples)

    # ── Q-matrix diagonals for diagnostic plots ──────────────────────────
    N = box.N
    Q_like_D  = np.full(N**3, wf.D_like)
    Q_prior_D = wf.D_prior.cpu().numpy()

    # ── Samples analysis plots ───────────────────────────────────────────
    print('Plotting samples analysis...')
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    utils.plot_samples_analysis(
        delta_z127=delta_z127 / rescaling_factor,
        delta_z0=delta_z0,
        samples=samples_np / rescaling_factor,
        z_MAP=z_MAP_np / rescaling_factor,
        box=box,
        cosmo_params=COSMO_PARAMS.copy(),
        MAS=args.MAS,
        Q_like_D=Q_like_D,
        Q_prior_D=Q_prior_D,
        save_dir=plots_dir,
        run_name=run_label,
        save_csv=True,
    )
    print(f'Plots saved to {plots_dir}')
    plt.close('all')

    # ── Calibration diagnostics ──────────────────────────────────────────
    print('Running calibration diagnostics...')
    utils.plot_calibration_diagnostics(
        delta_z127=delta_z127 / rescaling_factor,
        z_MAP=z_MAP_np / rescaling_factor,
        samples=samples_np / rescaling_factor,
        box=box,
        Q_like_D=Q_like_D,
        Q_prior_D=Q_prior_D,
        save_dir=plots_dir,
        run_name=run_label,
        save_csv=True,
    )
    print(f'Calibration diagnostics saved to {os.path.join(plots_dir, "calibration")}')
    plt.close('all')

    print(f'\nDone. All outputs saved to {output_dir}')


if __name__ == '__main__':
    main()
