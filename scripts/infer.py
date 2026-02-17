"""
Gaussian NPE Inference Script for Cosmological IC Reconstruction

Loads a trained Gaussian NPE model, generates posterior samples for a
given target observation, and produces diagnostic plots.

Usage:
    python scripts/infer.py --model_dir ./runs/20260216_153000_baseline

    python scripts/infer.py \
        --model_dir ./runs/20260216_153000_baseline \
        --run_name rerun_more_samples \
        --output_dir ./runs/infer_outputs \
        --target_path ./Quijote_target/Quijote_sample0.pt \
        --num_samples 200 \
        --MAS PCS \
        --noise_seed 42
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime

from gaussian_npe import utils, Gaussian_NPE_Network


# ── Quijote fiducial cosmology (Planck 2018) ────────────────────────────
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
        description='Run inference with a trained Gaussian NPE model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Model ────────────────────────────────────────────────────────────
    parser.add_argument(
        '--model_dir', type=str, required=True,
        help='Path to a training run directory (must contain model.pt and config.json)',
    )

    # ── Run identification ───────────────────────────────────────────────
    parser.add_argument(
        '--run_name', type=str, default='',
        help='Name for this inference run (appended to timestamp in output folder). '
             'If not set, reuses the training run name with an "infer" prefix.',
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Base output directory. If not set, outputs are saved '
             'inside model_dir/infer/{timestamp}_{run_name}/',
    )

    # ── Target & sampling ────────────────────────────────────────────────
    parser.add_argument(
        '--target_path', type=str, default=None,
        help='Path to the target observation .pt file '
             '(must contain delta_z0 and delta_z127 keys). '
             'If not set, uses the target_path from the training config.',
    )
    parser.add_argument(
        '--num_samples', type=int, default=100,
        help='Number of posterior samples to draw',
    )
    parser.add_argument(
        '--MAS', type=str, default=None,
        help='Mass assignment scheme for Pylians (e.g. PCS). None = no correction',
    )
    parser.add_argument(
        '--noise_seed', type=int, default=42,
        help='Random seed for the observational noise added to the target field',
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load training config ─────────────────────────────────────────────
    config_path = os.path.join(args.model_dir, 'config.json')
    with open(config_path, 'r') as f:
        train_config = json.load(f)
    print(f'Loaded training config from {config_path}')

    # Resolve target path: CLI flag > training config
    target_path = args.target_path or train_config.get('target_path')
    if target_path is None:
        raise ValueError(
            'No target_path specified and none found in training config. '
            'Pass --target_path explicitly.'
        )

    # ── Timestamp & output directory ─────────────────────────────────────
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.run_name:
        run_label = f"{timestamp}_{args.run_name}"
    else:
        train_run_name = train_config.get('run_name', '')
        suffix = f"infer_{train_run_name}" if train_run_name else 'infer'
        run_label = f"{timestamp}_{suffix}"

    if args.output_dir is not None:
        output_dir = os.path.join(args.output_dir, run_label)
    else:
        output_dir = os.path.join(args.model_dir, 'infer', run_label)
    os.makedirs(output_dir, exist_ok=True)

    print(f'Run: {run_label}')
    print(f'Output directory: {output_dir}')

    # ── Device ───────────────────────────────────────────────────────────
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # ── Box & cosmology ──────────────────────────────────────────────────
    box = utils.Power_Spectrum_Sampler(BOX_PARAMS, device=device)
    print(f'k_Nq = {box.k_Nq:.4f} h/Mpc, k_F = {box.k_F:.6f} h/Mpc')

    Dz_approx = (
        utils.growth_D_approx(COSMO_PARAMS, Z_IC)
        / utils.growth_D_approx(COSMO_PARAMS, 0)
    )
    rescaling_factor = train_config.get('rescaling_factor', Dz_approx)
    print(f'Rescaling factor: {rescaling_factor:.6f}')

    prior = box.get_prior_Q_factors(
        lambda k: torch.tensor(
            utils.get_pk_class(COSMO_PARAMS, 0, np.array(k)), device=device,
        )
    )

    # ── Reconstruct network & load weights ───────────────────────────────
    network = Gaussian_NPE_Network(
        box, prior,
        sigma_noise=train_config.get('sigma_noise', 0.0),
        rescaling_factor=rescaling_factor,
        k_cut=train_config.get('k_cut', 0.03),
        w_cut=train_config.get('w_cut', 0.001),
    )
    network.float().to(device)

    model_path = os.path.join(args.model_dir, 'model.pt')
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    network.load_state_dict(state_dict)
    network.eval()
    print(f'Model loaded from {model_path}')

    # ── Save inference config ────────────────────────────────────────────
    infer_config = {
        'timestamp': timestamp,
        'run_name': run_label,
        'model_dir': args.model_dir,
        'target_path': target_path,
        'num_samples': args.num_samples,
        'noise_seed': args.noise_seed,
        'sigma_noise': sigma_noise,
        'MAS': args.MAS,
        'device': device,
        'rescaling_factor': rescaling_factor,
        'train_config': train_config,
    }
    with open(os.path.join(output_dir, 'infer_config.json'), 'w') as f:
        json.dump(infer_config, f, indent=2)

    # ── Load target observation ──────────────────────────────────────────
    print(f'Loading target from {target_path}...')
    sample0 = torch.load(target_path, weights_only=False)
    delta_z0 = sample0['delta_z0'].astype('f')
    delta_z127 = sample0['delta_z127'].astype('f')

    sigma_noise = train_config.get('sigma_noise', 0.0)
    rng = np.random.default_rng(args.noise_seed)
    delta_obs = delta_z0 + rng.standard_normal(delta_z0.shape).astype('f') * sigma_noise
    print(f'Added observational noise (sigma={sigma_noise}, seed={args.noise_seed})')

    # ── Generate MAP & posterior samples ──────────────────────────────────
    print(f'Generating {args.num_samples} posterior samples...')
    with torch.no_grad():
        z_MAP = network.get_z_MAP(torch.from_numpy(delta_obs).to(device).float())
        samples = network.sample(args.num_samples, z_MAP=z_MAP)

    z_MAP_np = z_MAP.cpu().numpy()

    # ── Plot ─────────────────────────────────────────────────────────────
    print('Plotting analysis...')
    plots_dir = os.path.join(output_dir, 'plots')
    utils.plot_samples_analysis(
        delta_z127=delta_z127 / rescaling_factor,
        delta_z0=delta_z0,
        samples=np.array(samples) / rescaling_factor,
        z_MAP=z_MAP_np / rescaling_factor,
        box=box,
        cosmo_params=COSMO_PARAMS.copy(),
        MAS=args.MAS,
        Q_like_D=network.Q_like.D.detach().cpu().numpy(),
        Q_prior_D=network.Q_prior.D.detach().cpu().numpy(),
        save_dir=plots_dir,
        run_name=run_label,
    )
    print(f'Plots saved to {os.path.join(plots_dir, run_label)}')
    plt.close('all')

    print(f'\nInference complete. All outputs saved to {output_dir}')


if __name__ == '__main__':
    main()
