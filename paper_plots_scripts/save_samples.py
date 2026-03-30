"""
Save posterior samples from a trained Gaussian NPE model to disk.

Loads a pre-trained model, applies observational noise to the target field,
draws posterior samples, and saves each as an individual .npy file for
downstream analysis and paper figure generation.

Output layout:
    {output_dir}/{RUN_NAME}/
        sample_0000.npy   # shape (128, 128, 128), float32, physical delta_z127
        sample_0001.npy
        ...
        z_MAP.npy         # MAP estimate, same shape and units
        config.json       # run configuration for reproducibility

RUN_NAME is derived from the model directory basename, e.g.
    20260302_154321_WienerIsotropicD

Usage:
    python paper_plots_scripts/save_samples.py \
        --model_dir ./runs/20260302_154321_WienerIsotropicD

    python paper_plots_scripts/save_samples.py \
        --model_dir ./runs/20260302_154321_WienerIsotropicD \
        --target_path ./Quijote_target/Quijote_sample0_wout_MAK.pt \
        --num_samples 1000 \
        --noise_seed 42 \
        --output_dir /gpfs/scratch1/shared/osavchenko/mnras_paper/samples
"""

import os
import json
import argparse

import numpy as np
import torch
from datetime import datetime

from gaussian_npe import (
    utils,
    Gaussian_NPE_Network,
    Gaussian_NPE_UNet_Only,
    Gaussian_NPE_WienerNet,
    Gaussian_NPE_LearnableFilter,
    Gaussian_NPE_SmoothFilter,
    Gaussian_NPE_Iterative,
    Gaussian_NPE_LH,
    Gaussian_NPE_CustomUNet,
    Gaussian_NPE_IsotropicD,
    Gaussian_NPE_WienerIsotropicD,
    Gaussian_NPE_Default_IsotropicD,
    Gaussian_NPE_Poisson,
)

NETWORK_CLASSES = {
    'default': Gaussian_NPE_Network,
    'UNet_Only': Gaussian_NPE_UNet_Only,
    'WienerNet': Gaussian_NPE_WienerNet,
    'LearnableFilter': Gaussian_NPE_LearnableFilter,
    'SmoothFilter': Gaussian_NPE_SmoothFilter,
    'Iterative': Gaussian_NPE_Iterative,
    'LH': Gaussian_NPE_LH,
    'CustomUNet': Gaussian_NPE_CustomUNet,
    'IsotropicD': Gaussian_NPE_IsotropicD,
    'WienerIsotropicD': Gaussian_NPE_WienerIsotropicD,
    'default_IsotropicD': Gaussian_NPE_Default_IsotropicD,
    'Poisson': Gaussian_NPE_Poisson,
}

# ── Quijote fiducial cosmology (Planck 2018) ─────────────────────────────────
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

Z_IC = 127


def parse_args():
    parser = argparse.ArgumentParser(
        description='Save posterior samples from a trained Gaussian NPE model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--model_dir', type=str, required=True,
        help='Path to trained run folder (must contain config.json and model.pt)',
    )
    parser.add_argument(
        '--target_path', type=str, default=None,
        help='Path to target observation .pt file. '
             'If not set, falls back to the path stored in the training config.',
    )
    parser.add_argument(
        '--num_samples', type=int, default=1000,
        help='Number of posterior samples to draw and save',
    )
    parser.add_argument(
        '--output_dir', type=str,
        default='/gpfs/scratch1/shared/osavchenko/mnras_paper/samples',
        help='Base output directory; samples are saved to {output_dir}/{RUN_NAME}/',
    )
    parser.add_argument(
        '--noise_seed', type=int, default=42,
        help='RNG seed for observational noise applied to the target field',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load training config ──────────────────────────────────────────────
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

    # ── Output directory ──────────────────────────────────────────────────
    run_name = os.path.basename(os.path.abspath(args.model_dir))
    out_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f'Saving {args.num_samples} samples to: {out_dir}/')

    # ── Device ────────────────────────────────────────────────────────────
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # ── Box & cosmology ───────────────────────────────────────────────────
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
            utils.get_pk_class(COSMO_PARAMS, 0, k.detach().cpu().numpy()), device=device,
        )
    )

    # ── Reconstruct network & load weights ────────────────────────────────
    network_name = train_config.get('network', 'default')
    NetworkClass = NETWORK_CLASSES[network_name]
    print(f'Network: {NetworkClass.__name__}')
    n_bar       = train_config.get('n_bar', 5e-4)
    galaxy_bias = train_config.get('galaxy_bias', 1.5)

    net_kwargs = dict(
        sigma_noise=train_config.get('sigma_noise', 0.0),
        rescaling_factor=rescaling_factor,
    )
    # Only pass k_cut/w_cut to networks that accept them
    if network_name not in ('UNet_Only', 'WienerNet', 'Iterative', 'CustomUNet',
                             'IsotropicD', 'WienerIsotropicD', 'Poisson', 'LH'):
        net_kwargs['k_cut'] = train_config.get('k_cut', 0.03)
        net_kwargs['w_cut'] = train_config.get('w_cut', 0.003)

    if network_name == 'Poisson':
        network = NetworkClass(box, prior,
                               n_bar=n_bar, galaxy_bias=galaxy_bias,
                               rescaling_factor=rescaling_factor)
    else:
        network = NetworkClass(box, prior, **net_kwargs)
    network.float().to(device)

    model_path = os.path.join(args.model_dir, 'model.pt')
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    network.load_state_dict(state_dict)
    network.eval()
    print(f'Model loaded from {model_path}')

    # ── Load target observation ───────────────────────────────────────────
    print(f'Loading target from {target_path}...')
    sample0 = torch.load(target_path, weights_only=False)
    delta_z0 = sample0['delta_z0'].astype('f')

    sigma_noise = train_config.get('sigma_noise', 0.0)
    rng = np.random.default_rng(args.noise_seed)
    if network_name == 'Poisson':
        V_voxel = (box.box_size / box.N) ** 3
        N_bar   = n_bar * V_voxel
        N_mean  = (N_bar * (1 + galaxy_bias * delta_z0)).clip(min=0)
        delta_obs = (rng.poisson(N_mean).astype('f') / (N_bar * galaxy_bias)
                     - 1.0 / galaxy_bias)
        print(f'Applied Poisson noise (n_bar={n_bar}, galaxy_bias={galaxy_bias}, seed={args.noise_seed})')
    else:
        delta_obs = delta_z0 + rng.standard_normal(delta_z0.shape).astype('f') * sigma_noise
        print(f'Added Gaussian noise (sigma={sigma_noise}, seed={args.noise_seed})')

    # ── MAP estimate ──────────────────────────────────────────────────────
    print('Computing MAP estimate...')
    with torch.no_grad():
        z_MAP = network.get_z_MAP(torch.from_numpy(delta_obs).to(device).float())
    z_MAP_np = z_MAP.cpu().numpy()
    np.save(os.path.join(out_dir, 'z_MAP.npy'), z_MAP_np)
    print(f'z_MAP saved.')

    # ── Draw & save posterior samples ─────────────────────────────────────
    print(f'Drawing {args.num_samples} posterior samples...')
    with torch.no_grad():
        for i in range(args.num_samples):
            sample = network.sample(1, z_MAP=z_MAP)[0]   # (128, 128, 128) float32
            np.save(os.path.join(out_dir, f'sample_{i:04d}.npy'), sample)
            if (i + 1) % 100 == 0:
                print(f'  {i + 1}/{args.num_samples} samples saved')

    # ── Save run config ───────────────────────────────────────────────────
    run_config = {
        'timestamp': datetime.now().strftime('%y%m%d_%H%M%S'),
        'run_name': run_name,
        'model_dir': os.path.abspath(args.model_dir),
        'target_path': target_path,
        'num_samples': args.num_samples,
        'noise_seed': args.noise_seed,
        'sigma_noise': sigma_noise,
        'device': device,
        'rescaling_factor': rescaling_factor,
        'train_config': train_config,
    }
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(run_config, f, indent=2)

    print(f'\nDone. {args.num_samples} samples + z_MAP saved to {out_dir}/')


if __name__ == '__main__':
    main()
