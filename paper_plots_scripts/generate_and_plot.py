"""
Generate posterior samples and immediately produce diagnostic plots.

Loads a trained model, draws posterior samples, saves each to disk, then
plots using the in-memory samples (no disk reload). This avoids the I/O
cost of reading large sample arrays back from disk.

Samples are saved to:
    {output_dir}/{RUN_NAME}/sample_0000.npy  ...  sample_{N-1:04d}.npy
    {output_dir}/{RUN_NAME}/z_MAP.npy
    {output_dir}/{RUN_NAME}/config.json

Plots are saved to:
    paper_plots_scripts/{RUN_NAME}/

Usage:
    python paper_plots_scripts/generate_and_plot.py \
        --model_dir paper_test_runs/runs/20260301_215801_net_IsotropicD

    python paper_plots_scripts/generate_and_plot.py \
        --model_dir paper_test_runs/runs/20260301_215801_net_IsotropicD \
        --num_samples 1000 \
        --noise_seed 42 \
        --output_dir /gpfs/scratch1/shared/osavchenko/mnras_paper/samples \
        --use_latex
"""

import os
import json
import argparse
from datetime import datetime

import numpy as np
import torch

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
        description='Generate posterior samples and produce diagnostic plots',
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
        help='Number of posterior samples to draw',
    )
    parser.add_argument(
        '--output_dir', type=str,
        default='/gpfs/scratch1/shared/osavchenko/mnras_paper/samples',
        help='Base directory for saving sample .npy files',
    )
    parser.add_argument(
        '--noise_seed', type=int, default=42,
        help='RNG seed for observational noise',
    )
    parser.add_argument(
        '--MAS', type=str, default=None,
        help='Mass assignment scheme for Pylians (e.g. PCS). '
             'Overrides the value stored in the training config.',
    )
    parser.add_argument(
        '--use_latex', action='store_true', default=False,
        help='Use LaTeX rendering and scienceplots style for all plots',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    utils.configure_matplotlib_style(use_latex=args.use_latex)

    # ── Load training config ──────────────────────────────────────────────
    config_path = os.path.join(args.model_dir, 'config.json')
    with open(config_path, 'r') as f:
        train_config = json.load(f)
    print(f'Loaded training config from {config_path}')

    target_path = args.target_path or train_config.get('target_path')
    if target_path is None:
        raise ValueError(
            'No target_path specified and none found in training config. '
            'Pass --target_path explicitly.'
        )

    run_name = os.path.basename(os.path.abspath(args.model_dir))
    MAS = args.MAS if args.MAS is not None else train_config.get('MAS')
    print(f'Run: {run_name}')

    # ── Output directories ────────────────────────────────────────────────
    samples_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(samples_dir, exist_ok=True)
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir   = os.path.join(scripts_dir, run_name)
    os.makedirs(plots_dir, exist_ok=True)
    print(f'Samples → {samples_dir}/')
    print(f'Plots   → {plots_dir}/')

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

    # ── Network ───────────────────────────────────────────────────────────
    network_name = train_config.get('network', 'default')
    NetworkClass = NETWORK_CLASSES[network_name]
    print(f'Network: {NetworkClass.__name__}')
    n_bar       = train_config.get('n_bar', 5e-4)
    galaxy_bias = train_config.get('galaxy_bias', 1.5)

    net_kwargs = dict(
        sigma_noise=train_config.get('sigma_noise', 0.0),
        rescaling_factor=rescaling_factor,
    )
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

    # ── Precision matrix diagonals (for plotting) ─────────────────────────
    # Networks with a Q_prior/Q_like split (all except LH) expose both attributes.
    # LH deletes them and exposes only Q_post (a single learned precision matrix).
    with torch.no_grad():
        if hasattr(network, 'Q_like') and hasattr(network, 'Q_prior'):
            Q_like_D  = network.Q_like.D.detach().cpu().numpy()
            Q_prior_D = network.Q_prior.D.detach().cpu().numpy()
        else:
            # Single Q_post (e.g. LH): treat Q_post as D_like, D_prior = 0
            Q_like_D  = network.Q_post.D.detach().cpu().numpy()
            Q_prior_D = np.zeros_like(Q_like_D)

    Q_like_obj = getattr(network, 'Q_like', None) or network.Q_post
    Q_like_k_nodes = (
        Q_like_obj._log_k_nodes.exp().detach().cpu().numpy()
        if hasattr(Q_like_obj, '_log_k_nodes') else None
    )
    Q_like_D_nodes = (
        Q_like_obj.log_D_nodes.exp().detach().cpu().numpy()
        if hasattr(Q_like_obj, 'log_D_nodes') else None
    )

    # ── Target observation ────────────────────────────────────────────────
    print(f'Loading target from {target_path}...')
    sample0    = torch.load(target_path, weights_only=False)
    delta_z0   = sample0['delta_z0'].astype('f')
    delta_z127 = sample0['delta_z127'].astype('f')

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
    np.save(os.path.join(samples_dir, 'z_MAP.npy'), z_MAP_np)
    print('z_MAP saved.')

    # ── Generate & save samples (keep in memory for plotting) ─────────────
    print(f'Drawing {args.num_samples} posterior samples...')
    samples_list = []
    with torch.no_grad():
        for i in range(args.num_samples):
            s = network.sample(1, z_MAP=z_MAP)[0]   # (N, N, N) float32 numpy
            np.save(os.path.join(samples_dir, f'sample_{i:04d}.npy'), s)
            samples_list.append(s)
            if (i + 1) % 100 == 0:
                print(f'  {i + 1}/{args.num_samples} samples saved')

    # Save run config
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
    with open(os.path.join(samples_dir, 'config.json'), 'w') as f:
        json.dump(run_config, f, indent=2)
    print(f'Samples saved to {samples_dir}/')

    # ── Stack in-memory samples (no disk reload) ──────────────────────────
    print('Stacking samples for plotting...')
    samples_arr = np.array(samples_list)   # (num_samples, N, N, N)
    del samples_list                        # free the list references

    # Rescale to internal space (plotting functions expect internal units)
    samples_int    = samples_arr / rescaling_factor
    z_MAP_int      = z_MAP_np   / rescaling_factor
    delta_z127_int = delta_z127 / rescaling_factor

    # ── Plot samples analysis ─────────────────────────────────────────────
    print('Running plot_samples_analysis...')
    utils.plot_samples_analysis(
        delta_z127=delta_z127_int,
        delta_z0=delta_z0,
        samples=samples_int,
        z_MAP=z_MAP_int,
        box=box,
        cosmo_params=COSMO_PARAMS.copy(),
        MAS=MAS,
        Q_like_D=Q_like_D,
        Q_prior_D=Q_prior_D,
        Q_like_k_nodes=Q_like_k_nodes,
        Q_like_D_nodes=Q_like_D_nodes,
        save_dir=plots_dir,
        run_name=run_name,
        save_csv=True,
    )

    # ── Plot calibration diagnostics ──────────────────────────────────────
    print('Running plot_calibration_diagnostics...')
    utils.plot_calibration_diagnostics(
        delta_z127=delta_z127_int,
        z_MAP=z_MAP_int,
        samples=samples_int,
        box=box,
        Q_like_D=Q_like_D,
        Q_prior_D=Q_prior_D,
        save_dir=plots_dir,
        run_name=run_name,
        save_csv=True,
        rescaling_factor=rescaling_factor,
    )

    print(f'\nDone. Plots saved to {plots_dir}/')


if __name__ == '__main__':
    main()
