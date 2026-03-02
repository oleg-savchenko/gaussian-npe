"""
Generate diagnostic plots from pre-computed posterior samples.

Loads posterior samples saved by save_samples.py, reconstructs the network
(to obtain precision matrix diagonals), and produces:
  - plot_samples_analysis  : P(k), T(k), C(k), field slices, Q diagonals
  - plot_calibration_diagnostics : χ², log-probability, per-mode scatter

Plots are saved to paper_plots_scripts/{RUN_NAME}/.
CSVs with scalar diagnostics are saved to the respective diagnostics/ subfolders.

Usage:
    python paper_plots_scripts/plot_from_samples.py \
        --samples_dir /gpfs/scratch1/shared/osavchenko/mnras_paper/samples/20260302_154321_WienerIsotropicD

    python paper_plots_scripts/plot_from_samples.py \
        --samples_dir /gpfs/scratch1/shared/osavchenko/mnras_paper/samples/20260302_154321_WienerIsotropicD \
        --use_latex
"""

import os
import json
import argparse

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
        description='Generate diagnostic plots from pre-computed posterior samples',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--samples_dir', type=str, required=True,
        help='Path to samples folder produced by save_samples.py '
             '(must contain sample_*.npy, z_MAP.npy, config.json)',
    )
    parser.add_argument(
        '--MAS', type=str, default=None,
        help='Mass assignment scheme for Pylians (e.g. PCS). '
             'Overrides the value stored in config.json.',
    )
    parser.add_argument(
        '--use_latex', action='store_true', default=False,
        help='Use LaTeX rendering and scienceplots style for all plots',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    utils.configure_matplotlib_style(use_latex=args.use_latex)

    # ── Load saved run config ─────────────────────────────────────────────
    config_path = os.path.join(args.samples_dir, 'config.json')
    with open(config_path, 'r') as f:
        run_config = json.load(f)
    train_config = run_config['train_config']
    print(f'Loaded config from {config_path}')

    run_name      = run_config['run_name']
    target_path   = run_config['target_path']
    num_samples   = run_config['num_samples']
    rescaling_factor = run_config['rescaling_factor']
    MAS = args.MAS if args.MAS is not None else train_config.get('MAS')
    print(f'Run: {run_name}')
    print(f'Rescaling factor: {rescaling_factor:.6f}')

    # ── Device ────────────────────────────────────────────────────────────
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # ── Box & cosmology ───────────────────────────────────────────────────
    box = utils.Power_Spectrum_Sampler(BOX_PARAMS, device=device)

    prior = box.get_prior_Q_factors(
        lambda k: torch.tensor(
            utils.get_pk_class(COSMO_PARAMS, 0, k.detach().cpu().numpy()), device=device,
        )
    )

    # ── Reconstruct network to get precision matrix diagonals ─────────────
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
                             'IsotropicD', 'WienerIsotropicD', 'Poisson'):
        net_kwargs['k_cut'] = train_config.get('k_cut', 0.03)
        net_kwargs['w_cut'] = train_config.get('w_cut', 0.001)

    if network_name == 'Poisson':
        network = NetworkClass(box, prior,
                               n_bar=n_bar, galaxy_bias=galaxy_bias,
                               rescaling_factor=rescaling_factor)
    else:
        network = NetworkClass(box, prior, **net_kwargs)
    network.float().to(device)

    model_path = os.path.join(run_config['model_dir'], 'model.pt')
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    network.load_state_dict(state_dict)
    network.eval()
    print(f'Model weights loaded from {model_path}')

    # ── Extract precision matrix diagonals ────────────────────────────────
    # Networks with a Q_prior/Q_like split expose both attributes.
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

    # ── Load target fields ────────────────────────────────────────────────
    print(f'Loading target from {target_path}...')
    sample0    = torch.load(target_path, weights_only=False)
    delta_z0   = sample0['delta_z0'].astype('f')
    delta_z127 = sample0['delta_z127'].astype('f')

    # ── Load pre-computed samples and z_MAP ───────────────────────────────
    print(f'Loading {num_samples} samples from {args.samples_dir}...')
    samples = np.array([
        np.load(os.path.join(args.samples_dir, f'sample_{i:04d}.npy'))
        for i in range(num_samples)
    ])   # (num_samples, N, N, N), physical delta_z127 units
    z_MAP_np = np.load(os.path.join(args.samples_dir, 'z_MAP.npy'))
    print(f'Samples shape: {samples.shape}')

    # Rescale to internal space (plotting functions expect internal units)
    samples_int  = samples  / rescaling_factor   # (num_samples, N, N, N)
    z_MAP_int    = z_MAP_np / rescaling_factor   # (N, N, N)
    delta_z127_int = delta_z127 / rescaling_factor  # (N, N, N)

    # ── Output directory ──────────────────────────────────────────────────
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(scripts_dir, run_name)
    os.makedirs(plots_dir, exist_ok=True)
    print(f'Saving plots to: {plots_dir}/')

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
    )

    print(f'\nDone. Plots saved to {plots_dir}/')


if __name__ == '__main__':
    main()
