"""
MAP Estimator + External Q-Matrix Posterior Sampling

Loads a trained MAP_MSE_Network to compute a MAP estimate from an observation,
then loads a separately trained Gaussian_NPE_IsotropicD to borrow its learned
Q matrices (Q_like, Q_prior) for posterior sampling.  This decouples the point
estimator from the posterior covariance model: anyone who has trained a MAP
estimator can plug in a known IsotropicD precision matrix to obtain full
posterior samples without retraining the complete pipeline.

Usage:
    python scripts/map_then_sample.py \\
        --map_dir runs/260330_163349_map_mse_test \\
        --posterior_dir paper_test_runs/runs/260303_224547_sweep_networks/260303_224627_net_IsotropicD \\
        --num_samples 100

    python scripts/map_then_sample.py \\
        --map_dir runs/260330_163349_map_mse_test \\
        --posterior_dir paper_test_runs/runs/260303_224547_sweep_networks/260303_224627_net_IsotropicD \\
        --target_path /home/osavchenko/Quijote/Quijote_target/Quijote_sample0_wout_MAK.pt \\
        --num_samples 100 \\
        --use_latex \\
        --paper_plot
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime

from gaussian_npe import (
    utils,
    MAP_MSE_Network,
    Gaussian_NPE_IsotropicD,
)

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

Z_IC = 127  # Quijote initial redshift


def parse_args():
    parser = argparse.ArgumentParser(
        description='MAP estimator + external IsotropicD Q-matrix posterior sampling',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Model directories ─────────────────────────────────────────────────
    parser.add_argument(
        '--map_dir', type=str, required=True,
        help='Path to a MAP_MSE_Network run directory (must contain model.pt and config.json)',
    )
    parser.add_argument(
        '--posterior_dir', type=str, required=True,
        help='Path to a Gaussian_NPE_IsotropicD run directory whose Q matrices are used for sampling',
    )

    # ── Run identification ────────────────────────────────────────────────
    parser.add_argument(
        '--run_name', type=str, default='map_then_sample',
        help='Label appended to timestamp in output folder',
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Base output directory. If not set, saves to map_dir/infer/{timestamp}_{run_name}/',
    )

    # ── Target & sampling ─────────────────────────────────────────────────
    parser.add_argument(
        '--target_path', type=str, default=None,
        help='Path to the target .pt file (must contain delta_z0 and delta_z127). '
             'If not set, uses target_path from the MAP training config.',
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
        help='Random seed for observational noise added to the target field',
    )

    # ── Plot options ──────────────────────────────────────────────────────
    parser.add_argument(
        '--use_latex', action='store_true', default=False,
        help='Use LaTeX rendering and scienceplots style for all plots',
    )
    parser.add_argument(
        '--paper_plot', action='store_true', default=False,
        help='Additionally produce the paper-quality P(k)/T(k)/C(k) PDF (fig2_2pt style)',
    )

    return parser.parse_args()


def main():
    args = parse_args()
    utils.configure_matplotlib_style(use_latex=args.use_latex)

    # ── Load MAP training config ──────────────────────────────────────────
    map_config_path = os.path.join(args.map_dir, 'config.json')
    with open(map_config_path, 'r') as f:
        map_config = json.load(f)
    print(f'Loaded MAP config from {map_config_path}')

    # ── Load IsotropicD (posterior) training config ───────────────────────
    iso_config_path = os.path.join(args.posterior_dir, 'config.json')
    with open(iso_config_path, 'r') as f:
        iso_config = json.load(f)
    print(f'Loaded posterior config from {iso_config_path}')

    # Resolve target path: CLI > MAP config
    target_path = args.target_path or map_config.get('target_path')
    if target_path is None:
        raise ValueError(
            'No target_path specified and none found in MAP training config. '
            'Pass --target_path explicitly.'
        )

    # ── Timestamp & output directory ──────────────────────────────────────
    timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
    run_label = f"{timestamp}_{args.run_name}"

    if args.output_dir is not None:
        output_dir = os.path.join(args.output_dir, run_label)
    else:
        output_dir = os.path.join(args.map_dir, 'infer', run_label)
    os.makedirs(output_dir, exist_ok=True)

    print(f'Run: {run_label}')
    print(f'Output directory: {output_dir}')

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
    rescaling_factor = map_config.get('rescaling_factor', Dz_approx)
    print(f'Rescaling factor (from MAP config): {rescaling_factor:.6f}')

    prior = box.get_prior_Q_factors(
        lambda k: torch.tensor(
            utils.get_pk_class(COSMO_PARAMS, 0, k.detach().cpu().numpy()), device=device,
        )
    )

    # ── Load MAP_MSE_Network ──────────────────────────────────────────────
    sigma_noise = map_config.get('sigma_noise', 0.0)
    map_net = MAP_MSE_Network(
        box,
        sigma_noise=sigma_noise,
        rescaling_factor=rescaling_factor,
    )
    map_net.float().to(device)
    map_model_path = os.path.join(args.map_dir, 'model.pt')
    map_net.load_state_dict(torch.load(map_model_path, map_location=device, weights_only=False))
    map_net.eval()
    print(f'MAP_MSE_Network loaded from {map_model_path}')

    # ── Load Gaussian_NPE_IsotropicD ──────────────────────────────────────
    iso_net = Gaussian_NPE_IsotropicD(
        box, prior,
        sigma_noise=iso_config.get('sigma_noise', 0.0),
        rescaling_factor=rescaling_factor,
    )
    iso_net.float().to(device)
    iso_model_path = os.path.join(args.posterior_dir, 'model.pt')
    iso_net.load_state_dict(torch.load(iso_model_path, map_location=device, weights_only=False))
    iso_net.eval()
    print(f'Gaussian_NPE_IsotropicD loaded from {iso_model_path}')

    # ── Load target observation ───────────────────────────────────────────
    print(f'Loading target from {target_path}...')
    sample0 = torch.load(target_path, weights_only=False)
    delta_z0   = sample0['delta_z0'].astype('f')
    delta_z127 = sample0['delta_z127'].astype('f')
    delta_z0   -= delta_z0.mean()
    delta_z127 -= delta_z127.mean()

    rng = np.random.default_rng(args.noise_seed)
    delta_obs = delta_z0 + rng.standard_normal(delta_z0.shape).astype('f') * sigma_noise
    print(f'Added observational noise (sigma={sigma_noise}, seed={args.noise_seed})')

    # ── Save inference config ─────────────────────────────────────────────
    infer_config = {
        'timestamp': timestamp,
        'run_name': run_label,
        'map_dir': args.map_dir,
        'posterior_dir': args.posterior_dir,
        'target_path': target_path,
        'num_samples': args.num_samples,
        'noise_seed': args.noise_seed,
        'sigma_noise': sigma_noise,
        'MAS': args.MAS,
        'device': device,
        'rescaling_factor': rescaling_factor,
        'map_config': map_config,
        'iso_config': iso_config,
    }
    with open(os.path.join(output_dir, 'infer_config.json'), 'w') as f:
        json.dump(infer_config, f, indent=2)

    # ── Get MAP estimate from MAP_MSE_Network ─────────────────────────────
    print('Computing MAP estimate...')
    with torch.no_grad():
        z_MAP = map_net.get_z_MAP(
            torch.from_numpy(delta_obs).to(device).float()
        )  # physical units, CPU tensor

    # ── Draw posterior samples using IsotropicD Q matrices ────────────────
    print(f'Drawing {args.num_samples} posterior samples using IsotropicD Q matrices...')
    with torch.no_grad():
        samples = iso_net.sample(args.num_samples, z_MAP=z_MAP.to(device))
    print(f'{args.num_samples} samples ready.')

    z_MAP_np   = z_MAP.cpu().numpy()                       # physical
    samples_np = np.array(samples) / rescaling_factor      # physical → internal

    # ── Extract Q-matrix diagonals from IsotropicD ────────────────────────
    with torch.no_grad():
        Q_like_D  = iso_net.Q_like.D.detach().cpu().numpy()
        Q_prior_D = iso_net.Q_prior.D.detach().cpu().numpy()
    Q_like_k_nodes = iso_net.Q_like._log_k_nodes.exp().detach().cpu().numpy()
    Q_like_D_nodes = iso_net.Q_like.log_D_nodes.exp().detach().cpu().numpy()

    # ── Plot ──────────────────────────────────────────────────────────────
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    if args.paper_plot:
        print('Generating paper-quality summary stats PDF...')
        _scripts_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.join(_scripts_dir, '..', 'paper_plots_scripts'))
        from fig2_2pt import plot_summary_stats
        # plot_summary_stats expects physical units
        plot_summary_stats(
            delta_z127=delta_z127,
            delta_z0=delta_z0,
            samples=np.array(samples),          # physical
            z_MAP=z_MAP_np,                     # physical
            box=box,
            cosmo_params=COSMO_PARAMS.copy(),
            MAS=args.MAS,
            rescaling_factor=rescaling_factor,
            save_dir=plots_dir,
            run_name=run_label,
        )
        plt.close('all')
    else:
        print('Plotting analysis...')
        utils.plot_samples_analysis(
            delta_z127=delta_z127 / rescaling_factor,
            delta_z0=delta_z0,
            samples=samples_np,
            z_MAP=z_MAP_np / rescaling_factor,
            box=box,
            cosmo_params=COSMO_PARAMS.copy(),
            MAS=args.MAS,
            Q_like_D=Q_like_D,
            Q_prior_D=Q_prior_D,
            Q_like_k_nodes=Q_like_k_nodes,
            Q_like_D_nodes=Q_like_D_nodes,
            save_dir=plots_dir,
            run_name=run_label,
            save_csv=True,
        )
        print(f'Plots saved to {plots_dir}')
        plt.close('all')

        print('Running calibration diagnostics...')
        utils.plot_calibration_diagnostics(
            delta_z127=delta_z127 / rescaling_factor,
            z_MAP=z_MAP_np / rescaling_factor,
            samples=samples_np,
            box=box,
            Q_like_D=Q_like_D,
            Q_prior_D=Q_prior_D,
            save_dir=plots_dir,
            run_name=run_label,
            save_csv=True,
            rescaling_factor=rescaling_factor,
        )
        print(f'Calibration diagnostics saved to {os.path.join(plots_dir, "calibration")}')
        plt.close('all')

    print(f'\nDone. All outputs saved to {output_dir}')


if __name__ == '__main__':
    main()
