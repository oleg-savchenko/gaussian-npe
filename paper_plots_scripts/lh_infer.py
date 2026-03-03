"""
Inference and diagnostic plots for a trained Gaussian_NPE_LH model.

Loads a trained LH run, reads one simulation from the LH ZarrStore (already
PCS-deconvolved and zero-mean), overrides the network's rescaling_factor with
the per-simulation value, generates posterior samples, and produces the standard
samples-analysis and calibration-diagnostics plots.

Samples are saved to:
    {output_dir}/{RUN_NAME}/sample_0000.npy  ...  sample_{N-1:04d}.npy
    {output_dir}/{RUN_NAME}/z_MAP.npy
    {output_dir}/{RUN_NAME}/config.json

Plots are saved to:
    paper_plots_scripts/{RUN_NAME}/

Usage:
    python paper_plots_scripts/lh_infer.py \
        --model_dir runs/260303_120000_LH_sigma_noise_1

    python paper_plots_scripts/lh_infer.py \
        --model_dir runs/260303_120000_LH_sigma_noise_1 \
        --store_path /gpfs/scratch1/shared/osavchenko/zarr_stores/Quijote_LH_res128_deconv_MAK \
        --index 5 \
        --num_samples 500 \
        --noise_seed 42 \
        --output_dir /gpfs/scratch1/shared/osavchenko/mnras_paper/samples \
        --use_latex
"""

import os
import json
import argparse
from datetime import datetime

import numpy as np
import swyft
import torch

from gaussian_npe import utils, Gaussian_NPE_LH

# ── Quijote fiducial box parameters ──────────────────────────────────────────
BOX_PARAMS = {
    'box_size': 1000.,   # Mpc/h
    'grid_res': 128,
    'h': 0.6711,
}

Z_IC = 127


def parse_args():
    parser = argparse.ArgumentParser(
        description='LH inference: load ZarrStore simulation, generate samples, plot diagnostics',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--model_dir', type=str, required=True,
        help='Path to trained LH run folder (must contain config.json and model.pt)',
    )
    parser.add_argument(
        '--store_path', type=str,
        default='/gpfs/scratch1/shared/osavchenko/zarr_stores/Quijote_LH_res128_deconv_MAK',
        help='Path to the LH ZarrStore (deconvolved + zero-mean fields)',
    )
    parser.add_argument(
        '--index', type=int, default=1,
        help='Simulation index to use as target observation',
    )
    parser.add_argument(
        '--num_samples', type=int, default=100,
        help='Number of posterior samples to draw',
    )
    parser.add_argument(
        '--output_dir', type=str,
        default='/gpfs/scratch1/shared/osavchenko/mnras_paper/samples',
        help='Base directory for saving sample .npy files',
    )
    parser.add_argument(
        '--noise_seed', type=int, default=42,
        help='RNG seed for Gaussian observational noise',
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

    run_name = os.path.basename(os.path.abspath(args.model_dir))
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

    # ── Box ───────────────────────────────────────────────────────────────
    box = utils.Power_Spectrum_Sampler(BOX_PARAMS, device=device)
    print(f'k_Nq = {box.k_Nq:.4f} h/Mpc, k_F = {box.k_F:.6f} h/Mpc')

    # ── Network ───────────────────────────────────────────────────────────
    network_name = train_config.get('network', 'LH')
    if network_name != 'LH':
        raise ValueError(
            f"lh_infer.py only supports the LH network, but model_dir contains "
            f"network='{network_name}'. Use generate_and_plot.py for other networks."
        )
    print(f'Network: Gaussian_NPE_LH')

    # No prior, no rescaling_factor — both are irrelevant for LH:
    # Q_prior/Q_like are deleted in __init__, and rescaling_factor is
    # overridden per-simulation before any inference call.
    network = Gaussian_NPE_LH(
        box,
        sigma_noise=train_config.get('sigma_noise', 0.0),
    )
    network.float().to(device)

    model_path = os.path.join(args.model_dir, 'model.pt')
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    network.load_state_dict(state_dict)
    network.eval()
    print(f'Model loaded from {model_path}')

    # ── Precision matrix diagonals (for plotting) ─────────────────────────
    with torch.no_grad():
        if hasattr(network, 'Q_like') and hasattr(network, 'Q_prior'):
            Q_like_D  = network.Q_like.D.detach().cpu().numpy()
            Q_prior_D = network.Q_prior.D.detach().cpu().numpy()
        else:
            # LH: single learned Q_post — treat as D_like, D_prior = 0
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

    # ── Load simulation from ZarrStore ────────────────────────────────────
    print(f'Loading simulation index -{args.index} (from end) from {args.store_path}...')
    store  = swyft.ZarrStore(args.store_path)
    sample = store[-args.index]

    delta_z0   = np.asarray(sample['delta_z0'],   dtype=np.float32)
    delta_z127 = np.asarray(sample['delta_z127'], dtype=np.float32)
    sim_params = np.asarray(sample['sim_params'], dtype=np.float32)   # [Om, Ob, h, ns, s8]
    rescaling_factor_sim = float(sample['rescaling_factor'][0])

    print(f'Simulation -{args.index}: '
          f'Omega_m={sim_params[0]:.4f}, Omega_b={sim_params[1]:.4f}, '
          f'h={sim_params[2]:.4f}, n_s={sim_params[3]:.4f}, sigma_8={sim_params[4]:.4f}')
    print(f'Per-simulation rescaling_factor: {rescaling_factor_sim:.6f}')

    # Reconstruct cosmology dict for this simulation (used by CLASS in plotting)
    cosmo_params_sim = {
        'h':          float(sim_params[2]),
        'Omega_b':    float(sim_params[1]),
        'Omega_cdm':  float(sim_params[0]) - float(sim_params[1]),
        'n_s':        float(sim_params[3]),
        'non linear': 'halofit',
        'sigma8':     float(sim_params[4]),
    }

    # ── Override rescaling_factor with per-simulation value ───────────────
    # The config stores the fiducial value used during training.
    # For a specific off-fiducial LH cosmology we must use the correct D(z127)/D(z0).
    network.rescaling_factor = rescaling_factor_sim
    print(f'Set network.rescaling_factor = {rescaling_factor_sim:.6f}')

    # ── Apply Gaussian noise ──────────────────────────────────────────────
    sigma_noise = train_config.get('sigma_noise', 0.0)
    rng = np.random.default_rng(args.noise_seed)
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
        'store_path': args.store_path,
        'sim_index': -args.index,
        'sim_params': sim_params.tolist(),
        'rescaling_factor_sim': rescaling_factor_sim,
        'num_samples': args.num_samples,
        'noise_seed': args.noise_seed,
        'sigma_noise': sigma_noise,
        'device': device,
        'train_config': train_config,
    }
    with open(os.path.join(samples_dir, 'config.json'), 'w') as f:
        json.dump(run_config, f, indent=2)
    print(f'Samples saved to {samples_dir}/')

    # ── Stack in-memory samples (no disk reload) ──────────────────────────
    print('Stacking samples for plotting...')
    samples_arr = np.array(samples_list)   # (num_samples, N, N, N)
    del samples_list

    # Rescale to internal space (plotting functions expect internal units)
    samples_int    = samples_arr / rescaling_factor_sim
    z_MAP_int      = z_MAP_np   / rescaling_factor_sim
    delta_z127_int = delta_z127 / rescaling_factor_sim

    # ── Plot samples analysis ─────────────────────────────────────────────
    # MAS=None: fields are already PCS-deconvolved in the ZarrStore builder
    print('Running plot_samples_analysis...')
    utils.plot_samples_analysis(
        delta_z127=delta_z127_int,
        delta_z0=delta_z0,
        samples=samples_int,
        z_MAP=z_MAP_int,
        box=box,
        cosmo_params=cosmo_params_sim,
        MAS=None,
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
