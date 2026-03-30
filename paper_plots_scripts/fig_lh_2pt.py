"""
IC two-point statistics (P(k), T(k), C(k)) for a Latin Hypercube test sample.

Loads a trained Gaussian_NPE_LH model, reads one of the last 10 held-out test
simulations from the LH ZarrStore, runs inference with the per-cosmology
rescaling_factor, and produces the same IC summary-stats plot as fig2_2pt.py.

The LH dataset has 2000 simulations; with n_train=1990 the last 10 (store[-10]
to store[-1]) are never seen during training and serve as the test set.

Output: {output_dir}/2_summary_stats_{RUN_NAME}.pdf

Usage:
    python paper_plots_scripts/fig_lh_2pt.py \\
        --model_dir runs/260303_195331_LH_sigma_noise_1_train_only_Q_post_UNet_Only

    python paper_plots_scripts/fig_lh_2pt.py \\
        --model_dir runs/260303_195331_LH_sigma_noise_1_train_only_Q_post_UNet_Only \\
        --index 3 \\
        --num_samples 100 \\
        --output_dir paper_plots_scripts/260303_195331_LH_sigma_noise_1_train_only_Q_post_UNet_Only
"""

import os
import json
import argparse

import numpy as np
import swyft
import torch

from gaussian_npe import utils, Gaussian_NPE_LH
from fig2_2pt import plot_summary_stats

BOX_PARAMS = {
    'box_size': 1000.,
    'grid_res': 128,
    'h': 0.6711,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='LH IC two-point statistics for a held-out test sample',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--model_dir', type=str, required=True,
        help='Path to trained LH run folder (must contain config.json and model.pt)',
    )
    parser.add_argument(
        '--store_path', type=str, default=None,
        help='ZarrStore path override. Defaults to store_path from config.json.',
    )
    parser.add_argument(
        '--index', type=int, default=1,
        help='Test sample: 1 = last (store[-1]), 2 = second-to-last, ..., 10 = first held-out.',
    )
    parser.add_argument(
        '--num_samples', type=int, default=100,
        help='Number of posterior samples to draw.',
    )
    parser.add_argument(
        '--noise_seed', type=int, default=42,
        help='RNG seed for Gaussian observational noise.',
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Where to save the PDF. Defaults to paper_plots_scripts/{run_name}/.',
    )
    parser.add_argument(
        '--no_latex', dest='use_latex', action='store_false', default=True,
        help='Disable LaTeX rendering.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    utils.configure_matplotlib_style(use_latex=args.use_latex)

    # ── Load training config ──────────────────────────────────────────────
    config_path = os.path.join(args.model_dir, 'config.json')
    with open(config_path) as f:
        train_config = json.load(f)

    run_name   = os.path.basename(os.path.abspath(args.model_dir))
    store_path = args.store_path or train_config['store_path']
    sigma_noise = train_config.get('sigma_noise', 0.0)
    print(f'Run: {run_name}')

    # ── Output directory ──────────────────────────────────────────────────
    if args.output_dir:
        save_dir = os.path.abspath(args.output_dir)
    else:
        scripts_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir    = os.path.join(scripts_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    # ── Device / box / network ────────────────────────────────────────────
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    box    = utils.Power_Spectrum_Sampler(BOX_PARAMS, device=device)

    network = Gaussian_NPE_LH(box, sigma_noise=sigma_noise)
    network.float().to(device)
    state_dict = torch.load(
        os.path.join(args.model_dir, 'model.pt'),
        map_location=device, weights_only=False,
    )
    network.load_state_dict(state_dict)
    network.eval()
    print(f'Model loaded from {args.model_dir}/model.pt')

    # ── Load test simulation from ZarrStore ───────────────────────────────
    print(f'Loading simulation store[-{args.index}] from {store_path}...')
    store  = swyft.ZarrStore(store_path)
    sample = store[-args.index]

    delta_z0   = np.asarray(sample['delta_z0'],   dtype=np.float32)
    delta_z127 = np.asarray(sample['delta_z127'], dtype=np.float32)
    sim_params = np.asarray(sample['sim_params'], dtype=np.float32)

    # Compute rescaling_factor from sim_params — more robust than reading the
    # stored value, which can be 0.0 for ZarrStore entries that were not fully written.
    cosmo_sim = {
        'Omega_cdm': float(sim_params[0]) - float(sim_params[1]),
        'Omega_b':   float(sim_params[1]),
    }
    rescaling_factor = (
        utils.growth_D_approx(cosmo_sim, 127)
        / utils.growth_D_approx(cosmo_sim, 0)
    )

    print(f'  Ωm={sim_params[0]:.4f}  Ωb={sim_params[1]:.4f}  '
          f'h={sim_params[2]:.4f}  n_s={sim_params[3]:.4f}  σ8={sim_params[4]:.4f}')
    print(f'  rescaling_factor = {rescaling_factor:.6f}')

    cosmo_params_sim = {
        'h':          float(sim_params[2]),
        'Omega_b':    float(sim_params[1]),
        'Omega_cdm':  float(sim_params[0]) - float(sim_params[1]),
        'n_s':        float(sim_params[3]),
        'non linear': 'halofit',
        'sigma8':     float(sim_params[4]),
    }

    # ── Set per-sample rescaling factor ───────────────────────────────────
    network.rescaling_factor = rescaling_factor

    # ── Add noise and compute MAP ─────────────────────────────────────────
    rng = np.random.default_rng(args.noise_seed)
    delta_obs = delta_z0 + rng.standard_normal(delta_z0.shape).astype('f') * sigma_noise
    print('Computing MAP estimate...')
    with torch.no_grad():
        z_MAP_np = network.get_z_MAP(
            torch.from_numpy(delta_obs).to(device).float()
        ).cpu().numpy()

    # ── Draw posterior samples ────────────────────────────────────────────
    print(f'Drawing {args.num_samples} posterior samples...')
    samples_list = []
    with torch.no_grad():
        for i in range(args.num_samples):
            s = network.sample(1, z_MAP=torch.from_numpy(z_MAP_np).to(device))[0]
            samples_list.append(s)
            if (i + 1) % 50 == 0:
                print(f'  {i + 1}/{args.num_samples}')
    samples_arr = np.array(samples_list)   # (num_samples, N, N, N)

    # ── Plot ──────────────────────────────────────────────────────────────
    # network.get_z_MAP and network.sample already return physical units
    # (delta_z127 amplitude). plot_summary_stats expects physical units,
    # same as fig2_2pt.py's main() which passes d['_int'] * rf.
    print('Plotting two-point statistics...')
    plot_summary_stats(
        delta_z127=delta_z127,
        delta_z0=delta_z0,
        samples=samples_arr,
        z_MAP=z_MAP_np,
        box=box,
        cosmo_params=cosmo_params_sim,
        MAS=None,
        rescaling_factor=rescaling_factor,
        save_dir=save_dir,
        run_name=run_name,
    )
    print(f'\nDone. Plot saved to {save_dir}/')


if __name__ == '__main__':
    main()
