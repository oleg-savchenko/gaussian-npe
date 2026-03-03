"""
Fit the learned D_like(k) diagonal of a trained Gaussian NPE network with an
analytical Gaussian model in log10(k) and produce a diagnostic plot.

Model:  D_like_fit(k) = A * exp(-((log10(k) - mu) / sigma)^2)
Output: D_post_fit(k) = D_like_fit(k) + D_prior(k)

The three fit parameters (A, mu, sigma) compactly characterise how much
information the network has learned at each scale.

Usage:
    python scripts/fit_D_spectrum.py --model_dir runs/20260220_152350_WienerNet_resumed

    python scripts/fit_D_spectrum.py \\
        --model_dir runs/20260220_152350_WienerNet_resumed \\
        --n_bins 50 \\
        --output_dir ./fits \\
        --save_csv
"""

import os

import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
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
)

NETWORK_CLASSES = {
    'default':         Gaussian_NPE_Network,
    'UNet_Only':       Gaussian_NPE_UNet_Only,
    'WienerNet':       Gaussian_NPE_WienerNet,
    'LearnableFilter': Gaussian_NPE_LearnableFilter,
    'SmoothFilter':    Gaussian_NPE_SmoothFilter,
    'Iterative':       Gaussian_NPE_Iterative,
    'LH':              Gaussian_NPE_LH,
    'CustomUNet':      Gaussian_NPE_CustomUNet,
}

BOX_PARAMS = {
    'box_size': 1000.,
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
        description='Fit D_like(k) of a trained network with a Gaussian in log10(k)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--model_dir', type=str, required=True,
        help='Training run directory (must contain model.pt and config.json)',
    )
    parser.add_argument(
        '--n_bins', type=int, default=40,
        help='Number of log-spaced k bins used for binning before fitting',
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Where to save the plot and CSV.  Default: model_dir/D_fit/',
    )
    parser.add_argument(
        '--save_csv', action='store_true', default=False,
        help='Also save a CSV with the binned data and fitted values',
    )
    parser.add_argument(
        '--use_latex', action='store_true', default=False,
        help='Use LaTeX rendering and scienceplots style',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    utils.configure_matplotlib_style(use_latex=args.use_latex)

    # ── Load training config ─────────────────────────────────────────────
    config_path = os.path.join(args.model_dir, 'config.json')
    with open(config_path, 'r') as f:
        train_config = json.load(f)
    print(f'Loaded config from {config_path}')

    # ── Output directory ─────────────────────────────────────────────────
    output_dir = args.output_dir or os.path.join(args.model_dir, 'D_fit')
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
    run_name = train_config.get('run_name', 'model')

    # ── Device ───────────────────────────────────────────────────────────
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # ── Box & prior ──────────────────────────────────────────────────────
    box = utils.Power_Spectrum_Sampler(BOX_PARAMS, device=device)
    Dz_approx = (
        utils.growth_D_approx(COSMO_PARAMS, Z_IC)
        / utils.growth_D_approx(COSMO_PARAMS, 0)
    )
    rescaling_factor = train_config.get('rescaling_factor', Dz_approx)

    prior = box.get_prior_Q_factors(
        lambda k: torch.tensor(
            utils.get_pk_class(COSMO_PARAMS, 0, np.array(k)), device=device,
        )
    )

    # ── Load network ─────────────────────────────────────────────────────
    network_name = train_config.get('network', 'default')
    NetworkClass = NETWORK_CLASSES[network_name]
    print(f'Network: {NetworkClass.__name__}')

    net_kwargs = dict(
        sigma_noise=train_config.get('sigma_noise', 0.0),
        rescaling_factor=rescaling_factor,
    )
    if network_name not in ('UNet_Only', 'WienerNet', 'Iterative', 'CustomUNet'):
        net_kwargs['k_cut'] = train_config.get('k_cut', 0.03)
        net_kwargs['w_cut'] = train_config.get('w_cut', 0.001)

    network = NetworkClass(box, prior, **net_kwargs)
    network.float().to(device)

    model_path = os.path.join(args.model_dir, 'model.pt')
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    network.load_state_dict(state_dict)
    network.eval()
    print(f'Loaded weights from {model_path}')

    # ── Extract D matrices ───────────────────────────────────────────────
    D_like  = network.Q_like.D.detach().cpu().numpy()
    D_prior = network.Q_prior.D.detach().cpu().numpy()
    k_flat  = box.k.cpu().numpy().flatten()
    k_Nq    = box.k_Nq

    mask = (k_flat > 1e-3) & (k_flat < k_Nq)

    # ── Fit ──────────────────────────────────────────────────────────────
    print(f'\nFitting D_like(k) with Gaussian in log10(k) using {args.n_bins} bins...')
    popt, perr, D_fit_func, k_bins, D_like_means, D_like_stds, D_prior_means, N_modes = \
        utils.fit_D_spectrum(
            k_flat[mask], D_like[mask], D_prior=D_prior[mask], n_bins=args.n_bins,
        )
    A, mu, sigma, c = popt
    dA, dmu, dsigma, dc = perr

    print(f'\nD_like fit  A * exp(-((log10(k) - mu) / sigma)^2) + c:')
    print(f'  A     = {A:.4f} +/- {dA:.4f}')
    print(f'  mu    = {mu:.4f} +/- {dmu:.4f}  (k_peak = {10**mu:.4f} h/Mpc)')
    print(f'  sigma = {sigma:.4f} +/- {dsigma:.4f}')
    print(f'  c     = {c:.4f} +/- {dc:.4f}')

    # ── Plot ─────────────────────────────────────────────────────────────
    k_smooth = np.geomspace(k_flat[mask].min(), k_flat[mask].max(), 500)
    valid_bins = np.isfinite(D_like_means) & np.isfinite(D_prior_means)
    D_prior_smooth = np.interp(
        np.log10(k_smooth),
        np.log10(k_bins[valid_bins]),
        D_prior_means[valid_bins],
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    # Raw scatter
    ax.scatter(k_flat[mask], D_like[mask], s=0.5, alpha=0.3, color='C0', rasterized=True)
    ax.scatter(k_flat[mask], D_prior[mask], s=0.5, alpha=0.3, color='C1', rasterized=True)
    ax.scatter(k_flat[mask], D_like[mask] + D_prior[mask], s=0.5, alpha=0.2,
               color='C2', rasterized=True)

    # Binned means with error bars
    k_bins_valid = k_bins[valid_bins]
    ax.errorbar(k_bins_valid, D_like_means[valid_bins], yerr=D_like_stds[valid_bins],
                fmt='o', ms=3, color='C0', elinewidth=0.8, capsize=2, zorder=3)

    # Fitted curves
    ax.plot(k_smooth, D_fit_func(k_smooth), 'C0-', lw=2,
            label=fr'$D_{{\rm like}}$ fit: $A$={A:.2f}, $k_0$={10**mu:.3f}, $\sigma$={sigma:.2f}, $c$={c:.2f}')
    ax.plot(k_smooth, D_fit_func(k_smooth) + D_prior_smooth, 'C2-', lw=2,
            label=r'$D_{\rm post}$ fit')

    # Invisible scatter handles for legend
    ax.scatter([], [], s=6, color='C0', label=r'$D_{\rm like}$')
    ax.scatter([], [], s=6, color='C1', label=r'$D_{\rm prior}$')
    ax.scatter([], [], s=6, color='C2', label=r'$D_{\rm posterior}$')
    ax.axvline(x=k_Nq, color='r', linestyle='--', lw=1, label=r'$k_{\rm Nyq}$')

    ax.set_xscale('log')
    ax.set_xlabel(r'$k~[h\,{\rm Mpc}^{-1}]$', fontsize=14)
    ax.set_ylabel(r'$D(k)$', fontsize=14)
    ax.set_title(fr'$D_{{like}}(k)$ fit — {run_name}', fontsize=12)
    ax.legend(loc='upper right', markerscale=4, fontsize=9)
    ax.grid(alpha=0.15)
    fig.tight_layout()

    plot_path = os.path.join(output_dir, f'{timestamp}_D_fit_{run_name}.png')
    fig.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'\nPlot saved to {plot_path}')

    # ── Optional CSV ─────────────────────────────────────────────────────
    if args.save_csv:
        D_like_fit_bins  = D_fit_func(k_bins)
        D_prior_bins     = np.where(np.isfinite(D_prior_means), D_prior_means, np.nan)
        D_post_fit_bins  = D_like_fit_bins + np.where(np.isfinite(D_prior_bins), D_prior_bins, 0.0)

        data = np.column_stack([
            k_bins, D_like_means, D_like_stds,
            D_like_fit_bins, D_prior_bins, D_post_fit_bins,
            N_modes.astype(float),
        ])
        csv_path = os.path.join(output_dir, f'{timestamp}_D_fit_{run_name}.csv')
        np.savetxt(
            csv_path, data, delimiter=',',
            header='k_bin_center,D_like_mean,D_like_std,D_like_fit,D_prior_mean,D_post_fit,N_modes',
            comments='',
        )

        # Human-readable summary
        txt_path = os.path.join(output_dir, f'{timestamp}_D_fit_{run_name}.txt')
        with open(txt_path, 'w') as f:
            f.write(f'D_like(k) Gaussian fit — {run_name}\n')
            f.write('=' * 50 + '\n\n')
            f.write('Model: A * exp(-((log10(k) - mu) / sigma)^2) + c\n\n')
            f.write(f'  A     = {A:.6f} +/- {dA:.6f}\n')
            f.write(f'  mu    = {mu:.6f} +/- {dmu:.6f}  (k_peak = {10**mu:.6f} h/Mpc)\n')
            f.write(f'  sigma = {sigma:.6f} +/- {dsigma:.6f}\n')
            f.write(f'  c     = {c:.6f} +/- {dc:.6f}\n')
        print(f'CSV  saved to {csv_path}')
        print(f'Text saved to {txt_path}')


if __name__ == '__main__':
    main()
