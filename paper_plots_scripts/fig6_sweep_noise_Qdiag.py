"""
Likelihood precision diagonal D_like(k) across noise levels from a sweep run.

Loads the trained IsotropicD network from each subdirectory of a noise sweep,
extracts the learned Q_like diagonal, and overlays all curves on one plot
colour-coded by sigma_noise. The prior D_prior is identical across runs and
is shown as a single dashed reference line.

Produces one figure saved to paper_plots_scripts/{SWEEP_NAME}/:
  - 6_sweep_noise_Qdiag_{SWEEP_NAME}.pdf

Usage:
    python paper_plots_scripts/fig6_sweep_noise_Qdiag.py \\
        --sweep_dir paper_test_runs/runs/260304_233941_sweep_noise

    python paper_plots_scripts/fig6_sweep_noise_Qdiag.py \\
        --sweep_dir paper_test_runs/runs/260304_233941_sweep_noise \\
        --no_latex
"""

import os
import json
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from gaussian_npe import utils
from _common import NETWORK_CLASSES, BOX_PARAMS, COSMO_PARAMS, Z_IC


def _load_Qdiag(model_dir):
    """Load a trained model and return Q diagonal arrays + box. No side effects."""
    with open(os.path.join(model_dir, 'config.json')) as f:
        cfg = json.load(f)

    device = 'cpu'
    box = utils.Power_Spectrum_Sampler(BOX_PARAMS, device=device)

    Dz_approx = (utils.growth_D_approx(COSMO_PARAMS, Z_IC)
                 / utils.growth_D_approx(COSMO_PARAMS, 0))
    rf = cfg.get('rescaling_factor', Dz_approx)

    prior = box.get_prior_Q_factors(
        lambda k: torch.tensor(
            utils.get_pk_class(COSMO_PARAMS, 0, k.detach().cpu().numpy()),
            device=device,
        )
    )

    network_name = cfg.get('network', 'default')
    NetworkClass = NETWORK_CLASSES[network_name]
    net_kwargs = dict(sigma_noise=cfg.get('sigma_noise', 0.0), rescaling_factor=rf)
    if network_name not in ('UNet_Only', 'WienerNet', 'Iterative', 'CustomUNet',
                             'IsotropicD', 'WienerIsotropicD', 'Poisson', 'LH'):
        net_kwargs['k_cut'] = cfg.get('k_cut', 0.03)
        net_kwargs['w_cut'] = cfg.get('w_cut', 0.003)

    network = NetworkClass(box, prior, **net_kwargs).float().to(device)
    state = torch.load(os.path.join(model_dir, 'model.pt'),
                       map_location=device, weights_only=False)
    network.load_state_dict(state)
    network.eval()

    with torch.no_grad():
        if hasattr(network, 'Q_like') and hasattr(network, 'Q_prior'):
            Q_like_D  = network.Q_like.D.detach().cpu().numpy()
            Q_prior_D = network.Q_prior.D.detach().cpu().numpy()
        else:
            Q_like_D  = network.Q_post.D.detach().cpu().numpy()
            Q_prior_D = np.zeros_like(Q_like_D)

    Q_like_obj = getattr(network, 'Q_like', None) or network.Q_post
    Q_like_k_nodes = (Q_like_obj._log_k_nodes.exp().detach().cpu().numpy()
                      if hasattr(Q_like_obj, '_log_k_nodes') else None)
    Q_like_D_nodes = (Q_like_obj.log_D_nodes.exp().detach().cpu().numpy()
                      if hasattr(Q_like_obj, 'log_D_nodes') else None)

    return dict(Q_like_D=Q_like_D, Q_prior_D=Q_prior_D,
                Q_like_k_nodes=Q_like_k_nodes, Q_like_D_nodes=Q_like_D_nodes,
                box=box)


def _radial_bin_mean(k_flat, values, n_bins=80):
    """Radial average of `values` in log-spaced k bins."""
    k_min = k_flat[k_flat > 0].min()
    k_max = k_flat.max()
    edges = np.logspace(np.log10(k_min), np.log10(k_max), n_bins + 1)
    idx   = np.digitize(k_flat, edges) - 1
    k_cen, v_cen = [], []
    for i in range(n_bins):
        sel = idx == i
        if sel.sum() > 0:
            k_cen.append(k_flat[sel].mean())
            v_cen.append(values[sel].mean())
    return np.array(k_cen), np.array(v_cen)


def plot_sweep_Qdiag(runs, box, save_dir, sweep_name):
    """Overlay D_like(k) curves for multiple noise levels.

    Parameters
    ----------
    runs : list of dicts, each with keys:
        sigma_noise, Q_like_k_nodes, Q_like_D_nodes, Q_like_D, Q_prior_D
        Sorted by sigma_noise ascending.
    box : utils.Power_Spectrum_Sampler
    save_dir : str
    sweep_name : str
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.rcParams['figure.facecolor'] = 'white'

    sigma_vals = np.array([r['sigma_noise'] for r in runs])
    norm       = mcolors.LogNorm(vmin=sigma_vals.min(), vmax=sigma_vals.max())
    cmap       = cm.plasma

    k_Nq   = box.k_Nq
    k_flat = box.k.cpu().numpy().flatten()
    mask   = (k_flat > 1e-3) & (k_flat < k_Nq)

    fig, ax = plt.subplots(figsize=(7, 5))

    # ── D_prior reference (same for all runs — use first) ─────────────────
    k_prior, d_prior = _radial_bin_mean(k_flat[mask], runs[0]['Q_prior_D'][mask])
    ax.plot(k_prior, d_prior, color='k', ls='--', lw=1.5,
            label=r'$D_{\rm prior}$', zorder=10)

    # ── D_like curves colour-coded by sigma_noise ─────────────────────────
    for r in runs:
        sigma  = r['sigma_noise']
        color  = cmap(norm(sigma))
        k_nodes = r['Q_like_k_nodes']
        D_nodes = r['Q_like_D_nodes']
        if k_nodes is not None and D_nodes is not None:
            ax.plot(k_nodes, D_nodes, color=color, lw=2.5, zorder=5)
            ax.scatter(k_nodes, D_nodes, color=color, s=18, zorder=6)
        else:
            # fallback: radial-bin scatter
            k_bin, d_bin = _radial_bin_mean(k_flat[mask], r['Q_like_D'][mask])
            ax.plot(k_bin, d_bin, color=color, lw=2.5, zorder=5)

    # ── k_Nyq line ────────────────────────────────────────────────────────
    ax.axvline(k_Nq, color='r', ls='--', lw=1,
               label=r'$k_{\rm Nyq}$', zorder=4)

    # ── Colorbar ──────────────────────────────────────────────────────────
    sm  = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r'$\sigma_{\rm noise}$', fontsize=13)
    cbar.set_ticks([0.1, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['0.1', '0.25', '0.5', '0.75', '1'])

    # ── Axes ──────────────────────────────────────────────────────────────
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$k~[h\,{\rm Mpc}^{-1}]$', fontsize=14)
    ax.set_ylabel(r'$D_{\rm like}(k)$',        fontsize=14)
    ax.set_title(r'Likelihood precision vs noise level')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.15)
    fig.tight_layout()

    out = os.path.join(save_dir, f'6_sweep_noise_Qdiag_{sweep_name}.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Q diagonal sweep-noise plot (fig 6)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--sweep_dir', type=str, required=True,
        help='Path to sweep directory containing one subdirectory per noise level.',
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Directory to save the plot. Defaults to paper_plots_scripts/{sweep_name}/.',
    )
    parser.add_argument(
        '--no_latex', dest='use_latex', action='store_false',
        help='Disable LaTeX rendering (LaTeX is on by default)',
    )
    parser.set_defaults(use_latex=True)
    return parser.parse_args()


def main():
    args = parse_args()
    utils.configure_matplotlib_style(use_latex=args.use_latex)

    sweep_dir  = os.path.abspath(args.sweep_dir)
    sweep_name = os.path.basename(sweep_dir)

    # ── Discover subdirectories ────────────────────────────────────────────
    subdirs = sorted([
        os.path.join(sweep_dir, d)
        for d in os.listdir(sweep_dir)
        if os.path.isfile(os.path.join(sweep_dir, d, 'config.json'))
        and os.path.isfile(os.path.join(sweep_dir, d, 'model.pt'))
    ])
    if not subdirs:
        raise FileNotFoundError(f'No valid run subdirectories found in {sweep_dir}')
    print(f'Found {len(subdirs)} runs in {sweep_dir}')

    # ── Load each model ────────────────────────────────────────────────────
    runs = []
    box  = None
    for subdir in subdirs:
        with open(os.path.join(subdir, 'config.json')) as f:
            cfg = json.load(f)
        sigma_noise = float(cfg['sigma_noise'])
        print(f'Loading sigma_noise={sigma_noise}  ({os.path.basename(subdir)})')

        d = _load_Qdiag(subdir)
        if box is None:
            box = d['box']

        runs.append(dict(
            sigma_noise=sigma_noise,
            Q_like_D=d['Q_like_D'],
            Q_prior_D=d['Q_prior_D'],
            Q_like_k_nodes=d['Q_like_k_nodes'],
            Q_like_D_nodes=d['Q_like_D_nodes'],
        ))

    runs.sort(key=lambda r: r['sigma_noise'])

    # ── Output directory ───────────────────────────────────────────────────
    if args.output_dir:
        save_dir = os.path.abspath(args.output_dir)
    else:
        scripts_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir    = os.path.join(scripts_dir, sweep_name)
    os.makedirs(save_dir, exist_ok=True)

    plot_sweep_Qdiag(runs, box, save_dir, sweep_name)
    print(f'\nDone. Plot saved to {save_dir}/')


if __name__ == '__main__':
    main()
