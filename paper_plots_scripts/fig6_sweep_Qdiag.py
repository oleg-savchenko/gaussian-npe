"""
Likelihood precision diagonal D_like(k) across sweep runs.

Produces a 1- or 2-panel figure:
  - Left panel  (always):  D_like(k) for different sigma_noise values
                            (--sweep_dir, plasma colormap)
  - Right panel (optional): D_like(k) for different n_train values
                            (--sweep_train_dir, viridis colormap)

D_prior is identical across all runs and shown as a single dashed reference.

Output: paper_plots_scripts/{OUTPUT_DIR}/6_sweep_Qdiag.pdf

Usage:
    python paper_plots_scripts/fig6_sweep_Qdiag.py \\
        --sweep_dir      paper_test_runs/runs/260304_233941_sweep_noise \\
        --sweep_train_dir paper_test_runs/runs/260328_234020_sweep_train \\
        --output_dir     paper_plots_scripts/260303_224627_net_IsotropicD
"""

import os
import json
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker

from gaussian_npe import utils
from _common import NETWORK_CLASSES, BOX_PARAMS, COSMO_PARAMS, Z_IC


# ── Model loading ─────────────────────────────────────────────────────────────

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


def _discover_runs(sweep_dir, param_key):
    """Return list of (param_value, model_dir) sorted by param_value."""
    runs = []
    for name in os.listdir(sweep_dir):
        subdir = os.path.join(sweep_dir, name)
        cfg_path   = os.path.join(subdir, 'config.json')
        model_path = os.path.join(subdir, 'model.pt')
        if not (os.path.isfile(cfg_path) and os.path.isfile(model_path)):
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)
        if param_key not in cfg:
            continue
        runs.append((float(cfg[param_key]), subdir))
    return sorted(runs, key=lambda x: x[0])


# ── Plotting ──────────────────────────────────────────────────────────────────

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


def _draw_Qdiag_panel(ax, fig, runs, box, cmap, norm, cb_label, cb_ticks):
    """Draw one D_like(k) sweep panel onto `ax`.

    Parameters
    ----------
    runs : list of dicts with keys Q_like_D, Q_prior_D, Q_like_k_nodes,
           Q_like_D_nodes, and a numeric `param` value (added by caller).
    """
    k_Nq   = box.k_Nq
    k_flat = box.k.cpu().numpy().flatten()
    mask   = (k_flat > 1e-3) & (k_flat < k_Nq)

    # D_prior reference (same for all runs — use first; no upper k cut)
    mask_prior = k_flat > 1e-3
    k_prior, d_prior = _radial_bin_mean(k_flat[mask_prior], runs[0]['Q_prior_D'][mask_prior])
    ax.plot(k_prior, d_prior, color='k', ls='--', lw=1.5,
            label=r'$D_{\rm prior}$', zorder=10)

    # D_like curves
    for r in runs:
        color   = cmap(norm(r['param']))
        k_nodes = r['Q_like_k_nodes']
        D_nodes = r['Q_like_D_nodes']
        if k_nodes is not None and D_nodes is not None:
            ax.plot(k_nodes, D_nodes, color=color, lw=2.5, zorder=5)
            ax.scatter(k_nodes, D_nodes, color=color, s=18, zorder=6)
        else:
            k_bin, d_bin = _radial_bin_mean(k_flat[mask], r['Q_like_D'][mask])
            ax.plot(k_bin, d_bin, color=color, lw=2.5, zorder=5)

    # k_Nyq
    ax.axvline(k_Nq, color='r', ls='--', lw=1, label=r'$k_{\rm Nyq}$', zorder=4)

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(cb_label, fontsize=13)
    cbar.ax.yaxis.set_major_locator(mticker.FixedLocator(cb_ticks))
    cbar.ax.yaxis.set_minor_locator(mticker.NullLocator())
    cbar.ax.yaxis.set_major_formatter(mticker.FixedFormatter([str(t) for t in cb_ticks]))

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$k~[h\,{\rm Mpc}^{-1}]$', fontsize=14)
    ax.set_ylabel(r'$D_{\rm like}(k)$', fontsize=14)
    ax.legend(loc='lower left', frameon=True, framealpha=0.9)
    ax.grid(alpha=0.15)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Q diagonal sweep plots',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--sweep_dir', type=str, required=True,
        help='Noise sweep directory (subdirs vary sigma_noise).',
    )
    parser.add_argument(
        '--sweep_train_dir', type=str, default=None,
        help='Training-size sweep directory (subdirs vary n_train). '
             'If given, adds a second panel.',
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Where to save the PDF. Defaults to paper_plots_scripts/{sweep_name}/.',
    )
    parser.add_argument(
        '--no_latex', dest='use_latex', action='store_false',
        help='Disable LaTeX rendering (on by default)',
    )
    parser.set_defaults(use_latex=True)
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    utils.configure_matplotlib_style(use_latex=args.use_latex)

    # ── Noise sweep ───────────────────────────────────────────────────────
    print(f'Noise sweep: {args.sweep_dir}')
    noise_pairs = _discover_runs(args.sweep_dir, 'sigma_noise')
    if not noise_pairs:
        raise FileNotFoundError(f'No valid runs found in {args.sweep_dir}')

    box = None
    noise_runs = []
    for sigma, subdir in noise_pairs:
        print(f'  Loading sigma_noise={sigma}  ({os.path.basename(subdir)})')
        d = _load_Qdiag(subdir)
        if box is None:
            box = d['box']
        d['param'] = sigma
        noise_runs.append(d)

    noise_vals  = [r['param'] for r in noise_runs]
    noise_norm  = mcolors.Normalize(vmin=min(noise_vals), vmax=max(noise_vals))
    noise_cmap  = cm.plasma
    noise_ticks = [0.1, 0.25, 0.5, 0.75, 1.0]

    # ── Train sweep (optional) ────────────────────────────────────────────
    train_runs = []
    if args.sweep_train_dir:
        print(f'\nTrain sweep: {args.sweep_train_dir}')
        train_pairs = _discover_runs(args.sweep_train_dir, 'n_train')
        for n_train, subdir in train_pairs:
            print(f'  Loading n_train={int(n_train)}  ({os.path.basename(subdir)})')
            d = _load_Qdiag(subdir)
            d['param'] = n_train
            train_runs.append(d)

    # ── Figure ────────────────────────────────────────────────────────────
    n_panels = 2 if train_runs else 1
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(7 * n_panels, 5),
                             squeeze=False)
    plt.rcParams['figure.facecolor'] = 'white'

    _draw_Qdiag_panel(
        axes[0, 0], fig, noise_runs, box,
        cmap=noise_cmap, norm=noise_norm,
        cb_label=r'$\sigma_{\rm noise}$',
        cb_ticks=noise_ticks,
    )
    axes[0, 0].set_title(r'Noise sweep')

    if train_runs:
        n_vals = [r['param'] for r in train_runs]
        train_norm = mcolors.Normalize(vmin=min(n_vals), vmax=max(n_vals))
        _draw_Qdiag_panel(
            axes[0, 1], fig, train_runs, box,
            cmap=cm.viridis, norm=train_norm,
            cb_label=r'$N_{\rm train}$',
            cb_ticks=[250, 500, 1000, 1500, 2000],
        )
        axes[0, 1].set_title(r'Training size sweep')

    fig.tight_layout()

    # ── Output ────────────────────────────────────────────────────────────
    if args.output_dir:
        save_dir = os.path.abspath(args.output_dir)
    else:
        scripts_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir    = os.path.join(scripts_dir,
                                   os.path.basename(os.path.abspath(args.sweep_dir)))
    os.makedirs(save_dir, exist_ok=True)

    out = os.path.join(save_dir, '6_sweep_Qdiag.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved {out}')


if __name__ == '__main__':
    main()
