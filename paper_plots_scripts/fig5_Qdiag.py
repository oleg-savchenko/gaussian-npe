"""
Precision matrix diagonals D_like, D_prior, D_posterior vs wavenumber k.

Produces one figure saved to paper_plots_scripts/{RUN_NAME}/:
  - 6_Q_diagonals_{RUN_NAME}.pdf : scatter plot of D_like, D_prior, and
    D_posterior = D_like + D_prior as a function of k, with optional k-node
    markers for parametric likelihood models (e.g. SmoothFilter, IsotropicD).

No posterior samples are needed for this plot. Use --num_samples 0 (default)
to skip the sampling step and produce the figure immediately after loading the
trained network.

Usage:
    python paper_plots_scripts/fig5_Qdiag.py \\
        --model_dir paper_test_runs/runs/20260301_215801_net_IsotropicD

    python paper_plots_scripts/fig5_Qdiag.py \\
        --model_dir paper_test_runs/runs/20260301_215801_net_IsotropicD \\
        --no_latex
"""

import os
import argparse

import matplotlib.pyplot as plt

from gaussian_npe import utils
from _common import add_common_args, load_model_and_generate_samples


def plot_Q_diagonals(Q_like_D, Q_prior_D, box,
                     Q_like_k_nodes=None, Q_like_D_nodes=None,
                     save_dir='./plots', run_name=''):
    """Plot precision-matrix diagonals vs k (fig 6 of plot_samples_analysis).

    Parameters
    ----------
    Q_like_D : np.ndarray, shape (N^3,)
        Diagonal of the likelihood precision matrix in Hartley space.
    Q_prior_D : np.ndarray, shape (N^3,)
        Diagonal of the prior precision matrix.
    box : utils.Power_Spectrum_Sampler
    Q_like_k_nodes : np.ndarray or None
        k-space node positions for parametric likelihood models.
    Q_like_D_nodes : np.ndarray or None
        D values at the k-nodes.
    save_dir : str
    run_name : str
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.rcParams['figure.facecolor'] = 'white'

    k_Nq   = box.k_Nq
    k_flat = box.k.cpu().numpy().flatten()
    mask   = (k_flat < k_Nq) & (k_flat > 1e-3)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(k_flat[mask], Q_like_D[mask],  s=0.5, alpha=0.4,
               label=r'$D_{\rm like}$')
    ax.scatter(k_flat[mask], Q_prior_D[mask], s=0.5, alpha=0.4,
               label=r'$D_{\rm prior}$')
    ax.scatter(k_flat[mask], Q_like_D[mask] + Q_prior_D[mask], s=0.5, alpha=0.3,
               label=r'$D_{\rm posterior}$')
    if Q_like_k_nodes is not None and Q_like_D_nodes is not None:
        ax.scatter(Q_like_k_nodes, Q_like_D_nodes, marker='x', s=60,
                   color='k', linewidths=1.5, zorder=5,
                   label=r'$D_{\rm like}$ k-nodes')
    ax.axvline(x=k_Nq, color='r', linestyle='--', lw=1, label=r'$k_{\rm Nyq}$')

    ax.set_xscale('log')
    ax.set_xlabel(r'$k~[h\,{\rm Mpc}^{-1}]$', fontsize=16)
    ax.set_ylabel(r'$D(k)$', fontsize=16)
    ax.set_title(r'Precision matrix diagonals: $Q = U^T D\, U$', fontsize=16)
    leg = ax.legend(loc='upper right', markerscale=8, fontsize=13)
    for lh in leg.legend_handles:
        if hasattr(lh, 'set_sizes') and len(lh.get_sizes()) > 0:
            if lh.get_sizes()[0] > 500:
                lh.set_sizes([30])
    ax.grid(alpha=0.15)
    fig.tight_layout()

    out = os.path.join(save_dir, f'6_Q_diagonals_{run_name}.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Precision matrix diagonal plot (fig 5)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser, default_num_samples=0)
    return parser.parse_args()


def main():
    args = parse_args()
    utils.configure_matplotlib_style(use_latex=args.use_latex)
    d = load_model_and_generate_samples(args)
    rf = d['rescaling_factor']
    plot_Q_diagonals(
        Q_like_D=d['Q_like_D']  / rf**2,
        Q_prior_D=d['Q_prior_D'] / rf**2,
        box=d['box'],
        Q_like_k_nodes=d['Q_like_k_nodes'],
        Q_like_D_nodes=d['Q_like_D_nodes'] / rf**2 if d['Q_like_D_nodes'] is not None else None,
        save_dir=d['plots_dir'],
        run_name=d['run_name'],
    )
    print(f"\nDone. Plot saved to {d['plots_dir']}/")


if __name__ == '__main__':
    main()
