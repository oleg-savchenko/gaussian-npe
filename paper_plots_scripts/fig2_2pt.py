"""
Summary statistics: P(k), T(k), and C(k) of posterior samples vs true ICs.

Produces one figure saved to paper_plots_scripts/{RUN_NAME}/:
  - 2_summary_stats_{RUN_NAME}.pdf : 3-panel plot with power spectrum P(k),
    transfer function T(k) = sqrt(P_sample / P_true), and cross-correlation
    C(k) between samples and the true IC field.

Usage:
    python paper_plots_scripts/fig2_spectra.py \\
        --model_dir paper_test_runs/runs/20260301_215801_net_IsotropicD

    python paper_plots_scripts/fig2_spectra.py \\
        --model_dir paper_test_runs/runs/20260301_215801_net_IsotropicD \\
        --num_samples 200 \\
        --no_latex
"""

import os
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
import Pk_library as PKL

from gaussian_npe import utils
from _common import COSMO_PARAMS, add_common_args, load_model_and_generate_samples


# Module-level worker — must be at top level for ProcessPoolExecutor pickling
def _compute_sample_pk(args):
    """Compute XPk (auto + normalised cross) for one sample vs truth."""
    s_i, delta_z127, box_size, MAS = args
    Pk_i  = PKL.XPk([s_i, delta_z127], box_size, axis=0, MAS=[MAS, MAS], threads=1)
    pk_i  = Pk_i.Pk[:, 0, 0]
    xpk_i = Pk_i.XPk[:, 0, 0] / np.sqrt(Pk_i.Pk[:, 0, 0] * Pk_i.Pk[:, 0, 1])
    return pk_i, xpk_i


def plot_summary_stats(delta_z127, delta_z0, samples, z_MAP, box,
                       cosmo_params=None, MAS=None, rescaling_factor=1.0,
                       save_dir='./plots', run_name='', n_workers=None):
    """Plot P(k), T(k), and C(k) summary statistics (fig 2 of plot_samples_analysis).

    Parameters
    ----------
    delta_z127 : np.ndarray, shape (N, N, N)
        True IC field in internal units.
    delta_z0 : np.ndarray, shape (N, N, N)
        Observed final-conditions field (physical units).
    samples : np.ndarray, shape (n_samples, N, N, N) or None
        Posterior samples in internal units.
    z_MAP : np.ndarray, shape (N, N, N)
        MAP estimate in internal units.
    box : utils.Power_Spectrum_Sampler
    cosmo_params : dict, optional
        CLASS cosmological parameters. If given, adds a linear P(k) line.
    MAS : str, optional
        Pylians mass-assignment scheme.
    save_dir : str
    run_name : str
    n_workers : int, optional
        Max workers for ProcessPoolExecutor.
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.rcParams['figure.facecolor'] = 'white'

    delta_z127 = delta_z127.astype('f')
    delta_z0   = delta_z0.astype('f')
    z_MAP      = z_MAP.astype('f')
    k_Nq       = box.k_Nq
    color_samples = 'forestgreen'

    has_samples = samples is not None and np.asarray(samples).size > 0
    if has_samples:
        samples = np.asarray(samples).astype('f')

    # MAP auto-spectrum and cross-spectra
    Pk_MAP     = PKL.XPk([z_MAP, delta_z127], box.box_size, axis=0,
                          MAS=[MAS, MAS], threads=1)
    k_pylians  = Pk_MAP.k3D
    pk_MAP     = Pk_MAP.Pk[:, 0, 0]
    pk_ic      = Pk_MAP.Pk[:, 0, 1]
    tk_MAP     = np.sqrt(pk_MAP / pk_ic)
    xpk_MAP    = Pk_MAP.XPk[:, 0, 0] / np.sqrt(pk_MAP * pk_ic)

    # Cross-correlation between IC and final field (linear baseline)
    Pk_lin     = PKL.XPk([delta_z127, delta_z0], box.box_size, axis=0,
                          MAS=[MAS, MAS], threads=1)
    xpk_linear = Pk_lin.XPk[:, 0, 0] / np.sqrt(Pk_lin.Pk[:, 0, 0] * Pk_lin.Pk[:, 0, 1])

    # Sample P(k) / T(k) / C(k) — parallelised
    if has_samples:
        pk_args = [(samples[i], delta_z127, box.box_size, MAS) for i in range(len(samples))]
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            pk_results = list(executor.map(_compute_sample_pk, pk_args))
        pks  = np.array([r[0] for r in pk_results])
        xpks = np.array([r[1] for r in pk_results])
        tks  = np.sqrt(pks / pk_ic)
        pks_mean,  pks_std  = pks.mean(0),  pks.std(0)
        tks_mean,  tks_std  = tks.mean(0),  tks.std(0)
        xpks_mean, xpks_std = xpks.mean(0), xpks.std(0)

    # Optional linear theory P(k) from CLASS
    if cosmo_params is not None:
        k_lin = np.logspace(np.log10(1e-4), np.log10(10), 100)
        pk_class_z0 = utils.get_pk_class(cosmo_params, 0, k_lin) * rescaling_factor**2

    # ── Figure 2: 3-panel summary ─────────────────────────────────────────
    fig, axs = plt.subplots(3, sharex=True, sharey=False, height_ratios=[2, 1, 1])
    fig.set_size_inches(4, 8)

    # P(k)
    ax = axs[0]
    ax.plot(k_pylians, pk_ic, marker='.', markersize=0.5, lw=0.5,
            color='mediumblue', label='True', zorder=10)
    if has_samples:
        ax.plot(k_pylians, pks_mean, lw=0.5, color=color_samples, label='Samples')
        ax.fill_between(k_pylians,
                         pks_mean - pks_std, pks_mean + pks_std,
                         alpha=0.75, color=color_samples)
        ax.fill_between(k_pylians,
                         pks_mean - 2 * pks_std, pks_mean + 2 * pks_std,
                         alpha=0.25, color=color_samples)
    ax.plot(k_pylians, pk_MAP, color='m', label='MAP', alpha=0.75, lw=0.5)
    if cosmo_params is not None:
        ax.plot(k_lin, pk_class_z0, label='Linear', color='black', alpha=0.3, lw=0.5)
    ax.axvline(x=k_Nq, color='r', ls='--', lw=0.5, alpha=0.5, label=r'$k_{\rm{Nyq}}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(r'$P(k)$', fontsize=16)
    ax.legend(facecolor='white', edgecolor='none', framealpha=0.8)
    ax.set_ylim([5e-2, 5])
    ax.set_xlim(left=k_pylians[0], right=k_Nq + 0.075)
    ax.grid(which='both', alpha=0.125)

    # T(k)
    ax = axs[1]
    if has_samples:
        ax.plot(k_pylians, tks_mean, color=color_samples)
        ax.fill_between(k_pylians,
                         tks_mean - tks_std, tks_mean + tks_std,
                         alpha=0.75, color=color_samples)
        ax.fill_between(k_pylians,
                         tks_mean - 2 * tks_std, tks_mean + 2 * tks_std,
                         alpha=0.25, color=color_samples)
    ax.plot(k_pylians, tk_MAP, color='m', alpha=0.75, lw=0.5)
    ax.axvline(x=k_Nq, color='r', ls='--', lw=0.5, alpha=0.5)
    ax.axhline(1.0, color='k', ls='--', lw=0.5)
    ax.set_xscale('log')
    ax.set_ylabel(r'$T(k)$', fontsize=16)
    ax.set_ylim(0.93, 1.07)
    ax.set_yticks([0.95, 1.0, 1.05])
    ax.grid(which='both', alpha=0.1)

    # C(k)
    ax = axs[2]
    ax.plot(k_pylians, xpk_MAP, color='m', alpha=0.75, lw=0.5)
    ax.plot(k_pylians, xpk_linear, alpha=0.75, lw=0.5, color='orange', label=r'$z=0$')
    if has_samples:
        ax.plot(k_pylians, xpks_mean, color=color_samples, lw=0.25)
        ax.fill_between(k_pylians,
                         xpks_mean - xpks_std, xpks_mean + xpks_std,
                         alpha=0.75, color=color_samples)
        ax.fill_between(k_pylians,
                         xpks_mean - 2 * xpks_std, xpks_mean + 2 * xpks_std,
                         alpha=0.25, color=color_samples)
    ax.axvline(x=k_Nq, color='r', ls='--', lw=0.5, alpha=0.5)
    ax.axhline(1.0, color='k', ls='--', lw=0.5)
    ax.set_xscale('log')
    ax.set_ylabel(r'$C(k)$', fontsize=16)
    ax.set_xlabel(r'$k$ [$h / \rm{Mpc}$]', fontsize=14)
    ax.set_ylim([-0.2, 1.2])
    ax.set_yticks([0, 0.5, 1.0])
    ax.grid(which='both', alpha=0.1)
    ax.legend(facecolor='white', edgecolor='none', framealpha=0.9, loc='lower left')

    plt.subplots_adjust(hspace=0)
    out = os.path.join(save_dir, f'2_summary_stats_{run_name}.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='P(k) / T(k) / C(k) summary statistics plot (fig 2)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser, default_num_samples=200)
    return parser.parse_args()


def main():
    import time
    args = parse_args()
    utils.configure_matplotlib_style(use_latex=args.use_latex)

    t0 = time.time()
    d = load_model_and_generate_samples(args)
    t_samples = time.time() - t0
    n = args.num_samples
    print(f'Sampling: {t_samples:.1f}s total  |  {t_samples/n:.2f}s per sample  |  '
          f'estimated for 1000 samples: {1000*t_samples/n:.0f}s ({1000*t_samples/n/60:.1f}min)')

    rf = d['rescaling_factor']
    plot_summary_stats(
        delta_z127=d['delta_z127_int'] * rf,
        delta_z0=d['delta_z0'],
        samples=d['samples_int'] * rf if d['samples_int'] is not None else None,
        z_MAP=d['z_MAP_int'] * rf,
        box=d['box'],
        cosmo_params=COSMO_PARAMS.copy(),
        MAS=d['MAS'],
        rescaling_factor=rf,
        save_dir=d['plots_dir'],
        run_name=d['run_name'],
    )
    print(f"\nDone. Plot saved to {d['plots_dir']}/")


if __name__ == '__main__':
    main()
