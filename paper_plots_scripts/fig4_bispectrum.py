"""
Reduced bispectrum Q of posterior IC samples vs true ICs.

Produces one figure saved to paper_plots_scripts/{RUN_NAME}/:
  - 3_bispectrum_{RUN_NAME}.pdf : 3-panel plot:
      - Left:   isoceles  k₁ = k₂ = 0.1 h/Mpc, Q(θ)
      - Middle: squeezed  k₁ = 0.05, k₂ = 0.1 h/Mpc, Q(θ)
      - Right:  equilateral k₁ = k₂ = k₃ = k (θ = 120°), Q(k)
    Each panel shows the true field, MAP estimate, and sample mean ± 1σ / 2σ.

Note: bispectrum computation is expensive. Consider using fewer samples
(e.g. --num_samples 50) for quick iteration.

Usage:
    python paper_plots_scripts/fig4_bispectrum.py \\
        --model_dir paper_test_runs/runs/260303_224547_sweep_networks/260303_224627_net_IsotropicD

    python paper_plots_scripts/fig4_bispectrum.py \\
        --model_dir paper_test_runs/runs/260303_224547_sweep_networks/260303_224627_net_IsotropicD \\
        --num_samples 50 \\
        --no_latex
"""

import os
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
import Pk_library as PKL

from gaussian_npe import utils
from _common import add_common_args, load_model_and_generate_samples


# Module-level workers — must be at top level for ProcessPoolExecutor pickling
def _compute_sample_bk(args):
    """Compute reduced bispectrum Q(theta) for one sample."""
    s_i, box_size, k1, k2, theta, MAS = args
    BBk_i = PKL.Bk(s_i, box_size, k1, k2, theta, MAS, threads=1)
    return BBk_i.Q


def _compute_sample_bk_equilateral(args):
    """Compute Q(k) for equilateral triangles (k1=k2=k3=k) for one sample."""
    s_i, box_size, k_values, MAS = args
    theta_eq = np.array([2 * np.pi / 3])   # 120° — equilateral condition
    return np.array([
        PKL.Bk(s_i, box_size, k, k, theta_eq, MAS, threads=1).Q[0]
        for k in k_values
    ])


def plot_bispectrum(delta_z127, samples, z_MAP, box, MAS=None,
                    save_dir='./plots', run_name='', n_workers=None):
    """Plot reduced bispectrum in three panels.

    Parameters
    ----------
    delta_z127 : np.ndarray, shape (N, N, N)
        True IC field in physical units.
    samples : np.ndarray, shape (n_samples, N, N, N) or None
        Posterior samples. If None, MAP-only mode.
    z_MAP : np.ndarray, shape (N, N, N)
        MAP estimate.
    box : utils.Power_Spectrum_Sampler
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
    z_MAP      = z_MAP.astype('f')
    color_samples = 'forestgreen'

    has_samples = samples is not None and np.asarray(samples).size > 0
    if has_samples:
        samples = np.asarray(samples).astype('f')

    theta     = np.linspace(0, np.pi, 25)
    theta_deg = np.degrees(theta)

    # k values for equilateral panel: log-spaced from 2*k_F to 0.9*k_Nq
    k_eq = np.logspace(np.log10(2 * box.k_F), np.log10(0.9 * box.k_Nq), 15)
    theta_eq = np.array([2 * np.pi / 3])

    bispec_configs = [
        {'k1': 0.1,  'k2': 0.1,  'label': r'$k_1 = k_2 = 0.1\;h/\mathrm{Mpc}$'},
        {'k1': 0.05, 'k2': 0.1,  'label': r'$k_1 = 0.05,\; k_2 = 0.1\;h/\mathrm{Mpc}$'},
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # ── Panels 1 & 2: Q(θ) for fixed k configs ───────────────────────────
    for ax, cfg in zip(axes[:2], bispec_configs):
        Qk_true = PKL.Bk(delta_z127, box.box_size,
                         cfg['k1'], cfg['k2'], theta, MAS, threads=1).Q
        Qk_MAP  = PKL.Bk(z_MAP, box.box_size,
                         cfg['k1'], cfg['k2'], theta, MAS, threads=1).Q

        if has_samples:
            bk_args = [(samples[i], box.box_size, cfg['k1'], cfg['k2'], theta, MAS)
                       for i in range(len(samples))]
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                Qks = np.array(list(executor.map(_compute_sample_bk, bk_args)))

        ax.plot(theta_deg, Qk_true, marker='.', markersize=3, lw=2.5,
                color='mediumblue', label='True', zorder=10)
        if has_samples:
            Qks_mean, Qks_std = Qks.mean(0), Qks.std(0)
            ax.plot(theta_deg, Qks_mean, lw=2.5, color=color_samples, label='Samples')
            ax.fill_between(theta_deg,
                             Qks_mean - Qks_std, Qks_mean + Qks_std,
                             alpha=0.75, color=color_samples)
            ax.fill_between(theta_deg,
                             Qks_mean - 2 * Qks_std, Qks_mean + 2 * Qks_std,
                             alpha=0.25, color=color_samples)
        ax.plot(theta_deg, Qk_MAP, color='m', label='MAP', alpha=0.75, lw=2.5)

        ax.set_xlabel(r'$\theta$ [deg]', fontsize=14)
        ax.set_title(cfg['label'], fontsize=11)
        ax.set_ylim(-45, 45)
        ax.grid(alpha=0.15)
        ax.legend(facecolor='white', edgecolor='none', framealpha=0.8, fontsize=9)

    axes[0].set_ylabel(r'$Q(\theta)$', fontsize=16)

    # ── Panel 3: Q(k) for equilateral triangles (k1=k2=k3=k, θ=120°) ────
    ax = axes[2]

    Qk_eq_true = np.array([
        PKL.Bk(delta_z127, box.box_size, k, k, theta_eq, MAS, threads=1).Q[0]
        for k in k_eq
    ])
    Qk_eq_MAP = np.array([
        PKL.Bk(z_MAP, box.box_size, k, k, theta_eq, MAS, threads=1).Q[0]
        for k in k_eq
    ])

    if has_samples:
        bk_eq_args = [(samples[i], box.box_size, k_eq, MAS) for i in range(len(samples))]
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            Qks_eq = np.array(list(executor.map(_compute_sample_bk_equilateral, bk_eq_args)))

    ax.plot(k_eq, Qk_eq_true, marker='.', markersize=3, lw=2.5, color='mediumblue', label='True', zorder=10)
    if has_samples:
        Qks_eq_mean, Qks_eq_std = Qks_eq.mean(0), Qks_eq.std(0)
        ax.plot(k_eq, Qks_eq_mean, lw=2.5, color=color_samples, label='Samples')
        ax.fill_between(k_eq,
                         Qks_eq_mean - Qks_eq_std, Qks_eq_mean + Qks_eq_std,
                         alpha=0.75, color=color_samples)
        ax.fill_between(k_eq,
                         Qks_eq_mean - 2 * Qks_eq_std, Qks_eq_mean + 2 * Qks_eq_std,
                         alpha=0.25, color=color_samples)
    ax.plot(k_eq, Qk_eq_MAP, color='m', label='MAP', alpha=0.75, lw=2.5)

    ax.set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]', fontsize=14)
    ax.set_ylabel(r'$Q(k)$', fontsize=16)
    ax.set_title(r'Equilateral: $k_1 = k_2 = k_3 = k$', fontsize=11)
    ax.set_xscale('log')
    ax.grid(alpha=0.15)
    ax.legend(facecolor='white', edgecolor='none', framealpha=0.8, fontsize=9)

    fig.tight_layout()
    out = os.path.join(save_dir, f'3_bispectrum_{run_name}.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Reduced bispectrum Q plot (fig 4)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser, default_num_samples=50)
    return parser.parse_args()


def main():
    import time
    args = parse_args()
    utils.configure_matplotlib_style(use_latex=args.use_latex)
    d = load_model_and_generate_samples(args)
    rf = d['rescaling_factor']
    n = args.num_samples

    t0 = time.time()
    plot_bispectrum(
        delta_z127=d['delta_z127_int'] * rf,
        samples=d['samples_int'] * rf if d['samples_int'] is not None else None,
        z_MAP=d['z_MAP_int'] * rf,
        box=d['box'],
        MAS=d['MAS'],
        save_dir=d['plots_dir'],
        run_name=d['run_name'],
    )
    t_bk = time.time() - t0
    print(f'Bispectrum: {t_bk:.1f}s total  |  {t_bk/n:.2f}s per sample  |  '
          f'estimated for 1000 samples: {1000*t_bk/n:.0f}s ({1000*t_bk/n/60:.1f}min)')
    print(f"\nDone. Plot saved to {d['plots_dir']}/")


if __name__ == '__main__':
    main()
