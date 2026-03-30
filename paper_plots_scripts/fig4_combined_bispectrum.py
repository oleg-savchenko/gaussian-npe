"""
Combined reduced bispectrum Q: IC posterior (top row) and re-simulated final
fields (bottom row) in a single 2×3 figure.

Top row    — ICs vs true initial conditions:
    Q(θ) isosceles, Q(θ) squeezed, Q(k) equilateral of posterior IC samples
    and MAP vs true delta_z127.

Bottom row — re-simulated z=0 vs true final field:
    Same configurations for PPC re-simulated z=0 fields vs true delta_z0.

Output: paper_plots_scripts/{RUN_NAME}/4_combined_bispectrum_{RUN_NAME}.pdf

Usage:
    python paper_plots_scripts/fig4_combined_bispectrum.py \\
        --model_dir paper_test_runs/runs/260303_224547_sweep_networks/260303_224627_net_IsotropicD \\
        --num_samples 50 \\
        --samples_dir /gpfs/scratch1/shared/osavchenko/mnras_paper/samples/260303_224627_net_IsotropicD \\
        --output_dir paper_plots_scripts/260303_224627_net_IsotropicD

    # Limit PPC to 5 re-simulations:
    python paper_plots_scripts/fig4_combined_bispectrum.py \\
        --model_dir ... --samples_dir ... --num_ppc_samples 5

Note: bispectrum computation is expensive. Consider --num_samples 50 and
--num_ppc_samples 10 for quick iteration.
"""

import os
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
import Pk_library as PKL

from gaussian_npe import utils
from _common import add_common_args, load_model_and_generate_samples


# ── Module-level workers (must be top-level for ProcessPoolExecutor pickling) ─

def _compute_sample_bk(args):
    """Compute reduced bispectrum Q(theta) for one field."""
    s_i, box_size, k1, k2, theta, MAS = args
    BBk_i = PKL.Bk(s_i, box_size, k1, k2, theta, MAS, threads=1)
    return BBk_i.Q


def _compute_sample_bk_equilateral(args):
    """Compute Q(k) for equilateral triangles (k1=k2=k3=k) for one field."""
    s_i, box_size, k_values, MAS = args
    theta_eq = np.array([2 * np.pi / 3])   # 120° — equilateral condition
    return np.array([
        PKL.Bk(s_i, box_size, k, k, theta_eq, MAS, threads=1).Q[0]
        for k in k_values
    ])


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_combined_bispectrum(delta_z127, delta_z0, samples, z_MAP,
                              resim_fields, box, MAS=None,
                              rescaling_factor=1.0,
                              save_dir='./plots', run_name='', n_workers=None):
    """2×3 combined bispectrum figure.

    Parameters
    ----------
    delta_z127   : np.ndarray (N,N,N)  True IC field, physical units.
    delta_z0     : np.ndarray (N,N,N)  True z=0 density contrast.
    samples      : np.ndarray (B,N,N,N) or None  IC posterior samples, physical.
    z_MAP        : np.ndarray (N,N,N)  MAP IC estimate, physical units.
    resim_fields : list of np.ndarray (N,N,N)  PPC re-simulated z=0 fields.
    box          : utils.Power_Spectrum_Sampler
    rescaling_factor : float
        Unused — kept for API consistency. Bottom-row Q ylim uses autoscale
        because z=0 Q values (~[0.5, 2]) are set by non-linear structure
        formation, not by field amplitude rescaling.
    MAS          : str or None  Pylians mass-assignment scheme (IC fields).
    save_dir     : str
    run_name     : str
    n_workers    : int or None
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.rcParams['figure.facecolor'] = 'white'

    delta_z127 = delta_z127.astype('f')
    delta_z0   = delta_z0.astype('f')
    z_MAP      = z_MAP.astype('f')
    color_ic   = 'forestgreen'
    color_rs   = 'mediumpurple'

    has_samples = samples is not None and np.asarray(samples).size > 0
    if has_samples:
        samples = np.asarray(samples).astype('f')

    theta     = np.linspace(0, np.pi, 25)
    theta_deg = np.degrees(theta)

    k_eq     = np.logspace(np.log10(2 * box.k_F), np.log10(0.9 * box.k_Nq), 15)
    theta_eq = np.array([2 * np.pi / 3])

    bispec_configs = [
        {'k1': 0.1,  'k2': 0.1,  'label': r'$k_1 = k_2 = 0.1\;h/\mathrm{Mpc}$'},
        {'k1': 0.05, 'k2': 0.1,  'label': r'$k_1 = 0.05,\; k_2 = 0.1\;h/\mathrm{Mpc}$'},
    ]

    ylim_ic = (-45, 45)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Share x between rows for θ columns; hide top-row tick labels
    for col in [0, 1]:
        axes[1, col].sharex(axes[0, col])
        axes[0, col].tick_params(axis='x', which='both', labelbottom=False)
    axes[1, 2].sharex(axes[0, 2])
    axes[0, 2].tick_params(axis='x', which='both', labelbottom=False)

    # ── Top row: IC posterior ─────────────────────────────────────────────────

    for col, cfg in enumerate(bispec_configs):
        ax = axes[0, col]
        Qk_true = PKL.Bk(delta_z127, box.box_size,
                         cfg['k1'], cfg['k2'], theta, MAS, threads=1).Q
        Qk_MAP  = PKL.Bk(z_MAP, box.box_size,
                         cfg['k1'], cfg['k2'], theta, MAS, threads=1).Q

        if has_samples:
            bk_args = [(samples[i], box.box_size, cfg['k1'], cfg['k2'], theta, MAS)
                       for i in range(len(samples))]
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                Qks = np.array(list(executor.map(_compute_sample_bk, bk_args)))
            Qks_mean, Qks_std = Qks.mean(0), Qks.std(0)
            ax.plot(theta_deg, Qks_mean, lw=2.5, color=color_ic, label='Samples')
            ax.fill_between(theta_deg,
                            Qks_mean - Qks_std, Qks_mean + Qks_std,
                            alpha=0.75, color=color_ic)
            ax.fill_between(theta_deg,
                            Qks_mean - 2 * Qks_std, Qks_mean + 2 * Qks_std,
                            alpha=0.25, color=color_ic)

        ax.plot(theta_deg, Qk_true, marker='.', markersize=3, lw=2.5,
                color='mediumblue', label='True', zorder=10)
        ax.plot(theta_deg, Qk_MAP, color='m', label='MAP', alpha=0.75, lw=2.5)

        ax.set_title(cfg['label'], fontsize=11)
        ax.set_ylim(*ylim_ic)
        ax.grid(alpha=0.15)
        ax.legend(facecolor='white', edgecolor='none', framealpha=0.8, fontsize=9)

    axes[0, 0].set_ylabel(r'$Q(\theta)$', fontsize=16)

    # Top row, equilateral panel
    ax = axes[0, 2]
    Qk_eq_true = np.array([
        PKL.Bk(delta_z127, box.box_size, k, k, theta_eq, MAS, threads=1).Q[0]
        for k in k_eq
    ])
    Qk_eq_MAP = np.array([
        PKL.Bk(z_MAP, box.box_size, k, k, theta_eq, MAS, threads=1).Q[0]
        for k in k_eq
    ])

    if has_samples:
        bk_eq_args = [(samples[i], box.box_size, k_eq, MAS)
                      for i in range(len(samples))]
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            Qks_eq = np.array(list(executor.map(_compute_sample_bk_equilateral, bk_eq_args)))
        Qks_eq_mean, Qks_eq_std = Qks_eq.mean(0), Qks_eq.std(0)
        ax.plot(k_eq, Qks_eq_mean, lw=2.5, color=color_ic, label='Samples')
        ax.fill_between(k_eq,
                        Qks_eq_mean - Qks_eq_std, Qks_eq_mean + Qks_eq_std,
                        alpha=0.75, color=color_ic)
        ax.fill_between(k_eq,
                        Qks_eq_mean - 2 * Qks_eq_std, Qks_eq_mean + 2 * Qks_eq_std,
                        alpha=0.25, color=color_ic)

    ax.plot(k_eq, Qk_eq_true, marker='.', markersize=3, lw=2.5,
            color='mediumblue', label='True', zorder=10)
    ax.plot(k_eq, Qk_eq_MAP, color='m', label='MAP', alpha=0.75, lw=2.5)

    ax.set_title(r'Equilateral: $k_1 = k_2 = k_3 = k$', fontsize=11)
    ax.set_ylabel(r'$Q(k)$', fontsize=16)
    ax.set_xscale('log')
    ax.grid(alpha=0.15)
    ax.legend(facecolor='white', edgecolor='none', framealpha=0.8, fontsize=9)

    # ── Bottom row: re-simulated final fields ─────────────────────────────────

    resim_fields = [r.astype('f') for r in resim_fields]

    for col, cfg in enumerate(bispec_configs):
        ax = axes[1, col]
        Qk_true_ff = PKL.Bk(delta_z0, box.box_size,
                             cfg['k1'], cfg['k2'], theta, None, threads=1).Q

        rs_bk_args = [(resim_fields[i], box.box_size, cfg['k1'], cfg['k2'], theta, None)
                      for i in range(len(resim_fields))]
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            Qrs = np.array(list(executor.map(_compute_sample_bk, rs_bk_args)))
        Qrs_mean, Qrs_std = Qrs.mean(0), Qrs.std(0)

        ax.plot(theta_deg, Qk_true_ff, marker='.', markersize=3, lw=2.5,
                color='mediumblue', label='True', zorder=10)
        ax.plot(theta_deg, Qrs_mean, lw=2.5, color=color_rs, label='Re-simulated')
        ax.fill_between(theta_deg,
                        Qrs_mean - Qrs_std, Qrs_mean + Qrs_std,
                        alpha=0.75, color=color_rs)
        ax.fill_between(theta_deg,
                        Qrs_mean - 2 * Qrs_std, Qrs_mean + 2 * Qrs_std,
                        alpha=0.25, color=color_rs)

        ax.set_xlabel(r'$\theta$ [deg]', fontsize=14)
        pass  # autoscale — z=0 Q driven by non-linear structure formation
        ax.grid(alpha=0.15)
        ax.legend(facecolor='white', edgecolor='none', framealpha=0.8, fontsize=9)

    axes[1, 0].set_ylabel(r'$Q(\theta)$', fontsize=16)

    # Bottom row, equilateral panel
    ax = axes[1, 2]
    Qk_eq_true_ff = np.array([
        PKL.Bk(delta_z0, box.box_size, k, k, theta_eq, None, threads=1).Q[0]
        for k in k_eq
    ])

    rs_bk_eq_args = [(resim_fields[i], box.box_size, k_eq, None)
                     for i in range(len(resim_fields))]
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        Qrs_eq = np.array(list(executor.map(_compute_sample_bk_equilateral, rs_bk_eq_args)))
    Qrs_eq_mean, Qrs_eq_std = Qrs_eq.mean(0), Qrs_eq.std(0)

    ax.plot(k_eq, Qk_eq_true_ff, marker='.', markersize=3, lw=2.5,
            color='mediumblue', label='True', zorder=10)
    ax.plot(k_eq, Qrs_eq_mean, lw=2.5, color=color_rs, label='Re-simulated')
    ax.fill_between(k_eq,
                    Qrs_eq_mean - Qrs_eq_std, Qrs_eq_mean + Qrs_eq_std,
                    alpha=0.75, color=color_rs)
    ax.fill_between(k_eq,
                    Qrs_eq_mean - 2 * Qrs_eq_std, Qrs_eq_mean + 2 * Qrs_eq_std,
                    alpha=0.25, color=color_rs)

    ax.set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]', fontsize=14)
    ax.set_ylabel(r'$Q(k)$', fontsize=16)
    ax.set_xscale('log')
    ax.grid(alpha=0.15)
    ax.legend(facecolor='white', edgecolor='none', framealpha=0.8, fontsize=9)

    # Row labels via figure text on the left
    fig.text(0.01, 0.75, r'Initial conditions ($z=127$)',
             va='center', ha='left', fontsize=18, rotation=90)
    fig.text(0.01, 0.27, r'Final conditions ($z=0$)',
             va='center', ha='left', fontsize=18, rotation=90)

    fig.tight_layout(pad=2.0, rect=[0.03, 0, 1, 1])
    out = os.path.join(save_dir, f'4_combined_bispectrum_{run_name}.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Combined IC + re-simulated bispectrum (fig 4 combined)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser, default_num_samples=50)
    parser.add_argument(
        '--samples_dir', type=str, required=True,
        help='Directory containing sample_XXXX/ subdirs with emu_delta_z0.npy.',
    )
    parser.add_argument(
        '--num_ppc_samples', type=int, default=None,
        help='Max number of sample_XXXX/ dirs to load. Default: all.',
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Where to save the PDF. Defaults to paper_plots_scripts/{run_name}/.',
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import time
    args = parse_args()
    utils.configure_matplotlib_style(use_latex=args.use_latex)

    d  = load_model_and_generate_samples(args)
    rf = d['rescaling_factor']

    # ── Discover re-simulated fields ──────────────────────────────────────
    samples_dir  = os.path.abspath(args.samples_dir)
    fourier_dirs = sorted([
        os.path.join(samples_dir, entry)
        for entry in os.listdir(samples_dir)
        if entry.startswith('sample_')
        and os.path.isfile(os.path.join(samples_dir, entry, 'emu_delta_z0.npy'))
    ])
    if not fourier_dirs:
        raise FileNotFoundError(
            f'No sample_XXXX/ dirs with emu_delta_z0.npy found in {samples_dir}')
    if args.num_ppc_samples is not None:
        fourier_dirs = fourier_dirs[:args.num_ppc_samples]
    print(f'Loading {len(fourier_dirs)} re-simulated fields from {samples_dir}')
    resim_fields = [np.load(os.path.join(fd, 'emu_delta_z0.npy')) for fd in fourier_dirs]

    # ── Output directory ──────────────────────────────────────────────────
    if args.output_dir:
        save_dir = os.path.abspath(args.output_dir)
    else:
        save_dir = d['plots_dir']

    t0 = time.time()
    plot_combined_bispectrum(
        delta_z127=d['delta_z127_int'] * rf,
        delta_z0=d['delta_z0'],
        samples=d['samples_int'] * rf if d['samples_int'] is not None else None,
        z_MAP=d['z_MAP_int'] * rf,
        resim_fields=resim_fields,
        box=d['box'],
        MAS=d['MAS'],
        rescaling_factor=rf,
        save_dir=save_dir,
        run_name=d['run_name'],
    )
    t_bk = time.time() - t0
    n_ic  = args.num_samples
    n_ppc = len(fourier_dirs)
    print(f'Bispectrum: {t_bk:.1f}s total  |  IC: {n_ic} samples, PPC: {n_ppc} fields')
    print(f'\nDone. Plot saved to {save_dir}/')


if __name__ == '__main__':
    main()
