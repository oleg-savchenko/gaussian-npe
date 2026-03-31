"""
Individual-sample P(k) and T(k) diagnostic for re-simulated final fields.

Plots P(k) and T(k) of each individual re-simulated z=0 field as a thin
line, alongside the true z=0 field. No mean/sigma bands — the goal is to
see whether individual samples show a systematic P(k) bump or whether the
bump only appears in the ensemble mean.

Left column  — ICs vs true initial conditions:
    P(k), T(k) of individual IC posterior samples vs true delta_z127.

Right column — re-simulated z=0 vs true final field:
    P(k), T(k) of individual PPC re-simulated fields vs true delta_z0.

Output: {output_dir}/pk_individual_samples_{RUN_NAME}.pdf

Usage:
    python paper_plots_scripts/fig_pk_individual_samples.py \\
        --model_dir paper_test_runs/runs/260303_224547_sweep_networks/260303_224627_net_IsotropicD \\
        --num_samples 10 \\
        --samples_dir /gpfs/scratch1/shared/osavchenko/mnras_paper/samples/260303_224627_net_IsotropicD \\
        --output_dir paper_plots_scripts/260303_224627_net_IsotropicD

    # Use 20 PPC fields:
    python paper_plots_scripts/fig_pk_individual_samples.py \\
        --model_dir ... --samples_dir ... --num_ppc_samples 20 --num_samples 20
"""

import os
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
import Pk_library as PKL

from gaussian_npe import utils
from _common import COSMO_PARAMS, add_common_args, load_model_and_generate_samples


# ── Module-level workers ──────────────────────────────────────────────────────

def _compute_sample_pk(args):
    """Auto P(k) and normalised cross-correlation for one IC posterior sample."""
    s_i, delta_z127, box_size, MAS = args
    Pk_i  = PKL.XPk([s_i, delta_z127], box_size, axis=0, MAS=[MAS, MAS], threads=1)
    pk_i  = Pk_i.Pk[:, 0, 0]
    xpk_i = Pk_i.XPk[:, 0, 0] / np.sqrt(Pk_i.Pk[:, 0, 0] * Pk_i.Pk[:, 0, 1])
    return pk_i, xpk_i


def _compute_resim_pk(args):
    """Auto P(k) and normalised cross-correlation for one re-simulated field."""
    resim_i, delta_z0, box_size = args
    Pk_i  = PKL.XPk([resim_i, delta_z0], box_size, axis=0,
                    MAS=[None, None], threads=1)
    pk_i  = Pk_i.Pk[:, 0, 0]
    xpk_i = Pk_i.XPk[:, 0, 0] / np.sqrt(Pk_i.Pk[:, 0, 0] * Pk_i.Pk[:, 0, 1])
    return pk_i, xpk_i


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_individual_samples(delta_z127, delta_z0, samples, box,
                            resim_fields, MAS=None, rescaling_factor=1.0,
                            save_dir='./plots', run_name='', n_workers=None):
    """2×2 figure: P(k) and T(k) for individual samples (no mean/sigma bands).

    Parameters
    ----------
    delta_z127   : np.ndarray (N,N,N)  True IC field, physical units.
    delta_z0     : np.ndarray (N,N,N)  True z=0 density contrast.
    samples      : np.ndarray (B,N,N,N) or None  IC posterior samples, physical.
    box          : utils.Power_Spectrum_Sampler
    resim_fields : list of np.ndarray (N,N,N)  PPC re-simulated z=0 fields.
    MAS          : str or None   Pylians mass-assignment scheme (IC fields).
    rescaling_factor : float
    save_dir     : str
    run_name     : str
    n_workers    : int or None
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.rcParams['figure.facecolor'] = 'white'

    delta_z127 = delta_z127.astype('f')
    delta_z0   = delta_z0.astype('f')
    k_Nq       = box.k_Nq
    color_ic   = 'forestgreen'
    color_rs   = 'mediumpurple'

    has_samples = samples is not None and np.asarray(samples).size > 0
    if has_samples:
        samples = np.asarray(samples).astype('f')

    # ── IC computations ───────────────────────────────────────────────────────
    Pk_true_ic = PKL.Pk(delta_z127, box.box_size, axis=0, MAS=MAS, threads=1)
    k_ic   = Pk_true_ic.k3D
    pk_ic  = Pk_true_ic.Pk[:, 0]

    if has_samples:
        ic_args = [(samples[i], delta_z127, box.box_size, MAS)
                   for i in range(len(samples))]
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            ic_res = list(ex.map(_compute_sample_pk, ic_args))
        ic_pks  = np.array([r[0] for r in ic_res])
        ic_tks  = np.sqrt(ic_pks / pk_ic)
        ic_mean_pk = ic_pks.mean(0)
        ic_mean_tk = ic_tks.mean(0)

    # ── Re-simulation computations ────────────────────────────────────────────
    Pk_true_ff = PKL.Pk(delta_z0, box.box_size, axis=0, MAS=None, threads=1)
    k_ff    = Pk_true_ff.k3D
    pk_true = Pk_true_ff.Pk[:, 0]

    rs_args = [(r.astype('f'), delta_z0, box.box_size) for r in resim_fields]
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        rs_res = list(ex.map(_compute_resim_pk, rs_args))
    rs_pks  = np.array([r[0] for r in rs_res])
    rs_tks  = np.sqrt(rs_pks / pk_true)
    rs_mean_pk = rs_pks.mean(0)
    rs_mean_tk = rs_tks.mean(0)

    n_rs = len(resim_fields)
    n_ic = len(samples) if has_samples else 0

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        2, 2, figsize=(10, 6),
        gridspec_kw={'height_ratios': [2, 1], 'hspace': 0, 'wspace': 0.15},
    )
    axes[1, 0].sharex(axes[0, 0])
    axes[1, 1].sharex(axes[0, 1])
    axes[0, 0].tick_params(axis='x', which='both', labelbottom=False)
    axes[0, 1].tick_params(axis='x', which='both', labelbottom=False)

    xlim_ic = (k_ic[0], k_Nq + 0.075)
    xlim_ff = (k_ff[0], k_Nq + 0.075)

    sample_alpha = max(0.05, min(0.4, 3.0 / max(n_rs, n_ic, 1)))

    # ── Left: ICs ─────────────────────────────────────────────────────────────

    # [0,0] P(k) IC
    ax = axes[0, 0]
    ax.set_title(r'Initial conditions ($z=127$)', fontsize=13)
    if has_samples:
        for pk_i in ic_pks:
            ax.plot(k_ic, pk_i, lw=0.4, color=color_ic, alpha=sample_alpha)
        ax.plot(k_ic, ic_mean_pk, lw=1.2, color=color_ic, label=f'Samples mean (n={n_ic})')
    ax.plot(k_ic, pk_ic, lw=1.2, color='mediumblue', label='True', zorder=10)
    ax.axvline(x=k_Nq, color='r', ls='--', lw=0.7, alpha=0.6, label=r'$k_{\rm{Nyq}}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(r'$P(k)$', fontsize=16)
    ax.legend(facecolor='white', edgecolor='none', framealpha=0.8, fontsize=10)
    ax.set_xlim(*xlim_ic)
    ax.grid(which='both', alpha=0.125)

    # [1,0] T(k) IC
    ax = axes[1, 0]
    if has_samples:
        for tk_i in ic_tks:
            ax.plot(k_ic, tk_i, lw=0.4, color=color_ic, alpha=sample_alpha)
        ax.plot(k_ic, ic_mean_tk, lw=1.2, color=color_ic)
    ax.axvline(x=k_Nq, color='r', ls='--', lw=0.7, alpha=0.6)
    ax.axhline(1.0, color='k', ls='--', lw=0.7)
    ax.set_xscale('log')
    ax.set_ylabel(r'$T(k)$', fontsize=16)
    ax.set_xlabel(r'$k$ [$h / \rm{Mpc}$]', fontsize=14)
    ax.set_ylim(0.90, 1.10)
    ax.set_yticks([0.95, 1.0, 1.05])
    ax.grid(which='both', alpha=0.1)

    # ── Right: re-simulated final fields ──────────────────────────────────────

    # [0,1] P(k) final
    ax = axes[0, 1]
    ax.set_title(r'Final conditions ($z=0$)', fontsize=13)
    for pk_i in rs_pks:
        ax.plot(k_ff, pk_i, lw=0.4, color=color_rs, alpha=sample_alpha)
    ax.plot(k_ff, rs_mean_pk, lw=1.2, color=color_rs, label=f'Samples mean (n={n_rs})')
    ax.plot(k_ff, pk_true, lw=1.2, color='mediumblue', label='True', zorder=10)
    ax.axvline(x=k_Nq, color='r', ls='--', lw=0.7, alpha=0.6, label=r'$k_{\rm{Nyq}}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(facecolor='white', edgecolor='none', framealpha=0.8, fontsize=10)
    ax.set_xlim(*xlim_ff)
    ax.grid(which='both', alpha=0.125)

    # [1,1] T(k) final
    ax = axes[1, 1]
    for tk_i in rs_tks:
        ax.plot(k_ff, tk_i, lw=0.4, color=color_rs, alpha=sample_alpha)
    ax.plot(k_ff, rs_mean_tk, lw=1.2, color=color_rs)
    ax.axhline(1.0, color='k', ls='--', lw=0.7)
    ax.axvline(x=k_Nq, color='r', ls='--', lw=0.7, alpha=0.6)
    ax.set_xscale('log')
    ax.set_ylabel(r'$T(k)$', fontsize=16)
    ax.set_xlabel(r'$k$ [$h / \rm{Mpc}$]', fontsize=14)
    ax.set_ylim(0.90, 1.10)
    ax.set_yticks([0.95, 1.0, 1.05])
    ax.grid(which='both', alpha=0.1)

    out = os.path.join(save_dir, f'pk_individual_samples_{run_name}.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Individual-sample P(k)/T(k) diagnostic (no mean/sigma bands)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser, default_num_samples=10)
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

    t0 = time.time()
    d  = load_model_and_generate_samples(args)
    print(f'IC sampling: {time.time()-t0:.1f}s')

    rf       = d['rescaling_factor']
    run_name = d['run_name']

    # ── PPC re-simulated fields ───────────────────────────────────────────────
    samples_dir  = os.path.abspath(args.samples_dir)
    fourier_dirs = sorted([
        os.path.join(samples_dir, dn)
        for dn in os.listdir(samples_dir)
        if dn.startswith('sample_')
        and os.path.isfile(os.path.join(samples_dir, dn, 'emu_delta_z0.npy'))
    ])
    if not fourier_dirs:
        raise FileNotFoundError(
            f'No sample_XXXX/ dirs with emu_delta_z0.npy in {samples_dir}')
    if args.num_ppc_samples is not None:
        fourier_dirs = fourier_dirs[:args.num_ppc_samples]
    print(f'Loading {len(fourier_dirs)} re-simulated fields from {samples_dir}')
    resim_fields = [np.load(os.path.join(dn, 'emu_delta_z0.npy')) for dn in fourier_dirs]

    save_dir = os.path.abspath(args.output_dir) if args.output_dir else d['plots_dir']

    plot_individual_samples(
        delta_z127=d['delta_z127_int'] * rf,
        delta_z0=d['delta_z0'],
        samples=d['samples_int'] * rf if d['samples_int'] is not None else None,
        box=d['box'],
        resim_fields=resim_fields,
        MAS=d['MAS'],
        rescaling_factor=rf,
        save_dir=save_dir,
        run_name=run_name,
    )
    print(f'\nDone. Plot saved to {save_dir}/')


if __name__ == '__main__':
    main()
