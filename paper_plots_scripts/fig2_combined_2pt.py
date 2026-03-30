"""
Combined two-point statistics: IC posterior samples (left) and re-simulated
final fields (right) in a single 3×2 figure.

Left column  — ICs vs true initial conditions:
    P(k), T(k), C(k) of posterior IC samples and MAP vs true delta_z127.

Right column — re-simulated z=0 vs true final field:
    P(k), T(k), C(k) of PPC re-simulated fields vs true delta_z0.
    Also shows a Gaussian noise P(k) reference (sigma_noise from config.json)
    and CLASS linear + HaloFit theory curves.

Output: paper_plots_scripts/{RUN_NAME}/2_combined_2pt_{RUN_NAME}.pdf

Usage:
    python paper_plots_scripts/fig2_combined_2pt.py \\
        --model_dir paper_test_runs/runs/260303_224547_sweep_networks/260303_224627_net_IsotropicD \\
        --num_samples 100 \\
        --samples_dir /gpfs/scratch1/shared/osavchenko/mnras_paper/samples/260303_224627_net_IsotropicD \\
        --output_dir paper_plots_scripts/260303_224627_net_IsotropicD

    # Limit PPC to 5 re-simulations:
    python paper_plots_scripts/fig2_combined_2pt.py \\
        --model_dir ... --samples_dir ... --num_ppc_samples 5
"""

import os
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
import Pk_library as PKL

from gaussian_npe import utils
from _common import COSMO_PARAMS, add_common_args, load_model_and_generate_samples


# ── Module-level workers (must be top-level for ProcessPoolExecutor pickling) ─

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

def plot_combined_2pt(delta_z127, delta_z0, samples, z_MAP, box,
                      resim_fields, cosmo_params=None, MAS=None,
                      rescaling_factor=1.0, sigma_noise=1.0, noise_seed=42,
                      save_dir='./plots', run_name='', n_workers=None):
    """3×2 combined two-point statistics figure.

    Parameters
    ----------
    delta_z127   : np.ndarray (N,N,N)  True IC field, physical units.
    delta_z0     : np.ndarray (N,N,N)  True z=0 density contrast.
    samples      : np.ndarray (B,N,N,N) or None  IC posterior samples, physical.
    z_MAP        : np.ndarray (N,N,N)  MAP IC estimate, physical units.
    box          : utils.Power_Spectrum_Sampler
    resim_fields : list of np.ndarray (N,N,N)  PPC re-simulated z=0 fields.
    cosmo_params : dict or None  CLASS parameters.
    MAS          : str or None   Pylians mass-assignment scheme (IC fields).
    rescaling_factor : float
    sigma_noise  : float  Noise amplitude for reference P(k) on right panel.
    noise_seed   : int
    save_dir     : str
    run_name     : str
    n_workers    : int or None
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.rcParams['figure.facecolor'] = 'white'

    delta_z127 = delta_z127.astype('f')
    delta_z0   = delta_z0.astype('f')
    z_MAP      = z_MAP.astype('f')
    k_Nq       = box.k_Nq
    color_ic   = 'forestgreen'
    color_rs   = 'mediumpurple'

    has_samples = samples is not None and np.asarray(samples).size > 0
    if has_samples:
        samples = np.asarray(samples).astype('f')

    # ── IC computations (left column) ─────────────────────────────────────
    Pk_MAP    = PKL.XPk([z_MAP, delta_z127], box.box_size, axis=0,
                        MAS=[MAS, MAS], threads=1)
    k_ic      = Pk_MAP.k3D
    pk_MAP    = Pk_MAP.Pk[:, 0, 0]
    pk_ic     = Pk_MAP.Pk[:, 0, 1]
    tk_MAP    = np.sqrt(pk_MAP / pk_ic)
    xpk_MAP   = Pk_MAP.XPk[:, 0, 0] / np.sqrt(pk_MAP * pk_ic)

    Pk_lin    = PKL.XPk([delta_z127, delta_z0], box.box_size, axis=0,
                        MAS=[MAS, MAS], threads=1)
    xpk_linear = Pk_lin.XPk[:, 0, 0] / np.sqrt(
        Pk_lin.Pk[:, 0, 0] * Pk_lin.Pk[:, 0, 1])

    if has_samples:
        ic_args = [(samples[i], delta_z127, box.box_size, MAS)
                   for i in range(len(samples))]
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            ic_res = list(ex.map(_compute_sample_pk, ic_args))
        ic_pks  = np.array([r[0] for r in ic_res])
        ic_xpks = np.array([r[1] for r in ic_res])
        ic_tks  = np.sqrt(ic_pks / pk_ic)
        ic_pks_mean,  ic_pks_std  = ic_pks.mean(0),  ic_pks.std(0)
        ic_tks_mean,  ic_tks_std  = ic_tks.mean(0),  ic_tks.std(0)
        ic_xpks_mean, ic_xpks_std = ic_xpks.mean(0), ic_xpks.std(0)

    if cosmo_params is not None:
        k_lin       = np.logspace(np.log10(1e-4), np.log10(10), 100)
        pk_class_ic = utils.get_pk_class(cosmo_params, 0, k_lin) * rescaling_factor**2

    # ── Re-simulation computations (right column) ──────────────────────────
    Pk_true   = PKL.Pk(delta_z0, box.box_size, axis=0, MAS=None, threads=1)
    k_ff      = Pk_true.k3D
    pk_true   = Pk_true.Pk[:, 0]

    rs_args = [(r.astype('f'), delta_z0, box.box_size) for r in resim_fields]
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        rs_res = list(ex.map(_compute_resim_pk, rs_args))
    rs_pks  = np.array([r[0] for r in rs_res])
    rs_xpks = np.array([r[1] for r in rs_res])
    rs_tks  = np.sqrt(rs_pks / pk_true)
    rs_pks_mean,  rs_pks_std  = rs_pks.mean(0),  rs_pks.std(0)
    rs_tks_mean,  rs_tks_std  = rs_tks.mean(0),  rs_tks.std(0)
    rs_xpks_mean, rs_xpks_std = rs_xpks.mean(0), rs_xpks.std(0)

    noise_field = np.random.default_rng(noise_seed).normal(
        0, sigma_noise, delta_z0.shape).astype('f')
    Pk_noise = PKL.Pk(noise_field, box.box_size, axis=0, MAS=None, threads=1)
    pk_noise = Pk_noise.Pk[:, 0]
    k_noise  = Pk_noise.k3D

    if cosmo_params is not None:
        k_class      = np.logspace(np.log10(k_ff[0]), np.log10(k_Nq), 500)
        pk_class_lin = utils.get_pk_class(cosmo_params, 0, k_class, non_lin=False)
        pk_class_nl  = utils.get_pk_class(cosmo_params, 0, k_class, non_lin=True)

    # ── Figure ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        3, 2, figsize=(8, 8),
        gridspec_kw={'height_ratios': [2, 1, 1], 'hspace': 0, 'wspace': 0.15},
    )
    for row in [1, 2]:
        axes[row, 0].sharex(axes[0, 0])
        axes[row, 1].sharex(axes[0, 1])
    for col in [0, 1]:
        for row in [0, 1]:
            axes[row, col].tick_params(axis='x', which='both', labelbottom=False)

    xlim_ic = (k_ic[0], k_Nq + 0.075)
    xlim_ff = (k_ff[0], k_Nq + 0.075)

    # ── Left column: ICs ──────────────────────────────────────────────────

    # [0,0] P(k) IC
    ax = axes[0, 0]
    ax.set_title(r'Initial conditions ($z=127$)', fontsize=13)
    ax.plot(k_ic, pk_ic, marker='.', markersize=0.5, lw=0.5,
            color='mediumblue', label='True', zorder=10)
    if has_samples:
        ax.plot(k_ic, ic_pks_mean, lw=0.5, color=color_ic, label='Samples')
        ax.fill_between(k_ic, ic_pks_mean - ic_pks_std, ic_pks_mean + ic_pks_std,
                        alpha=0.75, color=color_ic)
        ax.fill_between(k_ic, ic_pks_mean - 2*ic_pks_std, ic_pks_mean + 2*ic_pks_std,
                        alpha=0.25, color=color_ic)
    ax.plot(k_ic, pk_MAP, color='m', label='MAP', alpha=0.75, lw=0.5)
    if cosmo_params is not None:
        ax.plot(k_lin, pk_class_ic, color='black', alpha=0.3, lw=0.5, label='Linear')
    ax.axvline(x=k_Nq, color='r', ls='--', lw=0.5, alpha=0.5, label=r'$k_{\rm{Nyq}}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(r'$P(k)$', fontsize=16)
    ax.legend(facecolor='white', edgecolor='none', framealpha=0.8)
    ylim_ic_pk = (5e-2, 5)
    ax.set_ylim(ylim_ic_pk)
    ax.set_xlim(*xlim_ic)
    ax.grid(which='both', alpha=0.125)

    # [1,0] T(k) IC
    ax = axes[1, 0]
    if has_samples:
        ax.plot(k_ic, ic_tks_mean, color=color_ic)
        ax.fill_between(k_ic, ic_tks_mean - ic_tks_std, ic_tks_mean + ic_tks_std,
                        alpha=0.75, color=color_ic)
        ax.fill_between(k_ic, ic_tks_mean - 2*ic_tks_std, ic_tks_mean + 2*ic_tks_std,
                        alpha=0.25, color=color_ic)
    ax.plot(k_ic, tk_MAP, color='m', alpha=0.75, lw=0.5)
    ax.axvline(x=k_Nq, color='r', ls='--', lw=0.5, alpha=0.5)
    ax.axhline(1.0, color='k', ls='--', lw=0.5)
    ax.set_xscale('log')
    ax.set_ylabel(r'$T(k)$', fontsize=16)
    ax.set_ylim(0.93, 1.07)
    ax.set_yticks([0.95, 1.0, 1.05])
    ax.grid(which='both', alpha=0.1)

    # [2,0] C(k) IC
    ax = axes[2, 0]
    ax.plot(k_ic, xpk_MAP, color='m', alpha=0.75, lw=0.5)
    ax.plot(k_ic, xpk_linear, alpha=0.75, lw=0.5, color='orange', label=r'$z=0$')
    if has_samples:
        ax.plot(k_ic, ic_xpks_mean, color=color_ic, lw=0.25)
        ax.fill_between(k_ic, ic_xpks_mean - ic_xpks_std, ic_xpks_mean + ic_xpks_std,
                        alpha=0.75, color=color_ic)
        ax.fill_between(k_ic, ic_xpks_mean - 2*ic_xpks_std, ic_xpks_mean + 2*ic_xpks_std,
                        alpha=0.25, color=color_ic)
    ax.axvline(x=k_Nq, color='r', ls='--', lw=0.5, alpha=0.5)
    ax.axhline(1.0, color='k', ls='--', lw=0.5)
    ax.set_xscale('log')
    ax.set_ylabel(r'$C(k)$', fontsize=16)
    ax.set_xlabel(r'$k$ [$h / \rm{Mpc}$]', fontsize=14)
    ax.set_ylim([-0.2, 1.2])
    ax.set_yticks([0, 0.5, 1.0])
    ax.grid(which='both', alpha=0.1)
    ax.legend(facecolor='white', edgecolor='none', framealpha=0.9, loc='lower left')

    # ── Right column: re-simulated final fields ────────────────────────────

    # [0,1] P(k) final
    ax = axes[0, 1]
    ax.set_title(r'Final conditions ($z=0$)', fontsize=13)
    ax.plot(k_ff, pk_true, marker='.', markersize=0.5, lw=0.5,
            color='mediumblue', label='True', zorder=10)
    ax.plot(k_ff, rs_pks_mean, lw=0.5, color=color_rs, label='Re-simulated')
    ax.fill_between(k_ff, rs_pks_mean - rs_pks_std, rs_pks_mean + rs_pks_std,
                    alpha=0.75, color=color_rs)
    ax.fill_between(k_ff, rs_pks_mean - 2*rs_pks_std, rs_pks_mean + 2*rs_pks_std,
                    alpha=0.25, color=color_rs)
    ax.plot(k_noise, pk_noise, color='dimgray', lw=0.5, alpha=0.75,
            label=rf'Noise ($\sigma={sigma_noise}$)')
    if cosmo_params is not None:
        ax.plot(k_class, pk_class_lin, color='black', lw=0.5, alpha=0.3,
                label='CLASS linear')
        ax.plot(k_class, pk_class_nl, color='black', lw=0.5, alpha=0.6,
                ls='--', label='HaloFit')
    ax.axvline(x=k_Nq, color='r', ls='--', lw=0.5, alpha=0.5,
               label=r'$k_{\rm{Nyq}}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(facecolor='white', edgecolor='none', framealpha=0.8,
              loc='lower left', bbox_to_anchor=(0, 0.08))
    ax.set_ylim(ylim_ic_pk[0] / rescaling_factor**2,
                ylim_ic_pk[1] / rescaling_factor**2)
    ax.set_xlim(*xlim_ff)
    ax.grid(which='both', alpha=0.125)

    # [1,1] T(k) final
    ax = axes[1, 1]
    ax.plot(k_ff, rs_tks_mean, lw=0.5, color=color_rs)
    ax.fill_between(k_ff, rs_tks_mean - rs_tks_std, rs_tks_mean + rs_tks_std,
                    alpha=0.75, color=color_rs)
    ax.fill_between(k_ff, rs_tks_mean - 2*rs_tks_std, rs_tks_mean + 2*rs_tks_std,
                    alpha=0.25, color=color_rs)
    ax.axvline(x=k_Nq, color='r', ls='--', lw=0.5, alpha=0.5)
    ax.axhline(1.0, color='k', ls='--', lw=0.5)
    ax.set_xscale('log')
    ax.set_ylim(0.93, 1.07)
    ax.set_yticks([0.95, 1.0, 1.05])
    ax.grid(which='both', alpha=0.1)

    # [2,1] C(k) final
    ax = axes[2, 1]
    ax.plot(k_ff, rs_xpks_mean, lw=0.25, color=color_rs)
    ax.fill_between(k_ff, rs_xpks_mean - rs_xpks_std, rs_xpks_mean + rs_xpks_std,
                    alpha=0.75, color=color_rs)
    ax.fill_between(k_ff, rs_xpks_mean - 2*rs_xpks_std, rs_xpks_mean + 2*rs_xpks_std,
                    alpha=0.25, color=color_rs)
    ax.axvline(x=k_Nq, color='r', ls='--', lw=0.5, alpha=0.5)
    ax.axhline(1.0, color='k', ls='--', lw=0.5)
    ax.set_xscale('log')
    ax.set_xlabel(r'$k$ [$h / \rm{Mpc}$]', fontsize=14)
    ax.set_ylim([-0.2, 1.2])
    ax.set_yticks([0, 0.5, 1.0])
    ax.grid(which='both', alpha=0.1)

    out = os.path.join(save_dir, f'2_combined_2pt_{run_name}.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Combined IC + re-simulated final-field two-point statistics',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser, default_num_samples=200)
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

    # ── IC data (model + samples) ─────────────────────────────────────────
    t0 = time.time()
    d  = load_model_and_generate_samples(args)
    print(f'IC sampling: {time.time()-t0:.1f}s')

    rf          = d['rescaling_factor']
    sigma_noise = d['sigma_noise']
    run_name    = d['run_name']

    # ── PPC re-simulated fields ───────────────────────────────────────────
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

    # ── Output directory ──────────────────────────────────────────────────
    if args.output_dir:
        save_dir = os.path.abspath(args.output_dir)
    else:
        save_dir = d['plots_dir']

    plot_combined_2pt(
        delta_z127=d['delta_z127_int'] * rf,
        delta_z0=d['delta_z0'],
        samples=d['samples_int'] * rf if d['samples_int'] is not None else None,
        z_MAP=d['z_MAP_int'] * rf,
        box=d['box'],
        resim_fields=resim_fields,
        cosmo_params=COSMO_PARAMS.copy(),
        MAS=d['MAS'],
        rescaling_factor=rf,
        sigma_noise=sigma_noise,
        noise_seed=args.noise_seed,
        save_dir=save_dir,
        run_name=run_name,
    )
    print(f'\nDone. Plot saved to {save_dir}/')


if __name__ == '__main__':
    main()
