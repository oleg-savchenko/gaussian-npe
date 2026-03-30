"""
Two-point statistics of posterior re-simulated final fields vs true z=0.

Discovers all sample_XXXX/ subdirectories in --samples_dir, loads
emu_delta_z0.npy from each, and plots P(k) / T(k) / C(k) vs the true z=0
density field.  A Gaussian noise reference P(k) is overlaid on the P(k)
panel (dark gray, sigma_noise=1 by default).

Output: paper_plots_scripts/{RUN_NAME}/7_ppc_2pt_{RUN_NAME}.pdf

Usage:
    python paper_plots_scripts/fig7_ppc_2pt.py \\
        --samples_dir /gpfs/scratch1/shared/osavchenko/mnras_paper/samples/260303_224627_net_IsotropicD \\
        --target_path ./Quijote_target.pt \\
        --output_dir paper_plots_scripts/260303_224627_net_IsotropicD

    # Limit to first 5 re-simulations:
    python paper_plots_scripts/fig7_ppc_2pt.py \\
        --samples_dir /gpfs/scratch1/shared/osavchenko/mnras_paper/samples/260303_224627_net_IsotropicD \\
        --num_samples 5 \\
        --output_dir paper_plots_scripts/260303_224627_net_IsotropicD
"""

import os
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import matplotlib.pyplot as plt
import Pk_library as PKL

from gaussian_npe import utils
from _common import BOX_PARAMS, COSMO_PARAMS


BOX_SIZE = BOX_PARAMS['box_size']   # 1000 Mpc/h
NGRID    = BOX_PARAMS['grid_res']   # 128


# Module-level worker — must be at top level for ProcessPoolExecutor pickling
def _compute_resim_pk(args):
    """Auto P(k) and normalised cross-correlation for one re-simulated field."""
    resim_i, delta_z0, box_size = args
    Pk_i  = PKL.XPk([resim_i, delta_z0], box_size, axis=0,
                    MAS=[None, None], threads=1)
    pk_i  = Pk_i.Pk[:, 0, 0]
    xpk_i = Pk_i.XPk[:, 0, 0] / np.sqrt(Pk_i.Pk[:, 0, 0] * Pk_i.Pk[:, 0, 1])
    return pk_i, xpk_i


def plot_ppc_2pt(delta_z0, resim_fields, cosmo_params=None,
                 sigma_noise=1.0, noise_seed=42,
                 save_dir='.', run_name='', n_workers=None):
    """P(k), T(k), C(k) for posterior re-simulated final fields vs true z=0.

    Parameters
    ----------
    delta_z0     : np.ndarray (N,N,N)         True z=0 density contrast.
    resim_fields : list of np.ndarray (N,N,N)  Re-simulated fields from PPC.
    sigma_noise  : float   Noise amplitude for reference P(k).
    noise_seed   : int     RNG seed for noise realization.
    save_dir     : str
    run_name     : str
    n_workers    : int or None
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.rcParams['figure.facecolor'] = 'white'

    delta_z0    = delta_z0.astype('f')
    color_resim = 'mediumpurple'
    k_Nq        = np.pi * NGRID / BOX_SIZE   # Nyquist frequency h/Mpc

    # True field auto-spectrum
    Pk_true   = PKL.Pk(delta_z0, BOX_SIZE, axis=0, MAS=None, threads=1)
    k_pylians = Pk_true.k3D
    pk_true   = Pk_true.Pk[:, 0]

    # Re-simulated P(k) and C(k) — parallelised over samples
    args_list = [(r.astype('f'), delta_z0, BOX_SIZE) for r in resim_fields]
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_compute_resim_pk, args_list))
    pks  = np.array([r[0] for r in results])
    xpks = np.array([r[1] for r in results])
    tks  = np.sqrt(pks / pk_true)

    pks_mean,  pks_std  = pks.mean(0),  pks.std(0)
    tks_mean,  tks_std  = tks.mean(0),  tks.std(0)
    xpks_mean, xpks_std = xpks.mean(0), xpks.std(0)

    # Noise P(k) reference
    noise_field = np.random.default_rng(noise_seed).normal(
        0, sigma_noise, delta_z0.shape).astype('f')
    Pk_noise = PKL.Pk(noise_field, BOX_SIZE, axis=0, MAS=None, threads=1)
    pk_noise = Pk_noise.Pk[:, 0]
    k_noise  = Pk_noise.k3D

    # CLASS linear + HaloFit P(k) at z=0 — restrict to plotted k range
    if cosmo_params is not None:
        k_class      = np.logspace(np.log10(k_pylians[0]), np.log10(k_Nq), 500)
        pk_class_lin = utils.get_pk_class(cosmo_params, 0, k_class, non_lin=False)
        pk_class_nl  = utils.get_pk_class(cosmo_params, 0, k_class, non_lin=True)

    # ── Figure ────────────────────────────────────────────────────────────
    fig, axs = plt.subplots(3, sharex=True, sharey=False, height_ratios=[2, 1, 1])
    fig.set_size_inches(4, 8)

    # P(k)
    ax = axs[0]
    ax.plot(k_pylians, pk_true, marker='.', markersize=0.5, lw=0.5,
            color='mediumblue', label='True', zorder=10)
    ax.plot(k_pylians, pks_mean, lw=0.5, color=color_resim, label='Re-simulated')
    ax.fill_between(k_pylians, pks_mean - pks_std,     pks_mean + pks_std,
                    alpha=0.75, color=color_resim)
    ax.fill_between(k_pylians, pks_mean - 2 * pks_std, pks_mean + 2 * pks_std,
                    alpha=0.25, color=color_resim)
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
    ax.set_ylabel(r'$P(k)$', fontsize=16)
    ax.legend(facecolor='white', edgecolor='none', framealpha=0.8)
    ax.set_xlim(left=k_pylians[0], right=k_Nq + 0.075)
    ax.grid(which='both', alpha=0.125)

    # T(k)
    ax = axs[1]
    ax.plot(k_pylians, tks_mean, lw=0.5, color=color_resim)
    ax.fill_between(k_pylians, tks_mean - tks_std,     tks_mean + tks_std,
                    alpha=0.75, color=color_resim)
    ax.fill_between(k_pylians, tks_mean - 2 * tks_std, tks_mean + 2 * tks_std,
                    alpha=0.25, color=color_resim)
    ax.axvline(x=k_Nq, color='r', ls='--', lw=0.5, alpha=0.5)
    ax.axhline(1.0, color='k', ls='--', lw=0.5)
    ax.set_xscale('log')
    ax.set_ylabel(r'$T(k)$', fontsize=16)
    ax.set_ylim(0.93, 1.07)
    ax.set_yticks([0.95, 1.0, 1.05])
    ax.grid(which='both', alpha=0.1)

    # C(k)
    ax = axs[2]
    ax.plot(k_pylians, xpks_mean, lw=0.25, color=color_resim)
    ax.fill_between(k_pylians, xpks_mean - xpks_std,     xpks_mean + xpks_std,
                    alpha=0.75, color=color_resim)
    ax.fill_between(k_pylians, xpks_mean - 2 * xpks_std, xpks_mean + 2 * xpks_std,
                    alpha=0.25, color=color_resim)
    ax.axvline(x=k_Nq, color='r', ls='--', lw=0.5, alpha=0.5)
    ax.axhline(1.0, color='k', ls='--', lw=0.5)
    ax.set_xscale('log')
    ax.set_ylabel(r'$C(k)$', fontsize=16)
    ax.set_xlabel(r'$k$ [$h / \rm{Mpc}$]', fontsize=14)
    ax.set_ylim([-0.2, 1.2])
    ax.set_yticks([0, 0.5, 1.0])
    ax.grid(which='both', alpha=0.1)

    plt.subplots_adjust(hspace=0)
    out = os.path.join(save_dir, f'7_ppc_2pt_{run_name}.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='P(k)/T(k)/C(k) for posterior re-simulated final fields (fig 7)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--samples_dir', type=str, required=True,
        help='Directory containing sample_XXXX/ subdirs with emu_delta_z0.npy.',
    )
    parser.add_argument(
        '--target_path', type=str, default='./Quijote_target.pt',
        help='Path to target .pt file containing the true delta_z0 field.',
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Where to save the PDF. Defaults to paper_plots_scripts/{run_name}/.',
    )
    parser.add_argument(
        '--num_samples', type=int, default=None,
        help='Max number of sample_XXXX/ dirs to load. Default: all.',
    )
    parser.add_argument(
        '--sigma_noise', type=float, default=1.0,
        help='Noise amplitude for reference P(k) curve.',
    )
    parser.add_argument(
        '--noise_seed', type=int, default=42,
        help='RNG seed for the noise realization.',
    )
    parser.add_argument(
        '--no_latex', dest='use_latex', action='store_false',
        help='Disable LaTeX rendering (on by default).',
    )
    parser.set_defaults(use_latex=True)
    return parser.parse_args()


def main():
    args = parse_args()
    utils.configure_matplotlib_style(use_latex=args.use_latex)

    # ── Discover re-simulated fields ──────────────────────────────────────
    samples_dir = os.path.abspath(args.samples_dir)
    run_name    = os.path.basename(samples_dir)

    fourier_dirs = sorted([
        os.path.join(samples_dir, d)
        for d in os.listdir(samples_dir)
        if d.startswith('sample_')
        and os.path.isfile(os.path.join(samples_dir, d, 'emu_delta_z0.npy'))
    ])
    if not fourier_dirs:
        raise FileNotFoundError(
            f'No sample_XXXX/ dirs with emu_delta_z0.npy found in {samples_dir}')

    if args.num_samples is not None:
        fourier_dirs = fourier_dirs[:args.num_samples]
    print(f'Loading {len(fourier_dirs)} re-simulated fields from {samples_dir}')

    resim_fields = [np.load(os.path.join(d, 'emu_delta_z0.npy')) for d in fourier_dirs]

    # ── Load true delta_z0 ────────────────────────────────────────────────
    target   = torch.load(args.target_path, weights_only=False)
    delta_z0 = target['delta_z0'].astype('f')
    print(f'Loaded true delta_z0 from {args.target_path}')

    # ── Output directory ──────────────────────────────────────────────────
    if args.output_dir:
        save_dir = os.path.abspath(args.output_dir)
    else:
        scripts_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir    = os.path.join(scripts_dir, run_name)

    plot_ppc_2pt(
        delta_z0=delta_z0,
        resim_fields=resim_fields,
        cosmo_params=COSMO_PARAMS.copy(),
        sigma_noise=args.sigma_noise,
        noise_seed=args.noise_seed,
        save_dir=save_dir,
        run_name=run_name,
    )
    print(f'\nDone. Plot saved to {save_dir}/')


if __name__ == '__main__':
    main()
