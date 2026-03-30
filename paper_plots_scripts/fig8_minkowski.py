"""
Minkowski functionals of re-simulated final conditions (PPC).

Plots V0 (volume), V1 (surface), V2 (curvature), V3 (Euler characteristic)
of PPC re-simulated z=0 fields vs the true delta_z0 field, as a function of
threshold ν = (δ - μ)/σ.

Output: {output_dir}/8_minkowski_{RUN_NAME}.pdf

Usage:
    python paper_plots_scripts/fig8_minkowski.py \\
        --model_dir paper_test_runs/runs/260303_224547_sweep_networks/260303_224627_net_IsotropicD \\
        --samples_dir /gpfs/scratch1/shared/osavchenko/mnras_paper/samples/260303_224627_net_IsotropicD \\
        --output_dir paper_plots_scripts/260303_224627_net_IsotropicD

    # Limit to 5 PPC fields:
    python paper_plots_scripts/fig8_minkowski.py \\
        --model_dir ... --samples_dir ... --num_ppc_samples 5
"""

from __future__ import annotations

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

from gaussian_npe import utils
from _common import BOX_PARAMS, add_common_args, load_model_and_generate_samples

BOX_SIZE = BOX_PARAMS['box_size']   # 1000 Mpc/h


# ── Minkowski computation (ported from jax_nbody_emulator/scripts/utils.py) ──

def _count_cubical_complex_elements_periodic(mask: np.ndarray) -> tuple:
    """Count occupied vertices/edges/faces/cubes in a periodic 3D cubical complex."""
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 3 or not (m.shape[0] == m.shape[1] == m.shape[2]):
        raise ValueError(f"`mask` must be a cubic 3D array, got shape={m.shape}.")

    n3 = int(np.count_nonzero(m))
    if n3 == 0:
        return 0, 0, 0, 0

    n2 = 0
    for axis in range(3):
        faces = m | np.roll(m, shift=1, axis=axis)
        n2 += int(np.count_nonzero(faces))

    n1 = 0
    edge_x = m | np.roll(m, shift=1, axis=1)
    edge_x |= np.roll(m, shift=1, axis=2)
    edge_x |= np.roll(np.roll(m, shift=1, axis=1), shift=1, axis=2)
    n1 += int(np.count_nonzero(edge_x))

    edge_y = m | np.roll(m, shift=1, axis=0)
    edge_y |= np.roll(m, shift=1, axis=2)
    edge_y |= np.roll(np.roll(m, shift=1, axis=0), shift=1, axis=2)
    n1 += int(np.count_nonzero(edge_y))

    edge_z = m | np.roll(m, shift=1, axis=0)
    edge_z |= np.roll(m, shift=1, axis=1)
    edge_z |= np.roll(np.roll(m, shift=1, axis=0), shift=1, axis=1)
    n1 += int(np.count_nonzero(edge_z))

    n0_mask = m.copy()
    for sx in (0, 1):
        for sy in (0, 1):
            for sz in (0, 1):
                if sx == 0 and sy == 0 and sz == 0:
                    continue
                n0_mask |= np.roll(m, shift=(sx, sy, sz), axis=(0, 1, 2))
    n0 = int(np.count_nonzero(n0_mask))
    return n0, n1, n2, n3


def compute_minkowski_functionals(field, boxsize, *, thresholds=None, standardize=True):
    """Compute voxelized periodic Minkowski-functional densities for excursion sets.

    Returns a dict with keys: v0, v1, v2, v3, thresholds, mean, std.
      v0 : volume fraction
      v1 : surface-like density [h/Mpc]
      v2 : curvature-like density [h²/Mpc²]
      v3 : Euler-characteristic density [(Mpc/h)⁻³]
    """
    field = np.asarray(field, dtype=np.float32)
    if field.ndim != 3 or not (field.shape[0] == field.shape[1] == field.shape[2]):
        raise ValueError(f"`field` must be a cubic 3D array, got shape={field.shape}.")

    if thresholds is None:
        thresholds_arr = np.linspace(-3.0, 3.0, 41, dtype=np.float32)
    else:
        thresholds_arr = np.asarray(thresholds, dtype=np.float32).ravel()
        if thresholds_arr.size < 2:
            raise ValueError("`thresholds` must contain at least two values.")

    mu = float(np.mean(field))
    sigma = float(np.std(field))
    if standardize and sigma > 0.0:
        work_field = (field - np.float32(mu)) / np.float32(sigma)
    elif standardize:
        work_field = np.zeros_like(field, dtype=np.float32)
    else:
        work_field = np.asarray(field, dtype=np.float32)

    n = int(field.shape[0])
    voxel = float(boxsize) / float(n)
    box_volume = float(boxsize) ** 3

    v0 = np.empty_like(thresholds_arr, dtype=np.float64)
    v1 = np.empty_like(thresholds_arr, dtype=np.float64)
    v2 = np.empty_like(thresholds_arr, dtype=np.float64)
    v3 = np.empty_like(thresholds_arr, dtype=np.float64)

    for i, thr in enumerate(thresholds_arr):
        mask = work_field >= np.float32(thr)
        n0, n1, n2, n3 = _count_cubical_complex_elements_periodic(mask)

        m0 = (voxel**3) * float(n3)
        m1 = (voxel**2) * ((-2.0 / 3.0) * float(n3) + (2.0 / 9.0) * float(n2))
        m2 = voxel * ((2.0 / 3.0) * float(n3) - (4.0 / 9.0) * float(n2) + (2.0 / 9.0) * float(n1))
        m3 = float(n0 - n1 + n2 - n3)

        v0[i] = m0 / box_volume
        v1[i] = m1 / box_volume
        v2[i] = m2 / box_volume
        v3[i] = m3 / box_volume

    return {
        'thresholds': thresholds_arr.astype(np.float64),
        'v0': v0, 'v1': v1, 'v2': v2, 'v3': v3,
        'mean': mu, 'std': sigma,
    }


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_minkowski_summary(nu, truth, samples, save_dir, run_name):
    """2×2 Minkowski functional figure.

    Parameters
    ----------
    nu      : np.ndarray (n_thresholds,)  Threshold grid.
    truth   : dict with keys v0..v3       Minkowski of true delta_z0.
    samples : dict with keys v0..v3, each (n_ppc, n_thresholds)  PPC fields.
    save_dir : str
    run_name : str
    """
    os.makedirs(save_dir, exist_ok=True)
    color_rs = 'mediumpurple'

    keys     = ['v0', 'v1', 'v2', 'v3']
    ylabels  = [
        r'$V_0$',
        r'$V_1$ [$h/{\rm Mpc}$]',
        r'$V_2$ [$(h/{\rm Mpc})^2$]',
        r'$V_3$ $[(h/{\rm Mpc})^3]$',
    ]
    titles = ['Volume', 'Surface', 'Curvature', 'Euler characteristic']
    scale_factors = [1, 1e2, 1e4, 1e4]
    scale_labels  = [None, r'$\times 10^{-2}$', r'$\times 10^{-4}$', r'$\times 10^{-4}$']

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.0), sharex=True,
                             constrained_layout=True)
    plt.rcParams['figure.facecolor'] = 'white'
    for ax in axes.flat:
        ax.tick_params(axis='both', labelsize=14)

    for idx, key in enumerate(keys):
        ax = axes.ravel()[idx]
        arr = np.asarray(samples[key], dtype=np.float64)
        sc  = scale_factors[idx]
        mu  = arr.mean(axis=0) * sc
        sig = arr.std(axis=0)  * sc
        truth_vals = np.asarray(truth[key], dtype=np.float64) * sc

        ax.plot(nu, truth_vals,
                color='black', lw=1.4, label='True')
        ax.plot(nu, mu, color=color_rs, lw=1.2, label='Re-simulated')
        ax.fill_between(nu, mu - sig,       mu + sig,
                        color=color_rs, alpha=0.35, label=r'$\pm 1\sigma$')
        ax.fill_between(nu, mu - 2.0 * sig, mu + 2.0 * sig,
                        color=color_rs, alpha=0.16, label=r'$\pm 2\sigma$')

        ax.set_title(titles[idx], fontsize=16)
        ax.set_ylabel(ylabels[idx], fontsize=16)
        if scale_labels[idx] is not None:
            ax.text(0.03, 0.97, scale_labels[idx], transform=ax.transAxes,
                    fontsize=13, va='top', ha='left')
        ax.grid(alpha=0.15)
        if idx >= 2:
            ax.set_xlabel(r'$\nu = (\delta - \mu)/\sigma$', fontsize=16)

    axes.ravel()[0].legend(framealpha=0.9, fontsize=14, loc='best')

    out = os.path.join(save_dir, f'8_minkowski_{run_name}.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Minkowski functionals of PPC re-simulated final fields',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser, default_num_samples=0)
    parser.add_argument(
        '--samples_dir', type=str, required=True,
        help='Directory containing sample_XXXX/ subdirs with emu_delta_z0.npy.',
    )
    parser.add_argument(
        '--num_ppc_samples', type=int, default=None,
        help='Max number of sample_XXXX/ dirs to load. Default: all.',
    )
    parser.add_argument(
        '--nu_min', type=float, default=-3.0,
        help='Lower threshold for ν grid.',
    )
    parser.add_argument(
        '--nu_max', type=float, default=3.0,
        help='Upper threshold for ν grid.',
    )
    parser.add_argument(
        '--n_thresholds', type=int, default=41,
        help='Number of threshold points.',
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

    d = load_model_and_generate_samples(args)

    # ── Discover PPC fields ───────────────────────────────────────────────
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

    # ── Threshold grid ────────────────────────────────────────────────────
    nu_grid = np.linspace(args.nu_min, args.nu_max, args.n_thresholds, dtype=np.float32)

    # ── Minkowski: truth ──────────────────────────────────────────────────
    print('Computing truth Minkowski functionals...')
    t0 = time.time()
    mink_truth = compute_minkowski_functionals(
        d['delta_z0'], BOX_SIZE, thresholds=nu_grid, standardize=True)
    print(f'  done in {time.time() - t0:.1f}s')

    # ── Minkowski: PPC samples ────────────────────────────────────────────
    mink_samples = {k: [] for k in ('v0', 'v1', 'v2', 'v3')}
    for i, r in enumerate(resim_fields):
        print(f'Computing Minkowski for re-sim {i+1}/{len(resim_fields)}...',
              end='\r', flush=True)
        m = compute_minkowski_functionals(r, BOX_SIZE, thresholds=nu_grid, standardize=True)
        for k in mink_samples:
            mink_samples[k].append(m[k])
    print(f'\nAll Minkowski done in {time.time() - t0:.1f}s total')
    mink_samples = {k: np.array(v) for k, v in mink_samples.items()}

    # ── Plot ──────────────────────────────────────────────────────────────
    save_dir = os.path.abspath(args.output_dir) if args.output_dir else d['plots_dir']
    plot_minkowski_summary(
        nu=nu_grid.astype(np.float64),
        truth=mink_truth,
        samples=mink_samples,
        save_dir=save_dir,
        run_name=d['run_name'],
    )
    print(f'\nDone. Plot saved to {save_dir}/')


if __name__ == '__main__':
    main()
