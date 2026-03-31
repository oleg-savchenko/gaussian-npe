"""
Field slice visualisations of posterior samples vs true ICs.

Produces one figure saved to paper_plots_scripts/{RUN_NAME}/:
  - 4_field_slices_{RUN_NAME}.pdf : 2×3 grid
        [0,0] True initial IC   [0,1] IC Sample 1    [0,2] True final (z=0)
        [1,0] MAP               [1,1] IC Sample 2    [1,2] Resimulated (z=0)

    IC panels  : seismic cmap, ×1e2 scaling, vmin/vmax = [-2, 2].
    Final panels: inferno cmap, vmin=-0.5, vmax=2.5.
    Slicing    : field[60:73].mean(0) (13-slice average).

Usage:
    # First run — loads model, generates/saves z_MAP.npy, loads 2 IC samples
    # and the resimulated field from the ppc pipeline:
    python paper_plots_scripts/fig1_slices.py \\
        --model_dir paper_test_runs/runs/260303_224547_sweep_networks/260303_224627_net_IsotropicD \\
        --samples_dir /gpfs/scratch1/shared/osavchenko/mnras_paper/samples/260303_224627_net_IsotropicD \\
        --ppc_dir /gpfs/scratch1/shared/osavchenko/mnras_paper/samples/260303_224627_net_IsotropicD/sample_0001 \\
        --num_samples 2

    # After z_MAP.npy is saved to samples_dir: comment out the model-loading
    # block in main() and uncomment the fast-path block — no GPU needed.
"""

import os
import json
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt

from gaussian_npe import utils
from _common import add_common_args, load_model_and_generate_samples


def plot_field_slices(delta_z127, delta_z0, samples, z_MAP, emu_delta_z0,
                      save_dir, run_name):
    """2×3 publication figure: IC and final-field slices.

    Parameters
    ----------
    delta_z127 : ndarray (N,N,N)   True IC field, physical units.
    delta_z0   : ndarray (N,N,N)   True final field (z=0), δ units.
    samples    : list of 2 ndarrays (N,N,N)  IC posterior samples, physical units.
    z_MAP      : ndarray (N,N,N)   MAP IC estimate, physical units.
    emu_delta_z0 : ndarray (N,N,N) or None  Resimulated z=0 field; if None the
                   panel is left blank.
    save_dir   : str
    run_name   : str
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'

    extent = [0, 1000, 0, 1000]
    sl = slice(60, 73)               # slice range to average over

    # ── helpers ───────────────────────────────────────────────────────────
    def ic_slice(field):
        return 1e2 * field[sl].mean(0)

    def ff_slice(field):
        return field[sl].mean(0)

    # IC colorscheme: RdBu_r, symmetric about 0 at 99th percentile of |field|
    ic_absmax = float(np.percentile(np.abs(ic_slice(delta_z127)), 99))
    vmin_ic, vmax_ic = -ic_absmax, ic_absmax
    vmin_ff, vmax_ff = -0.5, 2.5
    label_mpc = r'$[{\rm Mpc} / h]$'
    cb_ticks_ic = [-2, -1, 0, 1, 2]
    cb_title_ic = r'$\times 10^{-2}$'

    fig, axes = plt.subplots(2, 3, figsize=(12, 10), layout='compressed')

    s1 = samples[0] if samples is not None and len(samples) > 0 else z_MAP
    s2 = samples[1] if samples is not None and len(samples) > 1 else z_MAP

    # ── [0,0] True initial IC ─────────────────────────────────────────────
    axes[0, 0].imshow(ic_slice(delta_z127), origin='lower', cmap='RdBu_r',
                      vmin=vmin_ic, vmax=vmax_ic, extent=extent)
    axes[0, 0].set_title(r'True initial', fontsize=22)
    axes[0, 0].set_ylabel(label_mpc, fontsize=14)
    axes[0, 0].set_xticklabels([])

    # ── [1,0] MAP ─────────────────────────────────────────────────────────
    axes[1, 0].imshow(ic_slice(z_MAP), origin='lower', cmap='RdBu_r',
                      vmin=vmin_ic, vmax=vmax_ic, extent=extent)
    axes[1, 0].set_title(r'MAP', fontsize=22)
    axes[1, 0].set_ylabel(label_mpc, fontsize=14)
    axes[1, 0].set_xlabel(label_mpc, fontsize=14)

    # ── [0,1] IC Sample 1 ─────────────────────────────────────────────────
    im_ic = axes[0, 1].imshow(ic_slice(s1), origin='lower', cmap='RdBu_r',
                               vmin=vmin_ic, vmax=vmax_ic, extent=extent)
    axes[0, 1].set_title(r'Sample 1', fontsize=22)
    axes[0, 1].set_xticklabels([])
    axes[0, 1].set_yticklabels([])
    clb = plt.colorbar(im_ic, ax=axes[0, 1], pad=0.006, aspect=30,
                       ticks=cb_ticks_ic)
    clb.ax.set_title(cb_title_ic, fontsize=12)

    # ── [1,1] IC Sample 2 ─────────────────────────────────────────────────
    im_ic2 = axes[1, 1].imshow(ic_slice(s2), origin='lower', cmap='RdBu_r',
                                vmin=vmin_ic, vmax=vmax_ic, extent=extent)
    axes[1, 1].set_title(r'Sample 2', fontsize=22)
    axes[1, 1].set_yticklabels([])
    axes[1, 1].set_xlabel(label_mpc, fontsize=14)
    clb2 = plt.colorbar(im_ic2, ax=axes[1, 1], pad=0.006, aspect=30,
                        ticks=cb_ticks_ic)
    clb2.ax.set_title(cb_title_ic, fontsize=12)

    # ── [0,2] True final (z=0) ────────────────────────────────────────────
    im_ff = axes[0, 2].imshow(ff_slice(delta_z0), origin='lower', cmap='inferno',
                               vmin=vmin_ff, vmax=vmax_ff, extent=extent)
    axes[0, 2].set_title(r'True final', fontsize=22)
    axes[0, 2].set_xticklabels([])
    axes[0, 2].set_yticklabels([])
    plt.colorbar(im_ff, ax=axes[0, 2], pad=0.006, aspect=30)

    # ── [1,2] Resimulated final (z=0) ────────────────────────────────────
    if emu_delta_z0 is not None:
        im_rs = axes[1, 2].imshow(ff_slice(emu_delta_z0), origin='lower',
                                   cmap='inferno', vmin=vmin_ff, vmax=vmax_ff,
                                   extent=extent)
        plt.colorbar(im_rs, ax=axes[1, 2], pad=0.006, aspect=30)
    else:
        axes[1, 2].set_visible(False)
    axes[1, 2].set_title(r'Resimulated', fontsize=22)
    axes[1, 2].set_yticklabels([])
    axes[1, 2].set_xlabel(label_mpc, fontsize=14)

    out = os.path.join(save_dir, f'4_field_slices_{run_name}.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Field slice plots (2×3 publication layout)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser, default_num_samples=2)
    parser.add_argument(
        '--samples_dir', type=str, default=None,
        help='Directory of pre-saved .npy IC samples (from save_samples.py). '
             'Samples are loaded from disk instead of generated. '
             'z_MAP.npy is saved here after the first run for fast reloading.',
    )
    parser.add_argument(
        '--ppc_dir', type=str, default=None,
        help='Path to a sample_XXXX/ folder from the ppc pipeline. '
             'Loads emu_delta_z0.npy for the resimulated panel. '
             'Should point to sample_0001 so the resimulation matches the bottom-centre IC panel.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    utils.configure_matplotlib_style(use_latex=args.use_latex)

    # ── Load model + target to get delta_z127, delta_z0 and z_MAP ─────────
    # Once z_MAP.npy has been saved to samples_dir, comment out this block
    # and uncomment the fast-path block below.
    orig_num = args.num_samples
    args.num_samples = 0   # skip sample generation; samples loaded from disk below
    d = load_model_and_generate_samples(args)
    args.num_samples = orig_num
    rf             = d['rescaling_factor']
    z_MAP_phys     = d['z_MAP_int'] * rf          # physical units
    delta_z127     = d['delta_z127_int'] * rf     # physical units
    delta_z0       = d['delta_z0']                # already physical
    run_name       = d['run_name']
    plots_dir      = d['plots_dir']

    # Save z_MAP to samples_dir for future fast loading (physical units)
    if args.samples_dir:
        map_path = os.path.join(args.samples_dir, 'z_MAP.npy')
        if not os.path.exists(map_path):
            np.save(map_path, z_MAP_phys)
            print(f'Saved z_MAP to {map_path}')

    # ── Fast path (no GPU / no network) — uncomment and comment block above ─
    # config_path = os.path.join(args.samples_dir, 'config.json')
    # with open(config_path) as f:
    #     saved_config = json.load(f)
    # rf = saved_config['rescaling_factor']
    # run_name  = os.path.basename(os.path.abspath(args.samples_dir))
    # plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), run_name)
    # os.makedirs(plots_dir, exist_ok=True)
    # sample0    = torch.load(args.target_path, weights_only=False)
    # delta_z127 = sample0['delta_z127'].astype('f')   # physical units
    # delta_z0   = sample0['delta_z0'].astype('f')
    # z_MAP_phys = np.load(os.path.join(args.samples_dir, 'z_MAP.npy'))

    # ── Load pre-saved IC samples ─────────────────────────────────────────
    if args.samples_dir:
        sample_files = sorted([
            f for f in os.listdir(args.samples_dir)
            if f.startswith('sample_') and f.endswith('.npy')
        ])
        # load sample_0000 (top-centre) and sample_0001 (bottom-centre)
        sample_files = sample_files[0:args.num_samples]
        samples = (
            [np.load(os.path.join(args.samples_dir, f)) for f in sample_files]
            if sample_files else None
        )
        print(f'Loaded {len(sample_files)} IC samples from {args.samples_dir}')
    else:
        samples = None

    # ── Load resimulated field ───────────────────────────────────────────
    emu_delta_z0 = None
    if args.ppc_dir:
        emu_path = os.path.join(args.ppc_dir, 'emu_delta_z0.npy')
        if os.path.exists(emu_path):
            emu_delta_z0 = np.load(emu_path)
            print(f'Loaded resimulated field from {emu_path}')
        else:
            print(f'Warning: emu_delta_z0.npy not found in {args.ppc_dir}')

    plot_field_slices(
        delta_z127=delta_z127,
        delta_z0=delta_z0,
        samples=samples,
        z_MAP=z_MAP_phys,
        emu_delta_z0=emu_delta_z0,
        save_dir=plots_dir,
        run_name=run_name,
    )
    print(f'\nDone. Plot saved to {plots_dir}/')


if __name__ == '__main__':
    main()
