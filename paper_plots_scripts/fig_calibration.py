"""
Posterior calibration diagnostics in Hartley space.

Calls utils.plot_calibration_diagnostics, which produces three figures saved
to {output_dir}/calibration/:
  - log_prob_histogram_{RUN_NAME}.png  : log p(z|x) distribution
  - chi2_per_k_{RUN_NAME}.png          : per-mode chi² binned by |k|
  - hartley_modes_{RUN_NAME}.png       : true vs MAP Hartley modes ±2σ

Usage:
    python paper_plots_scripts/fig_calibration.py \\
        --model_dir paper_test_runs/runs/260303_224547_sweep_networks/260303_224627_net_IsotropicD

    python paper_plots_scripts/fig_calibration.py \\
        --model_dir paper_test_runs/runs/... \\
        --num_samples 200 \\
        --output_dir paper_plots_scripts/260303_224627_net_IsotropicD
"""

import os
import argparse

from gaussian_npe import utils
from _common import add_common_args, load_model_and_generate_samples


def parse_args():
    parser = argparse.ArgumentParser(
        description='Posterior calibration diagnostics (Hartley space)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser, default_num_samples=200)
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Where to save the calibration/ subfolder. '
             'Defaults to paper_plots_scripts/{run_name}/.',
    )
    parser.add_argument(
        '--save_csv', action='store_true',
        help='Also save scalar metrics as CSV + txt in a diagnostics/ subfolder.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    utils.configure_matplotlib_style(use_latex=args.use_latex)

    d = load_model_and_generate_samples(args)

    save_dir = os.path.abspath(args.output_dir) if args.output_dir else d['plots_dir']

    utils.plot_calibration_diagnostics(
        delta_z127=d['delta_z127_int'],
        z_MAP=d['z_MAP_int'],
        samples=d['samples_int'],
        box=d['box'],
        Q_like_D=d['Q_like_D'],
        Q_prior_D=d['Q_prior_D'],
        save_dir=save_dir,
        run_name=d['run_name'],
        save_csv=args.save_csv,
        fmt='pdf',
    )

    print(f"\nDone. Plots saved to {os.path.join(save_dir, 'calibration')}/")


if __name__ == '__main__':
    main()
