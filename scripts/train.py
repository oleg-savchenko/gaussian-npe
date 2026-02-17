"""
Gaussian NPE Training Script for Cosmological IC Reconstruction

Trains a Gaussian Neural Posterior Estimation model on Quijote N-body
simulations, saves the model checkpoint, and produces diagnostic plots
(field slices + summary statistics).

Usage:
    python scripts/train.py --run_name my_experiment --max_epochs 30 --sigma_noise 0.1

    python scripts/train.py \
        --run_name baseline \
        --output_dir ./runs \
        --store_path /gpfs/scratch1/shared/osavchenko/zarr_stores/Quijote_fiducial_res128_deconv_MAK \
        --max_epochs 30 \
        --rescaling_factor 0.009908314998484411 \
        --k_cut 0.03 \
        --w_cut 0.001 \
        --sigma_noise 0.1 \
        --learning_rate 0.01 \
        --early_stopping_patience 5 \
        --lr_scheduler_patience 1 \
        --batch_size 8 \
        --val_fraction 0.2 \
        --num_workers 2 \
        --precision 32 \
        --target_path ./Quijote_target/Quijote_sample0.pt \
        --num_samples 100 \
        --MAS PCS
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import swyft
from datetime import datetime

from gaussian_npe import utils, Gaussian_NPE_Network

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers


# ── Quijote fiducial cosmology (Planck 2018) ────────────────────────────
BOX_PARAMS = {
    'box_size': 1000.,   # Mpc/h
    'grid_res': 128,
    'h': 0.6711,
}

COSMO_PARAMS = {
    'h': 0.6711,
    'Omega_b': 0.049,
    'Omega_cdm': 0.2685,
    'n_s': 0.9624,
    'non linear': 'halofit',
    'sigma8': 0.834,
}

Z_IC = 127  # Quijote initial redshift


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Gaussian NPE for cosmological IC reconstruction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Run identification ───────────────────────────────────────────────
    parser.add_argument(
        '--run_name', type=str, default='',
        help='Name for this run (appended to timestamp in output folder)',
    )
    parser.add_argument(
        '--output_dir', type=str, default='./runs',
        help='Base output directory',
    )

    # ── Data ─────────────────────────────────────────────────────────────
    parser.add_argument(
        '--store_path', type=str,
        default='/gpfs/scratch1/shared/osavchenko/zarr_stores/Quijote_fiducial_res128_deconv_MAK',
        help='Path to swyft ZarrStore with Quijote simulations',
    )

    # ── Training hyperparameters ─────────────────────────────────────────
    parser.add_argument(
        '--max_epochs', type=int, default=30,
        help='Maximum number of training epochs',
    )
    parser.add_argument(
        '--rescaling_factor', type=float, default=None,
        help='Rescaling factor for z=127 fields. '
             'If not set, computed as D(z=127)/D(z=0)',
    )
    parser.add_argument(
        '--k_cut', type=float, default=0.03,
        help='Sigmoidal high-pass filter cutoff k [h/Mpc]',
    )
    parser.add_argument(
        '--w_cut', type=float, default=0.001,
        help='Sigmoidal high-pass filter width',
    )
    parser.add_argument(
        '--sigma_noise', type=float, default=0.1,
        help='Gaussian noise std added to the observed field during training',
    )
    parser.add_argument(
        '--learning_rate', type=float, default=1e-2,
        help='Initial learning rate for AdamW optimizer',
    )
    parser.add_argument(
        '--early_stopping_patience', type=int, default=5,
        help='Early stopping patience (epochs without val_loss improvement)',
    )
    parser.add_argument(
        '--lr_scheduler_patience', type=int, default=1,
        help='ReduceLROnPlateau patience (epochs before reducing LR)',
    )
    parser.add_argument(
        '--batch_size', type=int, default=8,
        help='Training batch size',
    )
    parser.add_argument(
        '--val_fraction', type=float, default=0.2,
        help='Validation fraction',
    )
    parser.add_argument(
        '--num_workers', type=int, default=2,
        help='Number of data loader workers',
    )
    parser.add_argument(
        '--precision', type=int, default=32, choices=[16, 32],
        help='Training precision',
    )

    # ── Sampling & plotting ──────────────────────────────────────────────
    parser.add_argument(
        '--target_path', type=str,
        default='/home/osavchenko/Quijote/Quijote_target/Quijote_sample0_wout_MAK.pt',
        help='Path to the target observation .pt file '
             '(must contain delta_z0 and delta_z127 keys)',
    )
    parser.add_argument(
        '--num_samples', type=int, default=100,
        help='Number of posterior samples to draw for the diagnostic plots',
    )
    parser.add_argument(
        '--MAS', type=str, default=None,
        help='Mass assignment scheme for Pylians (e.g. PCS). None = no correction',
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ── Timestamp & output directory ─────────────────────────────────────
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_label = f"{timestamp}_{args.run_name}" if args.run_name else timestamp
    output_dir = os.path.join(args.output_dir, run_label)
    os.makedirs(output_dir, exist_ok=True)

    print(f'Run: {run_label}')
    print(f'Output directory: {output_dir}')

    # ── Device ───────────────────────────────────────────────────────────
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # ── Box & cosmology ──────────────────────────────────────────────────
    box = utils.Power_Spectrum_Sampler(BOX_PARAMS, device=device)
    print(f'k_Nq = {box.k_Nq:.4f} h/Mpc, k_F = {box.k_F:.6f} h/Mpc')

    Dz_approx = (
        utils.growth_D_approx(COSMO_PARAMS, Z_IC)
        / utils.growth_D_approx(COSMO_PARAMS, 0)
    )
    rescaling_factor = args.rescaling_factor if args.rescaling_factor is not None else Dz_approx
    print(f'Rescaling factor D(z={Z_IC})/D(z=0): {rescaling_factor:.6f}')

    prior = box.get_prior_Q_factors(
        lambda k: torch.tensor(
            utils.get_pk_class(COSMO_PARAMS, 0, np.array(k)), device=device,
        )
    )

    # ── Save config ──────────────────────────────────────────────────────
    config = {
        'timestamp': timestamp,
        'run_name': args.run_name,
        'device': device,
        'rescaling_factor': rescaling_factor,
        **{k: v for k, v in vars(args).items() if k != 'rescaling_factor'},
    }
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # ── Data ─────────────────────────────────────────────────────────────
    store = swyft.ZarrStore(args.store_path)
    print(f'Number of simulations in the store: {len(store)}')

    # ── Callbacks & logger ───────────────────────────────────────────────
    log_dir = os.path.join(output_dir, 'logs')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stopping = EarlyStopping(
        monitor='val_loss', min_delta=0., patience=3, verbose=False, mode='min',
    )
    checkpoint = ModelCheckpoint(
        dirpath=log_dir,
        filename='best_{epoch}_{val_loss:.2f}',
        monitor='val_loss', mode='min', save_last=True,
    )
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=log_dir, name='tb_logs', version=None,
    )

    # ── Network ──────────────────────────────────────────────────────────
    network = Gaussian_NPE_Network(
        box, prior,
        sigma_noise=args.sigma_noise,
        rescaling_factor=rescaling_factor,
        k_cut=args.k_cut,
        w_cut=args.w_cut,
        learning_rate=args.learning_rate,
        early_stopping_patience=args.early_stopping_patience,
        lr_scheduler_patience=args.lr_scheduler_patience,
    )
    network.float().to(device)

    # ── Trainer ──────────────────────────────────────────────────────────
    trainer = swyft.SwyftTrainer(
        accelerator='cuda' if device == 'cuda' else 'cpu',
        precision=args.precision,
        logger=tb_logger,
        max_epochs=args.max_epochs,
        callbacks=[lr_monitor, early_stopping, checkpoint],
    )
    dm = swyft.SwyftDataModule(
        store,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )

    # ── Train ────────────────────────────────────────────────────────────
    print(f'\nStarting training for up to {args.max_epochs} epochs...')
    trainer.fit(network, dm)
    print('Training complete.')

    # ── Save model ───────────────────────────────────────────────────────
    model_path = os.path.join(output_dir, 'model.pt')
    torch.save(network.state_dict(), model_path)
    print(f'Model saved to {model_path}')

    # ── Generate samples & plot ──────────────────────────────────────────
    print(f'\nGenerating {args.num_samples} posterior samples '
          f'from target {args.target_path}...')
    network.to(device).eval()

    sample0 = torch.load(args.target_path, weights_only=False)
    delta_z0 = sample0['delta_z0'].astype('f')
    delta_z127 = sample0['delta_z127'].astype('f')

    with torch.no_grad():
        z_MAP = network.get_z_MAP(torch.from_numpy(delta_z0).to(device).float())
        samples = network.sample(args.num_samples, z_MAP=z_MAP)

    z_MAP_np = z_MAP.cpu().numpy()

    print('Plotting analysis...')
    plots_dir = os.path.join(output_dir, 'plots')
    utils.plot_samples_analysis(
        delta_z127=delta_z127 / rescaling_factor,
        delta_z0=delta_z0,
        samples=np.array(samples) / rescaling_factor,
        z_MAP=z_MAP_np / rescaling_factor,
        box=box,
        cosmo_params=COSMO_PARAMS.copy(),
        MAS=args.MAS,
        save_dir=plots_dir,
        run_name=run_label,
    )
    print(f'Plots saved to {os.path.join(plots_dir, run_label)}')
    plt.close('all')

    print(f'\nRun complete. All outputs saved to {output_dir}')


if __name__ == '__main__':
    main()
