"""
data_scripts/quijote.py — Build deconvolved Quijote ZarrStore.

Reads raw Quijote density fields (delta_z0, delta_z127) from an existing
ZarrStore, deconvolves the mass-assignment kernel (default: PCS) in Fourier
space, subtracts the field mean to enforce delta-bar = 0, and writes the
corrected fields to a new ZarrStore ready for use in training.

Usage (from the project root):
    python data_scripts/quijote.py
"""

import numpy as np
import swyft
from gaussian_npe import utils

# ── Paths ────────────────────────────────────────────────────────────────────
INPUT_STORE_PATH  = '/home/osavchenko/Quijote/os240713-Swyft_Quijote_128_1Gpc'
OUTPUT_STORE_PATH = '/gpfs/scratch1/shared/osavchenko/zarr_stores/Quijote_fiducial_res128_deconv_MAK'

N_SIMULATIONS = 2000   # total samples to write
CHUNK_SIZE    = 32     # zarr chunk size (samples per chunk)
BATCH_SIZE    = 10     # swyft simulate batch size
INDEX         = 0      # simulation index used for diagnostic plots

# ── Quijote fiducial cosmology (Planck 2018) ─────────────────────────────────
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


# ── Simulator ─────────────────────────────────────────────────────────────────
class Simulator_Quijote_FromStore(swyft.Simulator):
    """Reads raw Quijote fields, deconvolves the PCS kernel, yields corrected fields."""

    def __init__(self, store_path=INPUT_STORE_PATH, indices=None):
        super().__init__()
        self.transform_samples = swyft.to_numpy32
        self.store = swyft.ZarrStore(store_path)
        if indices is None:
            indices = range(len(self.store))
        self._iter_indices = iter(indices)

    def sim_call(self):
        i      = next(self._iter_indices)
        sample = self.store[i]
        delta_z0   = utils.deconvolve_mas(np.asarray(sample['delta_z0'],   dtype=np.float32))
        delta_z127 = utils.deconvolve_mas(np.asarray(sample['delta_z127'], dtype=np.float32))
        delta_z0   -= delta_z0.mean()
        delta_z127 -= delta_z127.mean()
        return delta_z127, delta_z0

    def build(self, graph):
        graph.node(['delta_z127', 'delta_z0'], self.sim_call)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    box = utils.Power_Spectrum_Sampler(BOX_PARAMS, device='cpu')
    print(f'k_Nq = {box.k_Nq:.4f} h/Mpc,  k_F = {box.k_F:.6f} h/Mpc')
    Dz = (utils.growth_D_approx(COSMO_PARAMS, Z_IC)
          / utils.growth_D_approx(COSMO_PARAMS, 0))
    print(f'D(z={Z_IC})/D(z=0) ≈ {Dz:.5f}  (rescaling_factor)')

    # Infer output shapes/dtypes from one probe run (consumes index INDEX of input store)
    sim_probe = Simulator_Quijote_FromStore()
    shapes, dtypes = sim_probe.get_shapes_and_dtypes()
    print('shapes:', shapes)
    print('dtypes:', dtypes)

    # Initialise output store, then populate with a fresh simulator instance
    store_out = swyft.ZarrStore(OUTPUT_STORE_PATH)
    store_out.init(N_SIMULATIONS, CHUNK_SIZE, shapes, dtypes)

    sim = Simulator_Quijote_FromStore()
    store_out.simulate(sim, batch_size=BATCH_SIZE)

    print(f'Generating diagnostic plots for index {INDEX} ...')
    utils.plot_training_data(OUTPUT_STORE_PATH, index=INDEX, box=box,
                             cosmo_params=COSMO_PARAMS)
    print('Done.')
