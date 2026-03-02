"""
data_scripts/quijote_lh.py — Build deconvolved Quijote Latin Hypercube ZarrStore.

Reads raw Quijote LH density fields (delta_z0, delta_z127, sim_params) from an
existing ZarrStore whose fields were generated with the PCS mass-assignment scheme,
deconvolves the PCS kernel in Fourier space, subtracts the field mean, and
precomputes the per-sample linear growth factor ratio D(z=127)/D(z=0) from the
sample's cosmological parameters.  All four quantities are written to a new
ZarrStore ready for use in training Gaussian_NPE_LH.

sim_params layout (Quijote LH convention):
    [0] Omega_m   [1] Omega_b   [2] h   [3] n_s   [4] sigma_8

Usage (from the project root):
    python data_scripts/quijote_lh.py
"""

import os
import sys
import numpy as np
import swyft

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gaussian_npe import utils

# ── Paths ────────────────────────────────────────────────────────────────────
INPUT_STORE_PATH  = '/home/osavchenko/Quijote/os241125-Swyft_Quijote_LH_128_1Gpc'
OUTPUT_STORE_PATH = '/gpfs/scratch1/shared/osavchenko/zarr_stores/Quijote_LH_res128_deconv_MAK'

N_SIMULATIONS = 2000   # Quijote LH: 2000 cosmologies × 1 realization each
CHUNK_SIZE    = 32     # zarr chunk size (samples per chunk)
BATCH_SIZE    = 10     # swyft simulate batch size
INDEX         = 0      # simulation index used for diagnostic plots

# ── Box parameters (same grid as fiducial) ───────────────────────────────────
BOX_PARAMS = {
    'box_size': 1000.,   # Mpc/h
    'grid_res': 128,
    'h': 0.6711,         # used only for k_F / k_Nq info print
}

Z_IC = 127  # Quijote initial redshift


# ── Simulator ─────────────────────────────────────────────────────────────────
class Simulator_Quijote_LH_FromStore(swyft.Simulator):
    """Reads raw Quijote LH fields, deconvolves the PCS kernel, precomputes
    the per-sample growth factor ratio, and yields all four corrected outputs."""

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

        # Deconvolve PCS mass-assignment kernel and enforce zero mean
        delta_z0   = utils.deconvolve_mas(np.asarray(sample['delta_z0'],   dtype=np.float32))
        delta_z127 = utils.deconvolve_mas(np.asarray(sample['delta_z127'], dtype=np.float32))
        delta_z0   -= delta_z0.mean()
        delta_z127 -= delta_z127.mean()

        # Cosmological parameters for this simulation
        sim_params = np.asarray(sample['sim_params'], dtype=np.float32)  # (5,)
        Omega_m = float(sim_params[0])
        Omega_b = float(sim_params[1])

        # Precompute D(z_ic)/D(z=0) — growth factor ratio used as rescaling_factor.
        # Storing it avoids recomputing on every training batch in Gaussian_NPE_LH.forward.
        cosmo = {'Omega_cdm': Omega_m - Omega_b, 'Omega_b': Omega_b}
        resc  = utils.growth_D_approx(cosmo, Z_IC) / utils.growth_D_approx(cosmo, 0)
        rescaling_factor = np.array([float(resc)], dtype=np.float32)  # shape (1,)

        return delta_z127, delta_z0, sim_params, rescaling_factor

    def build(self, graph):
        graph.node(['delta_z127', 'delta_z0', 'sim_params', 'rescaling_factor'],
                   self.sim_call)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    box = utils.Power_Spectrum_Sampler(BOX_PARAMS, device='cpu')
    print(f'k_Nq = {box.k_Nq:.4f} h/Mpc,  k_F = {box.k_F:.6f} h/Mpc')

    # Infer output shapes/dtypes from one probe run (consumes index INDEX of input store)
    sim_probe = Simulator_Quijote_LH_FromStore()
    shapes, dtypes = sim_probe.get_shapes_and_dtypes()
    print('shapes:', shapes)
    print('dtypes:', dtypes)

    # Initialise output store, then populate with a fresh simulator instance
    store_out = swyft.ZarrStore(OUTPUT_STORE_PATH)
    store_out.init(N_SIMULATIONS, CHUNK_SIZE, shapes, dtypes)

    sim = Simulator_Quijote_LH_FromStore()
    store_out.simulate(sim, batch_size=BATCH_SIZE)

    # Diagnostic plot for index INDEX — reconstruct cosmo_params from its sim_params
    sample0 = swyft.ZarrStore(OUTPUT_STORE_PATH)[INDEX]
    sp = np.asarray(sample0['sim_params'])
    cosmo_params_0 = {
        'h':          float(sp[2]),
        'Omega_b':    float(sp[1]),
        'Omega_cdm':  float(sp[0]) - float(sp[1]),
        'n_s':        float(sp[3]),
        'non linear': 'halofit',
        'sigma8':     float(sp[4]),
    }
    print(f'Index {INDEX} cosmology: {cosmo_params_0}')
    print(f'rescaling_factor[{INDEX}] = {float(sample0["rescaling_factor"][0]):.6f}')
    utils.plot_training_data(OUTPUT_STORE_PATH, index=INDEX, box=box,
                             cosmo_params=cosmo_params_0)
    print('Done.')
