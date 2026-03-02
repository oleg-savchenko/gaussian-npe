"""
data_scripts/discodj_store.py — Build DiscoDJ ZarrStore for Gaussian NPE training.

Generates pairs of (delta_ic, delta_fin) at the Planck 2018 fiducial cosmology:
  delta_ic  — linear density field at z=0 amplitude (Gaussian random, from P(k))
  delta_fin — non-linear density field at z=0 (PM N-body from z=3 → z=0)

The DiscoDJ P(k) and timetables are computed once at simulator init time.
Per-sample: fresh ICs are drawn using a sequential seed (index = simulation number),
then one PM N-body run produces the corresponding final field.

Store key convention (compatible with train.py / Gaussian_NPE_Network):
  delta_z127 ← delta_ic   (target — linear ICs at z=0 amplitude)
  delta_z0   ← delta_fin  (observed — non-linear field at z=0)
No rescaling_factor needed (ICs already at z=0 amplitude; use rescaling_factor=1.0).

Usage (from the project root):
    python data_scripts/discodj_store.py
"""

import os

# Must be set before JAX initialises (which happens when discodj is imported).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import swyft
from discodj import DiscoDJ
from gaussian_npe import utils

# ── Paths ────────────────────────────────────────────────────────────────────
OUTPUT_STORE_PATH = '/gpfs/scratch1/shared/osavchenko/zarr_stores/discodj_fiducial_res128'

N_SIMULATIONS = 20     # total samples to write
CHUNK_SIZE    = 32     # zarr chunk size (samples per chunk)
BATCH_SIZE    = 10     # swyft simulate batch size
INDEX         = 0      # simulation index used for diagnostic plots

# ── Quijote fiducial cosmology (Planck 2018) ─────────────────────────────────

_FIDUCIAL_COSMO = {
    "Omega_c": 0.3175 - 0.0490,   # = 0.2685  (Omega_m=0.3175, Omega_b=0.0490)
    "Omega_b": 0.0490,
    "h":       0.6711,
    "n_s":     0.9624,
    "sigma8":  0.8340,
}

BOX_PARAMS: dict = {
    "dim":     3,
    "res":     128,
    "boxsize": 1000.0,   # Mpc/h
}

SIM_PARAMS: dict = {
    "a_ini":                1.0 / (1.0 + 3.0),   # z = 3  →  scale factor ≈ 0.25
    "a_end":                1.0 / (1.0 + 0.0),   # z = 0  (present day)
    "stepper":              "bullfrog",
    "method":               "pm",
    "res_pm":               2 * BOX_PARAMS["res"],  # 128  (2× up-sampled PM mesh)
    "time_var":             "D",
    "alpha":                1.5,
    "theta":                0.5,
    "antialias":            0,
    "grad_kernel_order":    4,
    "laplace_kernel_order": 0,
    "n_resample":           1,
    "n_steps":              1,
    "deconvolve":           False,
    "nlpt_order_ics":       2,    # used in dj.with_lpt(n_order=...)
    "worder":               2,    # used in dj.get_delta_from_pos(worder=...)
}


# ── Simulator ─────────────────────────────────────────────────────────────────

class Simulator_DiscoDJ(swyft.Simulator):
    """Generates (delta_ic, delta_fin) pairs via DiscoDJ PM simulation.

    A single GPU DiscoDJ template is built once at __init__ (timetables + P(k)).
    Each sim_call generates ICs and runs the N-body on the same device.
    """

    def __init__(self, indices=None):
        super().__init__()
        self.transform_samples = swyft.to_numpy32
        if indices is None:
            indices = range(N_SIMULATIONS)
        self._iter_indices = iter(indices)

        import jax
        _gpu = jax.devices()[0]   # GPU if available, CPU otherwise

        # Single template: timetables + P(k) on GPU — reused for every sim_call
        self._dj_base = (
            DiscoDJ(
                dim=BOX_PARAMS['dim'], res=BOX_PARAMS['res'],
                boxsize=BOX_PARAMS['boxsize'], device=_gpu,
                cosmo=_FIDUCIAL_COSMO,
            )
            .with_timetables()
            .with_linear_ps()
        )

    def sim_call(self):
        seed = next(self._iter_indices)

        # ── Generate linear ICs at z=0 amplitude ─────────────────────────
        dj_ic    = self._dj_base.with_ics(seed=seed)
        delta_ic = np.array(dj_ic.get_delta_linear(a=1), dtype=np.float32)

        # ── Run PM N-body simulation: z=3 → z=0 ──────────────────────────
        sp  = SIM_PARAMS
        dj  = self._dj_base.with_external_ics(delta=delta_ic)
        dj  = dj.with_lpt(n_order=sp['nlpt_order_ics'], grad_kernel_order=0)
        _skip      = {'nlpt_order_ics', 'worder'}
        run_params = {k: v for k, v in sp.items() if k not in _skip}
        X_sim, _, _ = dj.run_nbody(**run_params, use_diffrax=False)
        delta_fin = np.array(
            dj.get_delta_from_pos(
                X_sim, res=dj.res, worder=sp['worder'], antialias=1, deconvolve=True,
            ),
            dtype=np.float32,
        )

        # Convention: delta_z127 ← delta_ic,  delta_z0 ← delta_fin
        return delta_ic, delta_fin

    def build(self, graph):
        graph.node(['delta_z127', 'delta_z0'], self.sim_call)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # utils.Power_Spectrum_Sampler uses different key names than BOX_PARAMS
    box = utils.Power_Spectrum_Sampler(
        {'box_size': BOX_PARAMS['boxsize'],
         'grid_res': BOX_PARAMS['res'],
         'h':        _FIDUCIAL_COSMO['h']},
        device='cpu',
    )
    print(f'k_Nq = {box.k_Nq:.4f} h/Mpc,  k_F = {box.k_F:.6f} h/Mpc')

    # CLASS-format cosmology dict for plot_training_data
    COSMO_PARAMS = {
        'h':          _FIDUCIAL_COSMO['h'],
        'Omega_b':    _FIDUCIAL_COSMO['Omega_b'],
        'Omega_cdm':  _FIDUCIAL_COSMO['Omega_c'],
        'n_s':        _FIDUCIAL_COSMO['n_s'],
        'non linear': 'halofit',
        'sigma8':     _FIDUCIAL_COSMO['sigma8'],
    }

    # Infer output shapes/dtypes from one probe run (consumes index 0)
    sim_probe = Simulator_DiscoDJ()
    shapes, dtypes = sim_probe.get_shapes_and_dtypes()
    print('shapes:', shapes)
    print('dtypes:', dtypes)

    # Initialise output store, then populate with a fresh simulator instance
    store_out = swyft.ZarrStore(OUTPUT_STORE_PATH)
    store_out.init(N_SIMULATIONS, CHUNK_SIZE, shapes, dtypes)

    sim = Simulator_DiscoDJ()
    store_out.simulate(sim, batch_size=BATCH_SIZE)

    print(f'Generating diagnostic plots for index {INDEX} ...')
    utils.plot_training_data(OUTPUT_STORE_PATH, index=INDEX, box=box,
                             cosmo_params=COSMO_PARAMS, use_rescaling_factor=False)
    print('Done.')
