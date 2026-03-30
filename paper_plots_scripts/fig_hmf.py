"""
Halo Mass Function from saved posterior-predictive aggregate arrays.

Re-renders the HMF figure from a previously saved `*_hmf_arrays.npz` file
produced by the posterior re-simulation pipeline, without rerunning FoF halo
finding.  Optionally overlays a Tinker theoretical HMF line computed via
Pylians mass_function_library and CLASS.

Output: {output_dir}/hmf_{run_name}.pdf

Usage:
    python paper_plots_scripts/fig_hmf.py \\
        --arrays_path paper_plots_scripts/260303_224627_net_IsotropicD/posterior_resimulation/posterior_predictive_hmf_arrays.npz

    python paper_plots_scripts/fig_hmf.py \\
        --arrays_path paper_plots_scripts/260303_224627_net_IsotropicD/posterior_resimulation/posterior_predictive_hmf_arrays.npz \\
        --build_class_pk_fiducial \\
        --output_dir paper_plots_scripts/260303_224627_net_IsotropicD
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from gaussian_npe import utils

try:
    import mass_function_library as MFL
except Exception as exc:
    MFL = None
    _MFL_IMPORT_ERROR = exc
else:
    _MFL_IMPORT_ERROR = None

try:
    from classy import Class
except Exception as exc:
    Class = None
    _CLASS_IMPORT_ERROR = exc
else:
    _CLASS_IMPORT_ERROR = None

QUIJOTE_FIDUCIAL_CLASS = {
    'Omega_cdm': 0.3175 - 0.0490,
    'Omega_b':   0.0490,
    'h':         0.6711,
    'n_s':       0.9624,
    'sigma8':    0.8340,
}

DEFAULT_ARRAYS_PATH = (
    'paper_plots_scripts/260303_224627_net_IsotropicD/'
    'posterior_resimulation/posterior_predictive_hmf_arrays.npz'
)


# ── Tinker helpers ────────────────────────────────────────────────────────────

def compute_tinker_from_class_pk(
    *,
    mass_centers: np.ndarray,
    class_pk_table_path: str,
    omega_m: float,
    author: str,
    integration_bins: int,
) -> np.ndarray:
    """Compute dn/dlog10M using Pylians MF_theory from a CLASS linear P(k) table."""
    if MFL is None:
        raise ImportError('mass_function_library is unavailable.') from _MFL_IMPORT_ERROR
    table = np.loadtxt(class_pk_table_path)
    if table.ndim != 2 or table.shape[1] < 2:
        raise ValueError(f'Invalid CLASS P(k) table format in {class_pk_table_path}')
    k  = np.asarray(table[:, 0], dtype=np.float64)
    pk = np.asarray(table[:, 1], dtype=np.float64)
    masses = np.asarray(mass_centers, dtype=np.float64)
    dndm = np.asarray(
        MFL.MF_theory(
            k_in=k, Pk_in=pk, OmegaM=float(omega_m),
            Masses=masses, author=str(author),
            bins=int(integration_bins), z=0.0, delta=200.0,
        ),
        dtype=np.float64,
    )
    return dndm * masses * np.log(10.0)


def _build_fiducial_class_pk(*, z, kmin, kmax, n_modes):
    """Generate fiducial Quijote linear CLASS P(k) on a geometric k-grid."""
    if Class is None:
        raise ImportError('classy is unavailable.') from _CLASS_IMPORT_ERROR
    k_eval = np.geomspace(float(kmin), float(kmax), int(n_modes), dtype=np.float64)
    params = dict(QUIJOTE_FIDUCIAL_CLASS)
    h = float(params['h'])
    params.update({
        'output':        'mPk',
        'P_k_max_h/Mpc': float(np.max(k_eval)),
        'z_max_pk':      float(z),
    })
    cosmo = Class()
    try:
        cosmo.set(params)
        cosmo.compute()
        pk = h**3 * np.asarray(
            [cosmo.pk_lin(h * float(ki), float(z)) for ki in k_eval],
            dtype=np.float64,
        )
    finally:
        try:
            cosmo.struct_cleanup()
            cosmo.empty()
        except Exception:
            pass
    return k_eval, pk


def compute_tinker_from_fiducial_class(
    *,
    mass_centers: np.ndarray,
    omega_m: float,
    author: str,
    integration_bins: int,
    class_z: float,
    class_kmin: float,
    class_kmax: float,
    class_n_modes: int,
) -> np.ndarray:
    """Compute Tinker dn/dlog10M from an on-the-fly fiducial Quijote CLASS P(k)."""
    if MFL is None:
        raise ImportError('mass_function_library is unavailable.') from _MFL_IMPORT_ERROR
    k, pk = _build_fiducial_class_pk(
        z=class_z, kmin=class_kmin, kmax=class_kmax, n_modes=class_n_modes,
    )
    masses = np.asarray(mass_centers, dtype=np.float64)
    dndm = np.asarray(
        MFL.MF_theory(
            k_in=np.asarray(k, dtype=np.float64),
            Pk_in=np.asarray(pk, dtype=np.float64),
            OmegaM=float(omega_m), Masses=masses,
            author=str(author), bins=int(integration_bins),
            z=float(class_z), delta=200.0,
        ),
        dtype=np.float64,
    )
    return dndm * masses * np.log(10.0)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_hmf_summary(
    *,
    m_centers: np.ndarray,
    hmf_samples: np.ndarray,
    tinker_curve: np.ndarray | None,
    hmf_truth: np.ndarray | None,
    mass_resolution_limit: float | None,
    save_dir: str,
    run_name: str,
    ratio_ymin: float,
    ratio_ymax: float,
) -> None:
    """2-panel HMF figure: log-log HMF top, ratio bottom."""
    color_rs = 'mediumpurple'

    m_centers   = np.asarray(m_centers,   dtype=np.float64)
    hmf_samples = np.asarray(hmf_samples, dtype=np.float64)
    hmf_samples = np.where(hmf_samples > 0.0, hmf_samples, np.nan)

    mu    = np.nanmean(hmf_samples, axis=0)
    sigma = np.nanstd(hmf_samples,  axis=0)

    s1_lo = np.where((mu -       sigma) > 0.0, mu -       sigma, np.nan)
    s1_hi = np.where((mu +       sigma) > 0.0, mu +       sigma, np.nan)
    s2_lo = np.where((mu - 2.0 * sigma) > 0.0, mu - 2.0 * sigma, np.nan)
    s2_hi = np.where((mu + 2.0 * sigma) > 0.0, mu + 2.0 * sigma, np.nan)

    # Convert from (h/Mpc)^3 to (h/Gpc)^3: multiply by (1000)^3 = 1e9
    gpc_factor = 1e9
    mu    *= gpc_factor
    sigma *= gpc_factor
    s1_lo = np.where(s1_lo > 0.0, s1_lo * gpc_factor, np.nan)
    s1_hi = np.where(s1_hi > 0.0, s1_hi * gpc_factor, np.nan)
    s2_lo = np.where(s2_lo > 0.0, s2_lo * gpc_factor, np.nan)
    s2_hi = np.where(s2_hi > 0.0, s2_hi * gpc_factor, np.nan)

    tinker = None if tinker_curve is None else np.asarray(tinker_curve, dtype=np.float64) * gpc_factor
    truth  = None if hmf_truth   is None else np.asarray(hmf_truth,     dtype=np.float64) * gpc_factor

    fig, axes = plt.subplots(
        1, 1, figsize=(6.4, 5.5), constrained_layout=True,
    )
    axes = [axes]  # keep indexing consistent

    # ── Top panel: HMF ───────────────────────────────────────────────────
    ax = axes[0]
    if (mass_resolution_limit is not None
            and np.isfinite(float(mass_resolution_limit))
            and float(mass_resolution_limit) > 0.0):
        ax.axvspan(0.0, float(mass_resolution_limit),
                   color='#efe6e6', alpha=0.85, label='Mass resolution limit', zorder=0)

    ax.fill_between(m_centers, s2_lo, s2_hi,
                    color=color_rs, alpha=0.20, lw=0.0, label=r'$\pm 2\sigma$', zorder=1)
    ax.fill_between(m_centers, s1_lo, s1_hi,
                    color=color_rs, alpha=0.40, lw=0.0, label=r'$\pm 1\sigma$', zorder=2)
    ax.plot(m_centers, mu, color=color_rs, lw=2.0, label='Posterior resimulations', zorder=3)

    if tinker is not None:
        ax.plot(m_centers, tinker, color='0.5', lw=2.0, ls=':', label='Tinker', zorder=4)
    if truth is not None:
        ax.plot(m_centers, truth, color='k', ls='--', lw=1.8,
                marker='x', ms=6, mew=1.6, label='Ground truth', zorder=5)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(r'$dn/d\log_{10}M\ [(h/{\rm Gpc})^3]$')
    ax.set_xlabel(r'Mass [$M_\odot\,h^{-1}$]')
    ax.grid(alpha=0.25, which='both', ls=':')
    ax.legend(framealpha=0.95, fontsize=9, loc='best')

    # ── Bottom panel: ratio (commented out) ──────────────────────────────
    # ax = axes[1]
    # if truth is not None:
    #     denom = np.where(truth > 0.0, truth, np.nan)
    #     ax.plot(m_centers, np.ones_like(m_centers),
    #             color='k', ls='--', lw=1.4, marker='x', ms=5, mew=1.4)
    # elif tinker is not None:
    #     denom = np.where(tinker > 0.0, tinker, np.nan)
    #     ax.axhline(1.0, color='k', ls='--', lw=1.4)
    # else:
    #     denom = np.where(mu > 0.0, mu, np.nan)
    #     ax.axhline(1.0, color='k', ls='--', lw=1.4)
    #
    # if (mass_resolution_limit is not None
    #         and np.isfinite(float(mass_resolution_limit))
    #         and float(mass_resolution_limit) > 0.0):
    #     ax.axvspan(0.0, float(mass_resolution_limit),
    #                color='#efe6e6', alpha=0.85, zorder=0)
    #
    # ax.fill_between(m_centers, s2_lo / denom, s2_hi / denom,
    #                 color=color_rs, alpha=0.20, lw=0.0, zorder=1)
    # ax.fill_between(m_centers, s1_lo / denom, s1_hi / denom,
    #                 color=color_rs, alpha=0.40, lw=0.0, zorder=2)
    # ax.plot(m_centers, mu / denom, color=color_rs, lw=1.8, zorder=3)
    # ax.plot(m_centers, np.ones_like(m_centers), color='0.5', lw=1.6, ls=':', zorder=4)
    #
    # ax.set_xscale('log')
    # ax.set_ylim(float(ratio_ymin), float(ratio_ymax))
    # ax.set_ylabel('Ratio')
    # ax.set_xlabel(r'Mass [$M_\odot\,h^{-1}$]')
    # ax.grid(alpha=0.25, which='both', ls=':')

    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, f'hmf_{run_name}.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Replot HMF from saved aggregate arrays (no FoF recomputation).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--arrays_path', type=str, default=DEFAULT_ARRAYS_PATH,
        help='Path to *_hmf_arrays.npz produced by the posterior re-simulation pipeline.',
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Output directory for the PDF. Defaults to the directory of arrays_path.',
    )
    parser.add_argument(
        '--ratio_ymin', type=float, default=0.6,
        help='Lower y-limit of the ratio panel.',
    )
    parser.add_argument(
        '--ratio_ymax', type=float, default=1.4,
        help='Upper y-limit of the ratio panel.',
    )
    parser.add_argument(
        '--show_truth', action=argparse.BooleanOptionalAction, default=True,
        help='Overlay truth HMF if present in arrays file.',
    )
    parser.add_argument(
        '--show_tinker', action=argparse.BooleanOptionalAction, default=True,
        help='Overlay Tinker HMF (from arrays or computed from CLASS).',
    )
    parser.add_argument(
        '--class_pk_table_path', type=str, default=None,
        help='Optional CLASS linear P(k) table path. Computes Tinker HMF and overrides any existing hmf_tinker.',
    )
    parser.add_argument(
        '--build_class_pk_fiducial', action=argparse.BooleanOptionalAction, default=False,
        help='Build a CLASS linear P(k) on the fly using fiducial Quijote cosmology for the Tinker overlay.',
    )
    parser.add_argument('--class_z',       type=float, default=0.0,   help='Redshift for fiducial CLASS P(k).')
    parser.add_argument('--class_kmin',    type=float, default=1e-5,  help='kmin [h/Mpc] for fiducial CLASS P(k).')
    parser.add_argument('--class_kmax',    type=float, default=10.0,  help='kmax [h/Mpc] for fiducial CLASS P(k).')
    parser.add_argument('--class_n_modes', type=int,   default=4000,  help='Number of k-samples for fiducial CLASS P(k).')
    parser.add_argument('--omega_m',       type=float, default=0.3175, help='Omega_m for Tinker MF_theory.')
    parser.add_argument(
        '--tinker_author', type=str, default='Tinker',
        choices=('ST', 'Tinker', 'Tinker10', 'Crocce', 'Jenkins', 'Warren', 'Watson', 'Watson_FOF', 'Angulo'),
        help='Pylians mass-function fitting function.',
    )
    parser.add_argument(
        '--tinker_integration_bins', type=int, default=1000,
        help='Number of log-k bins used by Pylians MF_theory integration.',
    )
    parser.add_argument(
        '--no_latex', dest='use_latex', action='store_false', default=True,
        help='Disable LaTeX rendering.',
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    utils.configure_matplotlib_style(use_latex=args.use_latex)

    arrays_path = os.path.abspath(args.arrays_path)
    if not os.path.isfile(arrays_path):
        raise FileNotFoundError(f'Arrays file not found: {arrays_path}')

    # Derive run_name from the grandparent directory (e.g. 260303_224627_net_IsotropicD)
    run_name = os.path.basename(os.path.dirname(os.path.dirname(arrays_path)))

    if args.output_dir:
        save_dir = os.path.abspath(args.output_dir)
    else:
        save_dir = os.path.dirname(arrays_path)

    print(f'Run: {run_name}')
    print(f'Arrays: {arrays_path}')

    data = np.load(arrays_path)

    required = ('hmf_mass_centers', 'hmf_samples', 'mass_resolution_limit')
    missing = [k for k in required if k not in data.files]
    if missing:
        raise KeyError(f'Missing required keys in {arrays_path}: {missing}')

    m_centers            = np.asarray(data['hmf_mass_centers'],      dtype=np.float64)
    hmf_samples          = np.asarray(data['hmf_samples'],           dtype=np.float64)
    mass_resolution_limit = float(np.asarray(data['mass_resolution_limit'], dtype=np.float64))

    def _optional(key):
        if key not in data.files:
            return None
        arr = np.asarray(data[key], dtype=np.float64)
        return arr if arr.size > 0 else None

    hmf_truth  = _optional('hmf_truth')  if args.show_truth  else None
    hmf_tinker = _optional('hmf_tinker') if args.show_tinker else None

    # ── Optionally recompute Tinker ───────────────────────────────────────
    if args.class_pk_table_path is not None and args.build_class_pk_fiducial:
        raise ValueError('Use either --class_pk_table_path or --build_class_pk_fiducial, not both.')

    if args.class_pk_table_path is not None:
        pk_path = os.path.abspath(args.class_pk_table_path)
        if not os.path.isfile(pk_path):
            raise FileNotFoundError(f'CLASS P(k) table not found: {pk_path}')
        print(f'Computing Tinker HMF from CLASS P(k) table: {pk_path}')
        hmf_tinker = compute_tinker_from_class_pk(
            mass_centers=m_centers,
            class_pk_table_path=pk_path,
            omega_m=args.omega_m,
            author=args.tinker_author,
            integration_bins=args.tinker_integration_bins,
        )
    elif args.build_class_pk_fiducial:
        print('Computing Tinker HMF from fiducial Quijote CLASS P(k)...')
        hmf_tinker = compute_tinker_from_fiducial_class(
            mass_centers=m_centers,
            omega_m=args.omega_m,
            author=args.tinker_author,
            integration_bins=args.tinker_integration_bins,
            class_z=args.class_z,
            class_kmin=args.class_kmin,
            class_kmax=args.class_kmax,
            class_n_modes=args.class_n_modes,
        )

    # ── Shape validation ──────────────────────────────────────────────────
    if hmf_truth is not None and hmf_truth.shape != m_centers.shape:
        raise ValueError(f'hmf_truth shape mismatch: {hmf_truth.shape} vs {m_centers.shape}')
    if hmf_tinker is not None and hmf_tinker.shape != m_centers.shape:
        raise ValueError(f'hmf_tinker shape mismatch: {hmf_tinker.shape} vs {m_centers.shape}')

    # ── Plot ──────────────────────────────────────────────────────────────
    plot_hmf_summary(
        m_centers=m_centers,
        hmf_samples=hmf_samples,
        tinker_curve=hmf_tinker,
        hmf_truth=hmf_truth,
        mass_resolution_limit=mass_resolution_limit,
        save_dir=save_dir,
        run_name=run_name,
        ratio_ymin=args.ratio_ymin,
        ratio_ymax=args.ratio_ymax,
    )
    print(f'\nDone. Plot saved to {save_dir}/')


if __name__ == '__main__':
    main()
