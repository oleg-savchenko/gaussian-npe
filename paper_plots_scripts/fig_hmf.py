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
import json
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

RHO_CRIT_H2_MSUN_MPC3 = 2.77536627e11

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


# ── Poisson + cosmic-variance helpers ────────────────────────────────────────

def _window_tophat_spherical(x: np.ndarray) -> np.ndarray:
    """Spherical tophat window W(x)=3(sin x - x cos x)/x^3 with small-x expansion."""
    x = np.asarray(x, dtype=np.float64)
    w = np.empty_like(x, dtype=np.float64)
    small = np.abs(x) < 1.0e-4
    xs = x[small]
    # W(x) = 1 - x^2/10 + x^4/280 + O(x^6)
    w[small] = 1.0 - (xs * xs) / 10.0 + (xs ** 4) / 280.0
    xm = x[~small]
    w[~small] = 3.0 * (np.sin(xm) - xm * np.cos(xm)) / (xm ** 3)
    return w


def _sigma_r_from_linear_pk(k: np.ndarray, pk: np.ndarray, radius_mpc_h: float) -> float:
    """Linear-theory sigma(R) from P(k) using a spherical top-hat window."""
    k = np.asarray(k, dtype=np.float64)
    pk = np.asarray(pk, dtype=np.float64)
    if k.ndim != 1 or pk.ndim != 1 or k.size != pk.size:
        raise ValueError('k and pk must be 1D arrays of identical length.')
    if radius_mpc_h <= 0.0:
        raise ValueError('radius_mpc_h must be > 0.')
    w = _window_tophat_spherical(k * float(radius_mpc_h))
    integrand = (k ** 2) * pk * (w ** 2)
    return float(np.sqrt(np.trapz(integrand, k) / (2.0 * np.pi ** 2)))


def _mass_to_radius_mpc_h(mass_msun_h: np.ndarray, omega_m: float) -> np.ndarray:
    """Convert halo mass [Msun/h] to Lagrangian top-hat radius [Mpc/h]."""
    mass_msun_h = np.asarray(mass_msun_h, dtype=np.float64)
    rho_m = float(omega_m) * RHO_CRIT_H2_MSUN_MPC3
    return ((3.0 * mass_msun_h) / (4.0 * np.pi * rho_m)) ** (1.0 / 3.0)


def _st_halo_bias_from_sigma(sigma_m: np.ndarray) -> np.ndarray:
    """Sheth-Tormen-like large-scale halo bias approximation b(M)."""
    sigma_m = np.asarray(sigma_m, dtype=np.float64)
    delta_c = 1.686
    a = 0.707
    p = 0.3
    nu = delta_c / np.maximum(sigma_m, 1.0e-10)
    anu2 = a * (nu ** 2)
    return 1.0 + (anu2 - 1.0) / delta_c + (2.0 * p) / (delta_c * (1.0 + anu2 ** p))


def _infer_boxsize_from_sample_dirs(sample_dirs: list[str] | None, fallback_boxsize: float) -> float:
    """Infer boxsize [Mpc/h] from sample metadata when available."""
    if sample_dirs is None:
        return float(fallback_boxsize)
    for out_dir in sample_dirs:
        meta = os.path.join(str(out_dir), 'metadata.json')
        if not os.path.isfile(meta):
            continue
        try:
            payload = json.load(open(meta))
        except Exception:
            continue
        if 'boxsize_mpc_over_h' in payload:
            try:
                return float(payload['boxsize_mpc_over_h'])
            except Exception:
                pass
    return float(fallback_boxsize)


def compute_poisson_plus_cosmic_variance_band(
    *,
    mass_centers: np.ndarray,
    log_edges: np.ndarray,
    reference_hmf: np.ndarray,
    boxsize_mpc_h: float,
    omega_m: float,
    k: np.ndarray | None,
    pk: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a 1σ band around a reference HMF using:
      (σ_n / n)^2 ≈ 1/N_bin + [b(M) σ_V]^2.

    If (k, pk) is None, falls back to Poisson-only.
    """
    m = np.asarray(mass_centers, dtype=np.float64)
    edges = np.asarray(log_edges, dtype=np.float64)
    ref = np.asarray(reference_hmf, dtype=np.float64)
    if m.ndim != 1 or ref.ndim != 1 or m.size != ref.size:
        raise ValueError('mass_centers and reference_hmf must be 1D with identical shape.')
    if edges.ndim != 1 or edges.size != (m.size + 1):
        raise ValueError('log_edges must have length len(mass_centers)+1.')

    dlogm = np.diff(edges)
    volume = float(boxsize_mpc_h) ** 3
    n_bin = ref * volume * dlogm
    n_bin = np.where(n_bin > 0.0, n_bin, np.nan)
    rel_poisson = 1.0 / np.sqrt(n_bin)

    if (k is not None) and (pk is not None):
        r_mass = _mass_to_radius_mpc_h(m, omega_m=float(omega_m))
        sigma_m = np.asarray([_sigma_r_from_linear_pk(k, pk, float(r)) for r in r_mass], dtype=np.float64)
        b_m = _st_halo_bias_from_sigma(sigma_m)
        r_box_eff = (3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0)
        sigma_v = _sigma_r_from_linear_pk(k, pk, float(r_box_eff))
        rel_sample = b_m * float(sigma_v)
    else:
        rel_sample = np.zeros_like(rel_poisson)

    rel_tot = np.sqrt(rel_poisson ** 2 + rel_sample ** 2)
    lo = ref * (1.0 - rel_tot)
    hi = ref * (1.0 + rel_tot)
    lo = np.where(np.isfinite(lo) & (lo > 0.0), lo, np.nan)
    hi = np.where(np.isfinite(hi) & (hi > 0.0), hi, np.nan)
    return lo, hi


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_hmf_summary(
    *,
    m_centers: np.ndarray,
    hmf_samples: np.ndarray,
    tinker_curve: np.ndarray | None,
    hmf_truth: np.ndarray | None,
    mass_resolution_limit: float | None,
    comic_variance_band: tuple[np.ndarray, np.ndarray] | None,
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
    comic_band = None
    if comic_variance_band is not None:
        lo, hi = comic_variance_band
        comic_band = (
            np.asarray(lo, dtype=np.float64) * gpc_factor,
            np.asarray(hi, dtype=np.float64) * gpc_factor,
        )

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

    if comic_band is not None:
        ax.fill_between(
            m_centers,
            comic_band[0],
            comic_band[1],
            color='0.6',
            alpha=0.25,
            lw=0.0,
            label='Poisson + cosmic variance',
            zorder=1,
        )

    ax.fill_between(m_centers, s2_lo, s2_hi,
                    color=color_rs, alpha=0.20, lw=0.0, label=r'$\pm 2\sigma$', zorder=2)
    ax.fill_between(m_centers, s1_lo, s1_hi,
                    color=color_rs, alpha=0.40, lw=0.0, label=r'$\pm 1\sigma$', zorder=3)
    ax.plot(m_centers, mu, color=color_rs, lw=2.0, label='Posterior resimulations', zorder=4)

    if tinker is not None:
        ax.plot(m_centers, tinker, color='0.5', lw=2.0, ls=':', label='Tinker', zorder=5)
    if truth is not None:
        ax.plot(m_centers, truth, color='k', ls='--', lw=1.8,
                marker='x', ms=6, mew=1.6, label='Ground truth', zorder=6)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(r'$dn/d\log_{10}M\ [(h/{\rm Gpc})^3]$', fontsize=16)
    ax.set_xlabel(r'Mass [$M_\odot\,h^{-1}$]', fontsize=16)
    ax.tick_params(labelsize=16)
    ax.grid(alpha=0.25, which='both', ls=':')

    # Reorder legend: Ground truth, Posterior resimulations, 1σ, 2σ, Mass resolution limit
    handles, labels = ax.get_legend_handles_labels()
    order = ['Ground truth', 'Posterior resimulations',
             r'$\pm 1\sigma$', r'$\pm 2\sigma$', 'Poisson + cosmic variance',
             'Mass resolution limit', 'Tinker']
    label_to_handle = dict(zip(labels, handles))
    ordered_handles = [label_to_handle[l] for l in order if l in label_to_handle]
    ordered_labels  = [l for l in order if l in label_to_handle]
    ax.legend(
        ordered_handles,
        ordered_labels,
        framealpha=0.95,
        fontsize=16,
        loc='best',
        # Restrict the "best" search box to start 5% from the left edge.
        bbox_to_anchor=(0.05, 0.0, 0.95, 1.0),
        bbox_transform=ax.transAxes,
    )

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
        '--plot_comic_variance', dest='plot_comic_variance',
        action=argparse.BooleanOptionalAction, default=False,
        help='Overlay a semi-transparent gray 1σ band from Poisson + cosmic variance estimate.',
    )
    # Alias with corrected spelling; kept hidden to preserve requested CLI flag.
    parser.add_argument(
        '--plot_cosmic_variance', dest='plot_comic_variance',
        action=argparse.BooleanOptionalAction, help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--cv_boxsize_mpc_h', type=float, default=1000.0,
        help='Fallback box size [Mpc/h] used in cosmic-variance estimate when metadata is unavailable.',
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

    def _resolve(keys):
        """Return the value of the first key found in data, or raise."""
        for k in keys:
            if k in data.files:
                return np.asarray(data[k], dtype=np.float64)
        raise KeyError(f'None of {keys} found in {arrays_path}. Available: {data.files}')

    def _optional(*keys):
        for k in keys:
            if k in data.files:
                arr = np.asarray(data[k], dtype=np.float64)
                return arr if arr.size > 0 else None
        return None

    m_centers             = _resolve(['hmf_mass_centers', 'posterior_hmf_mass_centers'])
    hmf_log_edges         = _resolve(['hmf_log_edges',    'posterior_hmf_log_edges'])
    hmf_samples           = _resolve(['hmf_samples',      'posterior_hmf_samples'])
    mass_resolution_limit = float(_resolve(['mass_resolution_limit']))
    sample_output_dirs_arr = data['sample_output_dirs'] if 'sample_output_dirs' in data.files else None
    sample_output_dirs = None
    if sample_output_dirs_arr is not None:
        sample_output_dirs = [str(x) for x in np.asarray(sample_output_dirs_arr).tolist()]

    hmf_truth  = _optional('hmf_truth',  'truth_emulated_hmf',        'posterior_hmf_original_truth') if args.show_truth  else None
    hmf_tinker = _optional('hmf_tinker', 'posterior_hmf_tinker') if args.show_tinker else None

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

    # ── Optional Poisson + cosmic-variance band ─────────────────────────
    comic_variance_band = None
    if args.plot_comic_variance:
        ref_for_cv = hmf_truth if hmf_truth is not None else np.nanmean(hmf_samples, axis=0)
        ref_for_cv = np.asarray(ref_for_cv, dtype=np.float64)

        boxsize_cv = _infer_boxsize_from_sample_dirs(
            sample_dirs=sample_output_dirs,
            fallback_boxsize=float(args.cv_boxsize_mpc_h),
        )
        print(f'Comic-variance band: using boxsize={boxsize_cv:.3f} Mpc/h')

        k_cv = None
        pk_cv = None
        if args.class_pk_table_path is not None:
            table = np.loadtxt(os.path.abspath(args.class_pk_table_path))
            if table.ndim == 2 and table.shape[1] >= 2:
                k_cv = np.asarray(table[:, 0], dtype=np.float64)
                pk_cv = np.asarray(table[:, 1], dtype=np.float64)
                print('Comic-variance band: using CLASS P(k) from --class_pk_table_path')
        else:
            try:
                k_cv, pk_cv = _build_fiducial_class_pk(
                    z=args.class_z,
                    kmin=args.class_kmin,
                    kmax=args.class_kmax,
                    n_modes=args.class_n_modes,
                )
                print('Comic-variance band: using on-the-fly fiducial CLASS P(k)')
            except Exception as exc:
                print(f'Comic-variance band: CLASS unavailable ({exc!r}); falling back to Poisson-only.')

        comic_variance_band = compute_poisson_plus_cosmic_variance_band(
            mass_centers=m_centers,
            log_edges=hmf_log_edges,
            reference_hmf=ref_for_cv,
            boxsize_mpc_h=float(boxsize_cv),
            omega_m=float(args.omega_m),
            k=k_cv,
            pk=pk_cv,
        )

    # ── Plot ──────────────────────────────────────────────────────────────
    plot_hmf_summary(
        m_centers=m_centers,
        hmf_samples=hmf_samples,
        tinker_curve=hmf_tinker,
        hmf_truth=hmf_truth,
        mass_resolution_limit=mass_resolution_limit,
        comic_variance_band=comic_variance_band,
        save_dir=save_dir,
        run_name=run_name,
        ratio_ymin=args.ratio_ymin,
        ratio_ymax=args.ratio_ymax,
    )
    print(f'\nDone. Plot saved to {save_dir}/')


if __name__ == '__main__':
    main()
