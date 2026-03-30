"""
1-point PDF of posterior samples vs true ICs with skewness and kurtosis annotations.

Produces one figure saved to paper_plots_scripts/{RUN_NAME}/:
  - 1_1pt_pdf_{RUN_NAME}.pdf : 1-point probability density function of the true
    IC field (dashed black) overlaid with the mean sample PDF (solid green) and
    ±1σ / ±2σ uncertainty bands across posterior samples, annotated with mean,
    std, skewness (γ₁), and excess kurtosis (γ₂).

Usage:
    python paper_plots_scripts/fig3_1pt.py \\
        --model_dir paper_test_runs/runs/260303_224547_sweep_networks/260303_224627_net_IsotropicD

    python paper_plots_scripts/fig3_1pt.py \\
        --model_dir paper_test_runs/runs/260303_224547_sweep_networks/260303_224627_net_IsotropicD \\
        --num_samples 200 \\
        --no_latex
"""

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from scipy.ndimage import gaussian_filter1d

from gaussian_npe import utils
from _common import add_common_args, load_model_and_generate_samples


def plot_1pt_pdf(delta_z127, samples, z_MAP, save_dir, run_name,
                 nbins=120, smooth_sigma=1.5, xlim=None):
    """1-point PDF with ±1σ / ±2σ uncertainty bands across posterior samples.

    Parameters
    ----------
    delta_z127 : np.ndarray, shape (N, N, N)
        True IC field in internal units.
    samples : np.ndarray, shape (n_samples, N, N, N) or None
        Posterior samples in internal units. If None, uses z_MAP instead.
    z_MAP : np.ndarray, shape (N, N, N)
        MAP estimate in internal units.
    save_dir : str
    run_name : str
    nbins : int
    smooth_sigma : float
        Gaussian smoothing applied to each histogram before plotting.
    xlim : tuple or None
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.rcParams['figure.facecolor'] = 'white'

    color_samples = 'forestgreen'

    has_samples = samples is not None and np.asarray(samples).size > 0
    if has_samples:
        samples = np.asarray(samples).astype(np.float64)
    else:
        samples = z_MAP[None].astype(np.float64)

    B = samples.shape[0]
    x_true = delta_z127.ravel().astype(np.float64)
    x_samp = samples.reshape(B, -1)

    # Shared bin range
    if xlim is None:
        lo  = np.percentile(np.concatenate([x_true, x_samp.ravel()]), 0.1)
        hi  = np.percentile(np.concatenate([x_true, x_samp.ravel()]), 99.9)
        pad = 0.05 * (hi - lo)
        xlim = (lo - pad, hi + pad)

    edges   = np.linspace(xlim[0], xlim[1], nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # True histogram
    h_true, _ = np.histogram(x_true[np.isfinite(x_true)], bins=edges, density=True)

    # Per-sample histograms
    h_samps = np.empty((B, nbins))
    for b in range(B):
        xb = x_samp[b]
        h_samps[b], _ = np.histogram(xb[np.isfinite(xb)], bins=edges, density=True)

    h_mean = h_samps.mean(axis=0)
    h_std  = h_samps.std(axis=0)

    # Smooth
    if smooth_sigma and smooth_sigma > 0:
        h_true_plot = gaussian_filter1d(h_true, sigma=smooth_sigma)
        h_mean_plot = gaussian_filter1d(h_mean, sigma=smooth_sigma)
        h_std_plot  = gaussian_filter1d(h_std,  sigma=smooth_sigma)
    else:
        h_true_plot = h_true
        h_mean_plot = h_mean
        h_std_plot  = h_std

    def _sci(val, err=None, sig=1):
        """Format val (± err) as LaTeX (m ± e) × 10^n.

        Decimal places are driven by the error magnitude so the error never
        rounds to zero: e.g. val=1.1e-1, err=3e-3 → (1.10 ± 0.03)×10⁻¹.
        """
        ref = abs(val) if abs(val) > 1e-30 else (abs(err) if err and abs(err) > 1e-30 else 1.0)
        exp = int(np.floor(np.log10(ref)))
        m = val / 10**exp
        if err is not None:
            e = err / 10**exp
            dec = (max(sig, -int(np.floor(np.log10(e))) + sig - 1)
                   if e > 1e-30 else sig)
            return rf'({m:.{dec}f} \pm {e:.{dec}f}) \times 10^{{{exp}}}'
        return rf'{m:.{sig}f} \times 10^{{{exp}}}'

    # Statistics for annotations
    x_true_fin = x_true[np.isfinite(x_true)]
    mu_true  = float(x_true_fin.mean())
    sig_true = float(x_true_fin.std(ddof=1))
    g1_true  = float(skew(x_true_fin, bias=False))
    g2_true  = float(kurtosis(x_true_fin, fisher=True, bias=False))

    mu   = x_samp.mean(axis=1)
    sig  = x_samp.std(axis=1, ddof=1)
    g1   = skew(x_samp, axis=1, bias=False, nan_policy='omit')
    g2   = kurtosis(x_samp, axis=1, fisher=True, bias=False, nan_policy='omit')
    mu_mean,  mu_std  = float(np.mean(mu)),  float(np.std(mu,  ddof=1))
    sig_mean, sig_std = float(np.mean(sig)), float(np.std(sig, ddof=1))
    g1_mean,  g1_std  = float(np.mean(g1)),  float(np.std(g1,  ddof=1))
    g2_mean,  g2_std  = float(np.mean(g2)),  float(np.std(g2,  ddof=1))

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.2, 4.4), dpi=150)

    # ±2σ band
    ax.fill_between(centers,
                    h_mean_plot - 2 * h_std_plot,
                    h_mean_plot + 2 * h_std_plot,
                    alpha=0.25, color=color_samples)
    # ±1σ band
    ax.fill_between(centers,
                    h_mean_plot - h_std_plot,
                    h_mean_plot + h_std_plot,
                    alpha=0.75, color=color_samples)
    # Mean line
    ax.plot(centers, h_mean_plot, lw=2.5, color=color_samples,
            label=r'$p(\boldsymbol{z}_i\,|\,\boldsymbol{x}_{\rm obs})$')
    # Truth line
    ax.plot(centers, h_true_plot, lw=4.0, ls='--', color='k', alpha=0.75,
            label=r'$p(\boldsymbol{z}_{\rm truth})$')

    ax.set_xlim(*xlim)
    ax.set_xlabel(r'$\boldsymbol{z}$', fontsize=24)
    ax.set_ylabel('PDF', fontsize=24)

    # Samples annotation (top left)
    txt_samples = (
        r'$\boldsymbol{z}_{\rm samples}$' + '\n' +
        rf'$\mu = {_sci(mu_mean, 2*mu_std)}$' + '\n' +
        rf'$\sigma = {_sci(sig_mean, 2*sig_std)}$' + '\n' +
        rf'$\gamma_1 = {_sci(g1_mean, 2*g1_std)}$' + '\n' +
        rf'$\gamma_2 = {_sci(g2_mean, 2*g2_std)}$'
    )
    ax.text(0.04, 0.97, txt_samples, transform=ax.transAxes,
            va='top', ha='left', fontsize=10)

    # Truth annotation (top right)
    txt_truth = (
        r'$\boldsymbol{z}_{\rm truth}$' + '\n' +
        rf'$\mu = {_sci(mu_true, sig=3)}$' + '\n' +
        rf'$\sigma = {_sci(sig_true, sig=3)}$' + '\n' +
        rf'$\gamma_1 = {_sci(g1_true, sig=3)}$' + '\n' +
        rf'$\gamma_2 = {_sci(g2_true, sig=3)}$'
    )
    ax.text(0.96, 0.97, txt_truth, transform=ax.transAxes,
            va='top', ha='right', fontsize=10)

    ax.legend(loc='lower center', frameon=True, bbox_to_anchor=(0.5, 0.025))
    ax.grid(False)
    fig.tight_layout()

    out = os.path.join(save_dir, f'1_1pt_pdf_{run_name}.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='1-point PDF with uncertainty bands (fig 3)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser, default_num_samples=200)
    return parser.parse_args()


def main():
    args = parse_args()
    utils.configure_matplotlib_style(use_latex=args.use_latex)
    d = load_model_and_generate_samples(args)
    rf = d['rescaling_factor']
    plot_1pt_pdf(
        delta_z127=d['delta_z127_int'] * rf,
        samples=d['samples_int'] * rf if d['samples_int'] is not None else None,
        z_MAP=d['z_MAP_int'] * rf,
        save_dir=d['plots_dir'],
        run_name=d['run_name'],
    )
    print(f"\nDone. Plot saved to {d['plots_dir']}/")


if __name__ == '__main__':
    main()
