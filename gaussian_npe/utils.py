import numpy as np
import torch
import Pk_library as PKL
from classy import Class
import matplotlib.pyplot as plt
import os
from scipy.stats import skew, kurtosis
from scipy.ndimage import gaussian_filter1d  # just for smoothing the histogram curves

def get_pk_class(cosmo_params, z, k, non_lin = False):
    """While cosmo_params is the cosmological parameters, z is a single redshift,
    while k is an array of k values in units of h/Mpc.
    The returned power spectrum is in units of (Mpc/h)³.
    """
    h = cosmo_params['h']
    cosmo_params.update({
        'output': 'mPk',
        'P_k_max_h/Mpc': np.max(k),
        'z_max_pk': z,
    })
    cosmo = Class() 
    cosmo.set(cosmo_params)
    cosmo.compute()
    if non_lin:
        pk_class = h**3*np.array([cosmo.pk(h*ki, z) for ki in k])
    else:
        pk_class = h**3*np.array([cosmo.pk_lin(h*ki, z) for ki in k])
    return pk_class

def growth_D_approx(cosmo_params, z):
    Om0_m = cosmo_params['Omega_cdm'] + cosmo_params['Omega_b']
    Om0_L = 1. - Om0_m
    Om_m = Om0_m * (1.+z)**3 / (Om0_L + Om0_m * (1.+z)**3)
    Om_L = Om0_L/(Om0_L+Om0_m*(1.+z)**3)
    return ((1.+z)**(-1)) * (5. * Om_m/2.) / (Om_m**(4./7.) - Om_L + (1.+Om_m/2.)*(1.+Om_L/70.))

# def get_k(box_parameters, device='cuda'):
#     """Set up the 3D k-vector Fourier grid and calculate its magnitude for each point of the grid.
#     """
#     box_size = box_parameters['box_size']
#     N = box_parameters['grid_res']
#     d = box_size / (2*np.pi*N)
#     freq = torch.fft.fftfreq(N, d = d, device = device)
#     kx, ky, kz = torch.meshgrid(freq, freq, freq, indexing = 'ij')
#     k = (kx**2 + ky**2 + kz**2)**0.5
#     k[0,0,0] = k[0,0,1]*1e-9    # Offset to avoid singularities (i.e., now k has no entries with zeros)
#     return k

def hartley(x, dim = (-3, -2, -1)):
    """
    Calculates the Hartley transform of the input field.
    axes: which dimensions to perform transformation on.
    """
    fx = torch.fft.fftn(x, dim = dim, norm = 'ortho')
    return (fx.real - fx.imag)

class Power_Spectrum_Sampler:
    def __init__(self, box_parameters, device = 'cuda', dim = 3):
        self.box_size = box_parameters['box_size']
        self.N = box_parameters['grid_res']
        self.shape = dim * (self.N,)
        self.hartley_dim = tuple(range(-dim, 0, 1))
        self.dim = dim
        self.device = device
        self.k = self.get_k(device = device)
        self.k_Nq = np.pi * self.N / self.box_size
        self.k_F = 2 * np.pi / self.box_size

    def get_k(self, device = None):
        """Set up the 3D k-vector Fourier grid and calculate its magnitude for each point of the grid.
        """
        if device is None:
            device = self.device
        d = self.box_size / (2*np.pi*self.N)
        freq = torch.fft.fftfreq(self.N, d = d, device = device)
        kx, ky, kz = torch.meshgrid(freq, freq, freq, indexing = 'ij')
        k = (kx**2 + ky**2 + kz**2)**0.5
        k[0,0,0] = k[0,0,1]*1e-9    # Offset to avoid singularities (i.e., now k has no entries with zeros)
        return k

    def get_prior_Q_factors(self, pk):
        """Return components of the prior precision matrix.

        Q_prior = UT * D * U

        Returns:
            UT, D, U: Linear operator, tensor, linear operator.
        """
        D = (pk(self.k.cpu().flatten()) * (self.N/self.box_size)**self.dim)**-1
        U = lambda x: hartley(x, dim = self.hartley_dim).flatten(-len(self.shape), -1)
        UT = lambda x: hartley(x.unflatten(-1, self.shape), dim = self.hartley_dim)
        return UT, D, U

    def sample(self, num_samples, pk = None, prior = None):
        """Sample a Gaussian random field with a given power spectrum.
        """
        if prior is None:
            prior = self.get_prior_Q_factors(pk)
        UT, D, U = prior[0], prior[1], prior[2]
        if num_samples == 1:
            r = torch.randn(D.shape, device = self.device)
        else:
            r = torch.randn(num_samples, *D.shape, device = self.device)
        x = UT(r * D**-0.5)
        return x

    def top_hat_filter(self, x, k_min = None, k_max = None):
        """Sharp cutoff filter in Fourier space.
        """
        if k_max == None:
            mask = (self.k <= k_min)
        elif k_min == None:
            mask = (self.k >= k_max)
        else:
            mask = ((self.k <= k_min) & (self.k >= k_max))
        mask.to(self.device)
        return hartley(mask * hartley(x, dim = self.hartley_dim), dim = self.hartley_dim)
    
    def sigmoid_filter(self, x, k_cut, w_cut):
        """Sigmoidal high-pass filter in Fourier space centred at k_cut with width w_cut.
        """
        mask = torch.sigmoid((self.k - k_cut)/w_cut)
        mask.to(self.device)
        return hartley(mask * hartley(x, dim = self.hartley_dim), dim = self.hartley_dim)
    
    def get_pk_pylians(self, delta, MAS = 'PCS'):
        """
        Compute the power spectrum of an input field using the Pylians library.
        """
        Pk = PKL.Pk(delta, self.box_size, axis=0, MAS=MAS, threads=1, verbose=False)    # Compute power spectrum

        # Pk is a python class containing the 1D, 2D, and 3D power spectra
        k_pylians = Pk.k3D    # 3D P(k)
        pk_pylians = Pk.Pk[:, 0]    # Monopole
        return k_pylians, pk_pylians


def plot_samples_analysis(delta_z127, delta_z0, samples, z_MAP, box,
                          cosmo_params=None, MAS=None,
                          save_dir='./plots', run_name=''):
    """Plot field slices and summary statistics (P(k), T(k), C(k)) for posterior samples.

    Produces five figures:
      1. Field slices: true IC, one posterior sample, and residual (3x3 grid).
      2. Truth vs MAP vs posterior std (1x3).
      3. Summary statistics: power spectrum, transfer function, and cross-correlation
         of the samples and MAP estimate relative to the ground truth, with 1/2-sigma bands.
      4. 1-point PDF comparison with skewness and kurtosis annotations.
      5. Reduced bispectrum Q(theta) for two triangle configurations.

    Parameters
    ----------
    delta_z127 : np.ndarray, shape (N, N, N)
        True initial conditions field (float32).
    delta_z0 : np.ndarray, shape (N, N, N)
        Observed final conditions field (float32).
    samples : np.ndarray, shape (n_samples, N, N, N)
        Posterior samples (float32).
    z_MAP : np.ndarray, shape (N, N, N)
        MAP estimate of the initial conditions (float32).
    box : Power_Spectrum_Sampler
        Box object (used for k-grid info and power spectrum computation).
    cosmo_params : dict, optional
        Cosmological parameters for CLASS. If provided, the linear theory P(k) is shown.
    MAS : str, optional
        Mass assignment scheme for Pylians (e.g. 'PCS'). Default None (no correction).
    save_dir : str
        Directory to save the output plots.
    run_name : str
        Suffix appended to filenames.
    """
    out_dir = os.path.join(save_dir, run_name) if run_name else save_dir
    os.makedirs(out_dir, exist_ok=True)

    plt.rcParams['figure.facecolor'] = 'white'

    delta_z127 = delta_z127.astype('f')
    delta_z0 = delta_z0.astype('f')
    samples = np.asarray(samples).astype('f')
    z_MAP = z_MAP.astype('f')

    sample = samples[0]
    std = samples.std(axis=0)
    residual = sample - delta_z127

    # ── Figure 1: field slices (3×3) ──────────────────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(16, 18), sharey=True)
    vmin, vmax = -3, 3
    row_labels = ['True IC', 'Posterior sample', 'Residual']
    row_data = [delta_z127, sample, residual]
    row_vlims = [(vmin, vmax), (vmin, vmax), (vmin / 2, vmax / 2)]

    N = delta_z127.shape[0]
    slices = [N // 4, N // 4 + N // 8, N // 2]

    for row, (data, label, (lo, hi)) in enumerate(zip(row_data, row_labels, row_vlims)):
        for col, s in enumerate(slices):
            im = axes[row, col].imshow(data[s], origin='lower', cmap='seismic', vmin=lo, vmax=hi)
            axes[row, col].set_title(f'{label}, slice {s}')
            axes[row, col].set_xlabel('x (voxels)', fontsize=14)
            if col == 0:
                axes[row, col].set_ylabel('y (voxels)', fontsize=14)
        cbar_ax = fig.add_axes([
            axes[row, 2].get_position().x1 + 0.01,
            axes[row, 2].get_position().y0,
            0.01,
            axes[row, 2].get_position().height,
        ])
        plt.colorbar(im, cax=cbar_ax)

    fig.savefig(os.path.join(out_dir, f'field_slices_{run_name}.png'), bbox_inches='tight')

    # ── Figure 2: truth / MAP / posterior std ─────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
    slice_idx = N // 2

    panels = [
        (delta_z127[slice_idx], 'True field', 'seismic', vmin, vmax),
        (z_MAP[slice_idx], 'MAP estimate', 'seismic', vmin, vmax),
        (std[slice_idx], 'Posterior std', 'Purples', None, None),
    ]
    for ax, (data, title, cmap, lo, hi) in zip(axes, panels):
        im = ax.imshow(data, origin='lower', cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(f'{title}, slice {slice_idx}')
        ax.set_xlabel('x (voxels)', fontsize=14)
        cbar_ax = fig.add_axes([
            ax.get_position().x1 + 0.01,
            ax.get_position().y0,
            0.01,
            ax.get_position().height,
        ])
        plt.colorbar(im, cax=cbar_ax)
    axes[0].set_ylabel('y (voxels)', fontsize=14)

    fig.savefig(os.path.join(out_dir, f'truth_MAP_std_{run_name}.png'), bbox_inches='tight')

    # ── Compute summary statistics ────────────────────────────────────────
    box_cpu = Power_Spectrum_Sampler(
        {'box_size': box.box_size, 'grid_res': box.N}, device='cpu',
    )
    k_Nq = box_cpu.k_Nq

    # True field P(k)
    k_pylians, pk_ic = box_cpu.get_pk_pylians(delta_z127, MAS=MAS)

    # MAP P(k) and cross-correlation with truth
    _, pk_MAP = box_cpu.get_pk_pylians(z_MAP, MAS=MAS)
    tk_MAP = np.sqrt(pk_MAP / pk_ic)

    Pk_MAP = PKL.XPk([z_MAP, delta_z127], box_cpu.box_size, axis=0,
                      MAS=[MAS, MAS], threads=1)
    xpk_MAP = Pk_MAP.XPk[:, 0, 0] / (Pk_MAP.Pk[:, 0, 0] * Pk_MAP.Pk[:, 0, 1])**0.5

    # Cross-correlation between IC and final field (linear baseline)
    Pk_lin = PKL.XPk([delta_z127, delta_z0], box_cpu.box_size, axis=0,
                      MAS=[MAS, MAS], threads=1)
    xpk_linear = Pk_lin.XPk[:, 0, 0] / (Pk_lin.Pk[:, 0, 0] * Pk_lin.Pk[:, 0, 1])**0.5

    # Samples P(k), T(k), C(k)
    pks, tks, xpks = [], [], []
    for i in range(len(samples)):
        s_i = samples[i]
        _, pk_i = box_cpu.get_pk_pylians(s_i, MAS=MAS)
        Pk_i = PKL.XPk([s_i, delta_z127], box_cpu.box_size, axis=0,
                        MAS=[MAS, MAS], threads=1)
        pks.append(pk_i)
        tks.append(np.sqrt(pk_i / pk_ic))
        xpks.append(Pk_i.XPk[:, 0, 0] / (Pk_i.Pk[:, 0, 0] * Pk_i.Pk[:, 0, 1])**0.5)

    pks = np.array(pks)
    tks = np.array(tks)
    xpks = np.array(xpks)

    # Optional: linear theory P(k) from CLASS
    if cosmo_params is not None:
        k_lin = np.logspace(np.log10(1e-4), np.log10(10), 100)
        pk_class_z0 = get_pk_class(cosmo_params, 0, k_lin)

    # ── Figure 3: summary statistics ──────────────────────────────────────
    fig, axs = plt.subplots(3, sharex=True, sharey=False, height_ratios=[2, 1, 1])
    fig.set_size_inches(4, 8)
    color_samples = 'forestgreen'

    # ── P(k) ──
    ax = axs[0]
    ax.plot(k_pylians, pk_ic, marker='.', markersize=0.5, lw=0.5,
            label='True', zorder=10)
    ax.plot(k_pylians, pks.mean(0), lw=0.5, color=color_samples, label='Samples')
    ax.fill_between(k_pylians,
                     pks.mean(0) - pks.std(0), pks.mean(0) + pks.std(0),
                     alpha=0.75, color=color_samples)
    ax.fill_between(k_pylians,
                     pks.mean(0) - 2 * pks.std(0), pks.mean(0) + 2 * pks.std(0),
                     alpha=0.25, color=color_samples)
    ax.plot(k_pylians, pk_MAP, color='magenta', label='MAP', alpha=0.75, lw=0.5)
    if cosmo_params is not None:
        ax.plot(k_lin, pk_class_z0, label='Linear', color='black', alpha=0.3, lw=0.5)
    ax.axvline(x=k_Nq, color='r', ls='--', lw=0.5, alpha=0.5, label=r'$k_{\rm{Nyq}}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(r'$P(k)$', fontsize=16)
    ax.legend(facecolor='white', edgecolor='none', framealpha=0.8)
    ax.set_ylim([5e2, 5e4])
    ax.set_xlim(left=k_pylians[0], right=k_Nq + 0.075)
    ax.grid(which='both', alpha=0.125)

    # ── T(k) ──
    ax = axs[1]
    ax.plot(k_pylians, tks.mean(0), color=color_samples)
    ax.fill_between(k_pylians,
                     tks.mean(0) - tks.std(0), tks.mean(0) + tks.std(0),
                     alpha=0.75, color=color_samples)
    ax.fill_between(k_pylians,
                     tks.mean(0) - 2 * tks.std(0), tks.mean(0) + 2 * tks.std(0),
                     alpha=0.25, color=color_samples)
    ax.plot(k_pylians, tk_MAP, color='magenta', alpha=0.75, lw=0.5)
    ax.axvline(x=k_Nq, color='r', ls='--', lw=0.5, alpha=0.5)
    ax.axhline(1.0, color='k', ls='--', lw=0.5)
    ax.set_xscale('log')
    ax.set_ylabel(r'$T(k)$', fontsize=16)
    ax.set_ylim(0.93, 1.07)
    ax.set_yticks([0.95, 1.0, 1.05])
    ax.grid(which='both', alpha=0.1)

    # ── C(k) ──
    ax = axs[2]
    ax.plot(k_pylians, xpk_MAP, color='magenta', alpha=0.75, lw=0.5)
    ax.plot(k_pylians, xpk_linear, alpha=0.75, lw=0.5, color='orange', label=r'$z=0$')
    ax.plot(k_pylians, xpks.mean(0), color=color_samples, lw=0.25)
    ax.fill_between(k_pylians,
                     xpks.mean(0) - xpks.std(0), xpks.mean(0) + xpks.std(0),
                     alpha=0.75, color=color_samples)
    ax.fill_between(k_pylians,
                     xpks.mean(0) - 2 * xpks.std(0), xpks.mean(0) + 2 * xpks.std(0),
                     alpha=0.25, color=color_samples)
    ax.axvline(x=k_Nq, color='r', ls='--', lw=0.5, alpha=0.5)
    ax.axhline(1.0, color='k', ls='--', lw=0.5)
    ax.set_xscale('log')
    ax.set_ylabel(r'$C(k)$', fontsize=16)
    ax.set_xlabel(r'$k$ [$h / \rm{Mpc}$]', fontsize=14)
    ax.set_ylim([-0.2, 1.2])
    ax.set_yticks([0, 0.5, 1.0])
    ax.grid(which='both', alpha=0.1)
    ax.legend(facecolor='white', edgecolor='none', framealpha=0.9, loc='lower left')

    plt.subplots_adjust(hspace=0)
    fig.savefig(os.path.join(out_dir, f'summary_stats_{run_name}.png'), bbox_inches='tight')

    # ── Figure 4: 1-point PDF with skewness & kurtosis ───────────────────
    fig, ax = plot_1pt_pdf_with_skew_kurt(delta_z127, samples)
    fig.savefig(os.path.join(out_dir, f'1pt_pdf_{run_name}.png'), bbox_inches='tight')

    # ── Figure 5: reduced bispectrum Q(theta) ────────────────────────────
    theta = np.linspace(0, np.pi, 25)
    # Two triangle configs: equilateral (k1=k2=0.1) and squeezed (k1=0.05, k2=0.1)
    bispec_configs = [
        {'k1': 0.1, 'k2': 0.1, 'label': r'$k_1 = k_2 = 0.1\;h/\mathrm{Mpc}$'},
        {'k1': 0.05, 'k2': 0.1, 'label': r'$k_1 = 0.05,\; k_2 = 0.1\;h/\mathrm{Mpc}$'},
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    theta_deg = np.degrees(theta)

    for ax, cfg in zip(axes, bispec_configs):
        # True field bispectrum
        BBk_true = PKL.Bk(delta_z127, box_cpu.box_size,
                          cfg['k1'], cfg['k2'], theta, MAS, threads=1)
        Qk_true = BBk_true.Q

        # Sample bispectra
        Qks = []
        for i in range(len(samples)):
            BBk_i = PKL.Bk(samples[i], box_cpu.box_size,
                           cfg['k1'], cfg['k2'], theta, MAS, threads=1)
            Qks.append(BBk_i.Q)
        Qks = np.array(Qks)

        ax.plot(theta_deg, Qk_true, marker='.', markersize=3, lw=1,
                label='True', zorder=10)
        ax.plot(theta_deg, Qks.mean(0), lw=1, color=color_samples, label='Samples')
        ax.fill_between(theta_deg,
                         Qks.mean(0) - Qks.std(0),
                         Qks.mean(0) + Qks.std(0),
                         alpha=0.75, color=color_samples)
        ax.fill_between(theta_deg,
                         Qks.mean(0) - 2 * Qks.std(0),
                         Qks.mean(0) + 2 * Qks.std(0),
                         alpha=0.25, color=color_samples)

        ax.set_xlabel(r'$\theta$ [deg]', fontsize=14)
        ax.set_title(cfg['label'], fontsize=11)
        ax.grid(alpha=0.15)
        ax.legend(facecolor='white', edgecolor='none', framealpha=0.8, fontsize=9)

    axes[0].set_ylabel(r'$Q(\theta)$', fontsize=16)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'bispectrum_{run_name}.png'), bbox_inches='tight')

def plot_1pt_pdf_with_skew_kurt(
    delta,              # (N,N,N) true field
    samples,            # (B,N,N,N) reconstructed samples
    nbins=120,
    smooth_sigma=1.5,   # set to 0 to disable smoothing
    xlim=None,          # e.g. (-6, 6); if None, choose from data percentiles
    title=None,
):
    """
    Reproduce a "paper-style" 1-point PDF comparison plot:
      - solid line: mean PDF across reconstructed samples
      - dashed line: PDF of the true field
      - text: mean ± std over samples of skewness (γ1) and excess kurtosis (γ2)

    Notes:
      - Skewness/kurtosis are computed per sample (each sample is flattened to N^3 values).
      - The displayed PDF for samples is the mean of per-sample histograms (shared bins).
    """
    delta = np.asarray(delta)
    samples = np.asarray(samples)
    assert delta.ndim == 3, f"delta must be (N,N,N), got {delta.shape}"
    assert samples.ndim == 4, f"samples must be (B,N,N,N), got {samples.shape}"
    assert samples.shape[1:] == delta.shape, "samples and delta spatial shapes must match"

    B = samples.shape[0]
    x_true = delta.ravel().astype(np.float64)
    x_samp = samples.reshape(B, -1).astype(np.float64)  # (B, N^3)

    # Optional: restrict range robustly
    if xlim is None:
        lo = np.percentile(np.concatenate([x_true, x_samp.ravel()]), 0.1)
        hi = np.percentile(np.concatenate([x_true, x_samp.ravel()]), 99.9)
        pad = 0.05 * (hi - lo)
        xlim = (lo - pad, hi + pad)

    # Shared bins for all histograms
    edges = np.linspace(xlim[0], xlim[1], nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # True histogram (density)
    h_true, _ = np.histogram(x_true[np.isfinite(x_true)], bins=edges, density=True)

    # Per-sample histograms (density), then average
    h_samps = np.empty((B, nbins), dtype=np.float64)
    for b in range(B):
        xb = x_samp[b]
        xb = xb[np.isfinite(xb)]
        h_samps[b], _ = np.histogram(xb, bins=edges, density=True)
    h_mean = h_samps.mean(axis=0)

    # Smooth for a clean "paper" look (optional)
    if smooth_sigma and smooth_sigma > 0:
        h_true_plot = gaussian_filter1d(h_true, sigma=smooth_sigma)
        h_mean_plot = gaussian_filter1d(h_mean, sigma=smooth_sigma)
    else:
        h_true_plot = h_true
        h_mean_plot = h_mean

    # Skewness and excess kurtosis per sample (over voxels), then mean ± std over samples
    g1 = skew(x_samp, axis=1, bias=False, nan_policy="omit")  # (B,)
    g2 = kurtosis(x_samp, axis=1, fisher=True, bias=False, nan_policy="omit")  # (B,) excess kurtosis

    g1_mean, g1_std = float(np.mean(g1)), float(np.std(g1, ddof=1))
    g2_mean, g2_std = float(np.mean(g2)), float(np.std(g2, ddof=1))

    # Plot
    fig, ax = plt.subplots(figsize=(6.2, 4.4), dpi=150)

    ax.plot(centers, h_mean_plot, lw=2.0, label=r'$\pi(x_i\,|\,\mathbf{d})$')
    ax.plot(centers, h_true_plot, lw=2.0, ls='--', color='k', label=r'$\pi(x^{\mathrm{truth}})$')

    ax.set_xlim(*xlim)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$\pi(x_i\,|\,\mathbf{d})$')

    if title is not None:
        ax.set_title(title)

    txt = (
        rf'$\gamma_1 = {g1_mean:.1e} \pm {g1_std:.1e}$' + "\n" +
        rf'$\gamma_2 = {g2_mean:.1e} \pm {g2_std:.1e}$'
    )
    ax.text(0.06, 0.92, txt, transform=ax.transAxes, va='top', ha='left')

    ax.legend(loc='upper right', frameon=True)
    ax.grid(False)

    fig.tight_layout()
    return fig, ax
