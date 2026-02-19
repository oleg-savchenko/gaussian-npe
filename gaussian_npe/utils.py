import time
import numpy as np
import csv
import torch
import Pk_library as PKL
from classy import Class
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import skew, kurtosis, norm, kstest
from scipy.ndimage import gaussian_filter1d  # just for smoothing the histogram curves
from pytorch_lightning.callbacks import Callback


def configure_matplotlib_style(use_latex=False):
    """Set global matplotlib style. Call once at the start of a script before any plotting.

    Parameters
    ----------
    use_latex : bool
        If True, enable LaTeX rendering with Computer Modern serif fonts and the
        scienceplots 'science' style. Requires a working LaTeX installation and
        the scienceplots package. If False (default), use matplotlib's built-in
        rendering with no external dependencies.
    """
    if use_latex:
        import scienceplots  # noqa: F401 — registers the 'science' style
        plt.style.use('science')
        mpl.rc('text', usetex=True)
        mpl.rc('font', **{'family': 'serif', 'serif': ['cmr10']})
    else:
        mpl.rcdefaults()


def _compute_sample_pk(args):
    """Module-level worker: compute XPk (auto + cross) for one sample vs truth field."""
    s_i, delta_z127, box_size, MAS = args
    Pk_i = PKL.XPk([s_i, delta_z127], box_size, axis=0, MAS=[MAS, MAS], threads=1)
    pk_i = Pk_i.Pk[:, 0, 0]
    xpk_i = Pk_i.XPk[:, 0, 0] / np.sqrt(Pk_i.Pk[:, 0, 0] * Pk_i.Pk[:, 0, 1])
    return pk_i, xpk_i


def _compute_sample_bk(args):
    """Module-level worker: compute reduced bispectrum Q(theta) for one sample."""
    s_i, box_size, k1, k2, theta, MAS = args
    BBk_i = PKL.Bk(s_i, box_size, k1, k2, theta, MAS, threads=1)
    return BBk_i.Q

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

def hartley_np(x, axes=(-3, -2, -1)):
    """Numpy Hartley transform (matches the torch version)."""
    fx = np.fft.fftn(x, axes=axes, norm='ortho')
    return fx.real - fx.imag

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
        return k

    def get_prior_Q_factors(self, pk):
        """Return components of the prior precision matrix.

        Q_prior = UT * D * U

        Returns:
            UT, D, U: Linear operator, tensor, linear operator.
        """
        k_flat = self.k.cpu().flatten()
        # Replace k=0 with a dummy for the pk() call to avoid P(0)=0 → 1/0
        k_safe = k_flat.clone()
        k_safe[0] = self.k_F
        D = (pk(k_safe) * (self.N / self.box_size) ** self.dim) ** -1
        # monopole mode (k=0): mean overdensity is zero by construction on a
        # periodic box, so the prior precision is effectively infinite.
        D[0] = 1e12
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


def plot_training_curves(metrics_file, save_path, title='', config=None):
    """Plot training/validation loss and learning rate from a metrics CSV.

    Parameters
    ----------
    metrics_file : str
        Path to metrics.csv (one row per epoch: epoch,step,train_loss,val_loss,lr).
    save_path : str
        Full path (including filename) where the figure will be saved.
    title : str
        Figure suptitle.
    config : dict, optional
        Run configuration dict displayed as a text panel on the right.
    """
    if not os.path.exists(metrics_file):
        print(f'Metrics file not found: {metrics_file}')
        return

    with open(metrics_file, 'r') as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f'No data in metrics file: {metrics_file}')
        return

    rows = [r for r in rows if r['train_loss'] and r['val_loss']]
    epochs = np.array([int(r['epoch']) for r in rows])
    train_loss = np.array([float(r['train_loss']) for r in rows])
    val_loss = np.array([float(r['val_loss']) for r in rows])

    ncols = 3 if config is not None else 2
    width_ratios = [2, 2, 0.9] if config is not None else [1, 1]
    fig, axes = plt.subplots(
        1, ncols, figsize=(13 if config is not None else 10, 4),
        gridspec_kw={'width_ratios': width_ratios},
        constrained_layout=True,
    )
    ax1, ax2 = axes[0], axes[1]

    # Loss curves
    ax1.plot(epochs, train_loss, label='Train loss')
    ax1.plot(epochs, val_loss, label='Val loss')
    best = int(np.argmin(val_loss))
    ax1.scatter(epochs[best], val_loss[best],
                marker='*', s=120, color='goldenrod', zorder=10,
                label=f'Best val: {val_loss[best]:.4f}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(alpha=0.15)
    ax1.set_title(r'Training \& validation loss')

    # Learning rate (per epoch)
    if 'lr' in rows[0]:
        lr_vals = np.array([float(r['lr']) for r in rows])
        ax2.plot(epochs, lr_vals)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning rate')
        ax2.set_yscale('log')
        ax2.grid(alpha=0.15)
        ax2.set_title('Learning rate schedule')

    # Config panel
    if config is not None:
        ax3 = axes[2]
        ax3.set_axis_off()
        lines = []
        for k, v in config.items():
            if isinstance(v, float):
                v_str = f'{v:.6g}'
            elif isinstance(v, str) and len(v) > 35:
                v_str = '\u2026' + v[-32:]
            else:
                v_str = str(v)
            lines.append(f'{k}: {v_str}')
        # Pad to equal width so ha='center' (used to align the title above)
        # still renders as left-aligned text inside the box (monospace font).
        max_len = max(len(l) for l in lines)
        lines_padded = [l.ljust(max_len) for l in lines]
        ax3.text(
            0.5, 1.0, '\n'.join(lines_padded),
            transform=ax3.transAxes,
            fontsize=6.5,
            verticalalignment='top',
            horizontalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#f5f5f5',
                      edgecolor='#aaaaaa', alpha=0.9),
        )
        ax3.set_title('Config', fontsize=9)

    if title:
        fig.suptitle(title, fontsize=11)
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def plot_samples_analysis(delta_z127, delta_z0, samples, z_MAP, box,
                          cosmo_params=None, MAS=None,
                          Q_like_D=None, Q_prior_D=None,
                          save_dir='./plots', run_name='',
                          save_csv=False, n_workers=None):
    """Plot field slices and summary statistics (P(k), T(k), C(k)) for posterior samples.

    Produces up to six figures:
      1. Field slices: true IC, one posterior sample, and residual (3x3 grid).
      2. Truth vs MAP vs posterior std (1x3).
      3. Summary statistics: power spectrum, transfer function, and cross-correlation
         of the samples and MAP estimate relative to the ground truth, with 1/2-sigma bands.
      4. 1-point PDF comparison with skewness and kurtosis annotations.
      5. Reduced bispectrum Q(theta) for two triangle configurations.
      6. Precision matrix diagonals D_like, D_prior, D_posterior vs k (if provided).

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
    Q_like_D : np.ndarray, shape (N^3,), optional
        Diagonal of the likelihood precision matrix. If provided together with
        Q_prior_D, an additional Q-matrix diagnostic plot is produced.
    Q_prior_D : np.ndarray, shape (N^3,), optional
        Diagonal of the prior precision matrix.
    save_dir : str
        Directory to save the output plots.
    run_name : str
        Suffix appended to filenames.
    """
    out_dir = save_dir
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
    # XPk gives auto-spectra of both fields, so no separate PKL.Pk call needed
    Pk_MAP = PKL.XPk([z_MAP, delta_z127], box_cpu.box_size, axis=0,
                      MAS=[MAS, MAS], threads=1)
    pk_MAP = Pk_MAP.Pk[:, 0, 0]
    tk_MAP = np.sqrt(pk_MAP / pk_ic)
    xpk_MAP = Pk_MAP.XPk[:, 0, 0] / np.sqrt(Pk_MAP.Pk[:, 0, 0] * Pk_MAP.Pk[:, 0, 1])

    # Cross-correlation between IC and final field (linear baseline)
    Pk_lin = PKL.XPk([delta_z127, delta_z0], box_cpu.box_size, axis=0,
                      MAS=[MAS, MAS], threads=1)
    xpk_linear = Pk_lin.XPk[:, 0, 0] / np.sqrt(Pk_lin.Pk[:, 0, 0] * Pk_lin.Pk[:, 0, 1])

    # Samples P(k), T(k), C(k) — parallelised over samples
    pk_args = [(samples[i], delta_z127, box_cpu.box_size, MAS) for i in range(len(samples))]
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        pk_results = list(executor.map(_compute_sample_pk, pk_args))
    pks = np.array([r[0] for r in pk_results])
    xpks = np.array([r[1] for r in pk_results])
    tks = np.sqrt(pks / pk_ic)

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

        # Sample bispectra — parallelised over samples
        bk_args = [(samples[i], box_cpu.box_size, cfg['k1'], cfg['k2'], theta, MAS)
                   for i in range(len(samples))]
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            Qks = np.array(list(executor.map(_compute_sample_bk, bk_args)))

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

    # ── Figure 6: Q-matrix diagonals vs k ─────────────────────────────────
    if Q_like_D is not None and Q_prior_D is not None:
        k_flat = box.k.cpu().numpy().flatten()
        mask = (k_flat < k_Nq) & (k_flat > 1e-3)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(k_flat[mask], Q_like_D[mask], s=0.5, alpha=0.4,
                   label=r'$D_{\rm like}$')
        ax.scatter(k_flat[mask], Q_prior_D[mask], s=0.5, alpha=0.4,
                   label=r'$D_{\rm prior}$')
        ax.scatter(k_flat[mask], Q_like_D[mask] + Q_prior_D[mask], s=0.5, alpha=0.3,
                   label=r'$D_{\rm posterior}$')
        ax.axvline(x=k_Nq, color='r', linestyle='--', lw=1, label=r'$k_{\rm Nyq}$')

        ax.set_xscale('log')
        ax.set_xlabel(r'$k~[h\,{\rm Mpc}^{-1}]$', fontsize=14)
        ax.set_ylabel(r'$D(k)$', fontsize=14)
        ax.set_title(r'Precision matrix diagonals: $Q = U^T D\, U$')
        ax.legend(loc='upper right', markerscale=8)
        ax.grid(alpha=0.15)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'Q_diagonals_{run_name}.png'), bbox_inches='tight')

    # ── Optional: save diagnostic data files ─────────────────────────────────
    if save_csv:
        diag_dir = os.path.join(out_dir, 'diagnostics')
        os.makedirs(diag_dir, exist_ok=True)

        # Spectra CSV (k-dependent quantities)
        spectra_keys = [
            'k', 'pk_true', 'pk_MAP', 'pk_samples_mean', 'pk_samples_std',
            'tk_MAP', 'tk_samples_mean', 'tk_samples_std',
            'ck_MAP', 'ck_linear', 'ck_samples_mean', 'ck_samples_std',
        ]
        spectra_data = np.column_stack([
            k_pylians, pk_ic, pk_MAP, pks.mean(0), pks.std(0),
            tk_MAP, tks.mean(0), tks.std(0),
            xpk_MAP, xpk_linear, xpks.mean(0), xpks.std(0),
        ])
        np.savetxt(os.path.join(diag_dir, f'spectra_{run_name}.csv'),
                   spectra_data, delimiter=',',
                   header=','.join(spectra_keys), comments='')

        # Scalars CSV (1-point statistics)
        B = samples.shape[0]
        x_true_flat = delta_z127.ravel().astype(np.float64)
        x_samp = samples.reshape(B, -1).astype(np.float64)
        mu_samp = x_samp.mean(axis=1)           # (B,) per-sample spatial mean
        sig_samp = x_samp.std(axis=1, ddof=1)   # (B,) per-sample spatial std
        g1 = skew(x_samp, axis=1, bias=False, nan_policy='omit')
        g2 = kurtosis(x_samp, axis=1, fisher=True, bias=False, nan_policy='omit')

        scalar_keys = [
            'mean_true', 'mean_samples_mean', 'mean_samples_std',
            'std_true', 'std_samples_mean', 'std_samples_std',
            'skewness_true', 'skewness_samples_mean', 'skewness_samples_std',
            'kurtosis_true', 'kurtosis_samples_mean', 'kurtosis_samples_std',
        ]
        scalar_vals = [
            float(x_true_flat.mean()),
            float(np.mean(mu_samp)), float(np.std(mu_samp, ddof=1)),
            float(x_true_flat.std(ddof=1)),
            float(np.mean(sig_samp)), float(np.std(sig_samp, ddof=1)),
            float(skew(x_true_flat, bias=False)),
            float(np.mean(g1)), float(np.std(g1, ddof=1)),
            float(kurtosis(x_true_flat, fisher=True, bias=False)),
            float(np.mean(g2)), float(np.std(g2, ddof=1)),
        ]
        with open(os.path.join(diag_dir, f'scalars_samples_{run_name}.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(scalar_keys)
            writer.writerow(scalar_vals)

        # Human-readable txt  (uncertainties shown as ±2σ over samples)
        with open(os.path.join(diag_dir, f'samples_summary_{run_name}.txt'), 'w') as f:
            f.write(f'Samples summary: {run_name}\n')
            f.write(f'{"=" * 50}\n\n')
            f.write(f'Mean (truth):               {scalar_vals[0]:.6f}\n')
            f.write(f'Mean (samples mean):        {scalar_vals[1]:.6f} +/- {2*scalar_vals[2]:.6f}\n')
            f.write(f'Std  (truth):               {scalar_vals[3]:.6f}\n')
            f.write(f'Std  (samples mean):        {scalar_vals[4]:.6f} +/- {2*scalar_vals[5]:.6f}\n')
            f.write(f'Skewness (truth):           {scalar_vals[6]:.6f}\n')
            f.write(f'Skewness (samples mean):    {scalar_vals[7]:.6f} +/- {2*scalar_vals[8]:.6f}\n')
            f.write(f'Kurtosis (truth):           {scalar_vals[9]:.6f}\n')
            f.write(f'Kurtosis (samples mean):    {scalar_vals[10]:.6f} +/- {2*scalar_vals[11]:.6f}\n')


def plot_calibration_diagnostics(delta_z127, z_MAP, samples, box,
                                  Q_like_D, Q_prior_D,
                                  save_dir='./plots', run_name='',
                                  save_csv=False):
    """Posterior calibration diagnostics in Hartley space.

    Produces three figures and a summary text file:
      1. Log-probability histogram: distribution of log p(sample | x) with
         log p(z_true | x) marked.
      2. Per-mode chi-squared vs |k|: binned mean of D_post[k] * r_h[k]^2.
      3. True vs predicted Hartley modes: scatter with +/-2 sigma error bars.
      4. calibration_summary.txt: scalar metrics.

    All inputs must be in internal space (i.e. divided by rescaling_factor).

    Parameters
    ----------
    delta_z127 : np.ndarray, shape (N, N, N)
        True initial conditions field.
    z_MAP : np.ndarray, shape (N, N, N)
        MAP estimate.
    samples : np.ndarray, shape (n_samples, N, N, N)
        Posterior samples.
    box : Power_Spectrum_Sampler
        Box object (for k-grid, k_F, k_Nq).
    Q_like_D : np.ndarray, shape (N^3,)
        Diagonal of the likelihood precision matrix.
    Q_prior_D : np.ndarray, shape (N^3,)
        Diagonal of the prior precision matrix.
    save_dir : str
        Directory to save the output plots.
    run_name : str
        Suffix appended to filenames.
    """
    out_dir = os.path.join(save_dir, 'calibration')
    os.makedirs(out_dir, exist_ok=True)
    plt.rcParams['figure.facecolor'] = 'white'

    delta_z127 = delta_z127.astype('f')
    z_MAP = z_MAP.astype('f')
    samples = np.asarray(samples).astype('f')

    k_flat = box.k.cpu().numpy().flatten()
    k_Nq = box.k_Nq
    k_F = box.k_F

    # ── Precomputation ────────────────────────────────────────────────────
    D_post = Q_like_D + Q_prior_D               # (N³,)
    n_modes = D_post.size - 1                    # N³ - 1 (excluding k=0)
    D_k = D_post[1:]                             # (n_modes,)
    k_modes = k_flat[1:]                         # (n_modes,)

    z_true_h = hartley_np(delta_z127).ravel()    # (N³,)
    z_MAP_h = hartley_np(z_MAP).ravel()          # (N³,)
    r_true_h = (z_true_h - z_MAP_h)[1:]          # (n_modes,)

    # Log-probability computation (float64 for precision)
    log_D_k = np.log(D_k.astype(np.float64))
    logdet_half = 0.5 * log_D_k.sum()
    norm_const = -0.5 * n_modes * np.log(2 * np.pi)

    def _log_prob_np(r_h_excl_k0):
        """Log-prob from Hartley residual (k!=0 modes), shape (n_modes,)."""
        quad = -0.5 * (D_k * r_h_excl_k0**2).sum()
        return norm_const + logdet_half + quad

    log_p_true = _log_prob_np(r_true_h)

    sample_log_probs = np.empty(len(samples))
    for i, s in enumerate(samples):
        r_s_h = (hartley_np(s).ravel() - z_MAP_h)[1:]
        sample_log_probs[i] = _log_prob_np(r_s_h)

    # Expected log-prob: E[-0.5 * sum(D_k * r²)] = -0.5 * n_modes
    expected_log_p = norm_const + logdet_half - 0.5 * n_modes
    expected_std = np.sqrt(n_modes / 2.0)

    # ── Figure 1: log-probability histogram ───────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(sample_log_probs, bins=30, density=True, alpha=0.7,
            color='steelblue', edgecolor='white', label='Posterior samples')
    ax.axvline(log_p_true, color='red', ls='--', lw=2,
               label=f'log p(z_true) = {log_p_true:.0f}')
    ax.axvline(expected_log_p, color='grey', ls='--', lw=1.5,
               label=f'Expected mean = {expected_log_p:.0f}')

    ax.axvspan(expected_log_p - expected_std, expected_log_p + expected_std,
               alpha=0.1, color='grey')

    sample_mean = sample_log_probs.mean()
    sample_std = sample_log_probs.std(ddof=1)
    z_score_true = (log_p_true - sample_mean) / sample_std if sample_std > 0 else np.nan

    txt = (
        f'Sample mean: {sample_mean:.0f}\n'
        f'Sample std: {sample_std:.0f}\n'
        f'z-score(z_true): {z_score_true:.2f}\n\n'
        r'Well-calibrated $\Rightarrow$ log p(z$_{\rm true}$)' '\n'
        'within the sample distribution'
    )
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va='top', ha='left',
            fontsize=9, bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))

    ax.set_xlabel(r'$\log\, p(z \mid x)$', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.set_title('Log-posterior distribution')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'log_prob_histogram_{run_name}.png'), bbox_inches='tight')

    # ── Figure 2: per-mode chi-squared vs |k| ─────────────────────────────
    chi2_true = D_k * r_true_h**2

    chi2_samples = np.empty((len(samples), n_modes))
    for i, s in enumerate(samples):
        r_s_h = (hartley_np(s).ravel() - z_MAP_h)[1:]
        chi2_samples[i] = D_k * r_s_h**2
    chi2_sample_mean = chi2_samples.mean(axis=0)

    k_bin_edges = np.logspace(np.log10(k_F), np.log10(k_Nq), 31)
    k_bin_centers = np.sqrt(k_bin_edges[:-1] * k_bin_edges[1:])

    chi2_binned_true = np.full(len(k_bin_centers), np.nan)
    chi2_binned_samp = np.full(len(k_bin_centers), np.nan)
    expected_scatter = np.full(len(k_bin_centers), np.nan)
    for i in range(len(k_bin_centers)):
        in_bin = (k_modes >= k_bin_edges[i]) & (k_modes < k_bin_edges[i + 1])
        n_bin = in_bin.sum()
        if n_bin > 0:
            chi2_binned_true[i] = chi2_true[in_bin].mean()
            chi2_binned_samp[i] = chi2_sample_mean[in_bin].mean()
            expected_scatter[i] = np.sqrt(2.0 / n_bin)

    valid = ~np.isnan(chi2_binned_true)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(k_bin_centers[valid], chi2_binned_true[valid],
            'o-', markersize=4, lw=1.5, color='red', label=r'Truth: $\langle D_k\, r_k^2 \rangle_k$')
    ax.plot(k_bin_centers[valid], chi2_binned_samp[valid],
            's-', markersize=3, lw=1, color='steelblue', alpha=0.7,
            label=r'Samples (averaged)')
    ax.fill_between(k_bin_centers[valid],
                     1 - expected_scatter[valid], 1 + expected_scatter[valid],
                     alpha=0.2, color='grey', label=r'Expected $\pm 1\sigma$')
    ax.axhline(1.0, color='k', ls='--', lw=0.8)
    ax.axvline(x=k_Nq, color='r', ls='--', lw=0.5, alpha=0.5, label=r'$k_{\rm Nyq}$')

    reduced_chi2 = chi2_true.mean()
    txt = (
        f'Reduced $\\chi^2$ (truth) = {reduced_chi2:.3f}\n\n'
        r'Well-calibrated $\Rightarrow$ binned mean $\approx 1$' '\n'
        r'$> 1$: overconfident (too narrow)' '\n'
        r'$< 1$: underconfident (too wide)'
    )
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va='top', ha='left',
            fontsize=9, bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))

    ax.set_xscale('log')
    ax.set_xlabel(r'$k~[h\,{\rm Mpc}^{-1}]$', fontsize=14)
    ax.set_ylabel(r'$\langle D_{\rm post}(k)\, r_h(k)^2 \rangle$', fontsize=14)
    ax.set_title(r'Per-mode $\chi^2$ diagnostic')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'chi2_per_k_{run_name}.png'), bbox_inches='tight')

    # ── Figure 3: true vs predicted Hartley modes ─────────────────────────
    # Stratified mode selection: ~50 per k-bin, ~1000 total
    rng_modes = np.random.default_rng(0)
    k_sel_bins = np.logspace(np.log10(k_F), np.log10(k_Nq), 21)
    selected = []
    for i in range(len(k_sel_bins) - 1):
        in_bin = np.where((k_modes >= k_sel_bins[i]) & (k_modes < k_sel_bins[i + 1]))[0]
        n_pick = min(50, len(in_bin))
        if n_pick > 0:
            selected.append(rng_modes.choice(in_bin, n_pick, replace=False))
    selected = np.concatenate(selected)

    sel_true = z_true_h[1:][selected]
    sel_pred = z_MAP_h[1:][selected]
    sel_sigma = D_k[selected]**-0.5
    sel_k = k_modes[selected]

    within_2sigma = np.abs(sel_true - sel_pred) < 2 * sel_sigma
    coverage_2sigma = within_2sigma.mean()

    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(sel_true, sel_pred, c=np.log10(sel_k), cmap='viridis',
                    s=4, alpha=0.6, zorder=5)
    ax.errorbar(sel_true, sel_pred, yerr=2 * sel_sigma,
                fmt='none', ecolor='grey', elinewidth=0.5, alpha=0.5, zorder=1)

    lim = max(np.abs(sel_true).max(), np.abs(sel_pred).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'k--', lw=0.8, alpha=0.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')

    cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label(r'$\log_{10}\,k~[h\,{\rm Mpc}^{-1}]$', fontsize=12)

    txt = (
        f'2$\\sigma$ coverage: {coverage_2sigma:.1%}\n'
        f'(N modes shown: {len(selected)})\n\n'
        r'Well-calibrated $\Rightarrow$ ~95% coverage' '\n'
        'and symmetric scatter around diagonal'
    )
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va='top', ha='left',
            fontsize=9, bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))

    ax.set_xlabel(r'$z_{\rm true}^{(H)}(k)$', fontsize=14)
    ax.set_ylabel(r'$z_{\rm MAP}^{(H)}(k)$', fontsize=14)
    ax.set_title(r'True vs predicted Hartley modes ($\pm 2\sigma$)')
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'hartley_modes_{run_name}.png'), bbox_inches='tight')

    # ── Optional: save calibration summary ────────────────────────────────────
    if save_csv:
        diag_dir = os.path.join(out_dir, 'diagnostics')
        os.makedirs(diag_dir, exist_ok=True)

        # CSV
        scalar_keys = [
            'n_modes', 'reduced_chi2', 'log_p_true',
            'expected_mean_log_p', 'expected_std_log_p',
            'sample_mean_log_p', 'sample_std_log_p',
            'z_score_true', 'coverage_2sigma',
        ]
        scalar_vals = [
            n_modes, float(reduced_chi2), float(log_p_true),
            float(expected_log_p), float(expected_std),
            float(sample_mean), float(sample_std),
            float(z_score_true), float(coverage_2sigma),
        ]
        with open(os.path.join(diag_dir, f'scalars_calibration_{run_name}.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(scalar_keys)
            writer.writerow(scalar_vals)

        # Human-readable txt
        with open(os.path.join(diag_dir, f'calibration_summary_{run_name}.txt'), 'w') as f:
            f.write(f'Calibration summary: {run_name}\n')
            f.write(f'{"=" * 50}\n\n')
            f.write(f'N_modes (N^3 - 1):          {n_modes}\n')
            f.write(f'Reduced chi2 (truth):       {reduced_chi2:.6f}\n')
            f.write(f'log p(z_true | x):          {log_p_true:.2f}\n')
            f.write(f'Expected mean log p:        {expected_log_p:.2f}\n')
            f.write(f'Expected std log p:         {expected_std:.2f}\n')
            f.write(f'Sample mean log p:          {sample_mean:.2f}\n')
            f.write(f'Sample std log p:           {sample_std:.2f}\n')
            f.write(f'z-score(z_true):            {z_score_true:.4f}\n')
            f.write(f'2-sigma coverage (modes):   {coverage_2sigma:.4f} ({coverage_2sigma:.1%})\n')


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
      - text: mean ± 2σ over samples of μ, σ, skewness (γ1), and excess kurtosis (γ2)

    Notes:
      - All statistics are computed per sample (each sample is flattened to N^3 values).
      - Uncertainties in the annotation are ±2σ across the B posterior samples.
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

    # Mean and std per sample (over voxels), then mean ± 2*std over samples
    x_true_fin = x_true[np.isfinite(x_true)]
    mu = x_samp.mean(axis=1)   # (B,)
    sig = x_samp.std(axis=1, ddof=1)  # (B,)
    mu_true = float(x_true_fin.mean())
    sig_true = float(x_true_fin.std(ddof=1))
    g1_true = float(skew(x_true_fin, bias=False))
    g2_true = float(kurtosis(x_true_fin, fisher=True, bias=False))
    mu_mean, mu_std = float(np.mean(mu)), float(np.std(mu, ddof=1))
    sig_mean, sig_std = float(np.mean(sig)), float(np.std(sig, ddof=1))

    # Skewness and excess kurtosis per sample (over voxels), then mean ± 2*std over samples
    g1 = skew(x_samp, axis=1, bias=False, nan_policy="omit")  # (B,)
    g2 = kurtosis(x_samp, axis=1, fisher=True, bias=False, nan_policy="omit")  # (B,) excess kurtosis

    g1_mean, g1_std = float(np.mean(g1)), float(np.std(g1, ddof=1))
    g2_mean, g2_std = float(np.mean(g2)), float(np.std(g2, ddof=1))

    # Plot
    fig, ax = plt.subplots(figsize=(6.2, 4.4), dpi=150)

    ax.plot(centers, h_mean_plot, lw=2.0, label=r'$p(z_i\,|\,\mathbf{x}_{\rm obs})$')
    ax.plot(centers, h_true_plot, lw=2.0, ls='--', color='k', label=r'$p(z^{\rm truth})$')

    ax.set_xlim(*xlim)
    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'$p(z_i\,|\,\mathbf{x}_{\rm obs})$')

    if title is not None:
        ax.set_title(title)

    # Samples annotation (top left)
    txt_samples = (
        rf'$\mu = {mu_mean:.1e} \pm {2*mu_std:.1e}$' + "\n" +
        rf'$\sigma = {sig_mean:.3f} \pm {2*sig_std:.1e}$' + "\n" +
        rf'$\gamma_1 = {g1_mean:.1e} \pm {2*g1_std:.1e}$' + "\n" +
        rf'$\gamma_2 = {g2_mean:.1e} \pm {2*g2_std:.1e}$'
    )
    ax.text(0.04, 0.97, txt_samples, transform=ax.transAxes, va='top', ha='left', fontsize=8)

    # Truth annotation (top right)
    txt_truth = (
        rf'$\mu = {mu_true:.1e}$' + "\n" +
        rf'$\sigma = {sig_true:.3f}$' + "\n" +
        rf'$\gamma_1 = {g1_true:.1e}$' + "\n" +
        rf'$\gamma_2 = {g2_true:.1e}$'
    )
    ax.text(0.96, 0.97, txt_truth, transform=ax.transAxes, va='top', ha='right', fontsize=8)

    ax.legend(loc='lower center', frameon=True)
    ax.grid(False)

    fig.tight_layout()
    return fig, ax

def plot_amortization_test(z_MAPs, z_trues, box, Q_like_D, Q_prior_D,
                           save_dir='./plots', run_name='',
                           save_csv=False):
    """Amortization test: calibration diagnostics across many held-out observations.

    For each test pair (z_MAP_i, z_true_i), computes the chi-squared statistic

        chi2_i = sum_{k!=0} D_post[k] * (H(z_true_i - z_MAP_i))[k]^2

    Under a well-calibrated Gaussian posterior, chi2_i ~ chi2(N^3-1).
    With many observations this provides a powerful statistical test.

    Produces three figures and a summary text file:
      1. Per-mode chi2 vs |k| averaged over observations (should be ~1.0 everywhere).
      2. Chi2 z-score histogram (should be standard normal).
      3. PP-plot: empirical CDF of probability integral transform values
         (should follow the diagonal).
      4. amortization_summary.txt: scalar metrics.

    All inputs must be in internal space (i.e. divided by rescaling_factor).

    Parameters
    ----------
    z_MAPs : np.ndarray, shape (N_obs, N, N, N)
        MAP estimates for each observation.
    z_trues : np.ndarray, shape (N_obs, N, N, N)
        True initial conditions for each observation.
    box : Power_Spectrum_Sampler
        Box object (for k-grid, k_F, k_Nq).
    Q_like_D : np.ndarray, shape (N^3,)
        Diagonal of the likelihood precision matrix.
    Q_prior_D : np.ndarray, shape (N^3,)
        Diagonal of the prior precision matrix.
    save_dir : str
        Directory to save the output plots.
    run_name : str
        Suffix appended to filenames.
    """
    out_dir = os.path.join(save_dir, 'amortization')
    os.makedirs(out_dir, exist_ok=True)
    plt.rcParams['figure.facecolor'] = 'white'

    z_MAPs = np.asarray(z_MAPs, dtype='f')
    z_trues = np.asarray(z_trues, dtype='f')
    N_obs = len(z_MAPs)

    k_flat = box.k.cpu().numpy().flatten()
    k_Nq = box.k_Nq
    k_F = box.k_F

    # ── Precomputation ────────────────────────────────────────────────────
    D_post = (Q_like_D + Q_prior_D).astype(np.float64)
    n_modes = D_post.size - 1           # N^3 - 1 (excluding k=0)
    D_k = D_post[1:]                    # (n_modes,)
    k_modes = k_flat[1:]               # (n_modes,)

    # Accumulate per-mode chi2 mean and total chi2 per observation.
    # Running sums avoid storing the full (N_obs, n_modes) array.
    chi2_mode_sum = np.zeros(n_modes, dtype=np.float64)
    chi2_total = np.zeros(N_obs, dtype=np.float64)

    for i in range(N_obs):
        r_h = hartley_np(z_trues[i] - z_MAPs[i]).ravel().astype(np.float64)
        chi2_k = D_k * r_h[1:]**2
        chi2_mode_sum += chi2_k
        chi2_total[i] = chi2_k.sum()

    chi2_mode_mean = chi2_mode_sum / N_obs

    # Z-scores: (chi2 - dof) / sqrt(2*dof) ~ N(0,1) for large dof
    z_scores = (chi2_total - n_modes) / np.sqrt(2.0 * n_modes)

    # PIT values via normal approximation (exact for dof >> 1)
    pit_values = norm.cdf(z_scores)

    # KS test for uniformity of PIT values
    ks_stat, ks_pvalue = kstest(pit_values, 'uniform')

    # ── Figure 1: per-mode chi2 vs |k| (averaged over observations) ──────
    k_bin_edges = np.logspace(np.log10(k_F), np.log10(k_Nq), 31)
    k_bin_centers = np.sqrt(k_bin_edges[:-1] * k_bin_edges[1:])

    chi2_binned = np.full(len(k_bin_centers), np.nan)
    expected_sem = np.full(len(k_bin_centers), np.nan)
    for i in range(len(k_bin_centers)):
        in_bin = (k_modes >= k_bin_edges[i]) & (k_modes < k_bin_edges[i + 1])
        n_bin = in_bin.sum()
        if n_bin > 0:
            chi2_binned[i] = chi2_mode_mean[in_bin].mean()
            # Under H0, each mode contributes chi2(1) with var=2.
            # Binned mean over n_bin modes and N_obs observations:
            expected_sem[i] = np.sqrt(2.0 / (N_obs * n_bin))

    valid = ~np.isnan(chi2_binned)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(k_bin_centers[valid], chi2_binned[valid],
            'o-', markersize=4, lw=1.5, color='steelblue',
            label=r'$\langle D_k\, r_k^2 \rangle_{\mathrm{obs},\,k}$')
    ax.fill_between(k_bin_centers[valid],
                     1 - 2 * expected_sem[valid], 1 + 2 * expected_sem[valid],
                     alpha=0.15, color='grey', label=r'Expected $\pm 2\sigma$')
    ax.fill_between(k_bin_centers[valid],
                     1 - expected_sem[valid], 1 + expected_sem[valid],
                     alpha=0.3, color='grey', label=r'Expected $\pm 1\sigma$')
    ax.axhline(1.0, color='k', ls='--', lw=0.8)
    ax.axvline(x=k_Nq, color='r', ls='--', lw=0.5, alpha=0.5, label=r'$k_{\rm Nyq}$')

    global_reduced_chi2 = chi2_total.mean() / n_modes
    txt = (
        f'$N_{{\\rm obs}} = {N_obs}$\n'
        f'Global reduced $\\chi^2 = {global_reduced_chi2:.4f}$\n\n'
        r'Well-calibrated $\Rightarrow$ binned mean $\approx 1$'
    )
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va='top', ha='left',
            fontsize=9, bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))

    ax.set_xscale('log')
    ax.set_xlabel(r'$k~[h\,{\rm Mpc}^{-1}]$', fontsize=14)
    ax.set_ylabel(r'$\langle D_{\rm post}(k)\, r_h(k)^2 \rangle$', fontsize=14)
    ax.set_title(r'Per-mode $\chi^2$ (averaged over observations)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'chi2_per_k_amortized_{run_name}.png'), bbox_inches='tight')

    # ── Figure 2: chi2 z-score histogram ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(z_scores, bins=max(15, N_obs // 5), density=True, alpha=0.7,
            color='steelblue', edgecolor='white', label='Observations')

    zz = np.linspace(z_scores.min() - 0.5, z_scores.max() + 0.5, 200)
    ax.plot(zz, norm.pdf(zz), 'k-', lw=1.5, label=r'$\mathcal{N}(0, 1)$')

    z_mean = z_scores.mean()
    z_std = z_scores.std(ddof=1)
    txt = (
        f'$N_{{\\rm obs}} = {N_obs}$\n'
        f'Mean $= {z_mean:.3f}$  (expect 0)\n'
        f'Std $= {z_std:.3f}$  (expect 1)\n\n'
        r'$z = (\chi^2 - \nu) \,/\, \sqrt{2\nu}$'
    )
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va='top', ha='left',
            fontsize=9, bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))

    ax.set_xlabel(r'$\chi^2$ z-score', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.set_title(r'Total $\chi^2$ distribution across observations')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'chi2_zscore_hist_{run_name}.png'), bbox_inches='tight')

    # ── Figure 3: PP-plot (expected coverage) ─────────────────────────────
    pit_sorted = np.sort(pit_values)
    empirical_cdf = np.arange(1, N_obs + 1) / N_obs

    # Confidence band: Kolmogorov-Smirnov 95% band for N_obs samples
    ks_band = 1.36 / np.sqrt(N_obs)  # 95% KS critical value

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(pit_sorted, empirical_cdf, 'o-', markersize=3, lw=1.5,
            color='steelblue', label='Observed')
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, label='Ideal')
    ax.fill_between([0, 1],
                     [0 - ks_band, 1 - ks_band],
                     [0 + ks_band, 1 + ks_band],
                     alpha=0.15, color='grey', label=f'95% KS band')

    txt = (
        f'$N_{{\\rm obs}} = {N_obs}$\n'
        f'KS statistic $= {ks_stat:.4f}$\n'
        f'KS $p$-value $= {ks_pvalue:.4f}$\n\n'
        r'Well-calibrated $\Rightarrow$ follows diagonal'
    )
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va='top', ha='left',
            fontsize=9, bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))

    ax.set_xlabel('Nominal coverage (PIT quantile)', fontsize=14)
    ax.set_ylabel('Empirical coverage', fontsize=14)
    ax.set_title('PP-plot (expected coverage)')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'pp_plot_{run_name}.png'), bbox_inches='tight')

    # ── Optional: save diagnostic data files ─────────────────────────────────
    if save_csv:
        diag_dir = os.path.join(out_dir, 'diagnostics')
        os.makedirs(diag_dir, exist_ok=True)

        # CSV
        scalar_keys = [
            'n_obs', 'n_modes', 'global_reduced_chi2',
            'z_score_mean', 'z_score_std',
            'ks_stat', 'ks_pvalue',
        ]
        scalar_vals = [
            N_obs, n_modes, float(global_reduced_chi2),
            float(z_mean), float(z_std),
            float(ks_stat), float(ks_pvalue),
        ]
        with open(os.path.join(diag_dir, f'scalars_amortization_{run_name}.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(scalar_keys)
            writer.writerow(scalar_vals)

        # Human-readable txt
        with open(os.path.join(diag_dir, f'amortization_summary_{run_name}.txt'), 'w') as f:
            f.write(f'Amortization test summary: {run_name}\n')
            f.write(f'{"=" * 50}\n\n')
            f.write(f'N_obs:                      {N_obs}\n')
            f.write(f'N_modes (N^3 - 1):          {n_modes}\n')
            f.write(f'Global reduced chi2:        {global_reduced_chi2:.6f}\n')
            f.write(f'chi2 z-score mean:          {z_mean:.6f}  (expect 0)\n')
            f.write(f'chi2 z-score std:           {z_std:.6f}  (expect 1)\n')
            f.write(f'KS statistic:               {ks_stat:.6f}\n')
            f.write(f'KS p-value:                 {ks_pvalue:.6f}\n')


class MetricsCSVCallback(Callback):
    """Write one clean row per epoch: epoch, step, train_loss, val_loss, lr."""

    def __init__(self, filepath):
        self.filepath = filepath
        self._header_written = False
        self._t_start = None

    def on_train_start(self, trainer, _pl_module):
        self._t_start = time.time()

    def on_validation_epoch_end(self, trainer, _pl_module):
        m = trainer.callback_metrics
        epoch = trainer.current_epoch
        step = trainer.global_step

        train_loss = m['train_loss'].item() if 'train_loss' in m else ''
        val_loss = m['val_loss'].item() if 'val_loss' in m else ''
        lr = trainer.optimizers[0].param_groups[0]['lr']

        if not self._header_written:
            with open(self.filepath, 'w') as f:
                f.write('epoch,step,train_loss,val_loss,lr\n')
            self._header_written = True

        with open(self.filepath, 'a') as f:
            f.write(f'{epoch},{step},{train_loss},{val_loss},{lr}\n')

        train_str = f'{train_loss:.4f}' if train_loss != '' else 'n/a'
        val_str   = f'{val_loss:.4f}'   if val_loss   != '' else 'n/a'
        elapsed = time.time() - self._t_start if self._t_start is not None else 0.0
        h, m_e, s = int(elapsed // 3600), int(elapsed % 3600 // 60), int(elapsed % 60)
        print(f'Epoch {epoch:3d} | step {step:6d} | {h}h {m_e}m {s}s | train_loss {train_str} | val_loss {val_str} | lr {lr:.2e}')
