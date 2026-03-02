import torch
import torch.nn.functional as F
import numpy as np
import swyft
from map2map.models import UNet
from gaussian_npe.CustomUNET import UNet as CustomUNet
from gaussian_npe.matrices import (
    Precision_Matrix_From_Factors,
    Precision_Matrix_FFT,
    Precision_Matrix_Sum,
    Precision_Matrix_IsotropicNodes,
)
from gaussian_npe.utils import hartley


class Gaussian_NPE_Base(swyft.AdamWReduceLROnPlateau, swyft.SwyftModule):
    """Abstract base class for all Gaussian NPE architectures.

    Provides the shared Gaussian posterior machinery (Q matrices, log_prob,
    loss, sampling) and training loop (forward).  Subclasses must implement
    ``estimator(x)`` which returns the MAP estimate in internal space.
    """

    def __init__(self, box, prior, sigma_noise, rescaling_factor=1.0,
                 learning_rate=1e-2, early_stopping_patience=5, lr_scheduler_patience=1):
        super().__init__()
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.lr_scheduler_patience = lr_scheduler_patience

        self.box = box
        self.N = box.N
        self.sigma_noise = sigma_noise
        self.rescaling_factor = rescaling_factor
        self.unet = UNet(1, 1, hid_chan=16, bypass=False)

        self.Q_prior = Precision_Matrix_From_Factors(*prior)
        self.Q_like = Precision_Matrix_FFT(self.N)
        self.Q_post = Precision_Matrix_Sum(self.Q_like, self.Q_prior)

    def configure_optimizers(self):
        # Per-mode Hartley-space params: Q_like diagonal + any subclass filter/scale params.
        # These must NOT be weight-decayed — their optimal values are far from zero.
        per_mode = list(self.Q_like.parameters())
        for attr in ('scale', 'filter_logit', 'filter_logit_nodes'):
            if hasattr(self, attr):
                per_mode.append(getattr(self, attr))

        per_mode_ids = {id(p) for p in per_mode}
        other_params = [p for p in self.parameters() if id(p) not in per_mode_ids]

        optimizer = torch.optim.AdamW(
            [
                {'params': other_params, 'weight_decay': 1e-2},
                {'params': per_mode,     'weight_decay': 0.0},
            ],
            lr=self.learning_rate,
        )
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=getattr(self, 'lr_scheduler_factor', 0.1),
                patience=self.lr_scheduler_patience,
            ),
            'monitor': 'val_loss',
        }
        return dict(optimizer=optimizer, lr_scheduler=lr_scheduler)

    def estimator(self, x):
        raise NotImplementedError

    def forward(self, A, B):
        x = A['delta_z0']
        x += torch.randn_like(x) * self.sigma_noise    # add noise to the final field
        b = self.estimator(x)

        z = B['delta_z127'][:len(x)]
        z = self.rescaling_factor**-1 * z
        return self.loss(b, z)

    def log_prob(self, z_MAP, z):
        """Log-posterior on the (N³−1)-dimensional zero-mean subspace.

        On a periodic box the mean overdensity is zero by construction, so
        the k=0 (monopole) mode is not a degree of freedom.  The posterior is
        an (N³−1)-dimensional Gaussian over zero-mean fields:

            log p(z|x) = −(N³−1)/2·log(2π)
                         + ½ Σ_{k≠0} log D_post[k]
                         − ½ Σ_{k≠0} D_post[k]·r_h[k]²

        Both the quadratic and log-determinant terms exclude k=0 explicitly
        via indexing in Hartley space.

        Args:
            z_MAP: MAP estimate (B, N, N, N)
            z:     target field  (B, N, N, N)

        Returns:
            (B,) tensor of per-sample log-probabilities.
        """
        r = z - z_MAP
        r_h = self.Q_post.G(r)                          # (B, N³)
        D = self.Q_post.D                                # (N³,)

        # Exclude k=0 mode
        n_modes = D.numel() - 1
        quad   = -0.5 * (D[1:] * r_h[:, 1:]**2).sum(dim=-1)
        logdet =  0.5 * torch.log(D[1:]).sum()
        norm   = -0.5 * n_modes * torch.tensor(2 * torch.pi).log()

        return norm + logdet + quad

    def loss(self, z_MAP, z):
        """Per-mode normalised negative log-posterior, for training.

        Returns −log p / (N³−1) per sample, wrapped in swyft.AuxLoss.
        """
        n_modes = self.Q_post.D.numel() - 1
        return swyft.AuxLoss(-self.log_prob(z_MAP, z) / n_modes, 'z')

    def loss_legacy(self, z_MAP, z):
        """Original real-space loss over the full N³ grid (including k=0).

        Computes the quadratic and log-det terms in real space using the
        Q-matrix interface.  All N³ Hartley modes contribute, so the k=0
        mode — whose prior precision D_prior[0] is very large — can dominate
        the loss if the residual has a nonzero spatial mean (e.g. from
        training noise).
        """
        r = z - z_MAP
        loss = 0.5 * (r*(self.Q_like(r) + self.Q_prior(r))).mean(dim=(-3, -2, -1)) - 0.5 * torch.log(self.Q_like.D + self.Q_prior.D).mean(dim=-1)
        return swyft.AuxLoss(loss, 'z')

    def get_z_MAP(self, x_obs):
        """Returns the MAP estimation for a given x_obs.

        The spatial mean is subtracted to enforce the zero-mean overdensity
        constraint (δ̄ = 0 on a periodic box). The k=0 mode is excluded from
        the training loss, so the network never learns to zero it — this
        correction applies it analytically at inference time.
        """
        z = self.estimator(x_obs.unsqueeze(0)).squeeze(0).detach() * self.rescaling_factor
        return z

    def sample(self, num_samples, x_obs=None, z_MAP=None, to_numpy=True):
        """Samples the posterior for a given x_obs or z_MAP (provide exactly one)."""
        if (x_obs is None) == (z_MAP is None):
            raise ValueError("Provide exactly one of x_obs or z_MAP")
        if z_MAP is None:
            z_MAP = self.get_z_MAP(x_obs)
        D_post = (self.Q_prior.D + self.Q_like.D).detach() * self.rescaling_factor**-2
        std = D_post**-0.5

        draws = []
        for _ in range(num_samples):
            z = self.Q_post.G_T(std * torch.randn_like(D_post)) + z_MAP
            draws.append(z.cpu().numpy() if to_numpy else z.cpu())
        return draws


class Gaussian_NPE_Network(Gaussian_NPE_Base):
    """Default architecture: sigmoid high-pass filter + per-mode scale.

        mu(x) = G_T(scale · G(x + sigmoid_filter(UNet(x))))

    The sigmoid filter suppresses UNet corrections below k_cut, and the
    learnable per-mode scale factors adjust amplitudes in Hartley space.
    """

    def __init__(self, box, *args, k_cut=0.03, w_cut=0.001, **kwargs):
        super().__init__(box, *args, **kwargs)
        self.k_cut = k_cut
        self.w_cut = w_cut
        self.scale = torch.nn.Parameter(torch.ones(self.N**3))

    def estimator(self, x):
        p3d = 6*(20,)
        xx = F.pad(x.unsqueeze(0), p3d, "circular").squeeze(0)
        xx = self.unet(xx.unsqueeze(1)).squeeze(1)

        x = x + self.box.sigmoid_filter(xx, self.k_cut, self.w_cut)
        x = self.Q_post.G_T(self.Q_post.G(x) * self.scale)
        return x - x.mean(dim=(-3, -2, -1), keepdim=True)  # enforce zero-mean overdensity


class Gaussian_NPE_UNet_Only(Gaussian_NPE_Base):
    """Residual U-Net MAP estimator without any Fourier-space filtering.

    The MAP estimate is the input plus a learned U-Net correction:

        mu(x) = x + UNet(x)

    Since bypass=False in the UNet (no internal skip connection), the
    residual path is added explicitly here.  No sigmoid high-pass filter
    and no per-mode scale factors.  Serves as a minimal baseline to
    isolate the U-Net's contribution.
    """

    def estimator(self, x):
        p3d = 6 * (20,)
        xx = F.pad(x.unsqueeze(0), p3d, "circular").squeeze(0)
        x = x + self.unet(xx.unsqueeze(1)).squeeze(1)
        return x - x.mean(dim=(-3, -2, -1), keepdim=True)


class Gaussian_NPE_WienerNet(Gaussian_NPE_Base):
    """Wiener filter baseline + U-Net residual correction.

    Replaces the sigmoid high-pass filter and per-mode scale with a
    physically-motivated Wiener filter that provides a smooth, data-driven
    transition between prior-dominated and likelihood-dominated modes:

        mu(x) = WienerFilter(x; D_like, D_prior) + UNet(x)

    The Wiener weight w(k) = D_like(k) / (D_prior(k) + D_like(k)) naturally
    passes modes where the data is informative and suppresses where the prior
    dominates, without an arbitrary k_cut.
    """

    def estimator(self, x):
        # Wiener filter in Hartley space
        x_h = self.Q_post.G(x)                          # (B, N^3)
        D_prior = self.Q_prior.D                         # (N^3,)
        D_like = self.Q_like.D                           # (N^3,)
        w = D_like / (D_prior + D_like)                  # (N^3,)
        x_wiener = self.Q_post.G_T(x_h * w)             # (B, N, N, N)

        # U-Net learns nonlinear residual correction
        p3d = 6 * (20,)
        xx = F.pad(x.unsqueeze(0), p3d, "circular").squeeze(0)
        correction = self.unet(xx.unsqueeze(1)).squeeze(1)
        x = x_wiener + correction
        return x - x.mean(dim=(-3, -2, -1), keepdim=True)


class Gaussian_NPE_LearnableFilter(Gaussian_NPE_Base):
    """Learnable per-mode filter replacing the fixed sigmoid.

    Same architecture as the original but replaces the hard sigmoidal
    high-pass filter with a learnable weight w(k) in [0, 1] for each
    Fourier mode, initialized from the sigmoid:

        w(k) = sigmoid(filter_logit(k))

    where filter_logit is initialized as logit(sigmoid((k - k_cut) / w_cut)),
    i.e. (k - k_cut) / w_cut. This allows the network to learn an optimal
    per-mode blending during training while starting from the hand-tuned
    sigmoid baseline.
    """

    def __init__(self, box, *args, k_cut=0.03, w_cut=0.001, **kwargs):
        super().__init__(box, *args, **kwargs)
        self.scale = torch.nn.Parameter(torch.ones(self.N**3))

        # Initialize filter_logit so that sigmoid(filter_logit) == sigmoid((k - k_cut) / w_cut)
        # i.e. filter_logit = (k - k_cut) / w_cut
        k_flat = box.k.flatten()  # (N^3,)
        self.filter_logit = torch.nn.Parameter((k_flat - k_cut) / w_cut)

    def estimator(self, x):
        p3d = 6 * (20,)
        xx = F.pad(x.unsqueeze(0), p3d, "circular").squeeze(0)
        xx = self.unet(xx.unsqueeze(1)).squeeze(1)

        # Learnable per-mode filter (replaces fixed sigmoid)
        w = torch.sigmoid(self.filter_logit)                           # (N^3,)
        xx_h = hartley(xx, dim=self.box.hartley_dim).flatten(-3, -1)   # (B, N^3)
        xx_filtered = hartley(
            (w * xx_h).unflatten(-1, self.box.shape),
            dim=self.box.hartley_dim,
        )

        x = x + xx_filtered
        x = self.Q_post.G_T(self.Q_post.G(x) * self.scale)
        return x - x.mean(dim=(-3, -2, -1), keepdim=True)


class Gaussian_NPE_SmoothFilter(Gaussian_NPE_Base):
    """Learnable smooth filter as a function of |k|.

    Like LearnableFilter but parameterizes the filter as a smooth function
    of |k| using learnable logit values at logarithmically-spaced k-nodes
    with linear interpolation.  This enforces the physical isotropy of the
    problem (~20 DOF) while being more flexible than the 2-parameter sigmoid.

        w(k) = sigmoid(interp(log|k|; logit_nodes))

    Initialized from the same sigmoid baseline as the default network.
    """

    def __init__(self, box, *args, k_cut=0.03, w_cut=0.001, n_filter_nodes=20, **kwargs):
        super().__init__(box, *args, **kwargs)
        self.scale = torch.nn.Parameter(torch.ones(self.N**3))

        # Log-spaced nodes from k_F to k_Nq
        log_k_nodes = torch.linspace(
            float(torch.tensor(float(box.k_F)).log()),
            float(torch.tensor(float(box.k_Nq)).log()),
            n_filter_nodes,
        )

        # Initialize node logits from the sigmoid baseline
        self.filter_logit_nodes = torch.nn.Parameter(
            (log_k_nodes.exp() - k_cut) / w_cut
        )

        # Precompute fixed (N³, n_nodes) interpolation weight matrix
        log_k = torch.log(box.k.flatten().clamp(min=float(box.k_F)))
        log_k_nodes = log_k_nodes.to(log_k.device)
        idx = torch.searchsorted(log_k_nodes, log_k).clamp(1, n_filter_nodes - 1)
        frac = ((log_k - log_k_nodes[idx - 1])
                / (log_k_nodes[idx] - log_k_nodes[idx - 1])).clamp(0, 1)

        # Each mode interpolates between node[idx-1] and node[idx]
        self.register_buffer('_left_idx',  idx - 1)
        self.register_buffer('_right_idx', idx)
        self.register_buffer('_frac',      frac)

    def estimator(self, x):
        p3d = 6 * (20,)
        xx = F.pad(x.unsqueeze(0), p3d, "circular").squeeze(0)
        xx = self.unet(xx.unsqueeze(1)).squeeze(1)

        # Smooth learnable filter (interpolated from ~20 k-nodes)
        w = torch.sigmoid(
            (1 - self._frac) * self.filter_logit_nodes[self._left_idx]
            +      self._frac  * self.filter_logit_nodes[self._right_idx]
        )
        xx_h = hartley(xx, dim=self.box.hartley_dim).flatten(-3, -1)   # (B, N³)
        xx_filtered = hartley(
            (w * xx_h).unflatten(-1, self.box.shape),
            dim=self.box.hartley_dim,
        )

        x = x + xx_filtered
        x = self.Q_post.G_T(self.Q_post.G(x) * self.scale)
        return x - x.mean(dim=(-3, -2, -1), keepdim=True)


class Gaussian_NPE_Iterative(Gaussian_NPE_Base):
    """Iterative refinement with a 2-channel U-Net.

    Instead of a single U-Net pass, iteratively refines the MAP estimate
    by feeding both the current estimate and the observation into a
    2-channel U-Net:

        mu^(0) = x
        mu^(t+1) = mu^(t) + UNet([mu^(t), x])   for t = 0..T-1
        mu = G_T(alpha * G(mu^(T)))

    The U-Net sees [current_estimate, observation] as a 2-channel input,
    enabling it to compute residual corrections that account for both the
    current state and the data at each step.
    """

    def __init__(self, *args, num_iterations=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = torch.nn.Parameter(torch.ones(self.N**3))
        self.num_iterations = num_iterations
        # Override U-Net: 2 input channels (current estimate + observation)
        self.unet = UNet(2, 1, hid_chan=8, bypass=False)

    def estimator(self, x):
        mu = x
        p3d = 6 * (20,)
        for _ in range(self.num_iterations):
            inp = torch.cat([mu.unsqueeze(1), x.unsqueeze(1)], dim=1)  # (B, 2, N, N, N)
            inp_padded = F.pad(inp, p3d, "circular")
            correction = self.unet(inp_padded).squeeze(1)              # (B, N, N, N)
            mu = mu + correction

        mu = self.Q_post.G_T(self.Q_post.G(mu) * self.scale)
        return mu - mu.mean(dim=(-3, -2, -1), keepdim=True)


class Gaussian_NPE_LH(Gaussian_NPE_Network):
    """Variant for the Quijote Latin Hypercube dataset with varying cosmology.

    Instead of a fixed rescaling factor D(z_ic)/D(z=0) shared across all
    simulations, computes it per-sample from the cosmological parameters
    stored alongside each simulation.

    The ZarrStore must contain 'sim_params' of shape (5,) per sample:
        [Omega_m, Omega_b, h, n_s, sigma_8]

    The estimator (sigmoid filter + UNet + scale) is identical to
    Gaussian_NPE_Network.  The posterior precision Q_post is a single learned
    isotropic Precision_Matrix_IsotropicNodes — no Q_prior / Q_like split.

    Rationale: the 2000 LH cosmologies each have a different P(k), making a
    fixed fiducial Q_prior incorrect.  Instead of supplying per-sample P(k),
    Q_post is learned directly.  At the loss minimum,
    D_post(k) = 1 / E[|z_k − b_k|²], which is the true posterior precision
    regardless of the prior/likelihood decomposition.
    """

    def __init__(self, box, *args, n_nodes=32, **kwargs):
        super().__init__(box, *args, **kwargs)
        # Single learned Q_post — no prior/likelihood decomposition.
        self.Q_post = Precision_Matrix_IsotropicNodes(
            self.N, box.k.flatten(), n_nodes=n_nodes,
        )
        del self.Q_like   # N³ Precision_Matrix_FFT from parent — not used
        del self.Q_prior  # fiducial Precision_Matrix_From_Factors — not used

    # Z_IC = 127

    # @staticmethod
    # def _growth_D_tensor(Omega_m, z):
    #     """Tensor-compatible growth factor approximation D(z) for flat LCDM."""
    #     Omega_L = 1.0 - Omega_m
    #     Om_m = Omega_m * (1.0 + z) ** 3 / (Omega_L + Omega_m * (1.0 + z) ** 3)
    #     Om_L = Omega_L / (Omega_L + Omega_m * (1.0 + z) ** 3)
    #     return (1.0 / (1.0 + z)) * (5.0 * Om_m / 2.0) / (
    #         Om_m ** (4.0 / 7.0) - Om_L
    #         + (1.0 + Om_m / 2.0) * (1.0 + Om_L / 70.0)
    #     )

    # def _compute_rescaling(self, sim_params):
    #     """Compute D(z_ic)/D(0) per sample from sim_params.

    #     Args:
    #         sim_params: (B, 5) tensor [Omega_m, Omega_b, h, n_s, sigma_8]
    #     Returns:
    #         (B,) tensor of rescaling factors
    #     """
    #     Omega_m = sim_params[:, 0]
    #     return (
    #         self._growth_D_tensor(Omega_m, self.Z_IC)
    #         / self._growth_D_tensor(Omega_m, 0.0)
    #     )

    # def set_rescaling(self, sim_params):
    #     """Set rescaling_factor for inference from a single sim_params vector.

    #     Args:
    #         sim_params: array-like of shape (5,) [Omega_m, Omega_b, h, n_s, sigma_8]
    #     """
    #     from gaussian_npe.utils import growth_D_approx
    #     cosmo = {
    #         'Omega_cdm': float(sim_params[0]) - float(sim_params[1]),
    #         'Omega_b': float(sim_params[1]),
    #     }
    #     self.rescaling_factor = (
    #         growth_D_approx(cosmo, self.Z_IC) / growth_D_approx(cosmo, 0)
    #     )

    def sample(self, num_samples, x_obs=None, z_MAP=None, to_numpy=True):
        """Sample the posterior using Q_post directly (no Q_prior / Q_like split)."""
        if (x_obs is None) == (z_MAP is None):
            raise ValueError("Provide exactly one of x_obs or z_MAP")
        if z_MAP is None:
            z_MAP = self.get_z_MAP(x_obs)
        D_post = self.Q_post.D.detach() * self.rescaling_factor**-2
        std = D_post**-0.5
        std[0] = 0.0   # k=0 monopole never sampled (zero-mean constraint)
        draws = []
        for _ in range(num_samples):
            z = self.Q_post.G_T(std * torch.randn_like(D_post)) + z_MAP
            draws.append(z.cpu().numpy() if to_numpy else z.cpu())
        return draws

    def forward(self, A, B):
        x = A['delta_z0']
        x += torch.randn_like(x) * self.sigma_noise
        b = self.estimator(x)

        z = B['delta_z127'][:len(x)]
        # Rescaling factor precomputed per-sample by data_scripts/quijote_lh.py
        rescaling = B['rescaling_factor'][:len(x), 0]   # (B, 1) → (B,)
        z = z / rescaling[:, None, None, None]
        # [OLD — on-the-fly computation from sim_params, kept for reference]
        # sim_params = B['sim_params'][:len(x)]          # (B, 5)
        # rescaling = self._compute_rescaling(sim_params) # (B,)

        return self.loss(b, z)

class Gaussian_NPE_CustomUNet(Gaussian_NPE_Base):
    """UNet-Only estimator backed by a deeper CustomUNet (4 downsampling levels).

    Pure residual estimator — no sigmoid filter, no per-mode scale:

        mu(x) = x + CustomUNet(x)

    Uses a 4-level (num_encoding_blocks=5) CustomUNet with circular padding
    handled internally, so no external F.pad is needed.  Configuration:
      - num_encoding_blocks=5 → bottleneck at 8³, encoder RF ≈ 72 vox
      - out_channels_first_layer=2 → ~380K parameters
      - output RF ≈ 208 voxels = 1625 Mpc/h (covers fundamental mode)

    The deeper bottleneck (8³ vs. map2map's 32³) allows each bottleneck
    neuron to integrate roughly half the box volume, enabling reconstruction
    of large-scale modes that the 2-level map2map UNet handles only via skip
    connections.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unet = CustomUNet(
            in_channels=1,
            out_classes=1,
            dimensions=3,
            num_encoding_blocks=5,
            out_channels_first_layer=4,
            normalization=None,
            pooling_type='max',
            upsampling_type='conv',
            preactivation=False,
            residual=False,
            padding=1,
            padding_mode='circular',
            activation='ReLU',
            initial_dilation=None,
            dropout=0,
        )

    def estimator(self, x):
        xx = self.unet(x.unsqueeze(1)).squeeze(1)
        x = x + xx
        return x - x.mean(dim=(-3, -2, -1), keepdim=True)


class Gaussian_NPE_IsotropicD(Gaussian_NPE_Base):
    """Isotropic likelihood precision parameterized by k-node amplitudes.

    Replaces the full N³-element Precision_Matrix_FFT with a
    Precision_Matrix_IsotropicNodes that models D_like as a smooth function
    of |k| using n_nodes learnable log-amplitude values (default 32):

        D_like(|k|) = exp(interp(log|k|; log_D_nodes))

    Estimator is a plain UNet residual (no filter, no scale):

        mu(x) = x + UNet(x)

    Motivation: D_like(k) is empirically ~isotropic, so ~32 parameters
    capture the full spectral shape without N³ = 2M free values.
    """

    def __init__(self, box, *args, n_nodes=32, **kwargs):
        super().__init__(box, *args, **kwargs)
        self.Q_like = Precision_Matrix_IsotropicNodes(
            self.N, box.k.flatten(), n_nodes=n_nodes,
        )
        self.Q_post = Precision_Matrix_Sum(self.Q_like, self.Q_prior)

    def estimator(self, x):
        p3d = 6 * (20,)
        xx = F.pad(x.unsqueeze(0), p3d, "circular").squeeze(0)
        x = x + self.unet(xx.unsqueeze(1)).squeeze(1)
        return x - x.mean(dim=(-3, -2, -1), keepdim=True)


class Gaussian_NPE_Poisson(Gaussian_NPE_Base):
    """Gaussian NPE with Poisson galaxy shot-noise forward model.

    UNet-only estimator + isotropic D_like(|k|) likelihood precision.
    The Poisson forward model replaces the Gaussian noise injection:

        N_gal(r) ~ Poisson(N_bar * (1 + galaxy_bias * delta_z0(r)).clamp(0))
        x_eff(r) = (N_gal(r) / N_bar - 1) / galaxy_bias  ≈  delta_z0 + shot_noise

    where N_bar = n_bar * V_voxel.  The effective Gaussian-equivalent noise level
    is sigma_eff = 1 / (galaxy_bias * sqrt(N_bar)).
    """

    def __init__(self, box, prior, n_bar, galaxy_bias=1.5, n_nodes=32, **kwargs):
        V_voxel = (box.box_size / box.N) ** 3
        sigma_noise_eff = 1.0 / (galaxy_bias * (n_bar * V_voxel) ** 0.5)
        super().__init__(box, prior, sigma_noise=sigma_noise_eff, **kwargs)
        self.n_bar       = n_bar
        self.galaxy_bias = galaxy_bias
        self.V_voxel     = V_voxel
        
        self.Q_like = Precision_Matrix_IsotropicNodes(
            self.N, box.k.flatten(), n_nodes=n_nodes,
        )
        self.Q_post = Precision_Matrix_Sum(self.Q_like, self.Q_prior)

    def estimator(self, x):
        p3d = 6 * (20,)
        xx = F.pad(x.unsqueeze(0), p3d, "circular").squeeze(0)
        x = x + self.unet(xx.unsqueeze(1)).squeeze(1)
        return x - x.mean(dim=(-3, -2, -1), keepdim=True)

    def forward(self, A, B):
        x = A['delta_z0']
        N_bar  = self.n_bar * self.V_voxel
        N_mean = (N_bar * (1 + self.galaxy_bias * x)).clamp(min=0)
        N_obs  = torch.poisson(N_mean)
        x = N_obs / (N_bar * self.galaxy_bias) - 1.0 / self.galaxy_bias

        b = self.estimator(x)
        z = B['delta_z127'][:len(x)]
        z = self.rescaling_factor**-1 * z
        return self.loss(b, z)


class Gaussian_NPE_WienerIsotropicD(Gaussian_NPE_Base):
    """Wiener filter + UNet estimator with isotropic learnable likelihood precision.

    Combines the physically-motivated Wiener filter backbone of WienerNet with
    the compact isotropic D_like(k) parameterization of IsotropicD:

        D_like(|k|) = exp(interp(log|k|; log_D_nodes))   [32 learnable values]
        w(k) = D_like(k) / (D_prior(k) + D_like(k))       [Wiener weight]
        mu(x) = G_T(G(x) * w) + UNet(x)                   [filter + residual]

    Initialization: log_D_nodes = log(1/sigma_noise²) so that at the start of
    training the Wiener weight equals the exact analytic Wiener filter for
    white observation noise with variance sigma_noise².  Training then refines
    D_like(k) away from white noise if the data supports it.
    """

    def __init__(self, box, *args, n_nodes=32, **kwargs):
        super().__init__(box, *args, **kwargs)
        log_D_init = 0. #-2.0 * np.log(self.sigma_noise)   # log(1/sigma_noise²)
        self.Q_like = Precision_Matrix_IsotropicNodes(
            self.N, box.k.flatten(), n_nodes=n_nodes, log_D_init=log_D_init,
        )
        self.Q_post = Precision_Matrix_Sum(self.Q_like, self.Q_prior)

    def estimator(self, x):
        x_h = self.Q_post.G(x)
        D_prior = self.Q_prior.D
        D_like  = self.Q_like.D
        w = D_like / (D_prior + D_like)
        x_wiener = self.Q_post.G_T(x_h * w)

        p3d = 6 * (20,)
        xx = F.pad(x.unsqueeze(0), p3d, "circular").squeeze(0)
        correction = self.unet(xx.unsqueeze(1)).squeeze(1)
        x = x_wiener + correction
        return x - x.mean(dim=(-3, -2, -1), keepdim=True)


class Gaussian_NPE_Default_IsotropicD(Gaussian_NPE_Base):
    """Default (sigmoid filter + isotropic scale) architecture with isotropic Q_like.

    Combines the estimator structure of Gaussian_NPE_Network with the compact
    isotropic parameterizations of IsotropicD / WienerIsotropicD:

        D_like(|k|) = exp(interp(log|k|; log_D_nodes))     [n_nodes params]
        scale(|k|)  = exp(interp(log|k|; log_scale_nodes)) [n_nodes params]
        mu(x) = G_T(scale(|k|) · G(x + sigmoid_filter(UNet(x))))

    Both scale and D_like are interpolated on the same shell-snapped k-nodes,
    reusing Q_like's precomputed _left_idx / _right_idx / _frac buffers (no
    additional N³ buffers).
    Initialized: log_scale_nodes = 0 → scale = 1; log_D_nodes = 0 → D = 1.
    """

    def __init__(self, box, *args, k_cut=0.03, w_cut=0.001, n_nodes=32, **kwargs):
        super().__init__(box, *args, **kwargs)
        self.k_cut = k_cut
        self.w_cut = w_cut
        self.Q_like = Precision_Matrix_IsotropicNodes(
            self.N, box.k.flatten(), n_nodes=n_nodes,
        )
        self.Q_post = Precision_Matrix_Sum(self.Q_like, self.Q_prior)
        self.log_scale_nodes = torch.nn.Parameter(torch.zeros(n_nodes))

    @property
    def _isotropic_scale(self):
        """Interpolate log_scale_nodes → per-mode scale factors, shape (N³,)."""
        log_s = ((1 - self.Q_like._frac) * self.log_scale_nodes[self.Q_like._left_idx]
                 +     self.Q_like._frac  * self.log_scale_nodes[self.Q_like._right_idx])
        return log_s.exp()

    def estimator(self, x):
        p3d = 6 * (20,)
        xx = F.pad(x.unsqueeze(0), p3d, "circular").squeeze(0)
        xx = self.unet(xx.unsqueeze(1)).squeeze(1)
        x = x + self.box.sigmoid_filter(xx, self.k_cut, self.w_cut)
        x = self.Q_post.G_T(self.Q_post.G(x) * self._isotropic_scale)
        return x - x.mean(dim=(-3, -2, -1), keepdim=True)


class MAP_MSE_Network(swyft.AdamWReduceLROnPlateau, swyft.SwyftModule):
    """MAP estimator trained with mean-squared error loss.

    A point-estimate-only network that predicts the IC field via a residual
    U-Net without modelling the full posterior.  The loss is per-voxel MSE
    in internal (rescaled) space:

        mu(x) = x + UNet(x)
        loss  = MSE(mu(x), z_rescaled)

    No Q matrices or prior are used.  Serves as a baseline to compare
    against the Gaussian posterior networks.
    """

    def __init__(self, box, sigma_noise, rescaling_factor=1.0,
                 learning_rate=1e-2, early_stopping_patience=5, lr_scheduler_patience=1):
        super().__init__()
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.lr_scheduler_patience = lr_scheduler_patience

        self.box = box
        self.N = box.N
        self.sigma_noise = sigma_noise
        self.rescaling_factor = rescaling_factor
        self.unet = UNet(1, 1, hid_chan=16, bypass=False)

    def estimator(self, x):
        p3d = 6 * (20,)
        xx = F.pad(x.unsqueeze(0), p3d, "circular").squeeze(0)
        return x + self.unet(xx.unsqueeze(1)).squeeze(1)

    def forward(self, A, B):
        x = A['delta_z0']
        x += torch.randn_like(x) * self.sigma_noise
        b = self.estimator(x)

        z = B['delta_z127'][:len(x)]
        z = self.rescaling_factor**-1 * z
        return self.loss(b, z)

    def loss(self, z_MAP, z):
        """Per-voxel MSE in internal (rescaled) space."""
        loss = ((z_MAP - z)**2).mean(dim=(-3, -2, -1))
        return swyft.AuxLoss(loss, 'z')

    def get_z_MAP(self, x_obs):
        """Returns the MAP estimate for a given observation."""
        return self.estimator(x_obs.unsqueeze(0)).squeeze(0).detach() * self.rescaling_factor
