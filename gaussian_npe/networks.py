import torch
import torch.nn.functional as F
import swyft
from map2map.models import UNet
from gaussian_npe.Q_matrices import Precision_Matrix_From_Factors, Precision_Matrix_FFT
from gaussian_npe.utils import hartley

class Gaussian_NPE_Network(swyft.AdamWReduceLROnPlateau, swyft.SwyftModule):
    def __init__(self, box, prior, sigma_noise, rescaling_factor, k_cut, w_cut):
        super().__init__()
        self.learning_rate = 1e-2
        self.early_stopping_patience = 5
        self.lr_scheduler_patience = 1

        self.box = box
        self.N = box.N
        self.sigma_noise = sigma_noise
        self.rescaling_factor = rescaling_factor
        self.k_cut = k_cut
        self.w_cut = w_cut
        self.scale = torch.nn.Parameter(torch.ones(self.N**3))
        self.unet = UNet(1, 1, hid_chan=16, bypass=False)

        self.prior = prior
        self.Q_prior = Precision_Matrix_From_Factors(*prior)
        self.Q_like = Precision_Matrix_FFT(self.N)

    def estimator(self, x):
        p3d = 6*(20,)
        xx = F.pad(x.unsqueeze(0), p3d, "circular").squeeze(0)
        xx = self.unet(xx.unsqueeze(1)).squeeze(1)

        x = x + self.box.sigmoid_filter(xx, self.k_cut, self.w_cut)
        x = self.prior[0](self.prior[2](x) * self.scale)
        return x

    def forward(self, A, B):
        x = A['delta_z0']
        x += torch.randn_like(x) * self.sigma_noise    # add noise to the final field
        b = self.estimator(x)

        z = B['delta_z127'][:len(x)]
        z = self.rescaling_factor**-1 * z
        return self.loss(b, z)

    def loss(self, z_MAP, z):
        """Calculates the loss function."""
        loss = 0.5 * ((z - z_MAP)*(self.Q_like(z - z_MAP) + self.Q_prior(z - z_MAP))).mean(dim=(-3, -2, -1)) - 0.5 * torch.log(self.Q_like.D + self.Q_prior.D).mean(dim=-1)
        return swyft.AuxLoss(loss.reshape(-1), 'z')

    def get_z_MAP(self, x_obs):
        """Returns the MAP estimation for a given x_obs."""
        return self.estimator(x_obs.unsqueeze(0)).squeeze(0).detach() * self.rescaling_factor

    def sample(self, num_samples, x_obs = None, z_MAP = None, to_numpy=True):
        """Samples the posterior for a given x_obs (or z_MAP), Q_prior and Q_like."""
        if z_MAP is None:
            z_MAP = self.get_z_MAP(x_obs)
        D_prior = self.Q_prior.D.detach().cuda() * self.rescaling_factor**-2
        D_like = self.Q_like.D.detach().cuda() * self.rescaling_factor**-2

        if to_numpy:
            draws = [(self.prior[0]((D_prior + D_like)**-0.5 * torch.randn_like(D_prior)) + z_MAP).cpu().numpy() for _ in range(num_samples)]
        else:
            draws = [(self.prior[0]((D_prior + D_like)**-0.5 * torch.randn_like(D_prior)) + z_MAP).cpu() for _ in range(num_samples)]
        return draws


class Gaussian_NPE_WienerNet(Gaussian_NPE_Network):
    """Wiener filter baseline + U-Net residual correction.

    Replaces the sigmoid high-pass filter and per-mode scale with a
    physically-motivated Wiener filter that provides a smooth, data-driven
    transition between prior-dominated and likelihood-dominated modes:

        mu(x) = WienerFilter(x; D_like, D_prior) + UNet(x)

    The Wiener weight w(k) = D_like(k) / (D_prior(k) + D_like(k)) naturally
    passes modes where the data is informative and suppresses where the prior
    dominates, without an arbitrary k_cut.
    """

    def __init__(self, box, prior, sigma_noise, rescaling_factor, k_cut=0.03, w_cut=0.001):
        super().__init__(box, prior, sigma_noise, rescaling_factor, k_cut, w_cut)
        # scale parameter is inherited but unused; Wiener weights come from D_like/D_prior

    def estimator(self, x):
        # Wiener filter in Hartley space
        x_h = self.prior[2](x)                         # (B, N^3)
        D_prior = self.Q_prior.D                        # (N^3,)
        D_like = self.Q_like.D                          # (N^3,)
        w = D_like / (D_prior + D_like)                 # (N^3,)
        x_wiener = self.prior[0](x_h * w)              # (B, N, N, N)

        # U-Net learns nonlinear residual correction
        p3d = 6 * (20,)
        xx = F.pad(x.unsqueeze(0), p3d, "circular").squeeze(0)
        correction = self.unet(xx.unsqueeze(1)).squeeze(1)

        return x_wiener + correction


class Gaussian_NPE_LearnableFilter(Gaussian_NPE_Network):
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

    def __init__(self, box, prior, sigma_noise, rescaling_factor, k_cut=0.03, w_cut=0.001):
        super().__init__(box, prior, sigma_noise, rescaling_factor, k_cut, w_cut)

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
        x = self.prior[0](self.prior[2](x) * self.scale)
        return x


class Gaussian_NPE_Iterative(Gaussian_NPE_Network):
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

    def __init__(self, box, prior, sigma_noise, rescaling_factor,
                 k_cut=0.03, w_cut=0.001, num_iterations=3):
        super().__init__(box, prior, sigma_noise, rescaling_factor, k_cut, w_cut)
        self.num_iterations = num_iterations
        # Override U-Net: 2 input channels (current estimate + observation)
        self.unet = UNet(2, 1, hid_chan=16, bypass=False)

    def estimator(self, x):
        mu = x
        p3d = 6 * (20,)
        for _ in range(self.num_iterations):
            inp = torch.cat([mu.unsqueeze(1), x.unsqueeze(1)], dim=1)  # (B, 2, N, N, N)
            inp_padded = F.pad(inp, p3d, "circular")
            correction = self.unet(inp_padded).squeeze(1)              # (B, N, N, N)
            mu = mu + correction

        mu = self.prior[0](self.prior[2](mu) * self.scale)
        return mu


class Gaussian_NPE_LH(Gaussian_NPE_Network):
    """Variant for the Quijote Latin Hypercube dataset with varying cosmology.

    Instead of a fixed rescaling factor D(z_ic)/D(z=0) shared across all
    simulations, computes it per-sample from the cosmological parameters
    stored alongside each simulation.

    The ZarrStore must contain 'sim_params' of shape (5,) per sample:
        [Omega_m, Omega_b, h, n_s, sigma_8]

    The estimator, Q matrices, and loss are identical to Gaussian_NPE_Network.
    Only the rescaling in forward() is changed to be per-sample.

    For inference, call set_rescaling(sim_params) with the target's
    cosmological parameters before using get_z_MAP() / sample().
    """

    Z_IC = 127

    def __init__(self, box, prior, sigma_noise, rescaling_factor=1.0,
                 k_cut=0.03, w_cut=0.001):
        super().__init__(box, prior, sigma_noise, rescaling_factor, k_cut, w_cut)

    @staticmethod
    def _growth_D_tensor(Omega_m, z):
        """Tensor-compatible growth factor approximation D(z) for flat LCDM."""
        Omega_L = 1.0 - Omega_m
        Om_m = Omega_m * (1.0 + z) ** 3 / (Omega_L + Omega_m * (1.0 + z) ** 3)
        Om_L = Omega_L / (Omega_L + Omega_m * (1.0 + z) ** 3)
        return (1.0 / (1.0 + z)) * (5.0 * Om_m / 2.0) / (
            Om_m ** (4.0 / 7.0) - Om_L
            + (1.0 + Om_m / 2.0) * (1.0 + Om_L / 70.0)
        )

    def _compute_rescaling(self, sim_params):
        """Compute D(z_ic) per sample from sim_params.

        Args:
            sim_params: (B, 5) tensor [Omega_m, Omega_b, h, n_s, sigma_8]
        Returns:
            (B,) tensor of rescaling factors
        """
        Omega_m = sim_params[:, 0]
        return self._growth_D_tensor(Omega_m, self.Z_IC)

    def set_rescaling(self, sim_params):
        """Set rescaling_factor for inference from a single sim_params vector.

        Args:
            sim_params: array-like of shape (5,) [Omega_m, Omega_b, h, n_s, sigma_8]
        """
        from gaussian_npe.utils import growth_D_approx
        cosmo = {
            'Omega_cdm': float(sim_params[0]) - float(sim_params[1]),
            'Omega_b': float(sim_params[1]),
        }
        self.rescaling_factor = growth_D_approx(cosmo, self.Z_IC)

    def forward(self, A, B):
        x = A['delta_z0']
        x += torch.randn_like(x) * self.sigma_noise
        b = self.estimator(x)

        z = B['delta_z127'][:len(x)]
        sim_params = B['sim_params'][:len(x)]          # (B, 5)
        rescaling = self._compute_rescaling(sim_params) # (B,)
        z = z / rescaling[:, None, None, None]

        return self.loss(b, z)
