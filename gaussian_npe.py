import torch
import torch.nn.functional as F
import swyft
from map2map.models import UNet
from abc import ABC, abstractmethod
from gaussian_npe.utils import hartley

class GDG_Factor_Matrix(torch.nn.Module, ABC):
    """Parametrization of a symmetric positive definite matrix.
    
    Q = G_T * D * G
    
    Input and output shapes are (B, N, N, N), where shape = (N, N, N).
    """
    def __init__(self):
        super().__init__()     
    
    def forward(self, x):
        """(B, N, N, N) --> (B, N, N, N)"""
        x = self.G(x)
        x = self.D*x
        x = self.G_T(x)
        return x
    
    def get_factors(self):
        G_T = self.G_T
        G = self.G
        D = self.D
        return G_T, D, G

    @abstractmethod
    def G(self, x):
        pass
    
    @property
    @abstractmethod
    def D(self):
        pass

    @abstractmethod
    def G_T(self, x):
        pass

class Precision_Matrix_From_Factors(GDG_Factor_Matrix):
    """Parametrization of a symmetric positive definite matrix.
    
    Q = G_T * D * G
    
    Input and output shapes are (B, N, N, N), where shape = (N, N, N).
    """
    def __init__(self, G_T, D, G):
        super().__init__()
        self._G = G
        self._G_T = G_T
        self._D = torch.nn.Parameter(D, requires_grad = False)

    def G(self, x):
        return self._G(x)
    
    @property
    def D(self):
        return self._D
    
    def G_T(self, x):
        return self._G_T(x)

class Precision_Matrix_FFT(GDG_Factor_Matrix):
    """Parametrization of a symmetric positive definite matrix diagonal in Fourier space.
    
    Q = F_T * D * F
    
    Input and output shapes are (B, N, N, N), where shape = (N, N, N).
    """
    def __init__(self, N):
        super().__init__()
        self.N = N
        self.shape = 3 * (self.N,)
        # self.logD = torch.nn.Parameter(torch.ones(self.N**3))    # tried in the past, sqrt works better usually
        self.sqrt_D = torch.nn.Parameter(torch.ones(self.N**3))

    def G(self, x):
        x = hartley(x).flatten(-len(self.shape), -1)
        return x
    
    @property
    def D(self):
        # return torch.exp(self.logD)
        return self.sqrt_D**2

    def G_T(self, x):
        x = hartley(x.unflatten(-1, self.shape))
        return x

class Precision_Matrix_Single_Conv(GDG_Factor_Matrix):
    """Parametrization of a symmetric positive definite matrix.
    
    Q = G_T * D * G
    
    Input and output shapes are (B, N, N, N), where shape = (N, N, N).
    """
    def __init__(self, N, p = 1):
        super().__init__()
        
        self._conv1 = torch.nn.Conv3d(1, 1, 2*p+1, padding = p, bias = False)
        self._conv1.weight = torch.nn.Parameter(torch.ones_like(self._conv1.weight))
        self._conv1T = torch.nn.ConvTranspose3d(1, 1, 2*p+1, padding = p, bias = False)
        # self._logD = torch.nn.Parameter(torch.zeros(N**3)-1.)
        self.sqrt_D = torch.nn.Parameter(torch.ones(N**3))
        self.shape = (N, N, N)

    def G(self, x):
        x = self._conv1(x.unsqueeze(1)).squeeze(1)
        x = x.flatten(1, 3)
        return x
    
    @property
    def D(self):
        # return torch.exp(self.logD)
        return self.sqrt_D**2
    
    def G_T(self, x):
        self._conv1T.weight = self._conv1.weight
        x = x.unflatten(1, self.shape)
        x = self._conv1T(x.unsqueeze(1)).squeeze(1)
        return x

class Gaussian_NPE_Network(swyft.AdamWReduceLROnPlateau, swyft.SwyftModule):
    def __init__(self, box, prior, rescaling_factor, k_cut, w_cut):
        super().__init__()
        self.learning_rate = 1e-2
        self.early_stopping_patience = 5
        self.lr_scheduler_patience = 1

        self.box = box
        self.N = box.N
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
        x = self.prior[0](self.prior[2](x) * self.scale)    # the method also works fine without this factor, but it makes the results slightly better
        return x        

    def forward(self, A, B):
        x = A['delta_z0']
        b = self.estimator(x)

        z = B['delta_z127'][:len(x)]
        z = self.rescaling_factor**-1 * z
        return self.loss(b, z)

    def loss(self, z_MAP, z):
        """Calculates the loss function.
        """
        loss = 0.5 * ((z - z_MAP)*(self.Q_like(z - z_MAP) + self.Q_prior(z - z_MAP))).mean(dim=(-3, -2, -1)) - 0.5 * torch.log(self.Q_like.D + self.Q_prior.D).mean(dim=-1)
        return swyft.AuxLoss(loss.reshape(-1), 'z')
    
    def get_z_MAP(self, x_obs):
        """Returns the MAP estimation for a given x_obs.
        """
        return self.estimator(x_obs.unsqueeze(0)).squeeze(0).detach() * self.rescaling_factor

    def sample(self, num_samples, x_obs = None, z_MAP = None, to_numpy=True):
        """Samples the posterior for a given x_obs (or z_MAP), Q_prior and Q_like.
        """
        if z_MAP is None:
            z_MAP = self.get_z_MAP(x_obs)
        D_prior = self.Q_prior.D.detach().cuda() * self.rescaling_factor**-2
        D_like = self.Q_like.D.detach().cuda() * self.rescaling_factor**-2
        
        if to_numpy:
            draws = [(self.prior[0]((D_prior + D_like)**-0.5 * torch.randn_like(D_prior)) + z_MAP).cpu().numpy() for _ in range(num_samples)]
        else:
            draws = [(self.prior[0]((D_prior + D_like)**-0.5 * torch.randn_like(D_prior)) + z_MAP).cpu() for _ in range(num_samples)]
        return draws
