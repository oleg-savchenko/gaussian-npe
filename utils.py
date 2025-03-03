import numpy as np
import torch
import Pk_library as PKL
from classy import Class

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
