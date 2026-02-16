import torch
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
        self.sqrt_D = torch.nn.Parameter(torch.ones(self.N**3))

    def G(self, x):
        x = hartley(x).flatten(-len(self.shape), -1)
        return x

    @property
    def D(self):
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
        self.sqrt_D = torch.nn.Parameter(torch.ones(N**3))
        self.shape = (N, N, N)

    def G(self, x):
        x = self._conv1(x.unsqueeze(1)).squeeze(1)
        x = x.flatten(1, 3)
        return x

    @property
    def D(self):
        return self.sqrt_D**2

    def G_T(self, x):
        self._conv1T.weight = self._conv1.weight
        x = x.unflatten(1, self.shape)
        x = self._conv1T(x.unsqueeze(1)).squeeze(1)
        return x
