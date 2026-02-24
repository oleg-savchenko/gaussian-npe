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
    def __init__(self, G_T, D, G, learnable=False):
        super().__init__()
        self._G = G
        self._G_T = G_T
        if learnable:
            self._D = torch.nn.Parameter(D)
        else:
            self.register_buffer('_D', D)

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

    Two parameterizations of the positive diagonal D are supported:
      'exp'  (default): D = exp(log_D),  log_D initialized to 0  → D=1.
                        Gradient of the log-det term is exactly constant (=0.5),
                        since log(exp(u)) = u.  Natural for precision parameters.
      'sqrt':           D = sqrt_D**2,   sqrt_D initialized to 1 → D=1.
                        Legacy option; suffers from sign symmetry and non-uniform
                        log-det gradients.
    """
    def __init__(self, N, parameterization='exp'):
        super().__init__()
        self.N = N
        self.shape = 3 * (self.N,)
        self.parameterization = parameterization
        if parameterization == 'exp':
            self.log_D = torch.nn.Parameter(torch.zeros(self.N**3))
        elif parameterization == 'sqrt':
            self.sqrt_D = torch.nn.Parameter(torch.ones(self.N**3))
        else:
            raise ValueError(f"Unknown parameterization '{parameterization}'. Choose 'exp' or 'sqrt'.")

    def G(self, x):
        x = hartley(x).flatten(-len(self.shape), -1)
        return x

    @property
    def D(self):
        if self.parameterization == 'exp':
            return self.log_D.exp()
        return self.sqrt_D**2

    def G_T(self, x):
        x = hartley(x.unflatten(-1, self.shape))
        return x

class Precision_Matrix_IsotropicNodes(GDG_Factor_Matrix):
    """Isotropic D_like(k) parameterized via learnable log-amplitude nodes.

    D(|k|) = exp(linear interpolation of log_D_nodes at log-spaced k-nodes)

    Only n_nodes parameters (default 32) instead of N³.  Three (N³,) buffers
    hold precomputed interpolation indices and fractions (~8 MB total vs.
    ~8 MB for a full N³ parameter tensor — but critically, the spectral shape
    is smooth and isotropic by construction).

    G / G_T are identical to Precision_Matrix_FFT (Hartley transform).
    """

    def __init__(self, N, k_flat, n_nodes=32):
        """
        Args:
            N      : grid side length (k grid is N³)
            k_flat : (N³,) tensor of |k| values (box.k.flatten())
            n_nodes: number of learnable log-D nodes (default 32)
        """
        super().__init__()
        self.N = N
        self.shape = 3 * (N,)

        k_F  = float(k_flat[k_flat > 0].min())
        k_Nq = float(k_flat.max())

        log_k_nodes = torch.linspace(
            float(torch.tensor(k_F).log()),
            float(torch.tensor(k_Nq).log()),
            n_nodes,
        )
        log_k = torch.log(k_flat.clamp(min=k_F))
        log_k_nodes = log_k_nodes.to(log_k.device)

        idx  = torch.searchsorted(log_k_nodes, log_k).clamp(1, n_nodes - 1)
        frac = ((log_k - log_k_nodes[idx - 1])
                / (log_k_nodes[idx] - log_k_nodes[idx - 1])).clamp(0, 1)

        self.register_buffer('_left_idx',  idx - 1)
        self.register_buffer('_right_idx', idx)
        self.register_buffer('_frac',      frac)
        self.log_D_nodes = torch.nn.Parameter(torch.zeros(n_nodes))

    def G(self, x):
        return hartley(x).flatten(-3, -1)

    @property
    def D(self):
        log_D = ((1 - self._frac) * self.log_D_nodes[self._left_idx]
                 +      self._frac  * self.log_D_nodes[self._right_idx])
        return log_D.exp()

    def G_T(self, x):
        return hartley(x.unflatten(-1, self.shape))


class Precision_Matrix_Sum(GDG_Factor_Matrix):
    """Sum of two precision matrices diagonal in the same basis.

    Q = Q1 + Q2 = G_T * (D1 + D2) * G

    The two matrices must share the same basis G (e.g. both diagonal in
    Hartley space).  The eigenvalues D are computed dynamically as the sum
    of the component eigenvalues, so learnable parameters in Q1 or Q2 are
    tracked through the computation graph.
    """
    def __init__(self, Q1, Q2):
        super().__init__()
        # Store as a plain list (not nn.ModuleList) to avoid registering Q1
        # and Q2 as submodules here — they are already registered at the
        # parent network level, so using nn.Module attributes would double-
        # count their parameters in PyTorch Lightning's model summary.
        self._refs = [Q1, Q2]

    @property
    def Q1(self):
        return self._refs[0]

    @property
    def Q2(self):
        return self._refs[1]

    def G(self, x):
        return self.Q1.G(x)

    @property
    def D(self):
        return self.Q1.D + self.Q2.D

    def G_T(self, x):
        return self.Q1.G_T(x)


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
