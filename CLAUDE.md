# Gaussian NPE — Project Context for Claude Code

## What this project is

Simulation-based inference framework for reconstructing cosmological initial conditions (ICs) from late-time density fields. The posterior over primordial matter density fields is modeled as a Gaussian with a diagonal precision matrix in Hartley (real Fourier) space, enabling fast sampling.

Training data: Quijote N-body simulations (128³ grid, 1 Gpc/h box). Fields: `delta_z0` (observed, z=0) and `delta_z127` (target ICs, z=127).

## Key files

- `gaussian_npe/networks.py` — All network architectures (Base + 7 variants)
- `gaussian_npe/matrices.py` — Precision matrix classes: Q = G^T D G factorization
- `gaussian_npe/utils.py` — Hartley transform, power spectra (CLASS, Pylians), growth factor, all plotting/diagnostics
- `scripts/train.py` — Training script with argparse, NETWORK_CLASSES dict
- `scripts/infer.py` — Inference + amortization test, also has NETWORK_CLASSES dict
- `paper_test_runs/sweep_noise.py` — SLURM sweep over sigma_noise values (calls train.py)

When adding a new network class, it must be registered in three places: `networks.py` (define), `__init__.py` (export), and both `train.py`/`infer.py` (NETWORK_CLASSES dict + import).

## Architecture

### Class hierarchy

```
Gaussian_NPE_Base  (abstract — Q matrices, UNet, loss, sample, forward)
├── Gaussian_NPE_Network       (sigmoid filter + scale) — "default"
│   └── Gaussian_NPE_LH        (per-sample rescaling for Latin Hypercube)
├── Gaussian_NPE_UNet_Only     (x + UNet(x)) — minimal baseline
├── Gaussian_NPE_WienerNet     (Wiener filter + UNet correction)
├── Gaussian_NPE_LearnableFilter  (N³ learnable filter logits + scale)
├── Gaussian_NPE_SmoothFilter     (~20 log-spaced k-node filter + scale)
└── Gaussian_NPE_Iterative        (multi-step 2-channel UNet + scale)
```

### What lives where

- **Base** has: box, Q_prior, Q_like, Q_post, UNet(1→1), sigma_noise, rescaling_factor, learning_rate, early_stopping_patience, lr_scheduler_patience. Methods: forward, estimator (raises NotImplementedError), log_prob, loss, get_z_MAP, sample. Note: `prior` tuple is passed to `__init__` but only used to construct Q_prior — it is NOT stored as `self.prior`.
- **Network** adds: k_cut, w_cut, self.scale (N³ learnable params). Implements estimator with sigmoid filter.
- **UNet_Only, WienerNet** inherit from Base directly — no k_cut, w_cut, or scale.
- **LearnableFilter, SmoothFilter, Iterative** inherit from Base, define their own self.scale. They do NOT inherit from Network.
- **LH** is the only child of Network (reuses its sigmoid estimator, overrides forward for per-sample rescaling).

### Consequence for scripts

In `train.py` and `infer.py`, `k_cut`/`w_cut` are only passed to networks that accept them:
```python
if network_name not in ('UNet_Only', 'WienerNet', 'Iterative'):
    net_kwargs['k_cut'] = ...
    net_kwargs['w_cut'] = ...
```
If you add a new network inheriting from Base that doesn't use k_cut/w_cut, add it to this exclusion list.

## Core math concepts

### Precision matrix decomposition
Q_post = Q_prior + Q_like, where each Q = G^T D G. G is the Hartley transform, D is diagonal in Hartley space. Access via: `Q_post.G(x)` (forward transform), `Q_post.G_T(x_h)` (inverse), `Q_post.D` (diagonal).

### Hartley transform
H(x) = Re(FFT(x)) - Im(FFT(x)), with ortho normalization. Self-inverse. Implemented as `hartley()` (torch) and `hartley_np()` (numpy) in utils.py.

### k=0 mode exclusion
Mean overdensity δ̄=0 by construction on a periodic box. The posterior lives on an (N³−1)-dimensional zero-mean subspace. log_prob and calibration diagnostics exclude k=0 via `D[1:]`, `r_h[:, 1:]` indexing.

### Rescaling factor
Fields at z=127 are rescaled by D(z=127)/D(z=0) ≈ 0.0099. The network works in rescaled (internal) space; `get_z_MAP` and `sample` multiply back by rescaling_factor for physical-space output.

### Logit parameterization
Filters w(k) ∈ [0,1] are parameterized as w = sigmoid(logit) where logit is unconstrained. Initialized so that sigmoid(logit) matches the default sigmoid filter: logit = (k - k_cut) / w_cut.

### self.scale
N³ learnable per-mode amplitude factors in Hartley space, initialized to 1. Applied as: `G_T(G(x) * scale)`.

## UNet details
Uses `map2map.models.UNet` with `bypass=False` (no internal skip connection). This means residual connections must be added explicitly in the estimator (e.g., `x + UNet(x)` in UNet_Only). All networks use circular (periodic) padding via `F.pad(x, 6*(20,), "circular")` before the UNet.

## Conventions

- `self.prior` is a tuple `(G_T, D, G)` — the prior Q-factor triple. Prefer `self.Q_post.G`, `self.Q_post.G_T` over `self.prior[2]`, `self.prior[0]`.
- `sample()` takes exactly one of `x_obs` or `z_MAP` (mutual exclusion enforced).
- Plotting functions save PNGs to `plots/{run_name}/` with calibration in a `calibration/` subfolder and amortization in an `amortization/` subfolder.
- `plot_samples_analysis`, `plot_calibration_diagnostics`, and `plot_amortization_test` accept `save_csv=False`. When True, they save CSV + human-readable txt files to a `diagnostics/` subfolder within their respective output directories. `train.py` and `infer.py` pass `save_csv=True`.
- Config is saved as JSON at start of training and printed to stdout.
- **Script header strings**: Both `train.py` and `infer.py` have a top-of-file docstring with a prose description and CLI usage examples. When adding, removing, or renaming any CLI argument, update the header docstring in the same script to keep it in sync.

## Running

```bash
# Training
python scripts/train.py --run_name baseline --network default --max_epochs 30 --sigma_noise 0.1

# Inference (reads config.json from model_dir)
python scripts/infer.py --model_dir ./runs/20260216_153000_baseline

# Amortization test
python scripts/infer.py --model_dir ./runs/... --target_dir ./Quijote_target
```

## Dependencies
PyTorch, PyTorch Lightning, swyft, map2map, classy (CLASS), Pylians3, numpy, scipy, matplotlib.
