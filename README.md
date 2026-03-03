# Gaussian NPE

**Fast Sampling of Cosmological Initial Conditions with Gaussian Neural Posterior Estimation**

*O. Savchenko, G. Franco Abell&aacute;n, F. List, N. Anau Montel, C. Weniger*

A simulation-based inference (SBI) framework for reconstructing cosmological initial conditions from late-time density fields. The method models the posterior over primordial matter density fields as a Gaussian with a diagonal precision matrix in Fourier space, enabling generation of thousands of posterior samples within seconds on a single GPU.

## Method

Reconstructing the initial conditions (ICs) of the Universe from present-day observations requires exploring a multi-million-dimensional parameter space. This code implements a Gaussian Neural Posterior Estimation approach that makes this tractable by exploiting the structure of the problem.

### Gaussian posterior in Fourier space

Given an observed density field **x** at redshift z=0, we model the posterior over the initial conditions **z** (at z=127) as:

$$p(\mathbf{z} \mid \mathbf{x}) \propto \exp\left(-\frac{1}{2}(\mathbf{z} - \hat{\boldsymbol{\mu}}_\theta(\mathbf{x}))^\top \mathbf{Q}_\theta \, (\mathbf{z} - \hat{\boldsymbol{\mu}}_\theta(\mathbf{x}))\right)$$

where the posterior precision matrix decomposes as:

$$\mathbf{Q}_\theta = \mathbf{Q}^P + \mathbf{Q}^L_\theta$$

- **Q^P** is the prior precision, fixed from the linear matter power spectrum P(k) at z=127: Q^P = H^T diag(1/P(k)) H, where H is the Hartley transform.
- **Q^L_&theta;** is the learned likelihood precision, also diagonal in Fourier space: Q^L_&theta; = H^T D^L_&theta; H, with D^L_&theta; learned during training.

### MAP estimator

The posterior mean (MAP estimate) is computed by a U-Net with periodic (circular) padding:

$$\hat{\boldsymbol{\mu}}_\theta(\mathbf{x}) = \mathbf{H}^\top \left\lbrace \alpha_\theta(k) \odot \left[ \mathbf{H}\lbrace\mathbf{x}\rbrace + \sigma_{>k_\Lambda}\left(\mathbf{H}\lbrace\text{UNet}_\theta(\mathbf{x})\rbrace\right) \right] \right\rbrace$$

where &sigma;_{>k_&Lambda;} is a sigmoidal high-pass filter (default cutoff k_&Lambda; = 0.03 h/Mpc) that lets the U-Net correct only small-scale modes while preserving the large-scale linear relationship.

### Sampling

Once trained, posterior samples are drawn as:

$$\mathbf{z} = \hat{\boldsymbol{\mu}}_\theta(\mathbf{x}_\text{obs}) + \mathbf{H}^\top \left\lbrace (D^P + D^L_\theta)^{-1/2} \odot \boldsymbol{\epsilon} \right\rbrace$$

where **&epsilon;** is a white noise field. This requires only a single forward pass for the MAP and then trivial element-wise operations per sample.

## Repository structure

```
gaussian_npe/               Python package
    __init__.py
    matrices.py             Precision matrix parametrizations (Q = G^T D G)
    networks.py             Network architectures (Base + 7 variants)
    utils.py                Power spectra (CLASS, Pylians), growth factor,
                            Hartley transform, plotting & diagnostics
scripts/
    train.py                Training script with argparse, timestamped runs, checkpointing
    infer.py                Inference & amortization test script
```

## Network architectures

All architectures share the same Gaussian posterior machinery (precision matrices, loss, sampling) via `Gaussian_NPE_Base`. They differ only in how the MAP estimator is computed:

| Network | `--network` | Description |
|---------|-------------|-------------|
| `Gaussian_NPE_Network` | `default` | Sigmoid high-pass filter + learnable per-mode scale |
| `Gaussian_NPE_UNet_Only` | `UNet_Only` | Minimal baseline: x + UNet(x), no Fourier-space operations |
| `Gaussian_NPE_WienerNet` | `WienerNet` | Wiener filter + UNet residual correction |
| `Gaussian_NPE_LearnableFilter` | `LearnableFilter` | Learnable per-mode filter (N^3 free parameters) + scale |
| `Gaussian_NPE_SmoothFilter` | `SmoothFilter` | Learnable smooth isotropic filter (~20 log-spaced k-nodes) + scale |
| `Gaussian_NPE_Iterative` | `Iterative` | Multi-step 2-channel UNet refinement + scale |
| `Gaussian_NPE_LH` | `LH` | Default estimator with per-sample rescaling for Latin Hypercube runs |

Class hierarchy:

```
Gaussian_NPE_Base  (abstract — Q matrices, UNet, loss, sample, forward)
├── Gaussian_NPE_Network       (sigmoid filter + scale)
│   └── Gaussian_NPE_LH        (per-sample rescaling)
├── Gaussian_NPE_UNet_Only     (x + UNet(x))
├── Gaussian_NPE_WienerNet     (Wiener filter + UNet)
├── Gaussian_NPE_LearnableFilter  (N³ learnable filter + scale)
├── Gaussian_NPE_SmoothFilter     (~20 k-node smooth filter + scale)
└── Gaussian_NPE_Iterative        (iterative 2-channel UNet + scale)
```

## Training data

The code is designed for [Quijote N-body simulations](https://quijote-simulations.readthedocs.io) stored as a [swyft](https://github.com/undark-lab/swyft) ZarrStore containing pairs of density fields:

| Parameter | Value |
|-----------|-------|
| Box size | 1 Gpc/h |
| Grid resolution | 128^3 |
| N-body particles | 512^3 |
| Initial redshift | z = 127 |
| Final redshift | z = 0 |
| Cosmology | Planck 2018 fiducial (&Omega;_m=0.3175, h=0.6711, n_s=0.9624, &sigma;_8=0.834) |

A separate target observation is provided as a `.pt` file with keys `delta_z0` and `delta_z127`.

## Dependencies

- **PyTorch** and **PyTorch Lightning**
- **[swyft](https://github.com/undark-lab/swyft)** &mdash; SBI framework, data module, and trainer
- **[map2map](https://github.com/eelregit/map2map)** &mdash; U-Net architecture (`map2map.models.UNet`)
- **[CLASS](https://github.com/lesgourg/class_public)** (via `classy`) &mdash; Boltzmann solver for the linear matter power spectrum
- **[Pylians3](https://github.com/franciscovillaescusa/Pylians3)** &mdash; power spectrum, cross-spectrum, and bispectrum computation
- **NumPy**, **SciPy**, **Matplotlib**

## Usage

### Training

```bash
python scripts/train.py --run_name baseline --max_epochs 30 --sigma_noise 0.1
```

All hyperparameters can be set via command-line flags (see `python scripts/train.py --help`):

| Flag | Default | Description |
|------|---------|-------------|
| `--run_name` | `""` | Run identifier (appended to timestamp) |
| `--network` | `default` | Architecture (see table above) |
| `--store_path` | *(cluster path)* | Path to swyft ZarrStore |
| `--target_path` | *(cluster path)* | Target observation for diagnostics |
| `--max_epochs` | 30 | Maximum training epochs |
| `--sigma_noise` | 0.1 | Noise std added to observed field during training |
| `--k_cut` | 0.03 | High-pass filter cutoff [h/Mpc] |
| `--w_cut` | 0.001 | High-pass filter width |
| `--batch_size` | 8 | Training batch size |
| `--num_samples` | 100 | Posterior samples for post-training diagnostics |

Training outputs are saved to `./runs/{timestamp}_{run_name}/`:

```
runs/20260216_153000_baseline/
    config.json         Full configuration for reproducibility
    model.pt            Model state dict
    logs/               TensorBoard logs and Lightning checkpoints
    plots/              Diagnostic figures
```

Training takes approximately 1.5 hours on a single 40GB NVIDIA A100 GPU.

### Inference

Load a trained model and generate posterior samples for a (potentially new) observation:

```bash
python scripts/infer.py --model_dir ./runs/20260216_153000_baseline
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model_dir` | *(required)* | Path to training run directory |
| `--target_path` | from training config | Target observation `.pt` file |
| `--num_samples` | 100 | Number of posterior samples |
| `--run_name` | `""` | Inference run identifier |
| `--target_dir` | `None` | Directory of held-out `.pt` files for amortization test |

Inference outputs are saved to `{model_dir}/infer/{timestamp}_{run_name}/`.

Generating 1000 posterior samples takes less than 3 seconds on GPU.

#### Amortization test

To evaluate calibration across many held-out observations:

```bash
python scripts/infer.py \
    --model_dir ./runs/20260216_153000_baseline \
    --target_dir ./Quijote_target \
    --noise_seed 42
```

### Diagnostic plots

Both scripts produce diagnostic figures:

**Sample analysis** (`plot_samples_analysis`):

1. **Field slices** &mdash; true IC field, one posterior sample, and residual (3&times;3 grid at different slice depths)
2. **Truth / MAP / posterior std** &mdash; side-by-side comparison
3. **Summary statistics** &mdash; power spectrum P(k), transfer function T(k), and cross-correlation C(k) with 1&sigma;/2&sigma; bands from samples, plus MAP and linear theory curves
4. **1-point PDF** &mdash; voxel histogram with skewness and kurtosis annotations
5. **Reduced bispectrum** Q(&theta;) &mdash; two triangle configurations (equilateral and squeezed) with 1&sigma;/2&sigma; bands

**Calibration diagnostics** (`plot_calibration_diagnostics`, saved to `calibration/` subfolder):

6. **True vs predicted Hartley modes** &mdash; scatter plot with posterior error bars
7. **Per-mode &chi;&sup2;** &mdash; D[k]&middot;r_h[k]&sup2; vs expected &chi;&sup2;(1) distribution
8. **Calibration summary** &mdash; reduced &chi;&sup2;, z-score, skewness, kurtosis

## Citation

If you use this code, please cite:

```bibtex
@article{Savchenko2025,
    title={Fast Sampling of Cosmological Initial Conditions with Gaussian Neural Posterior Estimation},
    author={Savchenko, Oleg and Franco Abell{\'a}n, Guillermo and List, Florian and Anau Montel, Noemi and Weniger, Christoph},
    year={2025}
}
```

## License

See the repository for license information.
