# Comprehensive Analysis of Network Comparison Runs

**Date:** 2026-02-19
**Runs directory:** `paper_test_runs/runs/`

---

## 1. Run Overview

Six networks were trained on the same data with identical hyperparameters (except where noted):
- sigma_noise=0.1, lr=0.01, early_stopping_patience=5, lr_scheduler_patience=3, max_epochs=50
- batch_size=8 for all **except Iterative (batch_size=4)**

| Network | Best Epoch | Train Time | Stop Reason |
|---------|-----------|-----------|-------------|
| default | 46 | 2h 05m | Early stop (epoch 49) |
| UNet_Only | 38 | 1h 48m | Early stop (epoch 43) |
| WienerNet | 35 | 1h 42m | Early stop (epoch 40) |
| LearnableFilter | 42 | 2h 00m | Early stop (epoch 47) |
| SmoothFilter | 15 | 0h 53m | Early stop (epoch 20) |
| Iterative | 0 | 0h 38m | Early stop (epoch 5) |

---

## 2. Training Dynamics

### WienerNet — best start
Initial val_loss at epoch 0 is **19.49**, vs ~32.4 for every other network. The hard-coded Wiener
filter gives an excellent initialization that skips much of the early exploration phase. It also
shows smooth, monotonically decreasing validation loss through the entire LR=0.01 phase (rare
among these runs — most others show large spikes).

### default — steady long-haul winner
Converges more slowly than WienerNet but keeps improving all the way to epoch 46 before early
stopping. Notably still had room to improve; **max_epochs=50 was the binding constraint** (early
stopping fired at epoch 49, only 3 epochs after the best). Val loss spikes are present but brief
(e.g., epoch 27: 11.10) and self-correcting.

### UNet_Only & LearnableFilter — similar story, noisier
Both show moderate val_loss spikes during the LR=0.01 phase (UNet_Only epoch 18: 13.35;
LearnableFilter epoch 10: 11.63), stabilize after LR drops, and converge to similar plateaus
around epoch 30+. LearnableFilter stabilizes ~0.17 above UNet_Only.

### SmoothFilter — broken LR schedule
The most dramatic failure mode. Val_loss at LR=0.01 oscillates wildly:

```
epoch 13: 12.16 → epoch 15: 9.99 (best) → epoch 16: 12.67 → epoch 17: 10.47 → ...
```

The oscillations are too frequent for the scheduler (patience=3) to ever trigger a LR drop —
it needs 3 *consecutive* bad epochs, but the loss keeps recovering just enough. Training stops
at epoch 20 with LR still at 0.001 (first ever drop). **The network never gets to train at low LR.**

### Iterative — catastrophic overfitting
Train loss = 5.5 at epoch 0 (vs ~12 for everyone else), then keeps falling to 4.8 while val
loss explodes monotonically from 6.1 → 13.1 over 5 epochs. Classic severe overfitting with
zero generalization. The different loss scale is suspicious — may indicate an architectural
issue in how the loss is computed for this 2-channel design. Also uses half the batch size.

---

## 3. Quantitative Results

### 3a. Validation Loss (negative log-likelihood under posterior; lower = better)

| Network | Min Val Loss |
|---------|-------------|
| **default** | **9.6053** ← best |
| UNet_Only | 9.6664 |
| WienerNet | 9.7234 |
| LearnableFilter | 9.8307 |
| SmoothFilter | 9.9927 |
| Iterative | 6.0774 (epoch 0 only; meaningless) |

### 3b. Calibration Diagnostics (N_modes = 2,097,151; ideal chi2 = 1.0, ideal 2σ coverage = 0.9545)

| Network | Reduced chi2 | z-score(z_true) | 2σ coverage | log p(z_true\|x) |
|---------|-------------|-----------------|-------------|-----------------|
| WienerNet | **1.2200** | -220.33 | 0.9386 | -2,682,193 |
| LearnableFilter | 1.2244 | -230.24 | 0.9325 | -2,702,069 |
| UNet_Only | 1.2328 | **-213.76** | **0.9398** | -2,671,136 |
| default | 1.2359 | -254.11 | 0.9133 | **-2,657,981** |
| SmoothFilter | 1.2672 | -287.52 | 0.9229 | -2,737,832 |
| Iterative | 1.3786 | -359.72 | 0.8867 | -3,240,738 |

### 3c. Field Statistics (samples vs truth)

Skewness of truth: 0.0361; kurtosis: -0.0024 (nearly Gaussian, as expected for IC fields).

| Network | Skewness (samples) | Kurtosis (samples) |
|---------|-------------------|-------------------|
| UNet_Only | -0.0675 ± 0.0015 | -0.0778 ± 0.0027 |
| WienerNet | -0.0614 ± 0.0012 | -0.0763 ± 0.0029 |
| default | -0.0606 ± 0.0014 | -0.0880 ± 0.0027 |
| LearnableFilter | **-0.0572 ± 0.0012** | -0.0768 ± 0.0028 |
| SmoothFilter | -0.0584 ± 0.0014 | -0.0669 ± 0.0030 |
| Iterative | -0.0372 ± 0.0016 | -0.1317 ± 0.0032 |

All networks produce samples with slightly negative skewness and slightly negative kurtosis
relative to truth (truth skewness: +0.036, kurtosis: -0.002). Iterative is the worst outlier.

---

## 4. Verdict

### Overall winner: ambiguous between **default** and **WienerNet**, for different reasons.

**`default`** is the best on the primary training metric (val_loss = 9.6053, i.e., highest posterior
density at the MAP estimate) and on log p(z_true | x). It is the most expressive estimator. But it
pays a calibration price — the posterior is the most overconfident of the three stable networks
(2σ coverage only 91.3%, chi2 = 1.24).

**`WienerNet`** is physically best-motivated and produces the best-calibrated posterior (chi2 = 1.22,
coverage 93.9%). Its hard-coded Wiener filter enforces the correct signal-to-noise structure
analytically, which constrains the solution space and prevents overfitting of the spectral shape.
It also trains fastest and most stably. The trade-off is ~0.12 higher val_loss than `default` —
meaning its MAP estimate is slightly less sharp.

**`UNet_Only`** is a competitive dark horse: best 2σ coverage (93.98%), second-best z-score, and
trains stably. The absence of any spectral filter means the UNet must learn all spectral
structure from scratch, yet it does so almost as well as networks with explicit filters.

**`LearnableFilter`** is broadly similar to UNet_Only but with better chi2. It uses 128³ free
spectral logits — a huge filter parameter count — yet converges to almost the same place as
UNet_Only. This suggests the UNet is already doing most of the spectral work and the extra
spectral parameters may not be worth their cost.

**`SmoothFilter`** cannot be fairly judged — it stopped at epoch 15/LR=0.01, never reaching the
fine-tuning phase. Its metrics are substantially worse as a result (chi2=1.27, coverage=92.3%).

**`Iterative`** failed completely. Catastrophic overfitting from the very first epoch.

### Summary ranking

| Rank | Network | Justification |
|------|---------|---------------|
| 1 | **default** | Best val_loss and log p; strong MAP recovery |
| 2 | **WienerNet** | Best calibration; most physically grounded; stable training |
| 3 | **UNet_Only** | Best coverage; simple and stable; no filter needed |
| 4 | LearnableFilter | Similar to UNet_Only, marginally better chi2, heavier model |
| 5 | SmoothFilter | Untested at low LR; current metrics unfair |
| 6 | Iterative | Complete failure |

---

## 5. Observations and Hypotheses

### The default-vs-WienerNet calibration tension
`default` has lower val_loss but worse calibration. The sigmoid filter in `default` is unconstrained
and may be finding a spectral shape that maximizes training-set likelihood while slightly
over-sharpening the posterior (explaining coverage < 91.3%). WienerNet's analytic filter anchors
the spectral shape to the true signal/noise ratio, which likely prevents this.

### Loss scale anomaly in Iterative
Train loss starts at ~5.5 vs ~12 for other networks. This is suspicious. The 2-channel architecture
processes residuals differently, which may cause the log-prob to be evaluated on a different scale
(e.g., if the UNet output is being interpreted differently by the loss function). Worth investigating
before re-running.

### SmoothFilter's scheduling failure
The ~20 spiky filter nodes parameterized in log-k space create a loss landscape with many local
minima, causing the validation loss to oscillate at LR=0.01. The current LR scheduler isn't
equipped to handle this — it never fires. This is a *scheduler mismatch*, not a fundamental
architectural problem.

### 1-point PDF bias
All networks show samples with a slightly shifted/biased 1-point PDF (negative skewness vs
positive truth skewness). This is likely related to the k=0 mode: the network estimators do not
explicitly enforce zero spatial mean, so z_MAP can carry a nonzero constant component. After
division by rescaling_factor ≈ 0.0099, this is amplified by ~100x. Adding mean and std as
explicit diagnostic metrics would quantify this bias across networks.

---

## 6. Suggestions for Next Steps

### High priority

1. **Extend `default` training to 100 epochs.** It was still improving at epoch 46 and hit the
   50-epoch ceiling (early stopping fired at epoch 49, just 3 epochs after the best). This is
   the cheapest experiment with the highest expected payoff.

2. **Re-run SmoothFilter with `--learning_rate 0.001` and `--lr_scheduler_patience 5`.** The
   network clearly has a high-LR instability problem. Lower initial LR would likely fix the
   oscillation and allow it to reach its true potential. This architecture is interesting
   (compact 20-node spectral parameterization) and deserves a fair test.

3. **Debug Iterative's loss scale** before re-running. Check whether the 2-channel UNet's output
   is accidentally contributing to the likelihood term twice, or if the loss normalization differs.
   Once fixed, re-run at `--learning_rate 0.001` and `--batch_size 8`.

### Medium priority

4. **WienerNet + learnable scale**: WienerNet has the best calibration but leaves performance on
   the table. Adding a learnable per-mode scale (like `self.scale` in `default`) on top of the
   Wiener filter might combine the calibration benefit of the analytic filter with extra
   expressive power, without sacrificing spectral coherence.

5. **Noise sweep on the top-2 networks** (`default` and `WienerNet`) — vary sigma_noise and
   compare how calibration degrades. The WienerNet's analytic signal-to-noise filter should be
   more robust at low SNR (high noise) than the learned sigmoid filter.

6. **Amortization test** on `default` and `WienerNet` using `--target_dir` with multiple held-out
   targets. The calibration metrics above are computed on a single target field; the amortization
   test will reveal whether WienerNet's calibration advantage holds across the distribution.

### Lower priority

7. **LearnableFilter with fewer parameters**: The N³=2,097,152 spectral logits are almost
   certainly overkill. Try a version with k-binned logits (~100 bins, as in SmoothFilter) as a
   middle ground between SmoothFilter's smoothness and LearnableFilter's flexibility.

8. **LH (Latin Hypercube) network**: Not yet benchmarked. If varying cosmologies are of interest,
   this is the natural next experiment.

9. **Add mean and std to 1-point diagnostics**: Track the spatial mean (should be ~0) and
   standard deviation of the IC field samples and MAP, to quantify amplitude bias and
   the k=0 mode issue across networks.
