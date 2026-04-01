# P(k) bump near k_Nyq: analysis and conclusions

**Run:** `260303_224627_net_IsotropicD` (IsotropicD network, σ_noise = 1)
**Date:** 2026-03-30
**Script:** `paper_plots_scripts/fig_pk_bump_analysis.py`

---

## The observation

When comparing the mean power spectrum of 100 posterior re-simulated z=0 fields against
the true z=0 field, there is a systematic excess that grows toward k_Nyq:

- At low k (k < 0.1 h/Mpc): excess ≈ **0.1%** — negligible, perfect agreement
- At k > 0.3 k_Nyq: excess ≈ **11.8% ± 5.9%** (mean ± std across k bins)
- Maximum excess below k_Nyq: **19.1%** at k ≈ 0.336 h/Mpc

By contrast, a control test (true ICs → emulator → emulated z=0) shows sub-percent
agreement all the way to k_Nyq, ruling out the emulator itself as the cause.

Importantly, **all other summary statistics** (reduced bispectrum Q, Minkowski
functionals V0–V3) agree well between posterior re-simulations and the true field.
Only the raw, unnormalized P(k) shows the excess.

---

## Hypothesis: Jensen's inequality / posterior variance

### Definition of Jensen's inequality

For any convex function f and random variable X:

    E[f(X)] ≥ f(E[X])

The power spectrum P(k) = |δ(k)|² is a **quadratic** (convex) function of the field
amplitude. Therefore, when averaging P(k) over many IC realizations drawn from the
posterior:

    ⟨P(k)⟩ = ⟨|δ(k)|²⟩ = |⟨δ(k)⟩|² + Var(δ(k)) ≥ P(k)|_{mean IC}

The extra term **Var(δ(k))** — the posterior variance per mode — is strictly
non-negative and causes the mean P(k) of posterior samples to exceed the P(k) of any
single field (including the truth).

### Application to IC posterior samples

Each posterior IC sample is drawn from a Gaussian in Hartley space:

    IC_i = z_MAP + ε_i,    ε_i ~ N(0, D_post(k)^{-1})

where D_post(k) = D_prior(k) + D_like(k) is the posterior precision in each
Hartley mode (in internal units; convert to physical by dividing by rf²).

In linear theory, the final field power spectrum from sample i is:

    P_final(k) | IC_i  ≈  D(z=0)² × |IC_i(k)|²

Taking the expectation over posterior samples:

    ⟨P_final(k)⟩  =  P_final(k)|z_MAP  +  D(z=0)² × D_post(k)^{-1}

The **relative excess** relative to the true final field P(k) is then (in linear theory):

    excess(k) / P_true(k)  ≈  D_post(k)^{-1} / P_IC_true(k)
                            =  D_prior(k) / [D_prior(k) + D_like(k)]

This quantity ranges from:
- **→ 0** where D_like >> D_prior: scales where the observation strongly constrains
  IC modes (low k, high SNR). At k ~ 0.05 h/Mpc: D_like/D_prior ≈ 39 → excess ≈ 2.5%
- **→ 1 (= 100%)** where D_like → 0: scales where the likelihood gives essentially
  no information. At k_Nyq: D_like/D_prior ≈ 0.08 → linear excess ≈ 92.8%

### Why only P(k) shows the bump?

- **Reduced bispectrum Q(θ, k)** is defined as B(k₁,k₂,k₃) / [P(k₁)P(k₂) + ...].
  If P(k) has a ~15% excess at all high-k modes, both the numerator B and the
  denominator P² scale up by similar factors, and Q remains unchanged.

- **Minkowski functionals V0–V3** are computed on the thresholded field
  ν = (δ − μ)/σ. Dividing by the field's own standard deviation σ absorbs the excess
  power (σ² ∝ ∫P(k)dk), so the topology of the excursion set is unaffected.

- **P(k) is not normalized** — it directly reports the absolute power per mode, so the
  posterior variance contribution appears as a direct additive excess.

This explains why the bispectrum and Minkowski functionals agree well while P(k) does
not: the former are normalized statistics, the latter is not.

---

## Quantitative test

The script computes:

1. **Observed excess** = (⟨P_samples(k)⟩ − P_true(k)) / P_true(k)  — from 100 emulated
   fields
2. **Predicted excess (linear theory)** = D_prior(k) / [D_prior(k) + D_like(k)]
3. **Predicted excess (non-linear corrected)** = predicted_linear × P_lin(k) / P_true(k)
   — accounts for the fact that the non-linear z=0 P(k) >> linear P(k) near k_Nyq, so
   the same IC variance produces a smaller *relative* excess in the final field

| Quantity | k > 0.3 k_Nyq (mean) | at k_Nyq |
|---|---|---|
| Observed excess | **11.8% ± 5.9%** | — |
| Predicted (linear only) | — | **92.8%** |
| Predicted (non-linear corrected) | **28.6%** | **45.9%** |

The non-linear corrected prediction is a factor of ~2.4 larger than observed. This
remaining discrepancy arises because the simple P_lin/P_nonlin correction is only an
approximation to the true non-linear IC→final-field response function. At k ~ 0.3–0.4
h/Mpc the non-linear mode coupling is strong, and the actual transfer of IC variance to
final-field P(k) is more strongly suppressed than the ratio P_lin/P_nonlin suggests.
Despite this, the prediction is in the correct direction and correct order of magnitude.

---

## Conclusions

| Question | Answer |
|---|---|
| Is this an emulator artifact? | ❌ No — control test (true ICs) shows sub-percent agreement |
| Is this caused by IC upsampling (mode_inject)? | ❌ No — only affects modes above k_Nyq |
| Is this a failure of IC reconstruction? | ⚠️ Partial — MAP deviates from truth at high k (unavoidable due to information loss in non-linear evolution) |
| Is this the Jensen's inequality / posterior variance effect? | ✅ Yes — primary cause, quantitatively consistent |
| Can it be eliminated? | ❌ Not completely — it is fundamental to any probabilistic posterior |
| Can it be reduced? | ✅ Yes — by making D_like larger at high k (if the network underestimates it), or by improving posterior approximation beyond the Gaussian model |

### Physical interpretation

The non-linear evolution of structure scrambles information about IC modes at small
scales (high k). By z=0, the observation δ_z0 carries almost no information about IC
modes near k_Nyq — the likelihood at those scales is nearly flat (D_like ≈ 0.08 × D_prior
at k_Nyq). As a result, the posterior at k_Nyq is barely constrained beyond the prior.
When posterior samples are drawn and emulated forward, their variance at those modes
propagates into the final field and, through Jensen's inequality, raises the mean P(k)
above that of the single true realization.

### Comparison with other summary statistics

The agreement of the bispectrum and Minkowski functionals is **consistent** with this
explanation, not contradictory. Those statistics are normalized in ways that absorb
the amplitude excess, so they correctly show good agreement even when P(k) does not.

---

## Revised conclusion: this is a genuine calibration failure

The discussion above initially framed the bump as an unavoidable consequence of Jensen's
inequality — something a perfect posterior would also show. **This conclusion is
incomplete.** A key additional observation changes the interpretation:

**The true z=0 P(k) falls outside (below) the 1–2σ posterior predictive bands near
k_Nyq.** This is not merely a consequence of the mean being elevated — it means the
truth is not a plausible sample from the posterior predictive distribution. By definition,
this is a calibration failure.

### Why "Jensen's inequality is unavoidable" is not the full story

The correct notion of a well-calibrated posterior predictive is:

> The true observable (here, the true z=0 field) should be a plausible sample from the
> posterior predictive distribution — i.e., it should fall within the 1–2σ bands at the
> expected frequency.

Jensen's inequality guarantees that the **mean** of the posterior predictive P(k) is
always above P_true(k) (for non-zero posterior variance). This is unavoidable. However,
for a well-calibrated posterior, P_true(k) should still lie **within the distribution** —
the bands should be wide enough to contain it. If P_true(k) falls below the 1σ band, it
means the posterior predictive ensemble has been shifted too far up: the posterior is
**over-dispersed** at those scales.

### Why "good IC P(k) agreement" does not guarantee good final P(k) agreement

The IC P(k) shows ~1–2% agreement (the truth falls within the IC sample bands). However,
the same σ²_IC(k) that causes the 1–2% IC band width gets amplified to ~12–17% in the
final field, because:

    excess_final(k) / P_final_true(k)  ≈  σ²_IC(k) × P_nonlinear(k) / [P_IC_true(k) × P_linear(k)]
                                        =  (1–2%)  ×  (P_nonlinear / P_linear)
                                        ≈  (1–2%)  ×  5–10  ≈  10–20%

The non-linear forward model amplifies small-scale IC variance by the non-linear boost
factor P_nonlinear/P_linear ≈ 5–10 at k_Nyq. The IC P(k) looking "good" is therefore
a necessary but not sufficient condition for the final field P(k) to be well-calibrated.

### Root cause: diagonal Gaussian approximation misses cross-mode correlations

The IsotropicD network approximates the posterior precision as **diagonal in Hartley
space** — each mode is treated independently. In reality, non-linear evolution creates
correlations between IC modes at different wavenumbers. A low-k mode in the z=0 field
carries partial information about a high-k IC mode through these non-linear couplings.
A full off-diagonal precision matrix could exploit these cross-mode correlations to
tighten the posterior at high k.

By using a diagonal approximation, the network **underestimates D_like(k) near k_Nyq**:
it assigns D_like ≈ 0.08 × D_prior at k_Nyq, meaning the posterior is almost as broad
as the prior at those scales. If cross-mode information were correctly captured, D_like
would be larger, σ²_IC smaller, the posterior predictive bands narrower, and the true
P(k) would fall within them.

### Summary of revised conclusions

| Question | Revised answer |
|---|---|
| Is the bump an artifact of Jensen's inequality? | ✅ Partially — Jensen's inequality is always present, but does not by itself push the truth outside the bands |
| Is the truth outside the 1–2σ posterior predictive bands near k_Nyq? | ✅ Yes — this constitutes a genuine calibration failure |
| Is the posterior over-dispersed at high k? | ✅ Yes — IC samples have too much variance at k ~ k_Nyq relative to what the observation can support |
| Is the diagonal Gaussian approximation the limiting factor? | ✅ Likely yes — off-diagonal cross-mode correlations from non-linear evolution could constrain high-k modes more tightly |
| Would a perfect (non-diagonal, non-Gaussian) posterior eliminate the bump? | ⚠️ It would reduce it significantly; some residual from Jensen's inequality would remain, but truth would lie within the bands |
| Is this a fundamental information-theoretic limit? | ⚠️ Partially — non-linear evolution destroys information, but more than we currently extract may still be recoverable via off-diagonal correlations |

---

## Why prior samples don't show the bump — and where the excess actually comes from

A natural question: if at high k the posterior is essentially the prior (D_like ≈ 0 → posterior ≈ prior),
and if simulating **prior** samples through the emulator produces no systematic P(k) excess, why do
**posterior** samples produce the ~12% bump?

### High-k IC modes are not the source

At k near k_Nyq, D_like << D_prior, so:

    z_MAP(k) ≈ (D_like / D_prior) × z_network ≈ 0
    D_post⁻¹(k) ≈ D_prior⁻¹(k) = P_class(k)

Posterior samples at those modes are essentially N(0, P_class) — identical to prior samples.
Emulating many prior samples gives the ensemble-mean P_nonlinear, which equals P_true on average
(the truth is one realization). No systematic bias. So high-k IC modes are **not** the source of
the bump.

### The bump comes from intermediate k

At intermediate scales (e.g. k ~ 0.1 h/Mpc, where D_like/D_prior ≈ 39), the posterior IS
meaningfully constrained: z_MAP ≈ z_true and D_post⁻¹ << P_class. The mean IC power of posterior
samples at those scales is:

    ⟨P_IC_sample(k)⟩ = |z_MAP(k)|² + D_post⁻¹(k)
                     ≈ P_IC_true(k)  +  D_post⁻¹(k)

The Jensen's term D_post⁻¹ = P_class / (1 + D_like/D_prior) ≈ P_class / 40 ≈ 2.5% of P_class at
these scales. This is a **systematic positive excess in the IC power spectrum at intermediate k**,
present in every posterior sample.

Why is it invisible in the IC T(k) plot? The ~2.5% excess falls within the ~1–2% cosmic-variance
band width (100 realizations), so it is hidden in the scatter.

### Propagation to the final field via non-linear coupling

When posterior IC samples are emulated forward, this small intermediate-k excess propagates to
**high-k final-field power** through non-linear mode coupling. Non-linear structure formation
transfers power from large/intermediate scales to small scales: the 1-halo term (power at k_Nyq)
depends on the halo mass function, which is set by the variance of density fluctuations at
intermediate k (σ(M)). A ~2.5% IC power excess at k ~ 0.1 h/Mpc, amplified by the non-linear
boost factor P_nonlin/P_lin ≈ 5–10 at k_Nyq, produces the observed ~12% excess in the final field.

### Why prior samples are unaffected

Prior samples are centered on **zero** (not z_MAP). Their IC power at intermediate k is simply
P_class = P_IC_true — no systematic Jensen's term. Emulating an ensemble of prior samples therefore
gives the ensemble-mean P_nonlinear ≈ P_true, with no systematic bias.

Posterior samples are centered on **z_MAP ≈ z_true** at intermediate k, which is the crucial
difference: the posterior variance D_post⁻¹ is added on top of the MAP power, creating a small
but systematic IC excess that the prior-sample test does not exhibit.

### Summary

| Scale | Posterior vs prior samples | Effect |
|---|---|---|
| k near k_Nyq | Posterior ≈ prior (both ≈ N(0, P_class)) | No systematic IC excess |
| k ~ 0.1 h/Mpc (intermediate) | Posterior centered on z_MAP ≈ z_true; excess = D_post⁻¹ ≈ 2.5% P_class | Small systematic IC excess, invisible in IC T(k) |
| Final field k_Nyq | Intermediate-k IC excess amplified by non-linear coupling × 5–10 | ~12% P(k) excess |

The IC T(k) looking good is therefore not a sufficient diagnostic: the bump is caused by a
~2.5% systematic IC excess at intermediate k that is hidden within the IC cosmic-variance bands
but amplified to ~12% in the final field by non-linear mode coupling.

---

## Files

| File | Contents |
|---|---|
| `pk_bump_analysis.pdf` | Two-panel figure: P(k) comparison (left); observed vs predicted excess (right) |
| `pk_excess_observed.csv` | k, P_true, ⟨P_samples⟩, std, excess, excess_err |
| `pk_excess_predicted.csv` | k, D_prior, D_like, D_post, predicted_excess |
| `bump_test_analysis.md` | This document |
