"""
Diagnostic: P(k) bump near k_Nyq in posterior predictive re-simulations.

Tests the hypothesis that the ~17% excess arises from Jensen's inequality /
posterior variance: <P_final(k)|samples> = P_final(k)|z_MAP + D(z=0)² × D_post(k)⁻¹.

Produces two panels:
  - Left:  Observed relative excess  (mean_P_samples - P_true) / P_true  vs k
  - Right: Predicted excess from posterior variance: D_prior / (D_prior + D_like)

Both quantities should agree if the hypothesis is correct.

Usage:
    python paper_plots_scripts/fig_pk_bump_analysis.py \\
        --model_dir paper_test_runs/runs/260303_224547_sweep_networks/260303_224627_net_IsotropicD \\
        --samples_dir /gpfs/scratch1/shared/osavchenko/mnras_paper/samples/260303_224627_net_IsotropicD \\
        --output_dir paper_plots_scripts/260303_224627_net_IsotropicD/pk_bump_analysis
"""

import os
import sys
import json
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
import Pk_library as PKL

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gaussian_npe import utils
from _common import NETWORK_CLASSES, BOX_PARAMS, COSMO_PARAMS, Z_IC


# ── Helpers ───────────────────────────────────────────────────────────────────

def radial_bin_mean(k_flat, values, n_bins=60):
    k_min = k_flat[k_flat > 0].min()
    k_max = k_flat.max()
    edges = np.logspace(np.log10(k_min), np.log10(k_max), n_bins + 1)
    idx   = np.digitize(k_flat, edges) - 1
    k_cen, v_cen, counts = [], [], []
    for i in range(n_bins):
        sel = idx == i
        if sel.sum() > 0:
            k_cen.append(k_flat[sel].mean())
            v_cen.append(values[sel].mean())
            counts.append(sel.sum())
    return np.array(k_cen), np.array(v_cen), np.array(counts)


def compute_pk_pylians(field, box_size, axis=0, MAS=None):
    Pk = PKL.Pk(field.astype('f'), box_size, axis=axis, MAS=MAS, threads=4)
    return Pk.k3D, Pk.Pk[:, 0]


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--model_dir',   type=str,
                   default='paper_test_runs/runs/260303_224547_sweep_networks/260303_224627_net_IsotropicD')
    p.add_argument('--samples_dir', type=str,
                   default='/gpfs/scratch1/shared/osavchenko/mnras_paper/samples/260303_224627_net_IsotropicD')
    p.add_argument('--target_path', type=str, default='Quijote_target.pt')
    p.add_argument('--output_dir',  type=str,
                   default='paper_plots_scripts/260303_224627_net_IsotropicD/pk_bump_analysis')
    p.add_argument('--num_samples', type=int, default=None,
                   help='Max number of sample dirs to use (default: all found).')
    p.add_argument('--no_latex', dest='use_latex', action='store_false', default=True)
    return p.parse_args()


def main():
    args = parse_args()
    utils.configure_matplotlib_style(use_latex=args.use_latex)
    os.makedirs(args.output_dir, exist_ok=True)

    device     = 'cpu'
    box_size   = BOX_PARAMS['box_size']
    model_dir  = os.path.abspath(args.model_dir)

    # ── Load model and get D_prior, D_like ────────────────────────────────────
    with open(os.path.join(model_dir, 'config.json')) as f:
        cfg = json.load(f)

    box = utils.Power_Spectrum_Sampler(BOX_PARAMS, device=device)
    Dz_approx = (utils.growth_D_approx(COSMO_PARAMS, Z_IC)
                 / utils.growth_D_approx(COSMO_PARAMS, 0))
    rf = cfg.get('rescaling_factor', Dz_approx)

    prior = box.get_prior_Q_factors(
        lambda k: torch.tensor(
            utils.get_pk_class(COSMO_PARAMS, 0, k.detach().cpu().numpy()),
            device=device,
        )
    )

    network_name = cfg.get('network', 'default')
    NetworkClass = NETWORK_CLASSES[network_name]
    net_kwargs   = dict(sigma_noise=cfg.get('sigma_noise', 0.0), rescaling_factor=rf)
    if network_name not in ('UNet_Only', 'WienerNet', 'Iterative', 'CustomUNet',
                             'IsotropicD', 'WienerIsotropicD', 'Poisson', 'LH'):
        net_kwargs['k_cut'] = cfg.get('k_cut', 0.03)
        net_kwargs['w_cut'] = cfg.get('w_cut', 0.003)

    network = NetworkClass(box, prior, **net_kwargs).float().to(device)
    state   = torch.load(os.path.join(model_dir, 'model.pt'),
                         map_location=device, weights_only=False)
    network.load_state_dict(state)
    network.eval()

    with torch.no_grad():
        D_like_h  = network.Q_like.D.detach().cpu().numpy()   # internal units
        D_prior_h = network.Q_prior.D.detach().cpu().numpy()  # internal units
        D_post_h  = D_like_h + D_prior_h

    # Convert to physical units (divide by rf²)
    D_like_phys  = D_like_h  / rf**2
    D_prior_phys = D_prior_h / rf**2
    D_post_phys  = D_post_h  / rf**2

    # Radially bin D quantities using box.k
    k_3d = box.k.cpu().numpy().flatten()
    mask = k_3d > 0

    k_b, d_like_b,  _ = radial_bin_mean(k_3d[mask], D_like_phys[mask])
    _,   d_prior_b, _ = radial_bin_mean(k_3d[mask], D_prior_phys[mask])
    _,   d_post_b,  _ = radial_bin_mean(k_3d[mask], D_post_phys[mask])

    # Predicted relative excess (linear theory): D_prior / (D_prior + D_like)
    # In linear theory: excess(k)/P_true(k) = σ²_IC_phys / P_IC_true ≈ D_prior/(D_prior+D_like)
    # In non-linear regime the actual P_final >> linear, so multiply by P_lin/P_nonlin.
    # We apply this correction after loading the true z=0 field and computing CLASS P(k).
    predicted_excess_linear = d_prior_b / d_post_b

    k_Nq = box.k_Nq
    print(f'k_Nyq = {k_Nq:.4f} h/Mpc')
    print(f'Linear predicted excess at k_Nyq: {predicted_excess_linear[k_b < k_Nq][-1]*100:.1f}%')

    # CLASS linear P(k) at z=0 for non-linear correction
    k_class = np.logspace(np.log10(1e-3), np.log10(k_Nq * 1.1), 300)
    pk_class_lin = utils.get_pk_class(COSMO_PARAMS, 0, k_class, non_lin=False)

    # ── Load true z=0 field and compute its P(k) ─────────────────────────────
    print('Loading true z=0 field...')
    target = torch.load(os.path.abspath(args.target_path),
                        map_location='cpu', weights_only=False)
    delta_z0_arr = target['delta_z0']
    delta_z0 = (delta_z0_arr.numpy() if hasattr(delta_z0_arr, 'numpy') else np.array(delta_z0_arr)).astype('f')
    k_true, pk_true = compute_pk_pylians(delta_z0, box_size)
    print(f'True z=0 P(k): {len(k_true)} bins, k=[{k_true[0]:.4f}, {k_true[-1]:.4f}]')

    # Non-linear correction: P_lin(k) / P_true(k) evaluated on the k_b grid
    pk_class_lin_at_kb = np.interp(k_b, k_class, pk_class_lin)
    pk_true_at_kb      = np.interp(k_b, k_true,  pk_true)
    nonlin_correction  = pk_class_lin_at_kb / pk_true_at_kb  # < 1 at high k
    predicted_excess   = predicted_excess_linear * nonlin_correction
    print(f'Non-linear corrected predicted excess at k_Nyq: {predicted_excess[k_b < k_Nq][-1]*100:.1f}%')

    # ── Load posterior emulated fields and compute mean P(k) ──────────────────
    samples_dir = os.path.abspath(args.samples_dir)
    emu_dirs = sorted([
        os.path.join(samples_dir, dn)
        for dn in os.listdir(samples_dir)
        if dn.startswith('sample_')
        and os.path.isfile(os.path.join(samples_dir, dn, 'emu_delta_z0.npy'))
    ])
    if args.num_samples is not None:
        emu_dirs = emu_dirs[:args.num_samples]
    print(f'Loading {len(emu_dirs)} emulated fields...')

    sample_pks = []
    for i, d in enumerate(emu_dirs):
        field = np.load(os.path.join(d, 'emu_delta_z0.npy')).astype('f')
        _, pk_i = compute_pk_pylians(field, box_size)
        sample_pks.append(pk_i)
        if (i + 1) % 10 == 0:
            print(f'  {i+1}/{len(emu_dirs)}')

    sample_pks = np.array(sample_pks)
    pk_mean    = sample_pks.mean(axis=0)
    pk_std     = sample_pks.std(axis=0)

    # Relative excess
    obs_excess      = (pk_mean - pk_true) / pk_true
    obs_excess_err  = pk_std / (pk_true * np.sqrt(len(emu_dirs)))

    # ── Save CSV ───────────────────────────────────────────────────────────────
    np.savetxt(
        os.path.join(args.output_dir, 'pk_excess_observed.csv'),
        np.column_stack([k_true, pk_true, pk_mean, pk_std, obs_excess, obs_excess_err]),
        header='k  pk_true  pk_mean_samples  pk_std_samples  excess  excess_err',
        delimiter=',',
    )
    np.savetxt(
        os.path.join(args.output_dir, 'pk_excess_predicted.csv'),
        np.column_stack([k_b, d_prior_b, d_like_b, d_post_b, predicted_excess]),
        header='k  D_prior  D_like  D_post  predicted_excess',
        delimiter=',',
    )

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plt.rcParams['figure.facecolor'] = 'white'

    color_obs  = 'mediumpurple'
    color_pred = 'forestgreen'

    # Panel 1: P(k) comparison
    ax = axes[0]
    ax.plot(k_true, pk_true, color='k', lw=1.5, label='True z=0')
    ax.plot(k_class, pk_class_lin, color='k', lw=1, ls=':', alpha=0.5, label='CLASS linear')
    ax.plot(k_true, pk_mean, color=color_obs, lw=1.5, label='Mean posterior samples')
    ax.fill_between(k_true, pk_mean - pk_std, pk_mean + pk_std,
                    color=color_obs, alpha=0.3)
    ax.axvline(k_Nq, color='r', ls='--', lw=1, label=r'$k_{\rm Nyq}$')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$k~[h\,{\rm Mpc}^{-1}]$', fontsize=14)
    ax.set_ylabel(r'$P(k)~[({\rm Mpc}/h)^3]$', fontsize=14)
    ax.set_title('Final field P(k)', fontsize=14)
    ax.legend(fontsize=11); ax.grid(alpha=0.15)

    # Panel 2: Observed vs predicted relative excess
    ax = axes[1]
    mask_plot = k_true < k_Nq * 1.05
    ax.plot(k_true[mask_plot], obs_excess[mask_plot],
            color=color_obs, lw=2, label='Observed: $(\\langle P \\rangle - P_{\\rm true}) / P_{\\rm true}$')
    ax.fill_between(k_true[mask_plot],
                    (obs_excess - obs_excess_err)[mask_plot],
                    (obs_excess + obs_excess_err)[mask_plot],
                    color=color_obs, alpha=0.25)
    mask_pred = k_b < k_Nq * 1.05
    ax.plot(k_b[mask_pred], predicted_excess[mask_pred],
            color=color_pred, lw=2, ls='--',
            label=r'Predicted (non-linear corr.): $\frac{D_{\rm prior}}{D_{\rm prior}+D_{\rm like}} \cdot \frac{P_{\rm lin}}{P_{\rm true}}$')
    ax.plot(k_b[mask_pred], predicted_excess_linear[mask_pred],
            color='orange', lw=1.5, ls=':',
            label=r'Predicted (linear only): $D_{\rm prior}/(D_{\rm prior}+D_{\rm like})$')
    ax.axvline(k_Nq, color='r', ls='--', lw=1, label=r'$k_{\rm Nyq}$')
    ax.axhline(0, color='k', ls=':', lw=0.8)
    ax.set_xscale('log')
    ax.set_xlabel(r'$k~[h\,{\rm Mpc}^{-1}]$', fontsize=14)
    ax.set_ylabel('Relative excess', fontsize=14)
    ax.set_title('Observed vs predicted P(k) excess', fontsize=14)
    ax.legend(fontsize=11); ax.grid(alpha=0.15)
    ax.set_ylim(-0.05, max(obs_excess[mask_plot].max(), predicted_excess[mask_pred].max()) * 1.2)

    fig.tight_layout()
    out = os.path.join(args.output_dir, 'pk_bump_analysis.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved {out}')

    # ── Print summary ─────────────────────────────────────────────────────────
    near_nyq = (k_true > 0.3 * k_Nq) & (k_true < k_Nq)
    print(f'\n=== Summary ===')
    print(f'Observed excess at k > 0.3 k_Nyq: {obs_excess[near_nyq].mean()*100:.1f}% +/- {obs_excess[near_nyq].std()*100:.1f}%')
    near_nyq_pred = (k_b > 0.3 * k_Nq) & (k_b < k_Nq)
    print(f'Predicted excess at k > 0.3 k_Nyq: {predicted_excess[near_nyq_pred].mean()*100:.1f}%')
    print(f'Hypothesis: {"SUPPORTED" if abs(obs_excess[near_nyq].mean() - predicted_excess[near_nyq_pred].mean()) < 0.10 else "PARTIAL / DISCREPANT"}')


if __name__ == '__main__':
    main()
