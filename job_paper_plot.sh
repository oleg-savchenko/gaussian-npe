#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
# #SBATCH --gpus=1
#SBATCH --partition=rome      #gpu_a100         #
#SBATCH --time=00:10:00
#SBATCH --output=./job_outputs/%x-%j-%N_slurm.out
#SBATCH --error=./job_outputs/R-%x.%j.err

module load 2024
module load Python/3.12.3-GCCcore-13.3.0
export OMP_NUM_THREADS=18
export PYTHONNOUSERSITE=0

cd /home/osavchenko/gaussian_npe

MODEL_DIR=paper_test_runs/runs/260303_224547_sweep_networks/260303_224627_net_IsotropicD

# python paper_plots_scripts/fig3_1pt.py \
#     --model_dir $MODEL_DIR \
#     --num_samples 1000

SAMPLES_DIR=/gpfs/scratch1/shared/osavchenko/mnras_paper/samples/260303_224627_net_IsotropicD
PPC_DIR=$SAMPLES_DIR/sample_0000
PLOTS_DIR=paper_plots_scripts/260303_224627_net_IsotropicD

# python paper_plots_scripts/fig1_slices.py \
#     --model_dir $MODEL_DIR \
#     --samples_dir $SAMPLES_DIR \
#     --ppc_dir $PPC_DIR \
#     --num_samples 2

# python paper_plots_scripts/fig2_2pt.py \
#     --model_dir $MODEL_DIR \
#     --num_samples 100

# python paper_plots_scripts/fig7_ppc_2pt.py \
#     --samples_dir $SAMPLES_DIR \
#     --output_dir $PLOTS_DIR

# python paper_plots_scripts/fig2_combined_2pt.py \
#     --model_dir $MODEL_DIR \
#     --num_samples 100 \
#     --samples_dir $SAMPLES_DIR \
#     --output_dir $PLOTS_DIR

# python paper_plots_scripts/fig4_bispectrum.py \
#     --model_dir $MODEL_DIR \
#     --num_samples 200

# python paper_plots_scripts/fig4_combined_bispectrum.py \
#     --model_dir $MODEL_DIR \
#     --num_samples 100 \
#     --samples_dir $SAMPLES_DIR \
#     --output_dir $PLOTS_DIR

# python paper_plots_scripts/fig_calibration.py \
#     --model_dir $MODEL_DIR \
#     --num_samples 100 \
#     --output_dir $PLOTS_DIR

# python paper_plots_scripts/fig8_minkowski.py \
#     --model_dir $MODEL_DIR \
#     --samples_dir $SAMPLES_DIR \
#     --output_dir $PLOTS_DIR

# LH_MODEL_DIR=runs/260303_195331_LH_sigma_noise_1_train_only_Q_post_UNet_Only
# LH_PLOTS_DIR=paper_plots_scripts/260303_195331_LH_sigma_noise_1_train_only_Q_post_UNet_Only
# python paper_plots_scripts/fig_lh_2pt.py \
#     --model_dir $LH_MODEL_DIR \
#     --index 6 \
#     --num_samples 1000 \
#     --ylim_pk 2e-2 15 \
#     --output_dir $LH_PLOTS_DIR

# python paper_plots_scripts/fig_hmf.py \
#     --arrays_path $PLOTS_DIR/posterior_resimulation/posterior_predictive_hmf_arrays.npz \
#     --output_dir $PLOTS_DIR

# python paper_plots_scripts/fig5_Qdiag.py \
#     --model_dir $MODEL_DIR

# python paper_plots_scripts/fig6_sweep_Qdiag.py \
#     --sweep_dir      paper_test_runs/runs/260304_233941_sweep_noise \
#     --sweep_train_dir paper_test_runs/runs/260328_234020_sweep_train \
#     --output_dir $PLOTS_DIR

python paper_plots_scripts/fig_hmf.py \
    --arrays_path paper_plots_scripts/260303_224627_net_IsotropicD/true_ic_forward_apples_to_apples/posterior_predictive_hmf_apples_to_apples_true_ic_comparison_arrays.npz \
    --output_dir paper_plots_scripts/260303_224627_net_IsotropicD/true_ic_forward_apples_to_apples \
    --build_class_pk_fiducial \
    # --plot_comic_variance

# python paper_plots_scripts/fig_pk_individual_samples.py \
#     --model_dir paper_test_runs/runs/260303_224547_sweep_networks/260303_224627_net_IsotropicD \
#     --num_samples 10 \
#     --samples_dir /gpfs/scratch1/shared/osavchenko/mnras_paper/samples/260303_224627_net_IsotropicD \
#     --output_dir paper_plots_scripts/260303_224627_net_IsotropicD \
#     --num_ppc_samples 10
