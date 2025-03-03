import numpy as np
import torch
import swyft
from gaussian_npe import utils, gaussian_npe

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers

print(f'torch.cuda.is_available() = {torch.cuda.is_available()}')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Parameters of the Quijote simulations box
box_parameters = {
        'box_size': 1000.,       #Mpc/h
        'grid_res': 128,         #resolution
        'h': 0.6711,
        }
box = utils.Power_Spectrum_Sampler(box_parameters, device = device)

# Defining the Nyquist and fundamental frequencies of the box
print(f'k_Nq = {box.k_Nq} h/Mpc')
print(f'k_F = {box.k_F} h/Mpc')

# Fiducial Quijote cosmological parameters for CLASS initialization, taken from Planck2018
cosmo_params = {
        'h': 0.6711,
        'Omega_b': 0.049,
        'Omega_cdm': 0.2685,
        # 'A_s': 2.1413e-09,
        'n_s': 0.9624,
        'non linear': 'halofit',
        'sigma8': 0.834,
    }

# Approximate analytical formula for the growth factor from Eisenstein-Hu (see formula A4 in arXiv:9709112)
z_Quijote = 127
Dz127_approx = utils.growth_D_approx(cosmo_params, z_Quijote)/utils.growth_D_approx(cosmo_params, 0)
print(f'Dz127_approx = {Dz127_approx}')

# Prior for the Gaussian NPE
prior = box.get_prior_Q_factors(lambda k: torch.tensor(utils.get_pk_class(cosmo_params, 0, np.array(k)), device = device))

# Read the ZarrStore with Quijote simulations
store = swyft.ZarrStore("./os240713-Swyft_Quijote_128_1Gpc")
print(f'Number of simulations in the training/validation set = {len(store)}.')

lr_monitor = LearningRateMonitor(logging_interval='step')
early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta = 0., patience=3, verbose=False, mode='min')
checkpoint_callback = ModelCheckpoint(dirpath='./logs/', filename='final_{epoch}_{val_loss:.2f}_{train_loss:.2f}', mode='min', save_last=True)
logger = pl_loggers.TensorBoardLogger(save_dir='./logs/', name='final_logs', version=None)

trainer = swyft.SwyftTrainer(accelerator = 'cuda', precision = 32, logger = logger, max_epochs = 30, callbacks=[lr_monitor, early_stopping_callback, checkpoint_callback])
dm = swyft.SwyftDataModule(store, val_fraction = 0.2, num_workers = 2, batch_size = 8)
network = gaussian_npe.Gaussian_NPE_Network(box, prior, rescaling_factor = Dz127_approx, k_cut = 0.03, w_cut = 0.001)

# Training for 30 epochs takes approximately 1.5 hours
network.float().cuda()
trainer.fit(network, dm)
