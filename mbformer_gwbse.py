"""
This script is used to show how to use the MBFormer models for GW-BSE training and inference.
It includes the following steps:
1. Data Preprocessing
2. VAE Training
3. GW Training
4. BSE Training
"""

results_dir = './results'

# %%
# Part 1 --------------------------------------------------------------------------------------------------
# Data Preprocessing (wavefunction, GW, BSE datasets)

print("\n\nPart 1: Data Preprocessing (wavefunction, GW, BSE datasets)\n\n")

from deep_gwbse.from_model.data import ToyDataSet

# wfdata: vae training dataset
wfdata = ToyDataSet.get_wfn_dataset(read=False, mbformer_data_dir=f'{results_dir}/dataset/', toy_data_path='./examples/flows')
wfdata = ToyDataSet.get_wfn_dataset(read=True, mbformer_data_dir=f'{results_dir}/dataset/')
# gwdata: gw training dataset
gwdata = ToyDataSet.get_gw_dataset(read=False, mbformer_data_dir=f'{results_dir}/dataset/', toy_data_path='./examples/flows')
gwdata = ToyDataSet.get_gw_dataset(read=True, mbformer_data_dir=f'{results_dir}/dataset/')
# bsedata: bse training dataset
bsedata = ToyDataSet.get_bse_dataset(read=False, mbformer_data_dir=f'{results_dir}/dataset/', toy_data_path='./examples/flows')
bsedata = ToyDataSet.get_bse_dataset(read=True, mbformer_data_dir=f'{results_dir}/dataset/')

print("Data preprocessing completed!")
print(f"WFN dataset: {len(wfdata) if hasattr(wfdata, '__len__') else 'created'}")
print(f"GW dataset: {len(gwdata) if hasattr(gwdata, '__len__') else 'created'}")
print(f"BSE dataset: {len(bsedata) if hasattr(bsedata, '__len__') else 'created'}")


# %%
# Part 2 --------------------------------------------------------------------------------------------------
# VAE Training

print("\n\nPart 2: VAE Training\n\n")

from deep_gwbse.from_model.data import ManyBodyData
from deep_gwbse.from_model.e2vaetrainer import WFNVAETrainer, wfn_collate_fn
from deep_gwbse.from_model.e2vae import EquivariantVAE
from deep_gwbse.from_model.trainer import Trainer
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import torch

# Basic configuration
config_model_path = f"{results_dir}/vae_e2_wfn.save" # (Optional) load from existing model
num_epoches = 20
beta = 0.2
train_val_split = 0.8

# Load dataset
wfdata = ManyBodyData.from_existing_dataset(f'{results_dir}/dataset/dataset_WFN.h5')
wfdata_train = wfdata[:int(len(wfdata)*train_val_split)]
wfdata_val = wfdata[int(len(wfdata)*train_val_split):]
dataloader_train = DataLoader(wfdata_train, batch_size=1, collate_fn=wfn_collate_fn)
dataloader_val = DataLoader(wfdata_val, batch_size=1, collate_fn=wfn_collate_fn)

# Initialize or load model
if os.path.exists(config_model_path):
    print("Loading model from", config_model_path)
    vae = Trainer.configure_model(EquivariantVAE, config_model_path)
else:
    vae = EquivariantVAE(input_channels=wfdata.info.cell_slab_truncation,
                            hidden_cnn_channels=[60,60,48,48,48],
                            hidden_pooling=[-1,0.66,-1,-1,0.66],
                            kernel_size=[7,5,5,3,3])

optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

# Train model
vae_trainer = WFNVAETrainer(vae, optimizer, beta=beta, model_name="vae_e2_wfn", save_path=config_model_path)
vae_trainer.load_model(load_best=True)
vae_trainer.train(num_epoches, dataloader_train, dataloader_val, continued=True)

# Evaluate and visualize
vae_trainer.load_model(load_best=True)
x, x_recon = vae_trainer.evaluate(dataloader_val)
sample_idxs = range(10)
i_channel = 3
fig, axes = plt.subplots(2, len(sample_idxs), figsize=(len(sample_idxs), 3))
for i_fig, i_sample in enumerate(sample_idxs):
    axes[0, i_fig].imshow(x[i_sample,  :, :].sum(axis=0))
    axes[0, i_fig].axis("off")
    axes[1, i_fig].imshow(x_recon[i_sample,  :, :].sum(axis=0))
    axes[1, i_fig].axis("off")

axes[0, 0].set_ylabel("Original")
axes[1, 0].set_ylabel("Reconstructed")
plt.savefig(f"{results_dir}/vae_e2_wfn_reconstructed.png")
print("Training completed! Model saved and reconstruction plot created.")


# %%
# Part 3 --------------------------------------------------------------------------------------------------
# GW Training
#!/usr/bin/env python
from deep_gwbse.from_model.data import ManyBodyData
from deep_gwbse.from_model.gwtrainer import GWTransformerTrainer, GWPredictTask, gw_collate_fn
from deep_gwbse.from_model.transformer import MBformer
from deep_gwbse.from_model.trainer import Trainer
from deep_gwbse.from_model.e2vae import EquivariantVAE
from deep_gwbse.from_model.wfnembedder import ManyBodyData_WFN_Embedder_pretrained, E2VAEEmbedder
import torch
import os
from functools import partial
from torch.utils.data import DataLoader

# Basic configuration
d_model = 48 # (Optional) load from existing model
num_epoches = 10
train_val_split = 0.8
dataset_dir = f'{results_dir}/dataset'
dataset_fname = 'dataset_GW.h5'
dataset_latent_fname = dataset_fname.split('.')[0] + '_latent.h5'

# Create or load latent dataset
if not os.path.exists(os.path.join(dataset_dir, dataset_latent_fname)):
    print(f"Latent dataset not found, creating new one")
    # create latent_dataset
    gwdata = ManyBodyData.from_existing_dataset(os.path.join(dataset_dir, dataset_fname))
    # Load pretrained VAE
    vae = Trainer.configure_model(EquivariantVAE, f"{results_dir}/vae_e2_wfn.save")
    eb = ManyBodyData_WFN_Embedder_pretrained(48, E2VAEEmbedder, model=vae, 
                                                model_name='vae_e2_wfn', 
                                                save_path=f'{results_dir}/vae_e2_wfn.save')
    gwdata = eb.create_latent_for_ManyBodyData_h5(gwdata, dataset_dir=dataset_dir, 
                                                    dataset_fname=dataset_latent_fname)
else:
    print(f"Latent dataset found, using {os.path.join(dataset_dir, dataset_latent_fname)}")
    # directly read the latent_dataset
    gwdata = ManyBodyData.from_existing_dataset(os.path.join(dataset_dir, dataset_latent_fname))

# Split data
gwdata_train = gwdata[:int(len(gwdata)*train_val_split)]
gwdata_val = gwdata[int(len(gwdata)*train_val_split):]
dataloader_train = DataLoader(gwdata_train, batch_size=35, collate_fn=gw_collate_fn, shuffle=True)
dataloader_val = DataLoader(gwdata_val, batch_size=35, collate_fn=gw_collate_fn, shuffle=True)

# Initialize model
enc2 = MBformer(d_input_src=d_model, d_input_tgt=d_model, d_model=d_model, 
                activation="gelu", num_encoder_layers=2, num_decoder_layers=2)
optimizer = torch.optim.Adam(enc2.parameters(), lr=1e-5)
lr_scheduler = None
loss = torch.nn.MSELoss()
additional_metrics = partial(torch.nn.functional.l1_loss, reduction='mean')

# Train model
gw_trainer_sigma = GWTransformerTrainer(enc2, loss, optimizer,  
                                            model_name="gw_transformer_sigma", 
                                            task=GWPredictTask.G0W0_energy,
                                            additional_metrics=additional_metrics, 
                                            scheduler=lr_scheduler,
                                            save_path=f"{results_dir}/gw_transformer_sigma.save")    
gw_trainer_sigma.load_model(True)
gw_trainer_sigma.train(num_epoches, dataloader_train, dataloader_val, continued=False)

# Evaluate
dataloader_val2 = DataLoader(gwdata_val, batch_size=1, collate_fn=gw_collate_fn, shuffle=False)
print("Validation evaluation:")
gw_trainer_sigma.evaluate_dataset(dataloader_val2, file_name_pred=f"{results_dir}/data_val_pred_gw.dat", 
                                    file_name_original=f"{results_dir}/data_val_orginal_gw.dat")
print("Training evaluation:")
gw_trainer_sigma.evaluate_dataset(dataloader_train, file_name_pred=f"{results_dir}/data_train_pred_gw.dat", 
                                    file_name_original=f"{results_dir}/data_train_orginal_gw.dat")

# parity plot for gw data
import matplotlib.pyplot as plt
import numpy as np

train_gwdata = np.loadtxt(f"{results_dir}/data_train_orginal_gw.dat")
val_gwdata = np.loadtxt(f"{results_dir}/data_val_orginal_gw.dat")
train_gwdata_pred = np.loadtxt(f"{results_dir}/data_train_pred_gw.dat")
val_gwdata_pred = np.loadtxt(f"{results_dir}/data_val_pred_gw.dat")

plt.figure(figsize=(10, 10))
plt.scatter(train_gwdata, train_gwdata_pred, color='blue', label='Training')
plt.scatter(val_gwdata, val_gwdata_pred, color='red', label='Validation')
plt.legend()
plt.savefig(f"{results_dir}/parity_plot_gw.png")   


print('GW training completed!')



# %%
# Part 4 --------------------------------------------------------------------------------------------------
# BSE Training

#!/usr/bin/env python
from deep_gwbse.from_model.data import ManyBodyData
from deep_gwbse.from_model.bsetrainer import bse_training_manager_for_deepgwbse_paper, BSEPredictTask
from deep_gwbse.from_model.trainer import Trainer
from deep_gwbse.from_model.e2vae import EquivariantVAE
from deep_gwbse.from_model.wfnembedder import ManyBodyData_WFN_Embedder_pretrained, E2VAEEmbedder
from deep_gwbse.from_model.basisassembly import ElectronHoleBasisAssembly_Concatenate
import torch
import shutil


# Load pretrained VAE
vae_path = f"{results_dir}/vae_e2_wfn.save"
vae = Trainer.configure_model(EquivariantVAE, vae_path)
eb = ManyBodyData_WFN_Embedder_pretrained(48, E2VAEEmbedder, model=vae, 
                                            model_name='vae_e2_wfn', 
                                            save_path=vae_path)

dataset_kwargs = {
    'dataset_dir': f'{results_dir}/dataset',
    'dataset_fname': 'dataset_BSE.h5',
    'dataset_latent_fname_suffix': '_latent_vae.h5',
}

# IMPORTANT:
# BSE Training has three tasks: eigenvalues, eigenvectors, and dipole
# Please check `bse_training_manager_for_deepgwbse_paper` in `deep_gwbse.from_model.bsetrainer` for more details and an example of how to use it!!!

"""Training for BSE eigenvalues with pretrained VAE"""
print("Training BSE eigenvalues model...")
bse = bse_training_manager_for_deepgwbse_paper(d_model=48, 
                                                model_name='bse_transformer_eval_vae_test', 
                                                task=BSEPredictTask.eigenvalues,
                                                BasisAssembly=ElectronHoleBasisAssembly_Concatenate, 
                                                use_Enc_Dec=True,
                                                optimizer=torch.optim.Adam, 
                                                loss=torch.nn.MSELoss, 
                                                lr=1e-3,
                                                save_path=f"{results_dir}/bse_transformer_eval_vae_test.save")

bse.load_data(eb=eb, data_slice=None, train_val_split=0.5, **dataset_kwargs)
bse.train(5, continued=False)

_, dataloader_val = bse.load_data(eb=eb, data_slice=slice(-20,-1), train_val_split=0.5, **dataset_kwargs)
bse.evaluate_dataset(dataloader_val, save_path=f"{results_dir}")

# shutil.rmtree(f"{results_dir}/bse_transformer_eval_vae_test.save")

"""Training for BSE eigenvectors with pretrained VAE"""
print("Training BSE eigenvectors model...")
bse = bse_training_manager_for_deepgwbse_paper(d_model=48, 
                                                model_name='bse_transformer_evec_vae_test', 
                                                task=BSEPredictTask.eigenvectors,
                                                BasisAssembly=ElectronHoleBasisAssembly_Concatenate,
                                                optimizer=torch.optim.Adam, 
                                                loss=torch.nn.CrossEntropyLoss, 
                                                lr=1e-3,
                                                save_path=f"{results_dir}/bse_transformer_evec_vae_test.save")
bse.load_data(eb=eb, data_slice=None, train_val_split=0.5, **dataset_kwargs)
bse.train(5, continued=False)

_, dataloader_val = bse.load_data(eb=eb, data_slice=slice(-20,-1), train_val_split=0.5, **dataset_kwargs)
# bse.evaluate_dataset(dataloader_val)

# shutil.rmtree(f"{results_dir}/bse_transformer_evec_vae_test.save")

"""Training for BSE dipole with pretrained VAE"""
print("Training BSE dipole model...")
bse = bse_training_manager_for_deepgwbse_paper(d_model=48, 
                                                model_name='bse_transformer_dipole_vae_test', 
                                                task=BSEPredictTask.dipole,
                                                BasisAssembly=ElectronHoleBasisAssembly_Concatenate,
                                                optimizer=torch.optim.Adam, 
                                                loss=torch.nn.MSELoss, 
                                                lr=1e-3,
                                                save_path=f"{results_dir}/bse_transformer_dipole_vae_test.save")
bse.load_data(eb=eb, data_slice=None, train_val_split=0.5, **dataset_kwargs)
bse.train(5, continued=False)

_, dataloader_val = bse.load_data(eb=eb, data_slice=slice(-20,-1), train_val_split=0.5, **dataset_kwargs)
# bse.evaluate_dataset(dataloader_val)

# shutil.rmtree(f"{results_dir}/bse_transformer_dipole_vae_test.save")

print("BSE training completed!")

