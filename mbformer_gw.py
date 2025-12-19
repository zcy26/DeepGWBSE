#!/usr/bin/env python
"""
MBFormer GW training script.

This script trains a transformer model for GW (G0W0) energy predictions
using pretrained VAE embeddings.

Usage:
    python mbformer_gw.py
"""

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


if __name__ == "__main__":
    d_model = 48
    num_epoches = 10
    train_val_split = 0.95
    dataset_dir = './dataset'
    dataset_fname = 'dataset_GW.h5'
    dataset_latent_fname = dataset_fname.split('.')[0] + '_latent.h5'

    # Create or load latent dataset
    if not os.path.exists(os.path.join(dataset_dir, dataset_latent_fname)):
        print(f"Latent dataset not found, creating new one")
        # create latent_dataset
        gwdata = ManyBodyData.from_existing_dataset(os.path.join(dataset_dir, dataset_fname))
        # Load pretrained VAE
        vae = Trainer.configure_model(EquivariantVAE, "./vae_e2_wfn.save")
        eb = ManyBodyData_WFN_Embedder_pretrained(48, E2VAEEmbedder, model=vae, 
                                                   model_name='vae_e2_wfn', 
                                                   save_path='./vae_e2_wfn.save')
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
                                                scheduler=lr_scheduler)    
    gw_trainer_sigma.load_model(True)
    gw_trainer_sigma.train(num_epoches, dataloader_train, dataloader_val, continued=False)

    # Evaluate
    dataloader_val2 = DataLoader(gwdata_val, batch_size=1, collate_fn=gw_collate_fn, shuffle=False)
    print("Validation evaluation:")
    gw_trainer_sigma.evaluate_dataset(dataloader_val2, file_name_pred="data_val_pred.dat", 
                                      file_name_original="data_val_orginal.dat")
    print("Training evaluation:")
    gw_trainer_sigma.evaluate_dataset(dataloader_train, file_name_pred="data_train_pred.dat", 
                                     file_name_original="data_train_orginal.dat")

    print('GW training completed!')

