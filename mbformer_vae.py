#!/usr/bin/env python
from deep_gwbse.from_model.data import ManyBodyData
from deep_gwbse.from_model.e2vaetrainer import WFNVAETrainer, wfn_collate_fn
from deep_gwbse.from_model.e2vae import EquivariantVAE
from deep_gwbse.from_model.trainer import Trainer
from torch.utils.data import DataLoader
import os
import torch
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Basic configuration
    config_model_path = "./vae_e2_wfn.save" # (Optional) load from existing model
    num_epoches = 20
    beta = 0.2
    train_val_split = 0.8

    # Load dataset
    wfdata = ManyBodyData.from_existing_dataset('./dataset/dataset_WFN.h5')
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
    vae_trainer = WFNVAETrainer(vae, optimizer, beta=beta, model_name="vae_e2_wfn")
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
    plt.savefig("vae_e2_wfn_reconstructed.png")
    print("Training completed! Model saved and reconstruction plot created.")

