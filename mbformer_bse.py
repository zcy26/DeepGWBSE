#!/usr/bin/env python
from deep_gwbse.from_model.data import ManyBodyData
from deep_gwbse.from_model.bsetrainer import bse_training_manager_for_deepgwbse_paper, BSEPredictTask
from deep_gwbse.from_model.trainer import Trainer
from deep_gwbse.from_model.e2vae import EquivariantVAE
from deep_gwbse.from_model.wfnembedder import ManyBodyData_WFN_Embedder_pretrained, E2VAEEmbedder
from deep_gwbse.from_model.basisassembly import ElectronHoleBasisAssembly_Concatenate
import torch
import shutil


if __name__ == "__main__":
    # Load pretrained VAE
    vae = Trainer.configure_model(EquivariantVAE, "./vae_e2_wfn.save")
    eb = ManyBodyData_WFN_Embedder_pretrained(48, E2VAEEmbedder, model=vae, 
                                             model_name='vae_e2_wfn', 
                                             save_path='./vae_e2_wfn.save')

    dataset_kwargs = {
        'dataset_dir': './dataset',
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
                                                    lr=1e-3)

    bse.load_data(eb=eb, data_slice=None, train_val_split=0.5, **dataset_kwargs)
    bse.train(5, continued=False)

    _, dataloader_val = bse.load_data(eb=eb, data_slice=slice(-20,-1), train_val_split=0.5, **dataset_kwargs)
    bse.evaluate_dataset(dataloader_val)

    shutil.rmtree("bse_transformer_eval_vae_test.save")

    """Training for BSE eigenvectors with pretrained VAE"""
    print("Training BSE eigenvectors model...")
    bse = bse_training_manager_for_deepgwbse_paper(d_model=48, 
                                                    model_name='bse_transformer_evec_vae_test', 
                                                    task=BSEPredictTask.eigenvectors,
                                                    BasisAssembly=ElectronHoleBasisAssembly_Concatenate,
                                                    optimizer=torch.optim.Adam, 
                                                    loss=torch.nn.CrossEntropyLoss, 
                                                    lr=1e-3)
    bse.load_data(eb=eb, data_slice=None, train_val_split=0.5, **dataset_kwargs)
    bse.train(5, continued=False)

    _, dataloader_val = bse.load_data(eb=eb, data_slice=slice(-20,-1), train_val_split=0.5, **dataset_kwargs)
    # bse.evaluate_dataset(dataloader_val)

    shutil.rmtree("bse_transformer_evec_vae_test.save")

    """Training for BSE dipole with pretrained VAE"""
    print("Training BSE dipole model...")
    bse = bse_training_manager_for_deepgwbse_paper(d_model=48, 
                                                    model_name='bse_transformer_dipole_vae_test', 
                                                    task=BSEPredictTask.dipole,
                                                    BasisAssembly=ElectronHoleBasisAssembly_Concatenate,
                                                    optimizer=torch.optim.Adam, 
                                                    loss=torch.nn.MSELoss, 
                                                    lr=1e-3)
    bse.load_data(eb=eb, data_slice=None, train_val_split=0.5, **dataset_kwargs)
    bse.train(5, continued=False)

    _, dataloader_val = bse.load_data(eb=eb, data_slice=slice(-20,-1), train_val_split=0.5, **dataset_kwargs)
    bse.evaluate_dataset(dataloader_val)

    shutil.rmtree("bse_transformer_dipole_vae_test.save")
    
    print("BSE training completed!")

