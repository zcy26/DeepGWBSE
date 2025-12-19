#%%
import math
from .trainer import Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F
from e2cnn import gspaces, nn as e2nn
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from .e2vae import EquivariantVAE, vae_loss
from torch.utils.data import DataLoader
from .data import ManyBodyData
from sklearn.metrics import r2_score

class WFNVAETrainer(Trainer):
    """
    For options in `kwargs`, see the `__init__` function of `Trainer`.
    """
    def __init__(self, model, optimizer, beta: float, 
                 model_name="vae_e2",
                 **kwargs):
        loss = lambda recon_x, x, mu, logvar: vae_loss(recon_x, x, mu, logvar, beta=beta)
        super().__init__(model, optimizer, loss, model_name=model_name, **kwargs)
        self.beta = beta

    def get_loss(self, input)->torch.Tensor:
        x, mask = input
        x = x.to(self.device)
        mask = ~mask.to(self.device)
        x_recon, mu, logvar = self.model(x)
        # Avoid directly multiplying the mask to x or x_recon to ensure proper loss calculation, accounting for normalization and NaN handling.
        return self.loss(x_recon[:, mask.squeeze()], x[:, mask.squeeze()], mu, logvar)
        
    def evaluate(self, input=None, mask=None, **kwargs):
        """
        # TODO: rewrite this function
        comment:
            i) it can only get the first batch of the dataloader
            ii) it can only get one batch of input
            iii) mask input doesn't make sense
        """

        self.model.eval()
        
        with torch.no_grad():
            if input is None:
                assert self.validation_dataloader is not None, "Must have a non-empty input"
                for x, loaded_mask in self.validation_dataloader:
                    input = x
                    mask = loaded_mask
                    break # By default, get only one batch
            if isinstance(input, DataLoader):
                for x, loaded_mask in input:
                    input = x
                    mask = loaded_mask
                    break # By default, get only one batch
            
            input = input.to(self.device)
            x_recon, _, _ = self.model(input)
            if mask is not None:
                mask_nan = torch.where(mask, np.nan, 1.0).to(self.device)
                x_recon = mask_nan * x_recon
                input = input.to(self.device)
                input = mask_nan * input
        
        input = input.cpu().numpy()
        x_recon = x_recon.cpu().numpy()
        return input, x_recon

# wfdata = ManyBodyData.from_existing_dataset('./dataset/dataset_semi.h5')

def wfn_collate_fn(batch):
    assert len(batch)==1, "Batch size should be 1 for WFN data"
    wfn = batch[0]["wfn"]
    nk, nb, X, Y, C = wfn.shape
    # Rearrange dimensions to make the z-coordinate the channel dimension,
    # treating each material as a batch due to varying wave function sizes.
    wfn = (wfn.reshape(nk*nb, X, Y, C)).transpose(0, 3, 1, 2)
    wfn = torch.from_numpy(wfn).float()
    scaling_factor = 4 # TODO: make the process determining the scaling factor automatic
    delta_X = scaling_factor * math.ceil(X / scaling_factor) - X
    delta_Y = scaling_factor * math.ceil(Y / scaling_factor) - Y
    wfn = F.pad(wfn, (0, delta_Y, 0, delta_X), mode="constant", value=np.nan)
    # Add a batch dimension to the mask for broadcasting across samples.
    mask = torch.isnan(wfn[0])[None, ...] 
    # Retain NaNs only to define unit cell boundaries; replace them with zeros afterward.
    wfn = torch.nan_to_num(wfn, nan=0.0)
    # Normalization: make sure that each sample in the batch is rescaled to 0-1.
    max_image = wfn.max(dim=1)[0].max(dim=1)[0].max(dim=1)[0]  # Extract max along each axis
    wfn = wfn / max_image[:, None, None, None]

    return wfn, mask

if __name__ == "__main__":
    config_model_path = "./vae_e2_wfn.save"
    num_epoches = 200
    beta = 0.0
    train_val_split = 0.8 # 

    # wfdata = ManyBodyData.from_existing_dataset('./dataset/1000_wfn_1/dataset_WFN_1000.h5')
    wfdata = ManyBodyData.from_existing_dataset('./dataset/dataset_WFN.h5')

    wfdata_train = wfdata[:int(len(wfdata)*train_val_split)]
    wfdata_val = wfdata[int(len(wfdata)*train_val_split):]
    dataloader_train = DataLoader(wfdata_train, batch_size=1, collate_fn=wfn_collate_fn)
    dataloader_val = DataLoader(wfdata_val, batch_size=1, collate_fn=wfn_collate_fn)

    if os.path.exists(config_model_path):
        print("Loading model from", config_model_path)
        vae = Trainer.configure_model(EquivariantVAE, config_model_path)
    else:
        vae = EquivariantVAE(input_channels=wfdata.info.cell_slab_truncation,
                                hidden_cnn_channels=[60,60,48,48,48],
                                hidden_pooling=[-1,0.66,-1,-1,0.66],
                                kernel_size=[7,5,5,3,3])
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)


    vae_trainer = WFNVAETrainer(vae, optimizer, beta=beta, model_name="vae_e2_wfn")
    vae_trainer.load_model(load_best=True)
    vae_trainer.train(num_epoches, dataloader_train, dataloader_val, continued=True)

    #%%
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
    plt.show()
    # %%