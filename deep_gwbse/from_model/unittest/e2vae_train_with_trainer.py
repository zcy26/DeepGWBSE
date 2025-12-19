#%%
from trainer import Trainer
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
from e2vae import EquivariantVAE, vae_loss

class VAETrainer(Trainer):
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
        x, _ = input
        x = x.to(self.device)
        x_recon, mu, logvar = self.model(x)
        return self.loss(x_recon, x, mu, logvar)
        
    def evaluate(self, input=None):
        self.model.eval()
        
        with torch.no_grad():
            if input is None:
                assert self.validation_dataloader is not None, "Must have a non-empty input"
                for x, _ in self.validation_dataloader:
                    input = x
                    break # By default, get only one batch
            if isinstance(input, DataLoader):
                for x, _ in input:
                    input = x
                    break # By default, get only one batch
            
            input = input.to(self.device)
            x_recon, _, _ = self.model(input)
        
        input = input.cpu().numpy()
        x_recon = x_recon.cpu().numpy()
        return input, x_recon
    


#%%

def test_train():
    # Model and training strategies
    num_epochs = 1
    beta = 0.02
    vae = EquivariantVAE(input_channels=1,
                        hidden_cnn_channels=[60,60,48,48,4],
                        hidden_pooling=[-1,0.66,-1,-1,0.66],
                        kernel_size=[7,5,5,3,3])
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

    # Data preparation
    # The images are normalized and converted to tensors
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure single channel
        transforms.ToTensor()
    ])
    train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Start training!
    vae_trainer = VAETrainer(vae, optimizer, beta=beta, model_name="vae_e2_minst")
    vae_trainer.train(num_epochs, train_loader, test_loader, continued=False)
    
    
# Mini-testing

def test_evaluate():

    beta = 0.02
    vae = EquivariantVAE(input_channels=1,
                        hidden_cnn_channels=[60,60,48,48,4],
                        hidden_pooling=[-1,0.66,-1,-1,0.66],
                        kernel_size=[7,5,5,3,3])
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure single channel
        transforms.ToTensor()
    ])

    train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Read data from file
    vae_trainer = VAETrainer(vae, optimizer, beta=beta, model_name="vae_e2_minst", overwrite=False)

    x, x_recon = vae_trainer.evaluate(test_loader)
    # Plot original vs reconstructed images
    fig, axes = plt.subplots(2, 10, figsize=(10, 3))
    for i in range(10):
        axes[0, i].imshow(x[i, 0], cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(x_recon[i, 0], cmap="gray")
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("Reconstructed")
    plt.show()

    test_id = 10
    sample_image = train_dataset[test_id][0]  # Get an MNIST sample (assuming dataset is loaded)

    # Test rotation equivariance for multiple angles
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    fig, axes = plt.subplots(len(angles), 4, figsize=(12, 3 * len(angles)))

    for i, angle in enumerate(angles):
        rotated_image = TF.rotate(sample_image, angle)
        # Blow, unsqueezing is for creating a fake batch that contains only one sample;
        # squeezing is to get the prediction from the fake output batch;
        # [0] is because a VAE has three outputs at the same time.
        # No need to send the dataset to or from the GPUs:
        # everything is taken care of in `evaluate`
        original_recon = vae_trainer.evaluate(input=sample_image.unsqueeze(0))[0].squeeze(0)
        rotated_recon = vae_trainer.evaluate(input=rotated_image.unsqueeze(0))[0].squeeze(0)
        
        # One more squeezing because of the useless "channel" dimension
        axes[i, 0].imshow(sample_image.squeeze(0), cmap="gray")
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(rotated_image.squeeze(0), cmap="gray")
        axes[i, 1].set_title(f"Rotated Input ({angle}°)")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(original_recon.squeeze(0), cmap="gray")
        axes[i, 2].set_title("Reconstruction")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(rotated_recon.squeeze(0), cmap="gray")
        axes[i, 3].set_title(f"Reconstruction of Rotated Input")
        axes[i, 3].axis("off")

    plt.tight_layout()
    plt.show()

def model_consistency():
    print("Loading the last model and the best model and check if they are the same.")
    vae_last = EquivariantVAE(input_channels=1,
                        hidden_cnn_channels=[60,60,48,48,4],
                        hidden_pooling=[-1,0.66,-1,-1,0.66],
                        kernel_size=[7,5,5,3,3])
    vae_best = EquivariantVAE(input_channels=1,
                        hidden_cnn_channels=[60,60,48,48,4],
                        hidden_pooling=[-1,0.66,-1,-1,0.66],
                        kernel_size=[7,5,5,3,3])
    vae_last.load_state_dict(torch.load("./vae_e2_minst.save/vae_e2_minst.pth"), strict=False)
    vae_best.load_state_dict(torch.load("./vae_e2_minst.save/vae_e2_minst_best.pth"), strict=False)
    for p1, p2 in zip(vae_last.parameters(), vae_best.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            print("The last model is not the best model.")
    return print("The last model is the best model.")


# %%

if __name__ == "__main__":
    test_train()
    test_evaluate()
    model_consistency()

# %%
