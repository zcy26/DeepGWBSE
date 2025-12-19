# %%
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model and optimizer
num_epochs = 10
beta = 0.02
vae = EquivariantVAE(input_channels=1,
                    hidden_cnn_channels=[60,60,48,48,4],
                    hidden_pooling=[-1,0.66,-1,-1,0.66],
                    kernel_size=[7,5,5,3,3]).to(device)



optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
# Data transformation: Normalize and convert to tensor
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure single channel
    transforms.ToTensor()
])

# Load MNIST dataset
train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
# Print model size



if os.path.exists("vae_e2_mnist.pth"):
    vae.load_state_dict(torch.load("vae_e2_mnist.pth"), strict=False)
else:
    print("No pretrained model found.")
    # Training loop
    for epoch in range(num_epochs):
        vae.train()
        total_loss = 0

        for x, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x = x.to(device)
            optimizer.zero_grad()
            x_recon, mu, logvar = vae(x)
            loss = vae_loss(x_recon, x, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader.dataset):.4f}")

    # Save model
    torch.save(vae.state_dict(), "vae_e2_mnist.pth")

#######################
vae.eval()

# Get sample images from test set
with torch.no_grad():
    for x, _ in test_loader:
        x = x.to(device)
        x_recon, _, _ = vae(x)
        break  # Get only one batch

# Convert to NumPy
x = x.cpu().numpy()
x_recon = x_recon.cpu().numpy()

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



# %%
# Example usage

test_id = 10
sample_image = train_dataset[10][0].to(device)  # Get an MNIST sample (assuming dataset is loaded)

# Test rotation equivariance for multiple angles
angles = [0, 45, 90, 135, 180, 225, 270, 315]
fig, axes = plt.subplots(len(angles), 4, figsize=(12, 3 * len(angles)))

for i, angle in enumerate(angles):
    rotated_image = TF.rotate(sample_image, angle)
    with torch.no_grad():
        original_recon = vae(sample_image.unsqueeze(0))[0].squeeze(0)
        rotated_recon = vae(rotated_image.unsqueeze(0))[0].squeeze(0)

    axes[i, 0].imshow(sample_image.squeeze(0).cpu(), cmap="gray")
    axes[i, 0].set_title("Original Image")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(rotated_image.squeeze(0).cpu(), cmap="gray")
    axes[i, 1].set_title(f"Rotated Input ({angle}°)")
    axes[i, 1].axis("off")

    axes[i, 2].imshow(original_recon.squeeze(0).cpu(), cmap="gray")
    axes[i, 2].set_title("Reconstruction")
    axes[i, 2].axis("off")

    axes[i, 3].imshow(rotated_recon.squeeze(0).cpu(), cmap="gray")
    axes[i, 3].set_title(f"Reconstruction of Rotated Input")
    axes[i, 3].axis("off")

plt.tight_layout()
plt.show()


# %%
