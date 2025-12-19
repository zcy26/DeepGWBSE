import torch
import torch.nn as nn
import torch.nn.functional as F
from e2cnn import gspaces, nn as e2nn
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from from_model.model_util import print_model_size, capture_config
import os
from tqdm import tqdm

class single_e2CNN_module(nn.Module):
    """
    e2CNN + BatchNorm + ReLU + 2stride_Pooling (optional)

    headLayer:
        input: torch tensor
        output: GeometricTensor
    
    midLayer:
        input: GeometricTensor
        output: GeoemtricTensor

    tailLayer:
        input: GeometricTensor
        output: torch tensor

    """
    def __init__(self, input_channels=1, output_channels=144, N_rotation=12, kernel_size=5, padding=2, sigma=0.66, 
                pooling: bool = False, headLayer: bool = True, tailLayer: bool = False):
        super().__init__()

        self.N_rotation = N_rotation
        self.headLayer = headLayer
        self.tailLayer = tailLayer
        self.r2_act = gspaces.Rot2dOnR2(N=N_rotation)

        if N_rotation != -1 and not tailLayer: # Discrete rotations
            assert output_channels % N_rotation == 0, "Output channels must be divisible by N_rotation"

        if headLayer:
            in_type = e2nn.FieldType(self.r2_act, input_channels*[self.r2_act.trivial_repr])
        else:
            assert input_channels % N_rotation == 0, "Input channels must be divisible by N_rotation if not headLayer" 
            in_type = e2nn.FieldType(self.r2_act, input_channels//N_rotation*[self.r2_act.regular_repr])

        if tailLayer:
            out_type = e2nn.FieldType(self.r2_act, output_channels*[self.r2_act.trivial_repr])
        else:
            out_type = e2nn.FieldType(self.r2_act, output_channels//N_rotation*[self.r2_act.regular_repr])     
  
        self.input_type = in_type   
        self.out_type = out_type


        self.cnn_block = e2nn.SequentialModule(
            e2nn.R2Conv(in_type, out_type, kernel_size=kernel_size, padding=padding, bias=False),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type, inplace=True),
        )

        if pooling:
            self.pool = e2nn.PointwiseAvgPoolAntialiased(out_type, sigma=sigma, stride=2)

    def forward(self, x):
        if self.headLayer:
            assert isinstance(x, torch.Tensor), "Input must be a torch tensor for headLayer"
            x = e2nn.GeometricTensor(x, self.input_type)
        else:
            assert isinstance(x, e2nn.GeometricTensor), "Input tensor must be e2nn.GeometricTensor provided for non-headLayer"
        x = self.cnn_block(x)
        if hasattr(self, 'pool'):
            x = self.pool(x)
        if self.tailLayer:
            x = x.tensor
        return x

class single_e2TransposeCNN_module(nn.Module):
    """
    e2_Transpose_CNN + 2stride (optional) + BatchNorm + ReLU

    headLayer:
        input: torch tensor
        output: GeometricTensor
    
    midLayer:
        input: GeometricTensor
        output: GeoemtricTensor

    tailLayer:
        input: GeometricTensor
        output: torch tensor


    """
    def __init__(self, input_channels=1, output_channels=144, N_rotation=12, kernel_size=5,
                doubleSize: bool = False, headLayer: bool = True, tailLayer: bool = False):
        super().__init__()

        self.N_rotation = N_rotation
        self.headLayer = headLayer
        self.tailLayer = tailLayer
        self.r2_act = gspaces.Rot2dOnR2(N=N_rotation)

        if N_rotation != -1 and not tailLayer: # Discrete rotations
            assert output_channels % N_rotation == 0, "Output channels must be divisible by N_rotation"

        if headLayer:
            in_type = e2nn.FieldType(self.r2_act, input_channels*[self.r2_act.trivial_repr])
        else:
            assert input_channels % N_rotation == 0, "Input channels must be divisible by N_rotation if not headLayer" 
            in_type = e2nn.FieldType(self.r2_act, input_channels//N_rotation*[self.r2_act.regular_repr])
        
        if tailLayer:
            out_type = e2nn.FieldType(self.r2_act, output_channels*[self.r2_act.trivial_repr])
        else:
            out_type = e2nn.FieldType(self.r2_act, output_channels//N_rotation*[self.r2_act.regular_repr])     

        self.input_type = in_type   
        self.out_type = out_type

        if doubleSize:
            output_padding = 1
            padding = (kernel_size - 1) // 2
            stride = 2
        else:
            output_padding = 0
            padding = kernel_size // 2
            stride = 1

        self.tcnn_block = e2nn.SequentialModule(
            e2nn.R2ConvTransposed(in_type, out_type, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=False),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type, inplace=True)
        )

    def forward(self, x):
        if self.headLayer:
            assert isinstance(x, torch.Tensor), "Input must be a torch tensor for headLayer"
            x = e2nn.GeometricTensor(x, self.input_type)
        else:
            assert isinstance(x, e2nn.GeometricTensor), "Input tensor must be e2nn.GeometricTensor provided for non-headLayer"

        x = self.tcnn_block(x)
        if self.tailLayer:
            x = x.tensor
        return x

class EquivariantEncoder_double_cnn(nn.Module):
    def __init__(self, input_channels=1, N_rotation=12,
                hidden_cnn_channels: list=[96, 48, 48, 12], 
                hidden_pooling: list=[-1.00, 0.66, -1.00, 0.66],
                kernel_size=[5,5,3,3]):
        """
        hidden_pooling: -1 for no pooling, 0.66 for 2 stride pooling (H_out = H_in/2, W_out = W_in/2)
        """
        super().__init__()

        self.input_channels = input_channels
        self.N_rotation = N_rotation
        self.hidden_cnn_channels = hidden_cnn_channels
        self.hidden_pooling = hidden_pooling
        self.hidden_pooling_bool = [False if i == -1 else True for i in hidden_pooling]
        self.kernel_size = kernel_size
        self.spatical_compression_factor = np.sum(self.hidden_pooling_bool) * 2 

        assert len(hidden_cnn_channels) == len(hidden_pooling), "Length of hidden_cnn_channels and hidden_pooling must be equal"
        assert len(hidden_cnn_channels) > 0, "At least one hidden layer must be present"

        if kernel_size is None:
            self.kernel_size = [3] * len(hidden_cnn_channels)

        self.padding = list(map(lambda x: x//2,self.kernel_size))

        self.mu_cnn_stack = []
        self.logvar_cnn_stack = []

        assert len(hidden_cnn_channels) > 1, "At least 2 hidden layers required"

        for i in range(len(hidden_cnn_channels)):

            kwargs = dict(output_channels=hidden_cnn_channels[i],
                                    N_rotation=N_rotation, kernel_size=self.kernel_size[i], padding=self.padding[i],
                                    sigma=hidden_pooling[i], pooling=self.hidden_pooling_bool[i])
            if i == 0:
                kwargs.update(dict(input_channels=input_channels, headLayer=True, tailLayer=False))

            elif i == len(hidden_cnn_channels) - 1: # Last layer
                kwargs.update(dict(input_channels=hidden_cnn_channels[i-1], headLayer=False, tailLayer=True))
            else:
                kwargs.update(dict(input_channels=hidden_cnn_channels[i-1], headLayer=False, tailLayer=False))

            self.mu_cnn_stack.append(single_e2CNN_module(**kwargs))
            self.logvar_cnn_stack.append(single_e2CNN_module(**kwargs))

        self.mu_cnn_stack = nn.ModuleList(self.mu_cnn_stack)
        self.logvar_cnn_stack = nn.ModuleList(self.logvar_cnn_stack)

        self.summary()
    def summary(self):
        print('==================== Encoder Summary ====================')
        print(f'Encoder hidden {len(self.hidden_cnn_channels)} layers:', [f"{self.input_channels}->"]+self.hidden_cnn_channels[:-1]+[f"->{self.hidden_cnn_channels[-1]}"])
        print('Encoder pooling:', self.hidden_pooling)
        print('Encoder kernel_size:', self.kernel_size)
        print('Encoder padding:', self.padding)

        channel_compression_rate =  self.hidden_cnn_channels[-1] / self.input_channels
        spatial_compression_rate = 1 / (4 ** np.sum(self.hidden_pooling_bool))
        print(f'Channel compression rate: {channel_compression_rate*100: .2f}%')
        print(f'Spatial compression rate: {spatial_compression_rate*100: .2f}%')
        print(f'Total compression rate: {channel_compression_rate * spatial_compression_rate*100: .2f}%')

        pass

    def forward(self, x):
        
        H, W = x.shape[-2:]
        assert H // self.spatical_compression_factor > 0 and W // self.spatical_compression_factor > 0, "Input size too small for the given hidden layers"

        mu = x
        logvar = x

        for i in range(len(self.mu_cnn_stack)):
            mu = self.mu_cnn_stack[i](mu)
            logvar = self.logvar_cnn_stack[i](logvar)
        return mu, logvar

class EquivariantDecoder(nn.Module):
    def __init__(self, input_channels=1, N_rotation=12,
                hidden_cnn_channels: list=[12, 48, 48, 1], 
                size_double_list: list=[True, False, True, False],
                kernel_size=[3,3,5,5]):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = hidden_cnn_channels[-1]
        self.hidden_cnn_channels = hidden_cnn_channels
        self.size_double_list = size_double_list
        self.kernel_size = kernel_size

        if kernel_size is None:
            self.kernel_size = [3] * len(hidden_cnn_channels)
        
        self.tcnn_stack = []
        for i in range(len(hidden_cnn_channels)):
            kwargs = dict( output_channels=hidden_cnn_channels[i],
                                    N_rotation=N_rotation, kernel_size=self.kernel_size[i], doubleSize=size_double_list[i])
            if i == 0: # First layer
                kwargs.update(dict(input_channels=input_channels, headLayer=True, tailLayer=False))
            elif i == len(hidden_cnn_channels) - 1: # Last layer
                kwargs.update(dict(input_channels=hidden_cnn_channels[i-1], headLayer=False, tailLayer=True))
            else: # Middle layers
                kwargs.update(dict(input_channels=hidden_cnn_channels[i-1], headLayer=False, tailLayer=False))

            self.tcnn_stack.append(single_e2TransposeCNN_module(**kwargs))
            
        self.tcnn_stack = nn.ModuleList(self.tcnn_stack)

        self.summary()

    def summary(self):
        print('==================== Decoder Summary ====================')
        print(f'Decoder hidden {len(self.hidden_cnn_channels)} layers:', [f"{self.input_channels}->"]+self.hidden_cnn_channels[:-1]+[f"->{self.hidden_cnn_channels[-1]}"])
        print('Decoder size doubling:', self.size_double_list)
        print('Decoder kernel_size:', self.kernel_size)

        channel_expansion_rate = self.hidden_cnn_channels[-1] / self.input_channels
        spatial_expansion_rate = np.prod([4 if double else 1 for double in self.size_double_list])
        print(f'Channel expansion rate: {channel_expansion_rate: .2f}')
        print(f'Spatial expansion rate: {spatial_expansion_rate: .2f}')
        print(f'Total expansion rate: {channel_expansion_rate * spatial_expansion_rate: .2f}')


    def forward(self, x):
        
        for i in range(len(self.tcnn_stack)):
            x = self.tcnn_stack[i](x)
        return x

class EquivariantVAE(nn.Module):
    @capture_config
    def __init__(self, input_channels=1, N_rotation=12,
                hidden_cnn_channels: list=[96, 48, 48, 12], 
                hidden_pooling: list=[-1.00, 0.66, -1.00, 0.66],
                kernel_size=[5,5,3,3]):
        """
        Decoder will mirror the encoder

        Note: Input_channels and hidden_cnn_channels[-1] could be any number, but other hidden_cnn_channels should be divisible by N_rotation
        """
        super().__init__()

        self.encoder = EquivariantEncoder_double_cnn(input_channels=input_channels, 
                                                     N_rotation=N_rotation, 
                                                     hidden_cnn_channels=hidden_cnn_channels, 
                                                     hidden_pooling=hidden_pooling, 
                                                     kernel_size=kernel_size)
        
        dec_input_channels = hidden_cnn_channels[-1]
        dec_hidden_cnn_channels = (hidden_cnn_channels[:-1])[::-1]
        dec_hidden_cnn_channels.append(input_channels)
        dec_kernel_size = kernel_size[::-1]
        size_double_list = np.where(np.array(hidden_pooling[::-1]) == -1, False, True)

        self.decoder = EquivariantDecoder(input_channels=dec_input_channels, N_rotation=N_rotation,
                                          hidden_cnn_channels=dec_hidden_cnn_channels, 
                                          size_double_list=size_double_list,
                                          kernel_size=dec_kernel_size)

        self.compression_rate = self.encoder.spatical_compression_factor

        self.summary()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std 

    def summary(self):
        # Print model parameters
        print('==================== VAE Summary ====================')
        # encoder_parameters = sum(p.numel() for p in self.encoder.parameters())
        # decoder_parameters = sum(p.numel() for p in self.decoder.parameters())
        # print(f'Encoder parameters: {encoder_parameters}')
        # print(f'Decoder parameters: {decoder_parameters}')
        # print(f'Total parameters: {encoder_parameters + decoder_parameters}')
        enc_para_number = print_model_size(self.encoder, "Encoder")
        dec_para_number = print_model_size(self.decoder, "Decoder")
        print(f'Total parameters: {enc_para_number + dec_para_number}')

    def forward(self, x):

        H, W = x.shape[-2:]
        if H % self.compression_rate != 0 and W % self.compression_rate != 0:
            print("Warning: Input size not divisible by compression rate. Reconstruction may not match input size")

        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar




class VAE(nn.Module):
    def __init__(self, latent_dim=60, input_channels=30):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 480, kernel_size=3, padding=1, padding_mode='circular'), # Adjust parameters as needed
            # Comment: the performance of VAE will increase obviously when increasing kernel_size, try using 3, 5, 7,...
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(480, 480, kernel_size=3, padding=1, padding_mode='circular'),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            # nn.Linear(392*4, latent_dim*2),
            nn.Linear(480, latent_dim*2),
            # nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 392*4),
            nn.ReLU(),
            nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(32, 60, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(60,input_channels, kernel_size=2, stride=2),
            # nn.Sigmoid()  # Sigmoid for values between 0 and 1
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        # 0. Input size (preprocess)
        N, C, H, W = x.size()
        x = self.encoder(x)
        mu, logvar = x.chunk(2, dim=1)
        z = self.reparameterize(mu, logvar)
        x = self.decoder(z)
        upsampling = nn.Upsample(size=(H, W))
        x = upsampling(x)
        return x, mu, logvar

    def get_latent_space(self, x):
        x = self.encoder(x)
        mu, logvar = x.chunk(2, dim=1)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def no_grad(self):
        for param in self.parameters():
            param.requires_grad = False
        return self


# Loss function
mse_loss = nn.MSELoss(reduction="sum")  # Matches F.mse_loss with sum
def vae_loss(recon_x, x, mu, logvar, beta=0.02):
    recon_loss = mse_loss(recon_x, x)
    kl_loss = -beta * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss





def unit_test():
    #random seed

    torch.manual_seed(36)

    device = "cpu" # For testing
    data = torch.randn(10, 1, 60, 60).to(device) # input image size should be divisible by 2^n (larger n the better, n determines hidden pooling layers)
    vae = EquivariantVAE(input_channels=1,
                         hidden_cnn_channels=[60,60,48,48,4],
                         hidden_pooling=[-1,0.66,-1,-1,0.66],
                         kernel_size=[7,5,5,3,3]).to(device)

    x_recon, mu, logvar = vae(data)
    print("\nUnit Test:")
    print_model_size(vae)
    if x_recon.shape == data.shape:
        print("Reconstruction Shape: Pass")
        loss = vae_loss(x_recon, data, mu, logvar)
        print(loss)
        print("shape data:", data.shape)
        print("shape x_recon:", x_recon.shape)
        print("shape mu:", mu.shape)
        print("shape logvar:", logvar.shape)
    else:
        print("Reconstruction Shape: Fail")
        print(f"Expected: {data.shape}, Got: {x_recon.shape}")

    return vae

if __name__ == "__main__":
    unit_test()

    
