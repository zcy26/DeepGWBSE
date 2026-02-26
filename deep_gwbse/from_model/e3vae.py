import torch
import torch.nn as nn
from escnn import gspaces
from escnn import nn as e3nn

from deep_gwbse.from_model.model_util import print_model_size, capture_config
import random
import numpy as np
import os


class single_e3CNN_module(nn.Module):

    def __init__(
        self,
        input_channels=1,
        output_channels=16,
        kernel_size=3,
        padding=1,
        stride=1,
        pooling=False,
        headLayer=True,
        tailLayer=False,
        irrep_order=0,   # l value (0=scalar,1=vector,...)
    ):
        super().__init__()

        self.headLayer = headLayer
        self.tailLayer = tailLayer

        # SO(3) action
        self.r3_act = gspaces.rot3dOnR3()

        irrep = self.r3_act.irrep(irrep_order)

        # ---------- Field types ----------
        if headLayer:
            in_type = e3nn.FieldType(
                self.r3_act,
                input_channels * [self.r3_act.trivial_repr]
            )
        else:
            in_type = e3nn.FieldType(
                self.r3_act,
                input_channels * [irrep]
            )

        if tailLayer:
            out_type = e3nn.FieldType(
                self.r3_act,
                output_channels * [self.r3_act.trivial_repr]
            )
        else:
            out_type = e3nn.FieldType(
                self.r3_act,
                output_channels * [irrep]
            )

        self.input_type = in_type
        self.out_type = out_type

        # ---------- CNN block ----------
        self.cnn_block = e3nn.SequentialModule(
            e3nn.R3Conv(
                in_type,
                out_type,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            e3nn.InnerBatchNorm(out_type),
            e3nn.ReLU(out_type, inplace=True),
        )

        if pooling:
            self.pool = e3nn.PointwiseAvgPoolAntialiased3D(
                out_type,
                sigma=0.6,
                stride=2
            )

    def forward(self, x):

        if self.headLayer:
            assert isinstance(x, torch.Tensor)
            x = e3nn.GeometricTensor(x, self.input_type)
        else:
            assert isinstance(x, e3nn.GeometricTensor)
        x = self.cnn_block(x)

        if hasattr(self, "pool"):
            x = self.pool(x)

        if self.tailLayer:
            x = x.tensor

        return x

class single_e3TransposeCNN_module(nn.Module):

    """
    e3_Transpose_CNN + optional upsampling + BatchNorm + ReLU

    headLayer:
        input: torch tensor
        output: GeometricTensor
    
    midLayer:
        input: GeometricTensor
        output: GeometricTensor

    tailLayer:
        input: GeometricTensor
        output: torch tensor
    """

    def __init__(
        self,
        input_channels=1,
        output_channels=16,
        kernel_size=3,
        doubleSize=False,
        headLayer=True,
        tailLayer=False,
        irrep_order=0,
    ):
        super().__init__()

        self.headLayer = headLayer
        self.tailLayer = tailLayer

        # SO(3) action
        self.r3_act = gspaces.rot3dOnR3()
        irrep = self.r3_act.irrep(irrep_order)

        # ---------- Field types ----------
        if headLayer:
            in_type = e3nn.FieldType(
                self.r3_act,
                input_channels * [self.r3_act.trivial_repr]
            )
        else:
            in_type = e3nn.FieldType(
                self.r3_act,
                input_channels * [irrep]
            )

        if tailLayer:
            out_type = e3nn.FieldType(
                self.r3_act,
                output_channels * [self.r3_act.trivial_repr]
            )
        else:
            out_type = e3nn.FieldType(
                self.r3_act,
                output_channels * [irrep]
            )

        self.input_type = in_type
        self.out_type = out_type

        # ---------- stride logic ----------
        if doubleSize:
            stride = 2
            padding = (kernel_size - 1) // 2
            output_padding = 1
        else:
            stride = 1
            padding = kernel_size // 2
            output_padding = 0

        # ---------- Transpose CNN block ----------
        self.tcnn_block = e3nn.SequentialModule(
            e3nn.R3ConvTransposed(
                in_type,
                out_type,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                bias=False,
            ),
            e3nn.InnerBatchNorm(out_type),
            e3nn.ReLU(out_type, inplace=True),
        )

    def forward(self, x):

        if self.headLayer:
            assert isinstance(x, torch.Tensor)
            x = e3nn.GeometricTensor(x, self.input_type)
        else:
            assert isinstance(x, e3nn.GeometricTensor)

        x = self.tcnn_block(x)

        if self.tailLayer:
            x = x.tensor

        return x

class EquivariantEncoder3D_double_cnn(nn.Module):

    def __init__(
        self,
        input_channels=1,
        irrep_order=0,
        hidden_cnn_channels=[32, 32, 64, 64],
        hidden_pooling=[False, True, False, True],
        kernel_size=[3,3,3,3],
    ):
        super().__init__()

        self.input_channels = input_channels
        self.irrep_order = irrep_order
        self.hidden_cnn_channels = hidden_cnn_channels
        self.hidden_pooling = hidden_pooling
        self.kernel_size = kernel_size

        assert len(hidden_cnn_channels) == len(hidden_pooling)
        assert len(hidden_cnn_channels) > 1

        self.padding = [k//2 for k in kernel_size]
        self.spatial_compression_factor = 2 ** sum(hidden_pooling)

        self.mu_cnn_stack = []
        self.logvar_cnn_stack = []

        for i in range(len(hidden_cnn_channels)):

            kwargs = dict(
                output_channels=hidden_cnn_channels[i],
                kernel_size=self.kernel_size[i],
                padding=self.padding[i],
                pooling=hidden_pooling[i],
                irrep_order=irrep_order
            )

            if i == 0:
                kwargs.update(dict(
                    input_channels=input_channels,
                    headLayer=True,
                    tailLayer=False
                ))

            elif i == len(hidden_cnn_channels)-1:
                kwargs.update(dict(
                    input_channels=hidden_cnn_channels[i-1],
                    headLayer=False,
                    tailLayer=True
                ))

            else:
                kwargs.update(dict(
                    input_channels=hidden_cnn_channels[i-1],
                    headLayer=False,
                    tailLayer=False
                ))

            self.mu_cnn_stack.append(single_e3CNN_module(**kwargs))
            self.logvar_cnn_stack.append(single_e3CNN_module(**kwargs))

        self.mu_cnn_stack = nn.ModuleList(self.mu_cnn_stack)
        self.logvar_cnn_stack = nn.ModuleList(self.logvar_cnn_stack)

        self.summary()

    def summary(self):

        print('==================== Encoder3D Summary ====================')
        print('hidden:', self.hidden_cnn_channels)
        print('pooling:', self.hidden_pooling)
        print('kernel_size:', self.kernel_size)

        channel_compression = self.hidden_cnn_channels[-1] / self.input_channels
        spatial_compression = 1 / (8 ** sum(self.hidden_pooling))

        print(f'Channel compression rate: {channel_compression*100:.2f}%')
        print(f'Spatial compression rate: {spatial_compression*100:.2f}%')
        print(f'Total compression rate: {(channel_compression*spatial_compression)*100:.2f}%')

    def forward(self, x):

        D,H,W = x.shape[-3:]
        assert D//self.spatial_compression_factor > 0

        mu = x
        logvar = x

        for i in range(len(self.mu_cnn_stack)):
            mu = self.mu_cnn_stack[i](mu)
            logvar = self.logvar_cnn_stack[i](logvar)

        return mu, logvar

class EquivariantDecoder3D(nn.Module):

    def __init__(
        self,
        input_channels=64,
        irrep_order=0,
        hidden_cnn_channels=[64,32,32,1],
        size_double_list=[True, False, True, False],
        kernel_size=[3,3,3,3]
    ):
        super().__init__()

        self.input_channels = input_channels
        self.hidden_cnn_channels = hidden_cnn_channels
        self.size_double_list = size_double_list
        self.kernel_size = kernel_size

        self.tcnn_stack = []

        for i in range(len(hidden_cnn_channels)):

            kwargs = dict(
                output_channels=hidden_cnn_channels[i],
                doubleSize=size_double_list[i],
                kernel_size=kernel_size[i],
                irrep_order=irrep_order
            )

            if i == 0:
                kwargs.update(dict(
                    input_channels=input_channels,
                    headLayer=True,
                    tailLayer=False
                ))

            elif i == len(hidden_cnn_channels)-1:
                kwargs.update(dict(
                    input_channels=hidden_cnn_channels[i-1],
                    headLayer=False,
                    tailLayer=True
                ))

            else:
                kwargs.update(dict(
                    input_channels=hidden_cnn_channels[i-1],
                    headLayer=False,
                    tailLayer=False
                ))

            self.tcnn_stack.append(
                single_e3TransposeCNN_module(**kwargs)
            )

        self.tcnn_stack = nn.ModuleList(self.tcnn_stack)

        self.summary()

    def summary(self):

        print('==================== Decoder3D Summary ====================')
        print('hidden:', self.hidden_cnn_channels)
        print('size doubling:', self.size_double_list)
        print('kernel_size:', self.kernel_size)

        spatial_expansion = np.prod(
            [8 if d else 1 for d in self.size_double_list]
        )

        print(f'Spatial expansion rate: {spatial_expansion:.2f}')

    def forward(self, x):

        for layer in self.tcnn_stack:
            x = layer(x)

        return x

class EquivariantVAE3D(nn.Module):
    @capture_config
    def __init__(self, input_channels=1, irrep_order=0,
                 hidden_cnn_channels: list=[32, 32, 64, 64],
                 hidden_pooling: list=[False, True, False, True],
                 kernel_size=[3, 3, 3, 3]):
        """
        Decoder will mirror the encoder

        Note: Input_channels and hidden_cnn_channels[-1] could be any number,
        but other hidden_cnn_channels should follow your design.
        """
        super().__init__()

        # ---------- Encoder ----------
        self.encoder = EquivariantEncoder3D_double_cnn(
            input_channels=input_channels,
            irrep_order=irrep_order,
            hidden_cnn_channels=hidden_cnn_channels,
            hidden_pooling=hidden_pooling,
            kernel_size=kernel_size
        )

        # ---------- Decoder ----------
        dec_input_channels = hidden_cnn_channels[-1]
        dec_hidden_cnn_channels = hidden_cnn_channels[:-1][::-1]
        dec_hidden_cnn_channels.append(input_channels)
        dec_kernel_size = kernel_size[::-1]
        size_double_list = hidden_pooling[::-1]  # True/False list

        self.decoder = EquivariantDecoder3D(
            input_channels=dec_input_channels,
            irrep_order=irrep_order,
            hidden_cnn_channels=dec_hidden_cnn_channels,
            size_double_list=size_double_list,
            kernel_size=dec_kernel_size
        )

        # Keep compression rate for input size checks
        self.compression_rate = self.encoder.spatial_compression_factor

        self.summary()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def summary(self):
        # Print model parameters
        print('==================== VAE3D Summary ====================')
        enc_para_number = print_model_size(self.encoder, "Encoder3D")
        dec_para_number = print_model_size(self.decoder, "Decoder3D")
        print(f'Total parameters: {enc_para_number + dec_para_number}')

    def forward(self, x):
        D, H, W = x.shape[-3:]

        # Warning if input not divisible by compression factor
        if D % self.compression_rate != 0 or H % self.compression_rate != 0 or W % self.compression_rate != 0:
            print("Warning: Input size not divisible by compression rate. Reconstruction may not match input size")

        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar


def test(model=None):
    device="cpu"
    import torch
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    import numpy as np

    def create_cylinder(size=32, radius=8, height=20):
        grid = np.zeros((size, size, size))

        cx = cy = size // 2
        z0 = size // 2 - height // 2
        z1 = size // 2 + height // 2

        for x in range(size):
            for y in range(size):
                if (x - cx)**2 + (y - cy)**2 <= radius**2:
                    grid[x, y, z0:z1] = 1.0

        return torch.tensor(grid).float()

    def rotate_volume(volume, angle_deg=45):
        B, C, D, H, W = volume.shape

        angle = np.deg2rad(angle_deg)

        R = torch.tensor([
            [ np.cos(angle), -np.sin(angle), 0, 0],
            [ np.sin(angle),  np.cos(angle), 0, 0],
            [ 0,              0,             1, 0]
        ], dtype=torch.float32, device=volume.device)

        R = R.unsqueeze(0)

        grid = F.affine_grid(R, volume.size(), align_corners=False)
        rotated = F.grid_sample(volume, grid, align_corners=False)

        return rotated

    # def show_volume_3d_2x2(x, x_rot, y, y_rot_input, threshold=0.05):

    #     import matplotlib.pyplot as plt
    #     import numpy as np

    #     def get_voxels(vol):
    #         data = vol[0, 0].detach().cpu().numpy()
    #         return data > threshold

    #     vols = [
    #         (get_voxels(x), "Original Cylinder"),
    #         (get_voxels(x_rot), "Rotated Cylinder"),
    #         (get_voxels(y), "Model Output"),
    #         (get_voxels(y_rot_input), "Output Rotated Input"),
    #     ]

    #     fig = plt.figure(figsize=(12, 12))

    #     for i, (vox, title) in enumerate(vols):
    #         ax = fig.add_subplot(2, 2, i + 1, projection='3d')
    #         ax.voxels(vox, edgecolor='k')
    #         ax.set_title(title)

    #         # cleaner look
    #         ax.set_xticks([])
    #         ax.set_yticks([])
    #         ax.set_zticks([])

    #     plt.tight_layout()
    #     plt.show()


    def show_volume_3d_2x2(x, x_rot, y, y_rot_input, threshold=0.05):
        import matplotlib.pyplot as plt
        import numpy as np
        import torch

        def get_numpy(vol):
            # replace NaNs with 0 first
            data = torch.nan_to_num(vol[0, 0], nan=0.0).detach().cpu().numpy()
            return data

        def get_voxels(vol):
            data = get_numpy(vol)
            return data > threshold

        vols = [
            (x, "Original Cylinder"),
            (x_rot, "Rotated Cylinder"),
            (y, "Model Output"),
            (y_rot_input, "Output Rotated Input"),
        ]

        # ---------- 3D voxel plots ----------
        fig = plt.figure(figsize=(12, 12))

        for i, (vol, title) in enumerate(vols):
            vox = get_voxels(vol)
            ax = fig.add_subplot(2, 2, i + 1, projection='3d')
            ax.voxels(vox, edgecolor='k')
            ax.set_title(title)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

        plt.tight_layout()
        plt.show()

        # ---------- 2D summed slice plots ----------
        fig2 = plt.figure(figsize=(12, 12))

        for i, (vol, title) in enumerate(vols):
            data = get_numpy(vol)

            # sum over last axis
            slice_2d = np.sum(data, axis=-3)

            ax = fig2.add_subplot(2, 2, i + 1)
            im = ax.imshow(slice_2d)
            ax.set_title(f"{title} (sum last axis)")
            ax.axis('off')

        plt.tight_layout()
        plt.show()



    if model is None:
        model = single_e3CNN_module(
            input_channels=1,
            output_channels=16,
            stride=1,
            pooling=True,
            headLayer=True,
            tailLayer=True,
            ).to(device)

    model.eval()

    x = create_cylinder(24, 3, 15)
    x = x.unsqueeze(0).unsqueeze(0).to(device)

    x_rot = rotate_volume(x, 45)

    with torch.no_grad():
        print("Test x.shape:", x.shape)
        print("Test x_rot.shape:", x_rot.shape)
        y = model(x)
        y_rot_input = model(x_rot)

    if isinstance(y, tuple):
        y = y[0]
        y_rot_input = y_rot_input[0]
    print('y shape:', y.shape)
    print('y_rot_input shape:', y_rot_input.shape)


    # rotate output
    # y_rot_output = rotate_volume(y, 45)


    show_volume_3d_2x2(
        x,
        x_rot,
        y,
        y_rot_input,
        threshold=0.1
    )

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    

if __name__ == "__main__":
    torch.backends.cudnn.enabled = False # This is a very tricky bug. It seems cudnn only supports 2-5 dimensions, 
    # but escnn will somehow increase one dimension when using gpu, use this could avoid bug!

    set_seed(41) # 41 is good
    model = single_e3CNN_module(
        input_channels=1,
        output_channels=16,
        kernel_size=3,
        pooling=True,
        headLayer=True,
        tailLayer=True
    )


    model_enc = EquivariantEncoder3D_double_cnn(
        input_channels=1,
        irrep_order=0,
        hidden_cnn_channels=[16, 32, 64, 128],
        hidden_pooling=[True, True, True, False],
        kernel_size=[3,3,3,3],
    )

    tmodel = single_e3TransposeCNN_module(
        input_channels=1,
        output_channels=16,
        kernel_size=3,
        doubleSize=True,
        headLayer=True,
        tailLayer=True
    )

    model_dec = EquivariantDecoder3D(
        input_channels=1,
        irrep_order=0,
        hidden_cnn_channels=[16,32,16,1],
        size_double_list=[True, True, False, False],
        kernel_size=[3,3,3,3]
    )

    e3vae = EquivariantVAE3D(
        input_channels=1,
        irrep_order=0,
        hidden_cnn_channels=[16, 32, 64, 128],
        hidden_pooling=[True, True, True, False],
        kernel_size=[3,3,3,3]
    )

    print("single_e3CNN_module:")
    print(model)

    print("EquivariantEncoder3D_double_cnn:")
    print(model_enc)

    print("single_e3TransposeCNN_module:")
    print(tmodel)

    print("EquivariantDecoder3D:")
    print(model_dec)

    # encoder test
    x = torch.randn(1, 1, 32, 32, 32)  # (B, C, D, H, W)
    out = model(x)                             # single_e3CNN_module test
    out_enc_mu, out_enc_logvar = model_enc(x)  # EquivariantEncoder3D_double_cnn test 
    
    # decoder test
    tx = torch.randn(1, 1, 16, 16, 16)  # small feature map
    tout = tmodel(tx)
    out_dec = model_dec(tx)

    # VAE test
    print("Testing VAE forward pass with random input...")
    x = torch.randn(12, 1, 26, 32, 202)
    x_recon, mu, logvar = e3vae(x)



    test(model=e3vae)

    # x = x.to('cuda')
    # e3vae = e3vae.to('cuda')
    # x_recon, mu, logvar = e3vae(x)

    # visualization (sanity check)
    # test(model=model_enc)