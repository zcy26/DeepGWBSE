import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, TensorDataset
from .data import ToyDataSet as d
import torch
import torch.nn as nn
import numpy as np


"""
This PositionalEmbedding is modified from APET:
paper:  https://pubs.acs.org/doi/10.1021/acs.jpclett.3c02036
        https://arxiv.org/abs/2411.16483 
github: https://github.com/emotionor/APET/tree/main
"""
##############################################################################################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, base: int = 10000, dim: int = 3):
        """
        :param d_model: state embedding size (d_model)
        base: sin/cos base frequency. Empirically, 10000 is a good choice. But more tests are needed.
        """
        super().__init__()
        self.org_d_model = d_model
        self.dim = dim
        assert dim in [1, 2, 3], "dim should be 1, 2 or 3"
        if dim == 1:
            assert d_model%2==0
            d_model = int(np.ceil(d_model / 2))
        elif dim == 2:
            assert d_model%4==0
            d_model = int(np.ceil(d_model / 4))
        elif dim == 3:
            assert d_model%6==0
            d_model = int(np.ceil(d_model / 6))

        self.inv_freq_sin = 1.0 / (base ** (torch.arange(0, d_model, 1).float() / d_model)).to(dtype=torch.float32)
        self.inv_freq_cos = 1.0 / (base ** ((torch.arange(0, d_model, 1).float()+1)/d_model)).to(dtype=torch.float32)
        self.d_model = d_model
        
    def forward(self, pos):
        """
        :param tensor: A 5d tensor of size (batch_size, nk, nb, dim), dim is dimension of the position
        :return: Positional Encoding Matrix of size (batch_size, nk, nb, d_model)
        """
        batch_size, nk, nb, dim = pos.shape
        assert dim == self.dim, f"dim should be {self.dim}, but got {dim}"

        self.inv_freq_cos = self.inv_freq_cos.to(pos.device)
        self.inv_freq_sin = self.inv_freq_sin.to(pos.device)

        if self.dim == 1:
            pos = pos[:,:,:,0]*2*np.pi
            sin_inp = torch.einsum("ijl,k->ijlk", pos, self.inv_freq_sin)
            cos_inp = torch.einsum("ijl,k->ijlk", pos, self.inv_freq_cos)
            emb = torch.stack((sin_inp.sin(), cos_inp.cos()), dim=-1)
        elif self.dim == 2:
            pos_x = pos[:,:,:,0]*2*np.pi
            pos_y = pos[:,:,:,1]*2*np.pi
            sin_inp_x = torch.einsum("ijl,k->ijlk", pos_x, self.inv_freq_sin)
            sin_inp_y = torch.einsum("ijl,k->ijlk", pos_y, self.inv_freq_sin)
            cos_inp_x = torch.einsum("ijl,k->ijlk", pos_x, self.inv_freq_cos)
            cos_inp_y = torch.einsum("ijl,k->ijlk", pos_y, self.inv_freq_cos)
            emb = torch.stack((sin_inp_x.sin(), cos_inp_x.cos(),
                            sin_inp_y.sin(), cos_inp_y.cos()), dim=-1)
        elif self.dim == 3:
            pos_x = pos[:,:,:,0]*2*np.pi
            pos_y = pos[:,:,:,1]*2*np.pi
            pos_z = pos[:,:,:,2]*2*np.pi
            sin_inp_x = torch.einsum("ijl,k->ijlk", pos_x, self.inv_freq_sin)
            sin_inp_y = torch.einsum("ijl,k->ijlk", pos_y, self.inv_freq_sin)
            sin_inp_z = torch.einsum("ijl,k->ijlk", pos_z, self.inv_freq_sin)
            cos_inp_x = torch.einsum("ijl,k->ijlk", pos_x, self.inv_freq_cos)
            cos_inp_y = torch.einsum("ijl,k->ijlk", pos_y, self.inv_freq_cos)
            cos_inp_z = torch.einsum("ijl,k->ijlk", pos_z, self.inv_freq_cos)
            emb = torch.stack((sin_inp_x.sin(), cos_inp_x.cos(),
                            sin_inp_y.sin(), cos_inp_y.cos(),
                            sin_inp_z.sin(), cos_inp_z.cos()), dim=-1)

        return torch.flatten(emb, -2, -1)
##############################################################################################################

class BandPositionalEmbeddings(nn.Module):
    """
    learnable positional embeddings for bands
    Conduction index [1, 2, 3, ...]
    Valence index [-1, -2, -3, ...]
    Here we use two embeddings to represent the position of condution and valence bands respectively
    Note: 0 never appears in the band index
    """
    def __init__(self, d_model: int, max_len: int):
        """
        max_len: maximum band index for conduction or valence bands, not total number of bands
        The total number of bands is 2*max_len
        """
        super().__init__()
        self.positional_embeddings_pos = nn.Embedding(num_embeddings=max_len, embedding_dim=d_model)  # overwrite this line
        self.positional_embeddings_neg = nn.Embedding(num_embeddings=max_len, embedding_dim=d_model)  # overwrite this line

        self.d_model = d_model

    def forward(self, pos: Tensor) -> Tensor:
        """
        input: pos: (batch_size, nk, nb, 1)
        return:     (batch_size, nk, nb, d_model)
        """
        assert (pos != 0).all(), "Band index should not contain 0"

        batch_size, nk, nb, _ = pos.shape
        pos = pos.squeeze(-1).reshape(batch_size*nk, nb) # we treat batch_size*nk as a new batch_size, nb is the sequence length

        # Treat positive and negative indices separately
        if (pos > 0).all():
            res = self.positional_embeddings_pos(pos)
        elif (pos < 0).all():
            res =  self.positional_embeddings_neg(-pos)
        else:
            positive_mask = pos > 0
            negative_mask = pos < 0
            pos_emb = self.positional_embeddings_pos(pos[positive_mask])
            neg_emb = self.positional_embeddings_neg(-pos[negative_mask])
            # create a new tensor with the same shape as pos
            emb = torch.zeros((pos.shape[0], pos.shape[1], self.d_model), device=pos.device)
            emb[positive_mask] = pos_emb
            emb[negative_mask] = neg_emb
            res = emb
        
        res = res.reshape(batch_size, nk, nb, self.d_model)
        return res

class PositionalEmbeddings_band_energy_kpt(nn.Module):
    """
    learnable positional embeddings for bands
    """
    def __init__(self, d_model: int, max_band: int, kpt_dim: int=3, base_kpt: int=10000, base_energy: int=10000):
        super().__init__()

        assert d_model % (kpt_dim + 1) == 0, "d_model should be divisible by (kpt_dim + 1)"

        d_model_kpt = int(d_model / (kpt_dim + 1) * kpt_dim)
        d_model_band = int(d_model / (kpt_dim + 1))

        self.kpt_pos_emb = PositionalEncoding(d_model_kpt, dim=kpt_dim, base=base_kpt)
        self.energy_pose_emb = PositionalEncoding(d_model_band, dim=1, base=base_energy)
        self.band_pos_emb = BandPositionalEmbeddings(d_model_band, max_band)

    def forward(self, kpt_pos: Tensor, band_pos: Tensor, energy_pos: Tensor) -> Tensor:
        """
        input pos : (batch_size, nk, nb, 1), (batch_size, nk, nb, 2) or (batch_size, nk, nb, 3)
        return:     (batch_size, nk, nb, d_model)       
        """
        # concatenate all the embeddings
        kpt_emb = self.kpt_pos_emb(kpt_pos)
        energy_emb = self.energy_pose_emb(energy_pos)
        band_emb = self.band_pos_emb(band_pos)
        return torch.cat((kpt_emb, energy_emb + band_emb), dim=-1)

class KWeightSampling(nn.Module):
    """
    KWeightSampling is used to sample k-points based on the weights
    """
    # TODO


if __name__ == "__main__":
    # unittest
    kpt_emb = PositionalEncoding(d.d_model, dim=3)
    band_emb = BandPositionalEmbeddings(d.d_model, d.nb_max)
    pos_emb_3Dkpt = PositionalEmbeddings_band_energy_kpt(d.d_model, d.nb_max, 3)
    pos_emb_2Dkpt = PositionalEmbeddings_band_energy_kpt(d.d_model, d.nb_max, 2)
    k_eb = kpt_emb( d.cond_kpt)
    b_eb = band_emb(d.cond_band_index)
    kb_eb_3d = pos_emb_3Dkpt(d.cond_kpt, d.cond_band_index, d.cond_band_energy)
    kb_eb_2d = pos_emb_2Dkpt(d.cond_kpt[...,:2], d.cond_band_index, d.cond_band_energy)
    print('d.d_model: ', d.d_model)
    print("d.cond_kpt.shape: ", d.cond_kpt.shape)
    print("d.cond_band_index.shape: ", d.cond_band_index.shape)
    print("d.cond_band_energy.shape: ", d.cond_band_energy.shape)
    print("shape of kpt embedding: ", k_eb.shape)
    print("shape of band embedding: ", b_eb.shape)
    print("shape of 3D kpt + band embedding: ", kb_eb_3d.shape)
    print("shape of 2D kpt + band embedding: ", kb_eb_2d.shape)

