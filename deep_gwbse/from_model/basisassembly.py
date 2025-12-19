import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, TensorDataset
from .data import ToyDataSet as d
import torch
import torch.nn as nn
import numpy as np


def b1b2_grid(nb1, nb2):
    """
    Generate a grid of indices for basis 1 and basis 2
    example:
        we consider two conduction bands (nc=2) and three valence bands (nv=3):
        c_basis = np.array([[1,1],[2,2]])
        v_basis = np.array([[-1,-1],[-2,-2],[-3,-3]])

        # |cv> = |c> x |v>
        bb1, bb2 = b1b2_grid(2, 3)
        bb1 = np.array([[0,1],[0,1],[0,1]])
        bb2 = np.array([[0,0],[1,1],[2,2]])
        
        c_basis[bb1,:].shape 
        => (3,2,2) # 
        c_basis[bb1,:][0] == c_basis[bb1,:][1] == c_basis[bb1,:][2] # axis=0 is repeated
        => True

        v_basis[bb2,:].shape
        => (3,2,2)
        v_basis[bb2,:][:,0] == v_basis[bb2,:][:,1] # axis=1 is repeated
        => True

        assemble = np.einsum('vci, vcj -> vcij', c_basis[bb1,:], v_basis[bb2,:]).reshape(*bb1.shape,-1) # shape=(3,2,4)

        assert assemble[2,0] == c_basis[0] tensor_product v_basis[2]

        # The above example shows how to generate the new basis by tensor product
        # (nk1, d) tensor_product (nk2, d) = (nk2, nk1, ...)
        # see ElectronHoleBasisAssembly for einsum
    """
    b1 = np.arange(nb1)
    b2 = np.arange(nb2)
    bb1, bb2 = np.meshgrid(b1, b2)
    return bb1, bb2


def sort_exciton_eigenvalues_by_eh_pair_energy(ele:list, hole:list, eigenvalues:torch.Tensor, eigenvectors:torch.Tensor=None,
                                               dipole:torch.Tensor=None) -> tuple:
    """
    Sort eigenvalues by eh pair energy
    make sure the shape and energy order of transformer output matches with the exciton eigenvalues
    Input:
        assert batch == 1
        ele[2]: [batch, kpt, nc, 1] # ele energy (ele[2]>0)
        hole[2]: [batch, kpt, nv, 1] # hole energy (hole[2]<0)
        eigenvalues: [batch, nS, 1]
        eigenvectors: [batch, nS, nk, nc, nv] # This is optional, if None, we only sort eigenvalues
        dipole: [batch, nS, 1]
    Output:
        eigenvalues_sorted_by_eh_pair_energy: [batch, (kpt, nv, nc), 1], # (kpt, nv, nc) is 'nS' sorted by eh pair energy
        eigenvectors_sorted_by_eh_pair_energy: [batch, (kpt, nv, nc), nk, nv, nc] # (kpt, nv, nc) is 'nS' sorted by eh pair energy
        dipole_sorted_by_eh_pair_energy: [batch, (kpt, nv, nc), 1] # (kpt, nv, nc) is 'nS' sorted by eh pair energy
    """
    assert ele[0].shape[0] == 1, f"ele[0].shape[0]: {ele[0].shape[0]}, only support batch size 1"
    assert hole[0].shape[0] == 1, f"hole[0].shape[0]: {hole[0].shape[0]}, only support batch size 1"
    nk = ele[1].shape[-3]
    nc = ele[1].shape[-2]
    nv = hole[1].shape[-2]
    b1, b2 = b1b2_grid(nc, nv) # [ele, hole]
    ele_energy = ele[3][...,b1,:]
    hole_energy = hole[3][...,b2,:]
    eh_pair_energy = ele_energy - hole_energy 
    assert (eh_pair_energy > 0).all(), "eh_pair_energy should be positive, input order: ele, hole"  
    eh_pair_energy_shape = eh_pair_energy.shape
    eh_pair_energy_indices = torch.argsort(eh_pair_energy.flatten())

    # sort eigenvalues by eh_pair_energy
    eigenvalues_sorted_by_eh_pair_energy = torch.zeros_like(eigenvalues.flatten())
    eigenvalues_sorted_by_eh_pair_energy[eh_pair_energy_indices] = eigenvalues.flatten()
    eigenvalues_sorted_by_eh_pair_energy = eigenvalues_sorted_by_eh_pair_energy.reshape(eh_pair_energy_shape)

    eigenvectors_sorted_by_eh_pair_energy = None
    dipole_sorted_by_eh_pair_energy = None

    if eigenvectors is not None:
        # return eigenvalues_sorted_by_eh_pair_energy, None
        kcv_shape = eigenvectors.shape[-3:]
        eigenvectors_sorted_by_eh_pair_energy = torch.zeros((nk*nc*nv, *kcv_shape), device=eigenvectors.device)
        eigenvectors_sorted_by_eh_pair_energy[eh_pair_energy_indices,:] = eigenvectors[0]
        eigenvectors_sorted_by_eh_pair_energy = eigenvectors_sorted_by_eh_pair_energy.reshape(*eh_pair_energy_shape[:4], *kcv_shape)
        eigenvectors_sorted_by_eh_pair_energy = eigenvectors_sorted_by_eh_pair_energy.permute(0,1,2,3,4,6,5)
    
    if dipole is not None:
        dipole_sorted_by_eh_pair_energy = torch.zeros_like(dipole.flatten())
        dipole_sorted_by_eh_pair_energy[eh_pair_energy_indices] = dipole.flatten()
        dipole_sorted_by_eh_pair_energy = dipole_sorted_by_eh_pair_energy.reshape(eh_pair_energy_shape)

    return eigenvalues_sorted_by_eh_pair_energy, eigenvectors_sorted_by_eh_pair_energy, dipole_sorted_by_eh_pair_energy

class PassBasisAssembly(nn.Module):
    nbasis = 1
    def __init__(self,):
        super().__init__()
    def get_dinput_from_dmodel(self, d_model):
        return d_model
    def forward(self,x):
        return x
        
class ElectronHoleBasisAssembly_TensorProduct(nn.Module):
    nbasis = 2
    finite_momentum = False
    def __init__(self,):
        """
        Tensorproduct basis assembly: |cvk> = |ck> x |vk>
        input: x1, x2 with shape (batch, nk, nband, d_model)
        output: x with shape (batch, nk, x2_band, x1_band, d_model^2)
        """
        super().__init__()
    def get_dinput_from_dmodel(self, d_model):
        d_input = np.sqrt(d_model)
        assert d_input.is_integer(), "The dimension of the input should be a square number"
        return int(d_input)

    def forward(self, x1, x2):
        assert x1.shape[-3] == x2.shape[-3], "The number of k-points should be the same"
        d_model1 = x1.shape[-1]
        d_model2 = x2.shape[-1]
        assert d_model1 == d_model2, "The dimension of the model should be the same"
        bb1, bb2 = b1b2_grid(x1.shape[-2], x2.shape[-2])
        cvk = torch.einsum("bkcvi, bkcvj -> bkcvij" , x1[..., bb1, :], x2[..., bb2, :])
        return cvk.view(*cvk.shape[:-2], -1)

class ElectronHoleBasisAssembly_Concatenate(nn.Module):
    nbasis = 2
    finite_momentum = False
    def __init__(self,):
        """
        Tensorproduct basis assembly: |cvk> = |ck> + |vk>
        input: x1, x2 with shape (batch, nk, nband, d_model)
        output: x with shape (batch, nk, x2_band, x1_band, d_model*2)
        """
        super().__init__()
    
    def get_dinput_from_dmodel(self, d_model):
        d_input = d_model // 2
        assert d_input * 2 == d_model, "The dimension of the input should be a even number"
        return d_input

    def forward(self, x1, x2):
        assert x1.shape[-3] == x2.shape[-3], "The number of k-points should be the same"
        bb1, bb2 = b1b2_grid(x1.shape[-2], x2.shape[-2])
        cvk = torch.cat([x1[..., bb1, :], 
                      x2[..., bb2, :]], dim=-1)

        return cvk


