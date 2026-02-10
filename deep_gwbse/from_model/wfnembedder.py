from abc import ABC, abstractmethod
from .data import ManyBodyData
import numpy as np
from .model_util import H5ls
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os
import torch
from os.path import join as pjoin
from .vaetrainer import wfn_collate_fn, WFNVAETrainer
from .e2vae import EquivariantVAE
from .trainer import Trainer

# class ManyBodyData_WFN_Embedder_pretrained:
#     """
#     This Embedder Class only support static pretrained embedder embedder
#     i.e. it cannot be integrated downstream model for training.
#     """
#     def __init__(self, latent_dim, latent_embedder , **kwargs):
#         """
#         Initialize the WFNEmbedder with any necessary parameters.
#         latent_dim (int): The dimension of the latent space.
#         latent_embedder (LatentEmbedderBASE): The latent embedder class.
#         kwargs:
#             ...
#         """
#         self.latent_embedder = latent_embedder(latent_dim, **kwargs)

#         pass
#     def create_latent_for_ManyBodyData(self, manybodydata: ManyBodyData, del_wfn_original=False)->ManyBodyData:
#         """
#         Abstract method to perform wavefunction embedding from ManyBodyData.
#         Args:
#             manybodydata (ManyBodyData): The ManyBodyData object, see ManyBodyData.py "WFN" datapoint for details
#                 ...
#                 wfn_datapoint = {...}
#                 ...
#             del_wfn_original (bool): Whether to delete the original WFN data (nk, nc, nx, ny, nz) after embedding.
                
#         Returns:
#             manybodydata (ManyBodyData): Modified ManyBodyData object.
#                 ...
#                 wfn_datapoint = {..., 'latent': (nk, nc_wfn+nv_wfn, latent) ,...}
#                 ...
#         Note: support `WFN`, `GW`, and `BSE`
#         """
#         assert manybodydata.info.dataset_type in ['WFN', 'GW', 'BSE'], "only support dataset of `WFN`, `GW`, and `BSE`"

#         # This function will add a "latent" key to each wfn_data point and this is IN-PLACE!
#         update_wfn_data = lambda wfn_data: wfn_data.update({'latent': self.latent_embedder.embed(wfn_data['wfn'])}) 
#         update_src_data = lambda data: data['src'].update({'latent': self.latent_embedder.embed(data['src']['wfn'])}) 
#         update_tgt_data = lambda data: data['tgt'].update({'latent': self.latent_embedder.embed(data['tgt']['wfn'])})

#         if manybodydata.info.dataset_type == 'WFN':
#             list(map(update_wfn_data, tqdm(manybodydata, desc='Embedding WFN'))) 
#             if del_wfn_original:
#                 list(map(lambda data: data.pop('wfn'), manybodydata))

#         elif manybodydata.info.dataset_type == 'GW':
#             list(map(update_src_data, tqdm(manybodydata, desc='Embedding GW src WFN')))
#             list(map(update_tgt_data, tqdm(manybodydata, desc='Embedding GW tgt WFN')))
#             if del_wfn_original:
#                 list(map(lambda data: data['src'].pop('wfn'), manybodydata))
#                 list(map(lambda data: data['tgt'].pop('wfn'), manybodydata))

#         elif manybodydata.info.dataset_type == 'BSE':
#             list(map(update_src_data, tqdm(manybodydata, desc='Embedding BSE src WFN')))   
#             if del_wfn_original:
#                 list(map(lambda data: data['src'].pop('wfn'), manybodydata)) 

#         else:
#             raise NotImplementedError("Task not implemented")

#         return manybodydata

#     def create_latent_for_ManyBodyData_h5(self, h5file, **kwargs)->ManyBodyData:
#         """
#         Abstract method to perform wavefunction embedding from HDF5 file.
#         Args:
#             h5file (str): Path to the HDF5 file.
#         Returns:
#         """
#         raise NotImplementedError

class ManyBodyData_WFN_Embedder_pretrained:
    def __init__(self, latent_dim, latent_embedder, **kwargs):
        self.latent_embedder = latent_embedder(latent_dim, **kwargs)
        self.del_wfn_original = False
        self.on_cuda = False
        if hasattr(self.latent_embedder, 'vaetrainer'):
            self.on_cuda = True
    
    def _embed_wfn(self, wfn_data):
        """Helper function to embed wavefunction data."""
        wfn_data['latent'] = self.latent_embedder.embed(wfn_data['wfn'])
        if self.del_wfn_original:
            wfn_data.pop('wfn', None)
        return wfn_data

    def _embed_src(self, data):
        """Helper function to embed source wavefunction."""
        data['src']['latent'] = self.latent_embedder.embed(data['src']['wfn'])
        if self.del_wfn_original:
            data['src'].pop('wfn', None)
        return data

    def _embed_tgt(self, data):
        """Helper function to embed target wavefunction."""
        data['tgt']['latent'] = self.latent_embedder.embed(data['tgt']['wfn'])
        if self.del_wfn_original:
            data['tgt'].pop('wfn', None)
        return data
    
    def create_latent_for_ManyBodyData(self, manybodydata, del_wfn_original=False)->ManyBodyData:
        """
        manybodydata -> manybodydata (latent created)
        """
        self.del_wfn_original = del_wfn_original
        assert manybodydata.info.dataset_type in ['WFN', 'GW', 'BSE'], "Only support dataset of `WFN`, `GW`, and `BSE`"
        
        if manybodydata.info.dataset_type == 'WFN':
            if not self.on_cuda:
                with Pool(processes=32) as pool:
                    manybodydata = list(tqdm(pool.imap(self._embed_wfn, manybodydata), total=len(manybodydata), desc='Embedding WFN'))
            else:
                manybodydata = list(tqdm(map(self._embed_wfn, manybodydata), total=len(manybodydata), desc='Embedding WFN'))

        elif manybodydata.info.dataset_type == 'GW':
            if not self.on_cuda:
                with Pool(processes=32) as pool:
                    manybodydata = list(tqdm(pool.imap(self._embed_src, manybodydata), total=len(manybodydata), desc='Embedding GW src WFN'))
                    manybodydata = list(tqdm(pool.imap(self._embed_tgt, manybodydata), total=len(manybodydata), desc='Embedding GW tgt WFN'))
            else:  
                manybodydata = list(tqdm(map(self._embed_src, manybodydata), total=len(manybodydata), desc='Embedding GW src WFN'))
                manybodydata = list(tqdm(map(self._embed_tgt, manybodydata), total=len(manybodydata), desc='Embedding GW tgt WFN'))

        elif manybodydata.info.dataset_type == 'BSE':
            print('debug:', len(manybodydata[0]))
            if len(manybodydata[0]) == 2 and 'src' in manybodydata[0]:
                # deprecated warning
                print("Warning: Old BSE data format detected, using 'src' key for embedding.")
                if not self.on_cuda:
                    with Pool(processes=32) as pool:
                        manybodydata = list(tqdm(pool.imap(self._embed_src, manybodydata), total=len(manybodydata), desc='Embedding BSE src WFN'))
                else:
                    manybodydata = list(tqdm(map(self._embed_src, manybodydata), total=len(manybodydata), desc='Embedding BSE src WFN'))
            elif len(manybodydata[0]) == 2 and 'tgt' in manybodydata[0]:
                if not self.on_cuda:
                    with Pool(processes=32) as pool:
                        manybodydata = list(tqdm(pool.imap(self._embed_tgt, manybodydata), total=len(manybodydata), desc='Embedding BSE tgt WFN'))
                else:
                    manybodydata = list(tqdm(map(self._embed_tgt, manybodydata), total=len(manybodydata), desc='Embedding BSE tgt WFN'))
            elif len(manybodydata[0]) == 3:
                if not self.on_cuda:
                    with Pool(processes=32) as pool:
                        manybodydata = list(tqdm(pool.imap(self._embed_src, manybodydata), total=len(manybodydata), desc='Embedding BSE src WFN'))
                        manybodydata = list(tqdm(pool.imap(self._embed_tgt, manybodydata), total=len(manybodydata), desc='Embedding BSE tgt WFN'))
                else:  
                    manybodydata = list(tqdm(map(self._embed_src, manybodydata), total=len(manybodydata), desc='Embedding BSE src WFN'))
                    manybodydata = list(tqdm(map(self._embed_tgt, manybodydata), total=len(manybodydata), desc='Embedding BSE tgt WFN'))                

        return manybodydata

    def create_latent_for_ManyBodyData_h5(self, manybodydata:ManyBodyData, dataset_dir:str='./', dataset_fname:str='./latent_mbdata.h5'):
        """
        manybodydata -> manybodydata, manybody_h5file (latent replacing wfn)
        """
        info = manybodydata.info
        self.info = info
        info.latent_created = True
        manybodydata = self.create_latent_for_ManyBodyData(manybodydata, del_wfn_original=True)

        # Create a new HDF5 file with the updated data
        ManyBodyData.init_dataset_h5(dataset_dir, dataset_fname, info, multiprocessing=False)

        # info.mats_id has the same order as manybodydata
        # see data.py ManyBodyData.load_dataset, ManyBodyData.mat_statistics, and ManyBodyData.process for details
        for i, mat_id in enumerate(info.mat_id):
            ManyBodyData.datapoint_interface_h5(pjoin(dataset_dir, dataset_fname), mat_id, manybodydata[i], mode='a')
        
        return manybodydata

class ManyBodyData_WFN_Embedder_trainable:
    """
    You should put this to your model!
    """
    def __init__(self): 
        raise NotImplementedError

class LatentEmbedderBASE(ABC):
    """
    Abstract base class for latent embedding methods.
    """
    @abstractmethod
    def embed(self, wfn_data: np.ndarray) -> np.ndarray:
        """
        Abstract method to perform embedding on the input data.
        Args:
            wfn_data: The input WFN data to be embedded. each batch is a material
                (nk, nc_wfn+nv_wfn, nx, ny, nz)
        Returns:
            representation of wfn_data.
                (nk, nc_wfn+nv_wfn, latent_dim)
        """
        pass

# Here is a simple example!
class SimpleSumXYEmbedder(LatentEmbedderBASE):
    def __init__(self, latent_dim, **kwargs):
        self.latent_dim = latent_dim
        self.kwargs = kwargs

    def embed(self, wfn_data: np.ndarray) -> np.ndarray:
        """
        This is just an example, we don't consider efficiency and physical meaning at all!
        input: wfn_data, see LateEmbedderBASE
        return: latent, see LateEmbedderBASE
        """
        wfn_data = np.copy(wfn_data) # don't change the original data
        assert len(wfn_data.shape) == 5, f"len(wfn_data.shape): {len(wfn_data.shape)}"
        nk, nc_nv, nx, ny, nz = wfn_data.shape
        # mid = np.argmax(np.nan_to_num(wfn_data,0).sum(axis=(0,1,2,3)))
        mid = nz //2
        left, right = mid - self.latent_dim//2, mid + self.latent_dim//2
        right = right if self.latent_dim%2 == 0 else right + 1
        assert left >= 0, f"left: {left}, nz: {nz}"
        assert right <= nz, f"right: {right}, nz: {nz}"
        # return wfn_data[..., left:right].sum(axis=(2,3))
        wfn_data = np.nan_to_num(wfn_data, 0)
        extracted = wfn_data[..., left:right].sum(axis=(2, 3))
        # norm_factor = np.sum(wfn_data) / np.sum(extracted) if np.sum(extracted) != 0 else 1
        # return np.ones((nk, nc_nv, self.latent_dim))
        return extracted / extracted.sum(axis=2, keepdims=True)



class E2VAEEmbedder(LatentEmbedderBASE):
    def __init__(self, latent_dim, model, model_name, save_path, **kwargs):
        """
        model: instantiazed
        """
        # self.latent_dim = latent_dim
        print("E2VAEEmbedder doesn't support customized latent_dim, it depends on the model")
        self.kwargs = kwargs
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        self.vaetrainer = WFNVAETrainer(model, 
                                        optimizer, # don't matter for evaluation
                                        0.02, # doesn't matter for evaluation 
                                        save_path=save_path,
                                        model_name=model_name, 
                                        **kwargs)
        # self.vaetrainer.device = torch.device('cpu')
        self.vaetrainer.load_model(load_best=True)
        self.vaetrainer.model.eval()

    @torch.no_grad()
    def embed(self, wfn_data: np.ndarray) -> np.ndarray:
        nk, nb, X, Y, C = wfn_data.shape
        batch = [{'wfn':wfn_data}] # since wfn_collate_fn only support batch size 1
        wfn_data, mask = wfn_collate_fn(batch)
        wfn_data = wfn_data.to(self.vaetrainer.device)
        mask = ~mask.to(self.vaetrainer.device)
        wfn_recon, mu, logvar = self.vaetrainer.model(wfn_data)

        # GOP layer
        mu = mu.mean(dim=(2,3))
        # normalize dim=1
        mu = mu / mu.sum(dim=1, keepdim=True)

        self.mu = mu
        self.wfn_data = wfn_data
        self.wfn_recon = wfn_recon
        self.mask = mask

        mu = mu.detach().cpu().numpy()

        return mu.reshape(nk, nb, -1) # (nk, nb, latent_dim)


class OtherEmbedder(LatentEmbedderBASE):
    def __init__(self, latent_dim, **kwargs):
        pass
    def embed(self, wfn_data: np.ndarray) -> np.ndarray:
        pass



if __name__ == "__main__":
    wfdata = ManyBodyData.from_existing_dataset('./dataset/dataset_WFN.h5')
    gwdata = ManyBodyData.from_existing_dataset('./dataset/dataset_GW.h5')
    bsedata = ManyBodyData.from_existing_dataset('./dataset/dataset_BSE.h5')

    eb = ManyBodyData_WFN_Embedder_pretrained(24, SimpleSumXYEmbedder)
    eb.create_latent_for_ManyBodyData_h5(wfdata, dataset_dir='./dataset', dataset_fname='dataset_WFN_latent.h5')
    eb.create_latent_for_ManyBodyData_h5(gwdata, dataset_dir='./dataset', dataset_fname='dataset_GW_latent.h5')
    eb.create_latent_for_ManyBodyData_h5(bsedata, dataset_dir='./dataset', dataset_fname='dataset_BSE_latent.h5')

    wfdata = eb.create_latent_for_ManyBodyData(wfdata, del_wfn_original=True)
    gwdata = eb.create_latent_for_ManyBodyData(gwdata, del_wfn_original=True)
    bsedata = eb.create_latent_for_ManyBodyData(bsedata, del_wfn_original=True)

    wfdata_h5 = ManyBodyData.from_existing_dataset('./dataset/dataset_WFN_latent.h5')
    gwdata_h5 = ManyBodyData.from_existing_dataset('./dataset/dataset_GW_latent.h5')
    bsedata_h5 = ManyBodyData.from_existing_dataset('./dataset/dataset_BSE_latent.h5')

    " unit test "
    assert np.allclose(gwdata[0]['src']['latent'],  wfdata[0]['latent'])
    assert abs(gwdata[0]['tgt']['latent'].sum() -  4) < 1e-6
    assert np.allclose(gwdata[0]['src']['latent'],  gwdata_h5[0]['src']['latent'])
    assert np.allclose(gwdata[0]['tgt']['latent'],  gwdata_h5[0]['tgt']['latent'])
    assert np.allclose(bsedata[0]['src']['latent'],  bsedata_h5[0]['src']['latent'])
    assert np.allclose(wfdata[0]['latent'],  wfdata_h5[0]['latent'])

    print('basic unit test passed!')

    """ unit test for E2VAEEmbedder """
    if os.path.exists('./vae_e2_wfn'+'.save'):
        # wfdata = ManyBodyData.from_existing_dataset('./dataset/1000_wfn_1/dataset_WFN_1000.h5')
        wfdata = ManyBodyData.from_existing_dataset('./dataset/dataset_WFN.h5')
        wfn_data = wfdata[0]['wfn']
        vae = Trainer.configure_model(EquivariantVAE, './vae_e2_wfn.save')
        vae_eb = E2VAEEmbedder(24,vae,"vae_e2_wfn",'./vae_e2_wfn.save')
        mu = vae_eb.embed(wfn_data)
        mu = vae_eb.mu
        mask = vae_eb.mask
        wfn_data = vae_eb.wfn_data
        wfn_recon = vae_eb.wfn_recon
        mask = ~mask
        mask_nan = torch.where(mask, np.nan, 1.0)
        wfn_recon = mask_nan * wfn_recon
        wfn_data = mask_nan * wfn_data
        wfn_data = wfn_data.cpu().numpy()
        wfn_recon = wfn_recon.cpu().numpy()
        mask = mask.cpu().numpy()
        plt.figure()
        plt.imshow(wfn_data[0].sum(0))
        plt.figure()
        plt.imshow(wfn_recon[0].sum(0))
        # assert mu.shape == ()

        # usage of E2VAEEmbedder
        vae = Trainer.configure_model(EquivariantVAE, "./vae_e2_wfn.save")
        eb = ManyBodyData_WFN_Embedder_pretrained(24, E2VAEEmbedder, model=vae, model_name='vae_e2_wfn', save_path='./vae_e2_wfn.save')
        eb.create_latent_for_ManyBodyData_h5(wfdata, dataset_dir='./dataset', dataset_fname='dataset_WFN_latent_vae.h5')

    else:
        print("vae_e2_wfn not found, skip E2VAEEmbedder test")


