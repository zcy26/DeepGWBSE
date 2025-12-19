import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, Delaunay
from scipy.interpolate import griddata
from .interface import wfn
import torch.nn.functional as F
from scipy.ndimage import zoom
import time
import torch
from scipy.interpolate import LinearNDInterpolator
from .data import ManyBodyData, ToyDataSet
from torch.utils.data import DataLoader
# from collect_tool import check_flows_status
from .trainer import Trainer
from .transformer import MBformerEncoder, MBformer
from .basisassembly import ElectronHoleBasisAssembly_Concatenate, sort_exciton_eigenvalues_by_eh_pair_energy, b1b2_grid, PassBasisAssembly
from .wfnembedder import ManyBodyData_WFN_Embedder_pretrained, SimpleSumXYEmbedder, E2VAEEmbedder
from enum import Enum
from .e2vae import EquivariantVAE
# from torchmetrics.regression import MeanAbsoluteError
from sklearn.metrics import mean_absolute_error, r2_score
from functools import partial
import os
import shutil


class BSEPredictTask(Enum):
    eigenvalues = 1
    eigenvectors = 2
    dipole = 3

class BSETransformerTrainer(Trainer):
    def __init__(self, model, loss ,optimizer, model_name="bse_transformer", use_Enc_Dec = False,
                task:BSEPredictTask=BSEPredictTask.eigenvalues, additional_metrics=None, **kwargs):
        
        if additional_metrics is not None:
            assert hasattr(self, "get_additional_loss"), "get_additional_loss function not implemented"
            # additional_metrics = additional_metrics.to(self.device)

        super().__init__(model, optimizer, loss, model_name=model_name, additional_metrics=additional_metrics,**kwargs)   

        self.use_Enc_Dec = use_Enc_Dec
        self.task = task
        print(f"Trainer Task: {task}")


    def get_loss(self, input:list)->torch.Tensor:
        """
        input: [ele, hole, eigenvalues, eigenvectors]
               see bse_collate_fn_encoder_only for details
        """

        if self.use_Enc_Dec:
            ele_src, ele, hole, eigenvalues, dipole, eigenvectors = input
            ele_src = [x.to(self.device) for x in ele_src]
        else:
            ele, hole, eigenvalues, dipole, eigenvectors = input
        ele = [x.to(self.device) for x in ele]
        hole = [x.to(self.device) for x in hole]
        eigenvalues = eigenvalues.to(self.device)
        dipole = dipole.to(self.device)
        eigenvectors = eigenvectors.to(self.device)
        kcv_prod = np.prod(eigenvalues.shape[-3:])
        kcv = eigenvalues.shape[-3:]

        if self.use_Enc_Dec:
            self.value, self.atten = self.model([ele, hole], [ele_src])
        else:
            self.value, self.atten = self.model([ele, hole])

        if self.task == BSEPredictTask.eigenvalues:
            # eigenvalues has shape of (batch, nS, 1), here we reorder the eigenvalues based on electron-hole pair energy
            self.eigenvalues_sorted_by_eh_pair_energy, _, _ = sort_exciton_eigenvalues_by_eh_pair_energy(ele, hole, eigenvalues, None, None)
            assert self.value.shape == self.eigenvalues_sorted_by_eh_pair_energy.shape, f"value.shape: {self.value.shape}, eigenvalues_sorted_by_eh_pair_energy.shape: {eigenvalues_sorted_by_eh_pair_energy.shape}. Make sure [ele, hole] order right"
            return self.loss(self.value, self.eigenvalues_sorted_by_eh_pair_energy)
        
        elif self.task == BSEPredictTask.eigenvectors:
            _, self.eigenvectors_sorted_by_eh_pair_energy, _ = sort_exciton_eigenvalues_by_eh_pair_energy(ele, hole, eigenvalues, eigenvectors, None) 
            assert self.atten.shape == self.eigenvectors_sorted_by_eh_pair_energy.shape, f"atten.shape: {self.atten.shape}, eigenvectors_sorted_by_eh_pair_energy.shape: {eigenvectors_sorted_by_eh_pair_energy.shape}. Make sure [ele, hole] order right"
            
            # soft_atten = F.softmax(self.atten.reshape(kcv_prod, kcv_prod), dim=-1)
            # log_atten = (self.atten.reshape(kcv_prod, kcv_prod)).log() ###?????
            atten = (self.atten.reshape(kcv_prod, kcv_prod))
            target = self.eigenvectors_sorted_by_eh_pair_energy.reshape(kcv_prod, kcv_prod)
            return self.loss(atten, target)
    
        elif self.task == BSEPredictTask.dipole:
            _, _, self.dipole_sorted_by_eh_pair_energy = sort_exciton_eigenvalues_by_eh_pair_energy(ele, hole, eigenvalues, None, dipole)
            assert self.value.shape == self.dipole_sorted_by_eh_pair_energy.shape, f"value.shape: {self.value.shape}, dipole_sorted_by_eh_pair_energy.shape: {self.dipole_sorted_by_eh_pair_energy.shape}. Make sure [ele, hole] order right"
            # value =torch.nn.functional.log_softmax(self.value + 1e-7, dim=1)
            value = self.value
            return self.loss(value, self.dipole_sorted_by_eh_pair_energy)

        else:
            raise NotImplementedError("Task not implemented")


    def get_additional_loss(self)->float:
        """
        This function provide additional metrics for the model
        For efficiency, we only calculate the additional metrics from last round of get_loss (validation)
        """
        if self.additional_metrics is None:
            return ""
        else:
            if self.task == BSEPredictTask.eigenvalues:
                return self.additional_metrics(self.value.ravel(), self.eigenvalues_sorted_by_eh_pair_energy.ravel())
            elif self.task == BSEPredictTask.eigenvectors:
                return self.additional_metrics(self.atten.ravel(), self.eigenvectors_sorted_by_eh_pair_energy.ravel())
                # raise NotImplementedError("Only support eigenvalues additional metrics task for now")
            elif self.task == BSEPredictTask.dipole:
                return self.additional_metrics(self.value.ravel(), self.dipole_sorted_by_eh_pair_energy.ravel())
            else:
                raise NotImplementedError("Task not implemented")


    @torch.no_grad()
    def evaluate(self, input=None, **kwargs):
        """
        input:
        - None: use the validation dataloader
        - data_labled: [ele, hole, eigenvalues, eigenvectors]
        - data_unlabled: [ele, hole]
        """
        self.model.eval()

        if input is None:
            assert self.validation_dataloader is not None, "Must have a non-empty input"
            for data in self.validation_dataloader:
                if self.use_Enc_Dec:
                    ele_src, ele, hole, _, _ = data
                else:
                    ele, hole, _, _, _ = data
                break  # By default, get only one batch
        elif isinstance(input,list) or isinstance(input, tuple):
            assert len(input)>=2, f"Input should be larger than 2, but got {len(input)}"
            if self.use_Enc_Dec:
                ele_src, ele, hole = input[:3]
                ele_src = [x.to(self.device) for x in ele_src]
            else:   
                ele, hole = input[:2]

        else:
            raise NotImplementedError("Input type not implemented")
    
        ele = [x.to(self.device) for x in ele]
        hole = [x.to(self.device) for x in hole]

        if self.use_Enc_Dec:
            value, atten = self.model([ele, hole], [ele_src])
        else:
            value, atten = self.model([ele, hole])

        if self.task == BSEPredictTask.eigenvalues:
            return value.cpu().numpy()
        elif self.task == BSEPredictTask.eigenvectors:
            return atten.cpu().numpy()
        elif self.task == BSEPredictTask.dipole:
            return value.cpu().numpy()
        else:
            raise NotImplementedError("Task not implemented")


def bse_collate_fn_encoder_only(batch):
    """
    input: 
        1. Old Version: only {'src', 'label'}
            {'src', 'label'}
           'src': see data/ManyBodyData.py "WFN" datapoint
           'laebel': {'eigenvalues', 'eigenvectors'}
        2. New Version: {'tgt', 'label'}
            {'tgt', 'label'}
           'tgt': see data/ManyBodyData.py "WFN" datapoint
           'laebel': {'eigenvalues', 'eigenvectors'}
    output: ele, hole, eigenvalues, eigenvectors
        ele: [latent, kpt, band_indices, el]
        hole: [latent, kpt, band_indices, el]
        eigenvalues: [nS, 1]
        dipole: [nS, 1]
        eigenvectors: [nS, nk, nc, nv]
    """
    # electron and hole partition
    # assert len(batch[0]) == 2, f"Batch should have 2 elements, but got {len(batch[0])}"
    if 'tgt' not in batch[0]:
        print("Old Version: using 'src' as input")
        src, label = batch[0]['src'], batch[0]['label'] # Old Version
    else:
        src, label = batch[0]['tgt'], batch[0]['label'] # New Version
    # print(src, label)

    nk = src['band_indices'].shape[0]
    nc = sum(src['band_indices'][0]>0)[0]
    nv = sum(src['band_indices'][0]<0)[0]

    # for key, value in src.items():
    ele_partition = np.where(src['band_indices']>0, True, False).squeeze()
    hole_partition = np.where(src['band_indices']<0, True, False).squeeze()


    ele = [torch.from_numpy(src['latent'][ele_partition].reshape(1,nk, nc, -1)).float(),
           torch.from_numpy(src['kpt'][ele_partition].reshape(1,nk, nc, -1)).float(), 
           torch.from_numpy(src['band_indices'][ele_partition].reshape(1,nk, nc, -1)).int(), 
           torch.from_numpy(src['el'][ele_partition].reshape(1,nk, nc, -1)).float()]
    hole = [torch.from_numpy(src['latent'][hole_partition].reshape(1,nk, nv, -1)).float(),
           torch.from_numpy(src['kpt'][hole_partition].reshape(1,nk, nv, -1)).float(), 
           torch.from_numpy(src['band_indices'][hole_partition].reshape(1,nk, nv, -1)).int(), 
           torch.from_numpy(src['el'][hole_partition].reshape(1,nk, nv, -1)).float()]

    eigenvalues = (torch.from_numpy(label['eigenvalues']).float())[None,...]

    if 'dipole_squared' in label: # dipole is not always present, especially for old dataset
        dipole = (torch.from_numpy(label['dipole_squared']).float()).reshape(*eigenvalues.shape)
    else:
        dipole = torch.ones_like(eigenvalues)  # Fallback if dipole_squared is not present
    
    eigenvectors = (torch.from_numpy(label['eigenvectors']).float())[None,...]

    # normalize eigenvectors
    # eigenvectors = eigenvectors / eigenvectors.amax(dim=(2, 3, 4), keepdim=True)
    # dipole = dipole / dipole.sum(dim=(1, 2), keepdim=True) # this is not a distribution, don't normalize it to one
    dipole = dipole / dipole.amax(dim=(1, 2), keepdim=True) # normalize dipole to max value
    eigenvectors = eigenvectors / eigenvectors.sum(axis=(2,3,4), keepdim=True)

    # eigenvectors = torch.log(eigenvectors + 1e-7)  # Avoid log(0)

    assert ele[0].shape[1] == nk, f"ele[0].shape[1]: {ele[0].shape[1]}, nk: {nk}"
    assert ele[0].shape[2] == nc, f"ele[0].shape[2]: {ele[0].shape[2]}, nc: {nc}"
    assert hole[0].shape[1] == nk, f"hole[0].shape[1]: {hole[0].shape[1]}, nk: {nk}"
    assert hole[0].shape[2] == nv, f"hole[0].shape[2]: {hole[0].shape[2]}, nv: {nv}"
    assert eigenvectors.shape[2] == nk, f"eigenvectors.shape[2]: {eigenvectors.shape[2]}, nk: {nk}"
    assert eigenvectors.shape[3] == nc, f"eigenvectors.shape[3]: {eigenvectors.shape[3]}, nc: {nc}"
    assert eigenvectors.shape[4] == nv, f"eigenvectors.shape[4]: {eigenvectors.shape[4]}, nv: {nv}"

    return ele, hole, eigenvalues, dipole, eigenvectors

def bse_collate_fn_encoder_decoder(batch):
    """
    This function wraps the bse_collate_fn_encoder_only function
    to generate both src and tgt for the encoder-decoder model.
    input: 
        {'src', 'tgt', 'label'}
        'src'/'tgt': see data/ManyBodyData.py "WFN" datapoint
        'laebel': {'eigenvalues', 'eigenvectors'}
    output: ele, hole, eigenvalues, eigenvectors
        ele: [latent, kpt, band_indices, el]
        hole: [latent, kpt, band_indices, el]
        eigenvalues: [nS, 1]
        dipole: [nS, 1]
        eigenvectors: [nS, nk, nc, nv]
    """
    assert len(batch[0]) == 3
    ele, hole, eigenvalues, dipole, eigenvectors = bse_collate_fn_encoder_only(batch)

    src = batch[0]['src']
    nk_src = src['band_indices'].shape[0]
    nb_src = sum(src['band_indices'][0]>-float('inf'))[0]
    ele_src_partition = np.where(src['band_indices']>-float('inf'), True, False).squeeze()
    ele_src = [torch.from_numpy(src['latent'][ele_src_partition].reshape(1,nk_src, nb_src, -1)).float(),
                torch.from_numpy(src['kpt'][ele_src_partition].reshape(1,nk_src, nb_src, -1)).float(),
                torch.from_numpy(src['band_indices'][ele_src_partition].reshape(1,nk_src, nb_src, -1)).int(),
                torch.from_numpy(src['el'][ele_src_partition].reshape(1,nk_src, nb_src, -1)).float()]

    assert ele_src[0].shape[1] == nk_src, f"ele_src[0].shape[1]: {ele_src[0].shape[1]}, nk_src: {nk_src}"
    assert ele_src[0].shape[2] == nb_src, f"ele_src[0].shape[2]: {ele_src[0].shape[2]}, nb_src: {nb_src}"

    return ele_src, ele, hole, eigenvalues, dipole, eigenvectors


class bse_training_manager_for_deepgwbse_paper:
    """
    Note: This class is designed to generate data and figure for deep-gwbse paper
    functionality:
     - train
     - evaluate and plot figure xx
    """
    def __init__(self, d_model:int, model_name:str, task:BSEPredictTask
                 , BasisAssembly=ElectronHoleBasisAssembly_Concatenate,
                 BasisAssembly_Encoder=PassBasisAssembly, use_Enc_Dec = False,
                 optimizer=torch.optim.Adam, loss=torch.nn.MSELoss, lr=1e-3,
                 **kwargs):
        self.d_model = d_model
        self.task = task

        self.use_Enc_Dec = use_Enc_Dec
        if not self.use_Enc_Dec:
            self.model = MBformerEncoder(d_input=d_model,
                                        d_model=d_model*2, 
                                        BasisAssembly=BasisAssembly,
                                        **kwargs)
        else:
            self.model = MBformer(d_input_src=d_model,
                                     d_input_tgt=d_model,
                                     d_model=d_model*2,
                                     BasisAssembly=BasisAssembly,
                                     BasisAssembly_Encoder=BasisAssembly_Encoder,
                                     **kwargs)

        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.loss = loss()
        self.additional_metrics = partial(torch.nn.functional.l1_loss, reduction='mean')
        # TODO
        self.trainer = BSETransformerTrainer(self.model, self.loss, self.optimizer,
                                            model_name=model_name, use_Enc_Dec=self.use_Enc_Dec,
                                            task=self.task,
                                            additional_metrics=self.additional_metrics,
                                            save_path=kwargs.get('save_path', None))
        self.trainer.load_model(load_best=True)

    def load_data(self, dataset_dir:str, dataset_fname:str,  dataset_latent_fname_suffix:str,
                 eb:ManyBodyData_WFN_Embedder_pretrained, data_slice:slice=None, train_val_split:float=0.2,**kwargs):

        self.dataset_dir = dataset_dir
        self.dataset_fname = dataset_fname
        self.dataset_latent_fname = dataset_fname.split('.')[0] + dataset_latent_fname_suffix

        if not os.path.exists(os.path.join(self.dataset_dir, self.dataset_latent_fname)):
            print(f"latent dataset not found, creating new one")
            # create latent_dataset
            self.bsedata = ManyBodyData.from_existing_dataset(os.path.join(self.dataset_dir, self.dataset_fname))
            self.bsedata = eb.create_latent_for_ManyBodyData_h5(self.bsedata, dataset_dir=self.dataset_dir, dataset_fname=self.dataset_latent_fname)
        else:
            print(f"latent dataset found, using {os.path.join(self.dataset_dir, self.dataset_latent_fname)}")
            # directly read the latent_dataset
            self.bsedata = ManyBodyData.from_existing_dataset(os.path.join(self.dataset_dir, self.dataset_latent_fname), data_slice=data_slice)

        self.bsedata_train = self.bsedata[:int(len(self.bsedata)*train_val_split)]
        self.bsedata_val = self.bsedata[int(len(self.bsedata)*train_val_split):]

        self.dataloader_train = DataLoader(self.bsedata_train, batch_size=1, collate_fn=bse_collate_fn_encoder_only if not self.use_Enc_Dec else bse_collate_fn_encoder_decoder)
        self.dataloader_val = DataLoader(self.bsedata_val, batch_size=1, collate_fn=bse_collate_fn_encoder_only if not self.use_Enc_Dec else bse_collate_fn_encoder_decoder)

        return self.dataloader_train, self.dataloader_val

    def train(self, num_epoches:int, continued:bool=True):
        self.trainer.load_model(True)
        self.trainer.train(num_epoches, self.dataloader_train, self.dataloader_val, continued=continued)

    def evaluate_dataset(self, dataloader=None, save_path:str=None):
        # TODO
        assert dataloader is not None, "dataloader is None, please load data first"
        loss = 0
        r2 = 0
        eval_original = torch.tensor([])
        eval_pred = np.array([])
        self.trainer.load_model(load_best=True)
        for d in dataloader:
            if self.use_Enc_Dec:
                ele_src, ele, hole, eigenvalues, dipole, eigenvectors = d
            else:
                ele, hole, eigenvalues, dipole, eigenvectors = d
            eigval_sort, eigvec_sort, dipole = sort_exciton_eigenvalues_by_eh_pair_energy(ele, hole, eigenvalues, None, dipole) 
            # break

            if self.trainer.task == BSEPredictTask.eigenvalues:    
                original = eigval_sort
            elif self.trainer.task == BSEPredictTask.dipole:
                original = dipole
            else:
                raise NotImplementedError("Task not implemented")

            loss += mean_absolute_error(self.trainer.evaluate(d).ravel(), original.ravel())
            r2 += r2_score(self.trainer.evaluate(d).ravel(), original.ravel())

            eval_original = torch.cat((eval_original, original.ravel()))
            eval_pred = np.concatenate((eval_pred, self.trainer.evaluate(d).ravel()), axis=0)

            print('\n')
            print("eigenval:", mean_absolute_error(self.trainer.evaluate(d).ravel(), original.ravel()), 'eV')
            print('r2:', r2_score(self.trainer.evaluate(d).ravel(), original.ravel()))

        print('\nMAE:', loss/len(dataloader))
        print('R2:', r2/len(dataloader))
        if save_path is not None:
            np.savetxt(os.path.join(save_path, 'data1_bse_original.dat'), eval_original)
            np.savetxt(os.path.join(save_path, 'data2_bse_pred.dat'), eval_pred)
            plt.scatter(eval_original, eval_pred)
            plt.savefig(os.path.join(save_path, 'bse_original_vs_pred.png'))
        else:
            np.savetxt('data1_bse_original.dat', eval_original)
            np.savetxt('data2_bse_pred.dat', eval_pred)
            plt.scatter(eval_original, eval_pred)
            plt.savefig('bse_original_vs_pred.png')


if __name__ == "__main__":  
    
    vae = Trainer.configure_model(EquivariantVAE, "./vae_e2_wfn.save")
    eb = ManyBodyData_WFN_Embedder_pretrained(48, E2VAEEmbedder, model=vae, model_name='vae_e2_wfn', 
                                              save_path='../vae_e2_wfn.save')

    dataset_kwargs = {
        'dataset_dir': './dataset',
        'dataset_fname': 'dataset_BSE.h5',
        'dataset_latent_fname_suffix': '_latent_vae.h5',
    }


    """Training for BSE eigenvalues with pretrained VAE"""
    bse = bse_training_manager_for_deepgwbse_paper(d_model=48, model_name='bse_transformer_eval_vae_test', task=BSEPredictTask.eigenvalues,
                                            BasisAssembly=ElectronHoleBasisAssembly_Concatenate, use_Enc_Dec=True,
                                            optimizer=torch.optim.Adam, loss=torch.nn.MSELoss, lr=1e-3)


    bse.load_data(eb=eb, data_slice=None, train_val_split=0.5, **dataset_kwargs)
    bse.train(5, continued=False)

    _, dataloader_val = bse.load_data(eb=eb, data_slice=slice(-20,-1), train_val_split=0.5, **dataset_kwargs)
    bse.evaluate_dataset(dataloader_val)

    shutil.rmtree("bse_transformer_eval_vae_test.save")

    """Training for BSE eigenvectors with pretrained VAE"""
    bse = bse_training_manager_for_deepgwbse_paper(d_model=48, model_name='bse_transformer_evec_vae_test', task=BSEPredictTask.eigenvectors,
                                            BasisAssembly=ElectronHoleBasisAssembly_Concatenate,
                                            optimizer=torch.optim.Adam, loss=torch.nn.CrossEntropyLoss, lr=1e-3)
    bse.load_data(eb=eb, data_slice=None, train_val_split=0.5, **dataset_kwargs)
    bse.train(5, continued=False)

    _, dataloader_val = bse.load_data(eb=eb, data_slice=slice(-20,-1), train_val_split=0.5, **dataset_kwargs)
    # bse.evaluate_dataset(dataloader_val)

    shutil.rmtree("bse_transformer_evec_vae_test.save")

    """Training for BSE dipole with pretrained VAE"""
    bse = bse_training_manager_for_deepgwbse_paper(d_model=48, model_name='bse_transformer_dipole_vae_test', task=BSEPredictTask.dipole,
                                            BasisAssembly=ElectronHoleBasisAssembly_Concatenate,
                                            optimizer=torch.optim.Adam, loss=torch.nn.MSELoss, lr=1e-3)
    bse.load_data(eb=eb, data_slice=None, train_val_split=0.5, **dataset_kwargs)
    bse.train(5, continued=False)

    _, dataloader_val = bse.load_data(eb=eb, data_slice=slice(-20,-1), train_val_split=0.5, **dataset_kwargs)
    bse.evaluate_dataset(dataloader_val)

    shutil.rmtree("bse_transformer_dipole_vae_test.save")