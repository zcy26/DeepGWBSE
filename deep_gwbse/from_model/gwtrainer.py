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
from enum import Enum
from sklearn.metrics import mean_absolute_error, r2_score
from .wfnembedder import ManyBodyData_WFN_Embedder_pretrained, SimpleSumXYEmbedder, E2VAEEmbedder
from functools import partial
import os
from .e2vae import EquivariantVAE


class GWPredictTask(Enum):
    G0W0_energy = 1
    updated_wavefunction = 2
    self_GW_energy = 3

class GWTransformerTrainer(Trainer):
    def __init__(self, model, loss ,optimizer, model_name="gw_transformer", task:GWPredictTask=GWPredictTask.G0W0_energy ,additional_metrics=None, **kwargs):
        if additional_metrics is not None:
            assert hasattr(self, "get_additional_loss"), "get_additional_loss function not implemented"
            # additional_metrics = additional_metrics.to(self.device)
        self.task = task
        super().__init__(model, optimizer, loss, model_name=model_name, additional_metrics=additional_metrics,**kwargs)  
    def get_loss(self, input:list)->torch.Tensor:
        """
        input: [src_data, tgt_data, corr]

        """
        src_data, tgt_data, corr, src_mask, tgt_mask = input
        src_data = [x.to(self.device) for x in src_data]
        tgt_data = [x.to(self.device) for x in tgt_data]
        corr = corr.to(self.device)
        src_mask = src_mask.to(self.device)
        tgt_mask = tgt_mask.to(self.device)
        value, _ = self.model([tgt_data],[src_data], tgt_mask=tgt_mask, src_mask=src_mask)
        self.value = value
        if self.task == GWPredictTask.G0W0_energy:
            self.corr = corr
            assert self.value.shape == corr.shape, f"Value shape {self.value.shape} does not match corr shape {corr.shape}"
            raw_loss = self.loss(self.value[tgt_mask], corr[tgt_mask])
            # masked = raw_loss * tgt_mask.unsqueeze(-1)
            self.addtion_mask = tgt_mask
            # loss = masked.sum() / tgt_mask.sum()
            loss = raw_loss
            return loss
        elif self.task == GWPredictTask.updated_wavefunction:
            raise NotImplementedError("Task not implemented")
        elif self.task == GWPredictTask.self_GW_energy:
            raise NotImplementedError("Task not implemented")
        else:
            raise NotImplementedError("Task not implemented")
    
    @torch.no_grad()
    def evaluate(self, input=None, **kwargs):
        self.model.eval()

        if input is None:
            assert self.validation_dataloader is not None, "Must have a non-empty input"
            for data in self.validation_dataloader:
                src_data, tgt_data, corr, src_mask, tgt_mask = data
                break  # By default, get only one batch
        elif isinstance(input,list) or isinstance(input, tuple):
            # assert len(input) == 3 or len(input) == 3, f"Input must be a list of two or three elements, but got {len(input)}"
            src_data, tgt_data, corr, src_mask, tgt_mask = input


        src_data = [x.to(self.device) for x in src_data]
        tgt_data = [x.to(self.device) for x in tgt_data]
        src_mask = src_mask.to(self.device)
        tgt_mask = tgt_mask.to(self.device)
        value, atten = self.model([tgt_data],[src_data], tgt_mask=tgt_mask, src_mask=src_mask)

        if self.task == GWPredictTask.G0W0_energy:
            return value.cpu().numpy() 
        else:
            raise NotImplementedError("Task not implemented")
        
    @torch.no_grad()
    def evaluate_dataset(self, dataloader,file_name_pred=None, file_name_original=None):
        loss = 0
        r2 = 0
        MAE_loss = 0
        eval_original = np.array([])
        eval_pred = np.array([])
        self.load_model(load_best=True)
        for d in dataloader:
            src, tgt, corr, src_mask, tgt_mask = d
            corr = corr.to(self.device)
            src_data = [x.to(self.device) for x in src]
            tgt_data = [x.to(self.device) for x in tgt]
            src_mask = src_mask.to(self.device)
            tgt_mask = tgt_mask.to(self.device)
            value, atten = self.model([tgt_data],[src_data], tgt_mask=tgt_mask, src_mask=src_mask)
            # value = value.cpu().numpy()
            # corr = corr.cpu().numpy()
            # value = value * tgt_mask.unsqueeze(-1).cpu().numpy()
            # corr = corr * tgt_mask.unsqueeze(-1).cpu().numpy()
            eval_original = np.append(eval_original, corr[tgt_mask].cpu().numpy().ravel())
            eval_pred = np.append(eval_pred, value[tgt_mask].cpu().numpy().ravel())
            # loss += mean_absolute_error(value[tgt_mask.cpu().numpy()].ravel(), corr[tgt_mask.cpu().numpy()].ravel())
            # r2 += r2_score(value[tgt_mask.cpu().numpy()].ravel(), corr[tgt_mask.cpu().numpy()].ravel())
            loss_local = self.loss(value[tgt_mask], corr[tgt_mask])
            MAE_loss_local = self.additional_metrics(value[tgt_mask], corr[tgt_mask])
            loss += loss_local.item()
            MAE_loss += MAE_loss_local.item()



            print('\n')
            print("loss_local:", loss_local.item())
            print("MAE_loss_local:", MAE_loss_local.item())
            # print("eigenval:", mean_absolute_error(value[tgt_mask.cpu().numpy()].ravel(), corr[tgt_mask.cpu().numpy()].ravel()), 'eV')
            # print('r2:', r2_score(value[tgt_mask.cpu().numpy()].ravel(), corr[tgt_mask.cpu().numpy()].ravel()))
            # print('MAE:', mean_absolute_error(value[tgt_mask.cpu().numpy()].ravel(), corr[tgt_mask.cpu().numpy()].ravel()), 'eV')
        # print('\nMAE:', loss/len(dataloader))
        # print('R2:', r2/len(dataloader))
        print('\nMAE:', MAE_loss/len(dataloader))
        print('Loss:', loss/len(dataloader))

        np.savetxt(file_name_original, eval_original)
        np.savetxt(file_name_pred, eval_pred)

    def get_additional_loss(self)->float:
        """
        This function provide additional metrics for the model
        For efficiency, we only calculate the additional metrics from last round of get_loss (validation)
        """
        if self.additional_metrics is None:
            return ""
        else:
            if self.task == GWPredictTask.G0W0_energy:
                # value = self.value * self.addtion_mask.unsqueeze(-1)
                # corr = self.corr * self.addtion_mask.unsqueeze(-1)
                return self.additional_metrics(self.value[self.addtion_mask], self.corr[self.addtion_mask])
            else:
                raise NotImplementedError("Task not implemented")

def _pad(t, target_nk, target_nb, pad_value=1):
    """
    Pad a 3-D tensor (nk, nb, d) to (target_nk, target_nb, d).
    """
    nk, nb, d = t.shape
    # pad sizes are given as (pad_last_dim_left, pad_last_dim_right, …, pad_first_dim_right)
    pad = (0, 0,                         # d  (no padding in last dim)
           0, target_nb - nb,            # nb
           0, target_nk - nk)            # nk
    return torch.nn.functional.pad(t, pad, value=pad_value)

def gw_collate_fn(batch):
    """
    Pads every sample in the batch to the largest nk / nband and
    returns masks so the model can ignore padding.
    """

    B = len(batch)                         # minibatch size

    #largest nk and nband for src / tgt in minibatch
    max_nk       = max(item['src']['band_indices'].shape[0] for item in batch)
    max_nb_src   = max(item['src']['band_indices'].shape[1] for item in batch)
    max_nb_tgt   = max(item['tgt']['band_indices'].shape[1] for item in batch)

    # fixed size
    d_latent = batch[0]['src']['latent'].shape[-1]
    d_kpt    = batch[0]['src']['kpt'   ].shape[-1]

    src_lat   = torch.zeros(B, max_nk, max_nb_src, d_latent, dtype=torch.float32)
    src_kpt   = torch.zeros(B, max_nk, max_nb_src, d_kpt   , dtype=torch.float32)
    src_band  = torch.zeros(B, max_nk, max_nb_src, 1       , dtype=torch.int32 )
    src_el    = torch.zeros(B, max_nk, max_nb_src, 1       , dtype=torch.float32)
    src_mask  = torch.zeros(B, max_nk, max_nb_src, dtype=torch.bool)  # 1 = real token

    tgt_lat   = torch.zeros(B, max_nk, max_nb_tgt, d_latent, dtype=torch.float32)
    tgt_kpt   = torch.zeros(B, max_nk, max_nb_tgt, d_kpt   , dtype=torch.float32)
    tgt_band  = torch.zeros(B, max_nk, max_nb_tgt, 1       , dtype=torch.int32 )
    tgt_el    = torch.zeros(B, max_nk, max_nb_tgt, 1       , dtype=torch.float32)
    tgt_mask  = torch.zeros(B, max_nk, max_nb_tgt, dtype=torch.bool)

    corr      = torch.zeros(B, max_nk, max_nb_tgt, 1, dtype=torch.float32)

    for i, item in enumerate(batch):
        src, tgt, label = item['src'], item['tgt'], item['label']

        nk_i, nb_src_i = src['band_indices'].shape[:2]
        nb_tgt_i       = tgt['band_indices'].shape[1]

        #src
        src_lat [i] = _pad(torch.as_tensor(src['latent'      ]), max_nk, max_nb_src)
        src_kpt [i] = _pad(torch.as_tensor(src['kpt'         ]), max_nk, max_nb_src)
        src_band[i] = _pad(torch.as_tensor(src['band_indices']), max_nk, max_nb_src)
        src_el  [i] = _pad(torch.as_tensor(src['el'          ]), max_nk, max_nb_src)
        src_mask[i, :nk_i, :nb_src_i] = True

        #tgt
        tgt_lat [i] = _pad(torch.as_tensor(tgt['latent'      ]), max_nk, max_nb_tgt)
        tgt_kpt [i] = _pad(torch.as_tensor(tgt['kpt'         ]), max_nk, max_nb_tgt)
        tgt_band[i] = _pad(torch.as_tensor(tgt['band_indices']), max_nk, max_nb_tgt)
        tgt_el  [i] = _pad(torch.as_tensor(tgt['el'          ]), max_nk, max_nb_tgt)
        tgt_mask[i, :nk_i, :nb_tgt_i] = True

        #corr
        corr    [i] = _pad(torch.as_tensor(label['corr']), max_nk, max_nb_tgt)

    src_data = [src_lat, src_kpt, src_band, src_el]
    tgt_data = [tgt_lat, tgt_kpt, tgt_band, tgt_el]

    return src_data, tgt_data, corr, src_mask, tgt_mask



if __name__ == "__main__":
    d_model = 48
    num_epoches = 150
    train_val_split = 0.95
    config_model_path = "./gw_transformer_sigma.save"
    dataset_dir = './all_dataset'
    dataset_fname =  'dataset_GW_0_1000_ncs1nvs1'
    #dataset_fname =  'dataset_GW_hbn_aug.h5'
    dataset_latent_fname = dataset_fname.split('.')[0] + '_latent.h5'

    if not os.path.exists(os.path.join(dataset_dir, dataset_latent_fname)):
        print(f"latent dataset not found, creating new one")
        # create latent_dataset
        gwdata = ManyBodyData.from_existing_dataset(os.path.join(dataset_dir, dataset_fname))
        # eb = ManyBodyData_WFN_Embedder_pretrained(d_model, SimpleSumXYEmbedder)
        vae = Trainer.configure_model(EquivariantVAE, "./vae_e2_wfn.save")
        eb = ManyBodyData_WFN_Embedder_pretrained(48, E2VAEEmbedder, model=vae, model_name='vae_e2_wfn', save_path='./vae_e2_wfn.save')
        gwdata = eb.create_latent_for_ManyBodyData_h5(gwdata, dataset_dir=dataset_dir, dataset_fname=dataset_latent_fname)

    else:
        print(f"latent dataset found, using {os.path.join(dataset_dir, dataset_latent_fname)}")
        # directly read the latent_dataset
        gwdata = ManyBodyData.from_existing_dataset(os.path.join(dataset_dir, dataset_latent_fname))


    gwdata_train = gwdata[:int(len(gwdata)*train_val_split)]
    gwdata_val = gwdata[int(len(gwdata)*train_val_split):]
    # gwdata_train = gwdata[70:]
    # gwdata_val = gwdata[0:70]
    dataloader_train = DataLoader(gwdata_train, batch_size=35, collate_fn=gw_collate_fn, shuffle=True, num_workers=os.cpu_count(), pin_memory=True, persistent_workers=True)
    dataloader_val = DataLoader(gwdata_val, batch_size=35, collate_fn=gw_collate_fn, shuffle=True, num_workers=os.cpu_count(), pin_memory=True, persistent_workers=True)
    
    # if os.path.exists(config_model_path):
    #     print("Loading model from", config_model_path)
    #     enc2 = Trainer.configure_model(MBformer, config_model_path)
    # else:
    enc2 = MBformer(d_input_src=d_model,d_input_tgt=d_model, d_model=d_model, activation="gelu", num_encoder_layers=2, num_decoder_layers=2)
    
    optimizer = torch.optim.Adam(enc2.parameters(), lr=1e-5)
    #lr_scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoches)
    lr_scheduler = None
    loss = torch.nn.MSELoss()
    additional_metrics = partial(torch.nn.functional.l1_loss, reduction='mean')

    gw_trainer_sigma = GWTransformerTrainer(enc2, loss, optimizer,  
                                                model_name="gw_transformer_sigma", 
                                                task=GWPredictTask.G0W0_energy,
                                                additional_metrics=additional_metrics, scheduler=lr_scheduler)    
    gw_trainer_sigma.load_model(True)
    gw_trainer_sigma.train(num_epoches, dataloader_train, dataloader_val, continued=True)


    dataloader_val2 = DataLoader(gwdata_val, batch_size=1, collate_fn=gw_collate_fn, shuffle=False, num_workers=os.cpu_count(), pin_memory=True, persistent_workers=True)
    print("validation")
    gw_trainer_sigma.evaluate_dataset(dataloader_val2, file_name_pred="data_val_pred.dat", file_name_original="data_val_orginal.dat")
    print("train")
    gw_trainer_sigma.evaluate_dataset(dataloader_train, file_name_pred="data_train_pred.dat", file_name_original="data_train_orginal.dat")

    print('Done')

    # loss = 0
    # gw_trainer_sigma.load_model(load_best=True)

    # for d in dataloader_val:
    #     src, tgt, corr, src_mask, tgt_mask = d
    #     loss += mean_absolute_error(gw_trainer_sigma.evaluate(d).ravel(), corr.ravel())

    #     print("sigma:", mean_absolute_error(gw_trainer_sigma.evaluate(d).ravel(), corr.ravel()), 'eV')

    # print('MAE:', loss/len(dataloader_val))
