import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from deep_gwbse.from_model.interface import wfn, eqp, AScvk
from deep_gwbse.from_model.model_util import time_watch, memory_watch
from pathos.multiprocessing import ProcessingPool as Pool
import os
from tqdm import tqdm
from os.path import join as pjoin
import h5py as h5
import logging
import numpy as np
import copy
from deep_gwbse.utils import check_flows_status
"""
Author:  Bowen Hou
Contact: bowen.hou@yale.edu
"""
class DataSetInfo:
    """
    Basic Info for the dataset
    dataset_type = WFN, GW, BSE

    This class saves all the hyperparameters (required and optional) for the dataset
    All the hyperparameters are saved in the __dict__ attribute and with assigned default values
    """
    def __init__(self, dataset_type: str='WFN', **kwargs):
        if isinstance(dataset_type, bytes):
            dataset_type = dataset_type.decode('utf-8')

        if dataset_type == 'WFN':
            self.dataset_type = 'WFN'
            self.wfn_base_set(**kwargs)

        if dataset_type == 'GW':
            self.dataset_type = 'GW'
            self.gw_base_set(**kwargs)

            # for src and tgt
            if kwargs.get('from_dft'):
                self.wfn_base_set(**kwargs)
            else:
                raise NotImplementedError("GW dataset from non-DFT is not implemented yet")
                self.vae_base_set(**kwargs)
     
        if dataset_type == 'BSE':
            self.dataset_type = 'BSE'
            self.bse_base_set(**kwargs)

            if kwargs.get('from_dft'):
                if not self.predict_only:
                    """
                    nc_bse and nv_bse are only used fro predict_only=True
                    Otherwise, they are automatically set to the number in AScvk
                    """
                    kwargs['nc_bse'] = kwargs.get('nc_bse', np.nan)
                    kwargs['nv_bse'] = kwargs.get('nv_bse', np.nan)
                self.wfn_base_set(**kwargs)
            else:
                raise NotImplementedError("BSE dataset from non-DFT is not implemented yet")
                self.vae_base_set(**kwargs)
                assert {"nc_wfn","nv_wfn"} <= set(kwargs.keys()), f"nc, nv are required kwargs for BSE dataset"
                self.nc_wfn = kwargs.get('nc_wfn')
                self.nv_wfn = kwargs.get('nv_wfn')
        
        # common attributes
        # TODO: rename it as mat_ids
        self.mat_id = kwargs.get('mat_id', []) # updated after processing
    
    def wfn_base_set(self, **kwargs):
        """
        see interface.py/wfn.get_dataset()
        """
        assert {"nc_wfn","nv_wfn"} <= set(kwargs.keys()), f"nc_wfn and nv_wfn are required kwargs for WFN dataset"
        self.nc_wfn = kwargs.get('nc_wfn')
        self.nv_wfn = kwargs.get('nv_wfn')
        self.useWignerXY = kwargs.get('useWignerXY', False)
        self.useWignerXYZ = kwargs.get('useWignerXYZ', False)
        self.cell_slab_truncation = kwargs.get('cell_slab_truncation', 40) # Required for Wigner
        self.AngstromPerPixel = kwargs.get('AngstromPerPixel', 0.1) # Required for Wigner
        self.AngstromPerPixel_z = kwargs.get('AngstromPerPixel_z', 0.1) # Required for Wigner
        self.upsampling_factor = kwargs.get('upsampling_factor', 1) # Required for Wigner

    def gw_base_set(self, **kwargs):
        assert {"nc_sigma","nv_sigma","nc_wfn","nv_wfn"} <= set(kwargs.keys()), f"nc, nv are required kwargs for GW dataset"
        self.nc_sigma = kwargs.get('nc_sigma')
        self.nv_sigma = kwargs.get('nv_sigma')      
        self.nc_wfn = kwargs.get('nc_wfn')
        self.nv_wfn = kwargs.get('nv_wfn')
        self.from_dft = kwargs.get('from_dft', True)
        self.predict_only = kwargs.get('predict_only', False)
    
    def bse_base_set(self, **kwargs):
        assert {"nc_wfn","nv_wfn"} <= set(kwargs.keys()), f"nc, nv are required kwargs for BSE dataset"
        self.from_dft = kwargs.get('from_dft', True)
        self.predict_only = kwargs.get('predict_only', False)
        self.nc_wfn = kwargs.get('nc_wfn')
        self.nv_wfn = kwargs.get('nv_wfn')
        if self.predict_only:
            assert {"nc_bse","nv_bse"} <= set(kwargs.keys()), f"nc, nv are required kwargs for BSE dataset"
            self.nc_bse = kwargs.get('nc_bse')
            self.nv_bse = kwargs.get('nv_bse')            

    def vae_base_set(self, **kwargs):
        pass

    def show_info(self,):
        print(f"\n======{str(self.dataset_type)} Dataset Info:=======")
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")
        print("Total number of data: ", len(self.mat_id),'\n\n')

class ManyBodyData(Dataset):
    """
    raw_data_dir(flows)/
    ├── mat-1
    |   ├──02-wfn
    |   ├──13-sigma
    |   |   └── eqp1.dat # (G0W0 corr.)
    |   | ...
    |   ├──17-wfn_fi
    |   ├──18-kernel 
    |   ├──19-absorption 
    ├── mat-2
    |   └──  ...
    └── ...
    
    dataset general format:
    dataset.h5
    ├── info: see DataSetInfo
    ├── mat-1/{datapoint1}
    ├── mat-2/{datapoint2}
    ├── mat-3/...
    """

    def __init__(self, flows_dir: str, dataset_dir: str, dataset_type: str='WFN',
                 dataset_fname: str='dataset.h5', multiprocessing: bool = False, load_dataset: bool = True, 
                 onlySave:bool=False, data_slice=None, operator = None, **kwargs):
        """
        :param **kwargs: all parameters related to specific dataset ['WFN','GW','BSE'], see DataSetInfo
        :param flows_dir: Path to the raw data directory (flows)
        :param dataset_dir: Path to the dataset directory
        :param dataset_type: Workflow to process data, support ['WFN', 'GW','BSE'] now.

            'WFN': used to train VAE model (unsupervised)
                -  required dir: '02-wfn'
                -  [optional] dir: 
                -  required kwargs: nc_wfn, nv_wfn
                -  [optional] kwargs : useWignerXY, cell_slab_truncation, AngstromPerPixel, AngstromPerPixel_z
                                     upsample_factor (This is highly recommened for fast_cK)
                -  datapoint (also see interface.py/wfn.get_dataset()):
                            {'wfn': (nk, nc_wfn+nv_wfn, nx, ny, nz), 
                             'kpt': (nk, nc_wfn+nv_wfn, 3), 
                             'occ': (nk, nc_wfn_nv_wfn, 1),
                             'el':  (nk, nc_wfn+nv_wfn, 1), 
                             'kpt_weights': (nk, nc_wfn+nv_wfn, 1), 
                             'kpt': (nk, nc_wfn+nv_wfn, 3),
                             'band_indices': (nk, nc_wfn+nv_wfn, 1), 
                             'band_indices_abs':(nk, nc_wfn+nv_wfn, 1)}

            'GW': used to train GW-Transformer (supervised)
                -  required dir: '02-wfn', 
                -  [optional] dir: '13-sigma', '05-band'[optional: predict_only]
                -  required kwargs: nc_wfn, nv_wfn, nc_sigma, nv_sigma 
                -  [optional] kwargs: from_dft: bool=True, # save wfn instead of VAE latent space
                                      predict_only:bool=False, 
                                      other parameters are same as WFN
                -  datapoint: 
                    from_dft:
                    - {'src':dict, 'tgt':dict, 'label':dict[optional: not created if predict_only]}
                    -  src: same as "WFN" datapoint
                    -  tgt: same as "WFN" datapoint, with nc_sigma, nv_sigma instead of nc_wfn, nv_wfn
                    -  label: {'mf': (nk, nc_sigma+nv_sigma, 1), 'qp': (nk, nc_sigma+nv_sigma, 1), 'corr': (nk, nc_sigma+nv_sigma, 1)}
                    not from_dft:
                    - {'src':dict, 'tgt':dict, 'label':dict[optional: not created if predict_only]}
                    - src: not implemented yet
                    - tgt: not implemented yet
                    - label: same as from_dft
                    
            'BSE': used to train BSE-Transformer (supervised)
                -  required dir: '17-wfn_fi',
                -  [optional] dir: '19-absorption'
                -  required kwargs: 
                -  [optional] kwargs: nc_wfn, nv_wfn 
                                        (only used if predict_only is True, otherwise, automatically set to the same shape of AScvk,
                                        see interface.py/bse.get_dataset() for more details)
                                      from_dft: bool=True, # save wfn instead of VAE latent space
                                      predict_only:bool=False, 
                                      other parameters are same as WFN
                - datapoint:
                   from_dft:
                   - {'src':dict, 'label':dict[optional: not created if predict_only]}
                   - src: same as "WFN" datapoint
                   - label: {'eigenvalue':(nS, 1), 'eigenvalues':(nS, nk, nc, nv)}
                            assert nS == nk * nc * nv

        :param dataset_name: name of the dataset
        :param multiprocessing: Whether to use multiprocessing to process data
        :param load_dataset: Whether to load existing dataset
        :param onlySave: Whether to only save the dataset without loading (used for creating large dataset)
        :param data_slice: slice of the data to load (only used for loading large dataset)
                           support format: slice, list, np.ndarray(int)

        Output: 
            self.data: [datapoint1, datapoint2, ...]
            self.info: DataSetInfo
            dataset.h5:
        """
        super(ManyBodyData, self).__init__()
        assert dataset_type in ['WFN','GW','BSE'], f"dataset_type should be one of ['WFN','GW','BSE']"

        self.multiprocessing = multiprocessing
        self.flows_dir = flows_dir
        self.dataset_dir = dataset_dir
        self.dataset_type = dataset_type
        self.dataset_fname = dataset_fname
        self.onlySave = onlySave
        self.load_large_dataset_inference = kwargs.get('load_large_dataset_inference', False)
        self.kwargs = kwargs
        self.operator = operator
        
        # dataset and hyperparameters
        # - required:
        self.data = None
        self.info = DataSetInfo(dataset_type=dataset_type, **kwargs)

        if load_dataset and os.path.exists(pjoin(dataset_dir, dataset_fname)):
            print(f"Loading existing dataset: {os.path.abspath(pjoin(dataset_dir, dataset_fname))}")
            self.load_dataset(data_slice=data_slice)
        else:
            if not os.path.exists(pjoin(dataset_dir, dataset_fname)):
                print(f"Dataset file not found: {os.path.abspath(pjoin(dataset_dir, dataset_fname))}")
            print(f"Creating new dataset: {os.path.abspath(pjoin(dataset_dir, dataset_fname))}")
            self.process()
        
        if not self.onlySave:
            assert self.data is not None, "Data is not loaded or processed"
        else:
            self.data = []
            print(f"Only saving dataset")
            
        self.info.show_info()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    @classmethod
    def from_existing_dataset(cls, existing_dataset_fname: str, data_slice:slice=None, **kwargs) -> 'ManyBodyData':
        assert os.path.exists(existing_dataset_fname), f"{existing_dataset_fname} does not exist"
        dataset_dir, dataset_fname = os.path.dirname(existing_dataset_fname), os.path.basename(existing_dataset_fname)
        print('Loading dataset info')
        info_dict = {}
        with h5.File(existing_dataset_fname, 'r') as f:
            for key, value in f['info'].items():
                info_dict[key] = value[()]

        dataset_type = info_dict.pop('dataset_type')
        info = DataSetInfo(dataset_type,
                           **info_dict)
        # info.show_info()
        return cls(flows_dir=None, 
                   dataset_dir=dataset_dir, 
                   dataset_type=info.dataset_type, 
                   load_dataset=True, 
                   dataset_fname=dataset_fname,
                   multiprocessing=False,
                   onlySave=False,
                   data_slice=data_slice,
                   **info_dict,
                   **kwargs)

    def load_dataset(self, data_slice=None):
        """
        load existing dataset
        data_slice: None, sclie, list, np.ndarray(int)
        """

        with h5.File(pjoin(self.dataset_dir, self.dataset_fname), 'r') as f:
            print("updating info from existing dataset")
            for key, value in f['info'].items():
                if key == 'dataset_type':
                    assert self.info.dataset_type == value[()].decode('utf-8'), f"Dataset type mismatch: set {self.info.dataset_type}, get {value[()].decode('utf-8')}"
                    self.info.__dict__[key] = value[()].decode('utf-8')
                    continue
                self.info.__dict__[key] = value[()]

        self.info.mat_id = self.info.mat_id[data_slice] if data_slice is not None else self.info.mat_id
        print("loading data")
        self.data = [self.datapoint_interface_h5(pjoin(self.dataset_dir, self.dataset_fname), mat_id, mode='r', load_large_dataset_inference=self.load_large_dataset_inference) for mat_id in self.info.mat_id]

        # print(f"Loading existing dataset: {os.path.abspath(self.data.filename)}")

    def process(self):
        """
        folder_list: List["flow-mat-1", "flow-mat-2"]
        """
        #==================General Setting==================#
        #====Filter valid folders===
        folder_list, self.info.mat_id = self.mat_statistics(self.flows_dir, self.dataset_type)

        #===initialize dataset h5 file===
        self.init_dataset_h5(dataset_dir=self.dataset_dir, dataset_fname=self.dataset_fname, info=self.info, multiprocessing=False)

        #==================Dataset Specific Setting==================#
        #===Get processor===
        if self.dataset_type == 'WFN':
            processor = self.process_worker_WFN
        elif self.dataset_type == 'GW':
            processor = self.process_worker_GW
        elif self.dataset_type == 'BSE':
            processor = self.process_worker_BSE
        
        #===Process data===

        def processor_return_None_wrapper(folder):
            processor(folder)
            return None

        if self.multiprocessing:
            with Pool(8) as pool: # It seems 32 or 16 works the best.
                if self.onlySave: # used to handle large dataset
                    list(tqdm(pool.imap(processor_return_None_wrapper, folder_list), total=len(folder_list), desc='Processing WFN data'))
                else:
                    self.data = list(tqdm(pool.imap(processor, folder_list), total=len(folder_list), desc='Processing WFN data'))
            self.merge_dataset_h5(list(map(lambda x: x.decode('utf-8'), self.info.mat_id)), save_original=False, dataset_fname=self.dataset_fname)
        else:
            if self.onlySave: # used to handle large dataset
                list(tqdm(map(processor_return_None_wrapper, folder_list), total=len(folder_list), desc='Processing WFN data'))
            else:
                self.data = [processor(folder) for folder in tqdm(folder_list, desc='Processing WFN data')]

    def mat_statistics(self, flows_dir:str, dataset_type:type='WFN')-> tuple[list, np.ndarray]:
        """
           use collect_tool.py/check_flows_status
        """
        folder_list = []
        flows_status = check_flows_status(flows_dir, False)

        for flow, status in flows_status.items():
            finished_tasks = set(status['Yes'].split(','))
            if dataset_type == 'WFN':
                if {'02-wfn'} <= finished_tasks:
                    folder_list.append(flow)

            elif dataset_type == 'GW':
                if not self.info.from_dft:
                    raise NotImplementedError
                else:
                    if not self.info.predict_only:
                        if {'02-wfn','13-sigma'} <= finished_tasks:
                            folder_list.append(flow)
                    else:
                        if {'02-wfn','05-band'} <= finished_tasks:
                            folder_list.append(flow)
                            
            elif dataset_type == 'BSE':
                if not self.info.from_dft:
                    raise NotImplementedError
                else:
                    if not self.info.predict_only:
                        if {'17-wfn_fi','19-absorption'} <= finished_tasks:
                            folder_list.append(flow)
                    else:
                        if {'17-wfn_fi'} <= finished_tasks:
                            folder_list.append(flow)
        
            else:
                raise Exception(f"Dataset type {dataset_type} is not supported")
            
        assert len(folder_list) > 0, f"No data found under {flows_dir}"
        print(f"Found {len(folder_list)} out of {len(flows_status)} materials for {dataset_type}")

        folder_list = sorted(folder_list)

        mat_ids = np.array([os.path.basename(folder) for folder in folder_list], dtype='S')

        return folder_list, mat_ids

    @staticmethod
    def init_dataset_h5(dataset_dir:str, dataset_fname:str, info:dict, multiprocessing: bool = False):
        """
        multiprocessing: 
            True: create dataset files for each material
                  h5: mat_id+dataset_fname
            False: create one dataset file for all materials
                  h5: dataset_fname
        """
        os.makedirs(dataset_dir, exist_ok=True)    
        if not multiprocessing:
            with h5.File(pjoin(dataset_dir, dataset_fname), 'w') as f:
                # put info dict into h5 file
                f.create_group('info')
                for key, value in info.__dict__.items():
                    # print('--debug:', key, value)
                    if value is None:
                        value = np.nan
                    f['info'].create_dataset(key, data=value)
                print(f"[Series]: creating dataset file: {os.path.abspath(f.filename)}")

        else:
            print(f"[Pool]: creating dataset files for {len(info.mat_id)} material")
            mat_id_list = list(map(lambda x: x.decode('utf-8'), info.mat_id))
            for mat_id in mat_id_list:
                with h5.File(pjoin(dataset_dir, mat_id+dataset_fname), 'w') as f:
                    # put info dict into h5 file
                    f.create_group('info')
                    for key, value in info.__dict__.items():
                        f['info'].create_dataset(key, data=value)
                    # print(f"Creating dataset file: {os.path.abspath(f.filename)}")

    def merge_dataset_h5(self, mat_id_list: list, save_original: bool = False, dataset_fname: str='dataset.h5'):
        """
        Merge dataset h5 files into one
        """
        # print("Merging dataset h5 files", [mat_id+dataset_fname for mat_id in mat_id_list])

        self.init_dataset_h5(dataset_dir=self.dataset_dir, dataset_fname=self.dataset_fname, info=self.info, multiprocessing=False)

        for mat_id in mat_id_list:
            with h5.File(pjoin(self.dataset_dir, mat_id+dataset_fname), 'r') as f:
                # This is not a class method, so we don't need further info check
                self.datapoint_interface_h5(pjoin(self.dataset_dir, dataset_fname), mat_id, f[mat_id], mode='a', )

            if not save_original:
                os.remove(pjoin(self.dataset_dir, mat_id+dataset_fname))

    def process_worker_WFN(self, folder:str)-> dict:
        """
        This function processes the WFN data for a single material
        this func: get kwargs -> get wfn file -> create datapoint -> save data to h5 file
        folder: flow folder (not flows)
        """
        # get info
        mat_id = os.path.basename(folder)
        info = copy.deepcopy(dict(self.info.__dict__))
        nc, nv = info.pop('nc_wfn'), info.pop('nv_wfn')
        operator = self.operator

        # get wfn file
        wfn_fname = pjoin(pjoin(folder, '02-wfn', "wfn.h5"))

        # create datapoint
        wf = wfn(wfn_fname)
        datapoint =  wf.get_dataset(nc=nc, nv=nv, operator = operator,**info)

        # save data to h5 file
        # if use multiprocessing, save data to mat_id+dataset_fname
        # else save data to dataset_fname
        if not self.multiprocessing:
            dataset_h5_fname = pjoin(self.dataset_dir, self.dataset_fname)
        else:
            dataset_h5_fname = pjoin(self.dataset_dir, mat_id+self.dataset_fname)

        self.datapoint_interface_h5(dataset_h5_fname, mat_id, datapoint, mode='a')

        return datapoint

    def process_worker_GW(self, folder:str)-> dict:
        """
        This function processes the GW data for a single material
        """

        # get kwargs
        datapoint = {}
        mat_id = os.path.basename(folder)
        info = copy.deepcopy(self.info.__dict__)
        nc_wfn, nv_wfn, nc_sigma, nv_sigma = info.pop('nc_wfn'), info.pop('nv_wfn'), \
                                     info.pop('nc_sigma'), info.pop('nv_sigma')

        if info.get('from_dft'):
            # build src
            wfn_fname = pjoin(pjoin(folder, '02-wfn', "wfn.h5"))
            wf = wfn(wfn_fname)
            datapoint_src =  wf.get_dataset(nc=nc_wfn, nv=nv_wfn, **info)
            datapoint['src'] = datapoint_src

            # build tgt & label
            if info.get('predict_only'):
                raise NotImplementedError
                wfn_fname = pjoin(pjoin(folder, '05-band', "wfn.h5"))
                wf = wfn(wfn_fname)
                datapoint_tgt = wf.get_dataset(nc=nc_sigma, nv=nv_sigma, **info)

            else:
                datapoint_tgt = wf.get_dataset(nc=nc_sigma, nv=nv_sigma, **info)
                eqp1 = eqp(pjoin(pjoin(folder, '13-sigma'), "eqp1.dat"))
                datapoint_eqp = eqp1.get_dataset()

                _, tgt_idx, label_idx = np.intersect1d(datapoint_tgt['band_indices_abs'][0], datapoint_eqp['band_indices_abs'][0], return_indices=True)
                assert len(tgt_idx) == len(datapoint_tgt['band_indices_abs'][0]), "selected nc_sigma, nv_sigma are not in the label"

                # ensure the same band indices for tgt and label
                for key, val in datapoint_eqp.items():
                    datapoint_eqp[key] = val[:,label_idx,:]
                
                datapoint['tgt'], datapoint['label'] = datapoint_tgt, datapoint_eqp

        else: # read from vae output 
            raise NotImplementedError

        # save data to h5 file
        if not self.multiprocessing:
            dataset_h5_fname = pjoin(self.dataset_dir, self.dataset_fname)
        else:
            dataset_h5_fname = pjoin(self.dataset_dir, mat_id+self.dataset_fname)
        
        self.datapoint_interface_h5(dataset_h5_fname, mat_id, datapoint, mode='a')

        return datapoint

    def process_worker_BSE(self, folder:str)-> dict:
        """
        This function processes the BSE data for a single material
        """
        pass

        datapoint = {}
        mat_id = os.path.basename(folder)
        # print(f'--debug: {mat_id} BSE_processor started')
        info = copy.deepcopy(self.info.__dict__)
        nc_wfn, nv_wfn = info.pop('nc_wfn'), info.pop('nv_wfn')

        # build src
        if info.get('from_dft'):
            wfn_fname = pjoin(pjoin(folder, '17-wfn_fi', "wfn.h5"))
            wf = wfn(wfn_fname)
            datapoint_src =  wf.get_dataset(nc=nc_wfn, nv=nv_wfn, **info)
            datapoint['src'] = datapoint_src

        # print(f'--debug: {mat_id} WFN dataset created')

        # build tgt and label
        if not info.get('predict_only'):
            if info.get('from_dft'):
                # build label first
                acv = AScvk(pjoin(folder, '19-absorption/eigenvectors.h5'))
                datapoint['label'] = acv.get_dataset()

                assert datapoint['label']['eigenvalues'].shape[0] == datapoint['label']['eigenvectors'].shape[0] 

                nS, nk, nc, nv, = datapoint['label']['eigenvectors'].shape

                assert nS == nk * nc * nv, f"nS = {nS}, nk = {nk}, nc = {nc}, nv = {nv} are not consistent"

                wfn_fname = pjoin(pjoin(folder, '17-wfn_fi', "wfn.h5"))
                wf = wfn(wfn_fname)
                datapoint_src =  wf.get_dataset(nc=nc, nv=nv, **info)
                datapoint['tgt'] = datapoint_src
            else:
                raise NotImplementedError

        else:
            nc_wfn, nv_wfn = info.pop('nc_bse'), info.pop('nv_bse')
            if info.get('from_dft'):
                # build src
                wfn_fname = pjoin(pjoin(folder, '17-wfn_fi', "wfn.h5"))
                wf = wfn(wfn_fname)
                datapoint_src =  wf.get_dataset(nc=nc_wfn, nv=nv_wfn, **info)
                datapoint['tgt'] = datapoint_src

            else:
                raise NotImplementedError
            
        # print('--debug: {mat_id} BSE dataset created')

        # save data to h5 file
        if not self.multiprocessing:
            dataset_h5_fname = pjoin(self.dataset_dir, self.dataset_fname)
        else:
            dataset_h5_fname = pjoin(self.dataset_dir, mat_id+self.dataset_fname)
        
        self.datapoint_interface_h5(dataset_h5_fname, mat_id, datapoint, mode='a')


        return datapoint

    @staticmethod
    def datapoint_interface_h5(dataset_h5_fname: str, mat_id: str, datapoint=None, mode: str = 'a', load_large_dataset_inference: bool = False):
        """
        Save or load a datapoint to/from an HDF5 file.
        
        Parameters:
        - datapoint(two types, only required in "a"): 
            dict -> Nested dictionary structure containing data
            h5.Group(dict like structure) -> HDF5 group object
        - dataset_h5_fname: str -> Path to the HDF5 file.
        - mat_id: str -> Identifier for the dataset inside HDF5.(first level dict)
        - mode: str -> 'a' for append/write, 'r' for read.
        
        Structure:
        - Unsupervised: {"xx": np.array, "yy": np.array}
        - Supervised: {"src": {"xx": np.array, ...}, "tgt": {...}, "label": {...}}
        
        HDF5 format:
        ```
        dataset.h5
        ├── info
        └── mat_id
            ├── xx
            ├── yy
            ├── src
            │   ├── xx
            │   ├── ...
            ├── tgt
            ├── label
        ```
        """

        assert mode in ['a', 'r'], "mode should be 'a' or 'r'"

        if mode == 'a':
            with h5.File(dataset_h5_fname, mode) as f:
                if mat_id in f:
                    del f[mat_id]
                f.create_group(mat_id)

                def write_data(group, data):
                    """Recursively writes data to HDF5, handling nested dictionaries."""
                    for key, val in data.items():
                        if isinstance(val, dict) or isinstance(val, h5.Group):  # Nested dictionary
                            subgroup = group.create_group(key)
                            write_data(subgroup, val)
                        else:
                            group.create_dataset(key, data=val)

                write_data(f[mat_id], datapoint)
            return

        elif mode == 'r':
            datapoint = {}

            def read_data(group):
                """Recursively reads HDF5 data into a nested dictionary."""
                data_dict = {}
                for key, item in group.items():
                    if isinstance(item, h5.Group):  # If it's a group, recurse
                        data_dict[key] = read_data(item)
                    else:
                        # data_dict[key] = item[()]  # Read dataset
                        # safe loading for large dataset:
                        if load_large_dataset_inference:
                            if key == 'wfn':
                                data_dict[key] = item # only load its reference, the actual data will be loaded when accessed
                            else:
                                data_dict[key] = item[()]
                        else:
                                data_dict[key] = item[()]
                return data_dict

            # with h5.File(dataset_h5_fname, 'r') as f:
            #     if mat_id in f:
            #         datapoint = read_data(f[mat_id])

            f = h5.File(dataset_h5_fname, 'r')
            if mat_id in f:
                datapoint = read_data(f[mat_id])

            return datapoint

    def summary(self):
        pass

class ToyDataSet(Dataset):

    """
    This is a toy dataset
    features:
        - Toy data for Transformer model
        - usage of ManyBodyData
            - WFN, GW, BSE

    Note: each material is a "sentenece" in the transformer model
    nk*nb: number of "words" in the "sentence"
    d_latent: dimension of the "word" embedding

    """
    toy_data_path = "../../examples/flows"
    mbformer_data_dir = "./dataset/"

    d_model = 24 # divisible by 24
    batch_size = 10 #
    nk_max = 12*12
    nc_max = 8
    nv_max = 2
    nb_max = nc_max + nv_max
    d_latent = 12

    # BSE data
    cond_embedding = torch.rand((batch_size, nk_max, nc_max, d_model)) # after VAE-Embeeding
    val_embedding = torch.rand((batch_size, nk_max, nv_max, d_model)) # after VAE-Embeeding
    cond_band_index = torch.arange(1,nc_max+1)[None, None, :, None].repeat(batch_size, nk_max, 1,1)
    val_band_index = torch.arange(-1,-nv_max-1,-1)[None, None, :, None].repeat(batch_size, nk_max, 1,1)
    cond_band_energy = torch.rand(nc_max)[None, None, :, None].repeat(batch_size, nk_max, 1,1)
    val_band_energy = torch.rand(nv_max)[None, None, :, None].repeat(batch_size, nk_max, 1,1)

    cond_kpt = torch.rand((batch_size, nk_max, 1, 3)).repeat(1, 1, nc_max, 1)
    val_kpt = torch.rand((batch_size, nk_max, 1, 3)).repeat(1, 1, nv_max, 1)
    cond_kpt_weight = torch.rand((batch_size, nk_max, 1, 1)).repeat(1, 1, nc_max, 1)
    val_kpt_weight = torch.rand((batch_size, nk_max, 1, 1)).repeat(1, 1, nv_max, 1)

    @classmethod
    def get_ele_data_batch(cls):
        return [cls.cond_embedding, cls.cond_kpt, cls.cond_band_index, cls.cond_band_energy]

    @classmethod
    def get_hole_data_batch(cls):
        return [cls.val_embedding, cls.val_kpt, cls.val_band_index, cls.val_band_energy]

    # The usage of ManyBodyData: [WFN, GW, BSE]
    # mat-5, mat-6, mat-7 (all of them are hBN)
    @staticmethod
    def get_wfn_dataset(mbformer_data_dir: str = './dataset/', toy_data_path: str = '../../examples/flows', read=True):
        if not os.path.exists(f'{mbformer_data_dir}/dataset_WFN.h5') or not read:
            return ManyBodyData(flows_dir=toy_data_path, dataset_dir=mbformer_data_dir, dataset_type='WFN', dataset_fname='dataset_WFN.h5',
                          load_dataset=False, cell_slab_truncation=30, useWignerXY=True, AngstromPerPixel=0.1,
                          AngstromPerPixel_z=0.2, upsampling_factor=2, multiprocessing=True,
                          nc_wfn=4, nv_wfn=2)
        return ManyBodyData.from_existing_dataset(f'{mbformer_data_dir}/dataset_WFN.h5')
    
    @staticmethod
    def get_wfn_3d_dataset(mbformer_data_dir: str = './dataset/', toy_data_path: str = '../../examples/flows', read=True):
        if not os.path.exists(f'{mbformer_data_dir}/dataset_WFN_3D.h5') or not read:
            return ManyBodyData(flows_dir=toy_data_path, dataset_dir=mbformer_data_dir, dataset_type='WFN', dataset_fname='dataset_WFN_3D.h5',
                          load_dataset=False, cell_slab_truncation=None, useWignerXYZ=True, AngstromPerPixel=0.1,
                          upsampling_factor=2, multiprocessing=True,
                          nc_wfn=4, nv_wfn=2)
        return ManyBodyData.from_existing_dataset(f'{mbformer_data_dir}/dataset_WFN_3D.h5')


    @staticmethod
    def get_gw_dataset(mbformer_data_dir: str = './dataset/', toy_data_path: str = '../../examples/flows', read=True):
        if not os.path.exists(f'{mbformer_data_dir}/dataset_GW.h5') or not read:
            return ManyBodyData(flows_dir=toy_data_path, dataset_dir=mbformer_data_dir, dataset_type='GW', dataset_fname='dataset_GW.h5',
                          load_dataset=False, cell_slab_truncation=30, useWignerXY=True,  AngstromPerPixel=0.1, 
                          AngstromPerPixel_z=0.2, upsampling_factor=2, multiprocessing=True,
                          nc_wfn=4, nv_wfn=2,nc_sigma=1, nv_sigma=1, from_dft=True, predict_only=False,)
        return ManyBodyData.from_existing_dataset(f'{mbformer_data_dir}/dataset_GW.h5')
    
    @staticmethod
    def get_bse_dataset(mbformer_data_dir: str = './dataset/', toy_data_path: str = '../../examples/flows', read=True):
        if not os.path.exists(f'{mbformer_data_dir}/dataset_BSE.h5') or not read:
            return ManyBodyData(flows_dir=toy_data_path, dataset_dir=mbformer_data_dir, dataset_type='BSE', dataset_fname='dataset_BSE.h5',
                            load_dataset=False, cell_slab_truncation=30, useWignerXY=True,  AngstromPerPixel=0.1, 
                            AngstromPerPixel_z=0.2, upsampling_factor=2, multiprocessing=True,
                            from_dft=True, predict_only=False, nc_wfn=4,nv_wfn=2)   
        return ManyBodyData.from_existing_dataset(f'{mbformer_data_dir}/dataset_BSE.h5')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    """WFN Usage"""
    # please see ToyDataSet.get_wfn_dataset() for how to use ManyBodyData (Two ways)
    wfdata = ToyDataSet.get_wfn_dataset(read=False)
    wfdata = ToyDataSet.get_wfn_dataset(read=True)

    wf3ddata = ToyDataSet.get_wfn_3d_dataset(read=False)
    wf3ddata = ToyDataSet.get_wfn_3d_dataset(read=True)

    """WFN Unit Test"""
    assert abs(wfdata[1]['wfn'][0,0,14,13,15] - 2.1230801376011337e-06) < 1e-10, "WFN Unit Test Failed"
    # assert abs(wf3ddata[1]['wfn'][0,0,13,13,110] - 0.00045705909086906754) < 1e-10, "WFN 3D Unit Test Failed"
    assert wf3ddata[1]['wfn'].shape == (2, 6, 26,29, 200), "WFN 3D shape mismatch"

    """GW Usage"""
    # please see ToyDataSet.get_gw_dataset() for how to use ManyBodyData (Two ways)
    gwdata = ToyDataSet.get_gw_dataset(read=False)
    gwdata = ToyDataSet.get_gw_dataset(read=True) 

    assert abs(gwdata[1]['src']['wfn'][0,0,14,13,15] - 2.1230801376011337e-06) < 1e-10, "GW Unit Test Failed"
    assert abs(gwdata[1]['tgt']['wfn'][0,0,14,13,15] - 1.261505271449588e-07) < 1e-10, "GW Unit Test Failed"
  

    """BSE Usage"""
    # predict only
    bsedata = ManyBodyData(flows_dir='../../examples/flows', dataset_dir='./dataset', dataset_type='BSE', dataset_fname='dataset_BSE.h5',
                            load_dataset=False, cell_slab_truncation=30, useWignerXY=True,  AngstromPerPixel=0.1, 
                            AngstromPerPixel_z=0.2, upsampling_factor=2, multiprocessing=True,
                            from_dft=True, predict_only=True, nc_wfn=6, nv_wfn=4, nc_bse=4, nv_bse=2)   

    assert abs(bsedata[1]['tgt']['wfn'][0,0,14,13,15] - 5.971020835603282e-07) < 1e-10, "BSE Unit Test Failed"

    # slice
    bsedata = ManyBodyData.from_existing_dataset('./dataset/dataset_BSE.h5', slice(1,2))
    assert abs(bsedata[0]['tgt']['wfn'][0,0,14,13,15] - 5.971020835603282e-07) < 1e-10
    assert len(bsedata) == 1, "BSE Unit Test Failed"

    # please see ToyDataSet.get_bse_dataset() for how to use ManyBodyData (Two ways)
    bsedata = ToyDataSet.get_bse_dataset(read=False)
    bsedata = ToyDataSet.get_bse_dataset(read=True)

    """onlySave Test"""
    bsedata = ManyBodyData(flows_dir='../../examples/flows', dataset_dir='./dataset', dataset_type='BSE', dataset_fname='dataset_BSE.h5',
                            load_dataset=False, cell_slab_truncation=30, useWignerXY=True,  AngstromPerPixel=0.1, 
                            AngstromPerPixel_z=0.2, upsampling_factor=2, multiprocessing=True, onlySave=True,
                            from_dft=True, predict_only=False, nc_wfn=6, nv_wfn=4, nc_bse=4, nv_bse=2) 
    assert len(bsedata) == 0, "onlySave Test Failed"

    bsedata = ManyBodyData.from_existing_dataset('./dataset/dataset_BSE.h5', slice(1,2))
    assert abs(bsedata[0]['tgt']['wfn'][0,0,14,13,15] - 5.971020835603282e-07) < 1e-10
    assert len(bsedata) == 1, "BSE Unit Test Failed"

    print("WFN: unit test passed")    
    print("GW: unit test passed")
    print("BSE: unit test passed")