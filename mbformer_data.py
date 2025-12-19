#!/usr/bin/env python
"""
MBFormer data preprocessing script.

This script preprocesses raw data from GW-BSE calculations to create datasets
for training MBFormer models (VAE, GW, and BSE).

Usage:
    python mbformer_data.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, Delaunay
from scipy.interpolate import griddata
from deep_gwbse.from_model.interface import wfn
import torch.nn.functional as F
from scipy.ndimage import zoom
import time
from scipy.interpolate import LinearNDInterpolator
from deep_gwbse.from_model.data import ManyBodyData, ToyDataSet
from torch.utils.data import DataLoader
import logging
from deep_gwbse.collect_tool import check_flows_status


if __name__ == "__main__":
    # Example usage with toy data
    # Uncomment and modify paths as needed
    
    # folder, id = ManyBodyData.mat_statistics("./examples/other/flows-semi/", "WFN")

    # Toy Data (see from_model/data.py for more details)
    wfdata = ToyDataSet.get_wfn_dataset(read=False, mbformer_data_dir='./dataset/', toy_data_path='./examples/flows')
    wfdata = ToyDataSet.get_wfn_dataset(read=True, mbformer_data_dir='./dataset/', toy_data_path='./examples/flows')
    gwdata = ToyDataSet.get_gw_dataset(read=False, mbformer_data_dir='./dataset/', toy_data_path='./examples/flows')
    gwdata = ToyDataSet.get_gw_dataset(read=True, mbformer_data_dir='./dataset/', toy_data_path='./examples/flows')
    bsedata = ToyDataSet.get_bse_dataset(read=False, mbformer_data_dir='./dataset/', toy_data_path='./examples/flows')
    bsedata = ToyDataSet.get_bse_dataset(read=True, mbformer_data_dir='./dataset/', toy_data_path='./examples/flows')

    # These are dataset configs we used in paper: https://arxiv.org/abs/2507.05480
    # Uncomment and modify paths as needed:
    # wfdata = ManyBodyData(flows_dir=".././examples/other/flows-semi/", dataset_dir='./dataset', dataset_type='WFN', dataset_fname='dataset_semi.h5',
    #                         load_dataset=False, nc_wfn=40, nv_wfn=2, cell_slab_truncation=30, useWignerXY=True, 
    #                     AngstromPerPixel=0.1, AngstromPerPixel_z=0.2, upsampling_factor=2, multiprocessing=True)    

    # gwdata = ManyBodyData(flows_dir='.././examples/other/flows-semi/', dataset_dir='./dataset', dataset_type='GW', dataset_fname='dataset_GW.h5',
    #                         load_dataset=False, cell_slab_truncation=30, useWignerXY=True,  AngstromPerPixel=0.1, 
    #                         AngstromPerPixel_z=0.2, upsampling_factor=2, multiprocessing=True,
    #                         nc_wfn=40, nv_wfn=2,nc_sigma=1, nv_sigma=1, from_dft=True, predict_only=False,)    

    # bsedata = ManyBodyData(flows_dir=".././examples/other/flows-semi/", dataset_dir='./dataset', dataset_type='BSE', dataset_fname='dataset_BSE_semi.h5',
    #                     load_dataset=False, cell_slab_truncation=30, useWignerXY=True,  AngstromPerPixel=0.1, 
    #                     AngstromPerPixel_z=0.2, upsampling_factor=2, multiprocessing=True,
    #                     from_dft=True, predict_only=False, nc_wfn=4,nv_wfn=2)

    print("Data preprocessing completed!")
    print(f"WFN dataset: {len(wfdata) if hasattr(wfdata, '__len__') else 'created'}")
    print(f"GW dataset: {len(gwdata) if hasattr(gwdata, '__len__') else 'created'}")
    print(f"BSE dataset: {len(bsedata) if hasattr(bsedata, '__len__') else 'created'}")

