#!/usr/bin/env python
from deep_gwbse.from_model.data import ToyDataSet

if __name__ == "__main__":
    wfdata = ToyDataSet.get_wfn_dataset(read=False, mbformer_data_dir='./dataset/', toy_data_path='./examples/flows')
    wfdata = ToyDataSet.get_wfn_dataset(read=True, mbformer_data_dir='./dataset/')
    gwdata = ToyDataSet.get_gw_dataset(read=False, mbformer_data_dir='./dataset/', toy_data_path='./examples/flows')
    gwdata = ToyDataSet.get_gw_dataset(read=True, mbformer_data_dir='./dataset/')
    bsedata = ToyDataSet.get_bse_dataset(read=False, mbformer_data_dir='./dataset/', toy_data_path='./examples/flows')
    bsedata = ToyDataSet.get_bse_dataset(read=True, mbformer_data_dir='./dataset/')

    print("Data preprocessing completed!")
    print(f"WFN dataset: {len(wfdata) if hasattr(wfdata, '__len__') else 'created'}")
    print(f"GW dataset: {len(gwdata) if hasattr(gwdata, '__len__') else 'created'}")
    print(f"BSE dataset: {len(bsedata) if hasattr(bsedata, '__len__') else 'created'}")

