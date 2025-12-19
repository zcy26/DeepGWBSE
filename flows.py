#!/usr/bin/env python
"""
DFT-GW-BSE workflow script for multiple materials.

This script creates workflows for multiple materials from a configuration file.
Usage:
    python flows.py -c config.json
"""

from deep_gwbse.flow import DFT_GW_HPRO_Flow
from deep_gwbse.from_bgwpy.core import Workflow
import os
import json
import copy


class Mat_Flows(Workflow):
    """
    Build workflow for Materials
    Input: input_dir
            |--- config.json
            |--- mat-1/stru.cif
            |--- mat-2/stru.cif
            |--- ...
    """
    def __init__(self, flow=DFT_GW_HPRO_Flow, **kwargs):
        # Get config
        input_dir_config = self.parse_config(**kwargs)
        kwargs = input_dir_config
        super().__init__(**kwargs)

        # Set up the workflow
        self.dirname = kwargs.pop('dirname', None)
        self.flow = flow

        # specify "dirname", "stru_file", "prefix" for each material
        # add task 
        for mat_stru_dir in self.mat_stru_dir:
            print(f"Adding flow for {mat_stru_dir}")
            self.add_task(self.flow(dirname=os.path.join(self.dirname, mat_stru_dir),
                                           stru_file=os.path.join(kwargs['stru_dir'], mat_stru_dir, 'stru.cif'),
                                           prefix=mat_stru_dir,
                                           **kwargs))

    def parse_config(self, **kwargs):
        """
        Parse the input directory
        """
        assert 'configfname' in kwargs, "configfname is required"
        config = json.load(open(kwargs['configfname'], 'r'))
        assert 'stru_dir' in config, "stru_dir is required in config"
        stru_dir = config['stru_dir']
        self.config_input = copy.deepcopy(config)
        self.mat_stru_dir = [d for d in os.listdir(stru_dir) if os.path.isdir(os.path.join(stru_dir, d))]
        assert len(self.mat_stru_dir) > 0, "No structure found"
        print(f"{len(self.mat_stru_dir)} structures found")
        self._input_parsed = True

        return config

    def write(self):
        """
        Write the workflow
        """
        super().write()
        with open(os.path.join(self.dirname, 'fpconfig.json'), 'w') as f:
            json.dump(self.config_input, f, indent=4)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Create workflows for multiple materials.')
    parser.add_argument('-c', '--config', type=str, default='./config/fpconfig.json', 
                       help='input config file')
    args = parser.parse_args()
    print(f"Using config file: {args.config}")

    assert os.path.exists(args.config), f"Config file not found: {args.config}"

    flows = Mat_Flows(configfname=args.config)
    flows.write()

