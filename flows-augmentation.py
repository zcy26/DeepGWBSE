#!/usr/bin/env python
"""
DFT-GW-BSE workflow augmentation script.

This script creates augmentation workflows for GW or BSE calculations from existing flows.
Usage:
    python flows-augmentation.py -c augconfig.json
"""

from deep_gwbse.flow import DFT_GW_HPRO_Flow, check_pseudo
from deep_gwbse.from_bgwpy.core import Workflow
from deep_gwbse.utils import check_flows_status
import os
import json
import copy
import numpy as np
from os.path import join as pjoin
import itertools
from typing import Tuple, List
from deep_gwbse.from_bgwpy.config import flavors
from deep_gwbse.from_bgwpy.config import is_dft_flavor_espresso, check_dft_flavor
from deep_gwbse.from_bgwpy.external import Structure
from deep_gwbse.from_bgwpy.BGW import EpsilonTask, SigmaTask, IneqpTask, KernelTask, AbsorptionTask
from deep_gwbse.from_bgwpy.QE import QeScfTask, QeBgwFlow, Qe2BgwTask, QeWfnTask
from deep_gwbse.from_bgwpy.DFT import WfnBgwFlow
from ase import Atoms
import ase.io
import subprocess
from deep_gwbse.fptask import (AobasisTask, HPROTask, PseudoBandTask, QeBgwFlow_NNS, 
                               EpsilonTask_NNS, SigmaTask_NNS, nns_helper_epsilon, 
                               ParaBandTask, QeBgwFlow_band, IneqpTask_plot)


class Mat_Flows_Augmentation(Workflow):
    """
    Build GW or BSE workflow from existing flows
    Input: flows
            |--- flow-1/
            |--- flow-2/
            |--- ...
    Output: flows/
            |--- flow-1/
            |--- flow-1-aug-bse-221-shift/
            ...
            |--- flow-1-aug-gw-221-shift/
            ...
    """
    def __init__(self, flow, **kwargs):
        self.parse_config(**kwargs)
        super().__init__(**kwargs)

        # set up the workflow
        # super will set dirname as './', we need to reset it here
        self.dirname = self.aug_config['dirname']
        self.flow = flow

        # BSE augmentation
        if self.aug_config['BSE_augmentation']:
            bse_org_fpconfig_of_each_aug = {}
            for i, bse_aug_fpconfig in enumerate(self.bse_aug_fpconfigs):
                if not bse_org_fpconfig_of_each_aug or self.bse_org_fpconfigs_of_each_aug[i]['dirname'] != bse_org_fpconfig_of_each_aug['dirname']:
                    bse_org_fpconfig_of_each_aug = self.bse_org_fpconfigs_of_each_aug[i]
                    print(f'\nReading original flow for {bse_org_fpconfig_of_each_aug["dirname"]}')
                    org_flow = DFT_GW_HPRO_Flow(**bse_org_fpconfig_of_each_aug)
                print(f"Adding flow for {bse_aug_fpconfig['dirname']}")
                # set up cross workflow link
                bse_aug_fpconfig.update(
                    charge_density_fname = org_flow.scftask.charge_density_fname,
                    data_file_fname = org_flow.scftask.data_file_fname,
                    spin_polarization_fname = org_flow.scftask.spin_polarization_fname,
                    eps0mat_fname = org_flow.epsilontask.eps0mat_fname,
                    epsmat_fname = org_flow.epsilontask.epsmat_fname,
                    wfn_co_fname=  org_flow.wfntask_ush.wfn_fname,
                    bsemat_fname = org_flow.kerneltask.bsemat_fname,
                    eqp_fname = org_flow.sigmatask.eqp1_fname
                )
                self.add_task(self.flow(aug_task='bse',
                                        **bse_aug_fpconfig))

        # GW augmentation
        if self.aug_config['GW_augmentation']:
            gw_org_fpconfig_of_each_aug = {}
            for i, gw_aug_fpconfig in enumerate(self.gw_aug_fpconfigs):
                if not gw_org_fpconfig_of_each_aug or self.gw_org_fpconfigs_of_each_aug[i]['dirname'] != gw_org_fpconfig_of_each_aug['dirname']:
                    gw_org_fpconfig_of_each_aug = self.gw_org_fpconfigs_of_each_aug[i]
                    print(f'\nReading original flow for {gw_org_fpconfig_of_each_aug["dirname"]}')
                    org_flow = DFT_GW_HPRO_Flow(**gw_org_fpconfig_of_each_aug)
                    self.org_flow = org_flow
                print(f"Adding flow for {gw_aug_fpconfig['dirname']}")
                # set up cross workflow link
                gw_aug_fpconfig.update(
                    charge_density_fname = org_flow.scftask.charge_density_fname,
                    data_file_fname = org_flow.scftask.data_file_fname,
                    spin_polarization_fname = org_flow.scftask.spin_polarization_fname,
                    wfn_co_fname=  org_flow.wfntask_ush.wfn_fname,
                    eqp_fname = org_flow.sigmatask.eqp1_fname
                )
                self.add_task(self.flow(aug_task='gw',
                                        **gw_aug_fpconfig))

    def parse_config(self, **kwargs):
        """
        input:
        function:
            1. find the qualified flows
            2. generate fpconfig list for augmentation
        config variables:
            aug_config: how to augment
            config_input: a deep copy of the input config, used to write the config
            bse_org_fpconfig: original fpconfig for each flow
            bse_aug_fpconfig: augmented fpconfig for each flow
        """
        assert 'configfname' in kwargs, "configfname is required"
        self.aug_config = json.load(open(kwargs['configfname'], 'r'))
        assert self.aug_config['BSE_augmentation'] or self.aug_config['GW_augmentation'], "BSE_augmentation or GW_augmentation is required"
        self.config_input = copy.deepcopy(self.aug_config)

        if self.aug_config['BSE_augmentation']:
            self.bse_folder_list, self.bse_mat_ids = Mat_Flows_Augmentation.mat_statistics(self.aug_config['dirname'], dataset_type='BSE')
            self.bse_org_fpconfigs = [json.load(open(pjoin(flow, 'config.json'), 'r')) for flow in self.bse_folder_list]
            self.bse_aug_fpconfigs, self.bse_org_fpconfigs_of_each_aug = self.create_aug_fpconfigs(self.bse_org_fpconfigs,
                                                               grid_size_augment=self.aug_config['BSE_nk_aug'],
                                                               grid_shift_augment=self.aug_config['BSE_k_shift_aug'],
                                                               org_grid=self.bse_org_fpconfigs[0]['ngkpt_fi'],
                                                               taskname='bse')

        if self.aug_config['GW_augmentation']:
            self.gw_folder_list, self.gw_mat_ids = Mat_Flows_Augmentation.mat_statistics(self.aug_config['dirname'], dataset_type='GW')
            self.gw_org_fpconfigs = [json.load(open(pjoin(flow, 'config.json'), 'r')) for flow in self.gw_folder_list]
            self.gw_aug_fpconfigs, self.gw_org_fpconfigs_of_each_aug = self.create_aug_fpconfigs(self.gw_org_fpconfigs,
                                                               grid_size_augment=self.aug_config['GW_nk_aug'],
                                                               grid_shift_augment=self.aug_config['GW_k_shift_aug'],
                                                               org_grid=self.gw_org_fpconfigs[0]['ngkpt'],
                                                               taskname='gw')

    def create_aug_fpconfigs(self, org_fpconfigs:list, grid_size_augment:int, grid_shift_augment:int, org_grid:int, taskname='bse'):
        """
        Input:
        Output: list[config_aug_dict1, config_aug_dict2, ...], list[config_org_dict1, config_org_dict2, ...]
        """
        grid_sizes, grid_shift, labels = Mat_Flows_Augmentation.grid_aug_2d(grid_size_augment=grid_size_augment,
                                                                            grid_shift_augment=grid_shift_augment, 
                                                                            org_grid=org_grid) 
        print(f'{len(labels)} data augmentations for {taskname}, applied to each of {len(org_fpconfigs)} flows.')

        bse_org_fpconfigs_of_each_aug = []
        aug_fpconfigs = []
        for i, org_fpconfig in enumerate(org_fpconfigs):
            for j in range(len(labels)):
                aug_fpconfig = copy.deepcopy(org_fpconfig)
                bse_org_fpconfig_of_each_aug = copy.deepcopy(org_fpconfig)
                bse_org_fpconfig_of_each_aug['stru_file'] = pjoin(aug_fpconfig['dirname'], 'stru.cif')
                bse_org_fpconfig_of_each_aug['pseudo_dir_source'] = pjoin(aug_fpconfig['dirname'], 'pp')
                aug_fpconfig['stru_file'] = pjoin(aug_fpconfig['dirname'], 'stru.cif')
                aug_fpconfig['pseudo_dir_source'] = pjoin(aug_fpconfig['dirname'], 'pp')
                if taskname == 'bse':
                    aug_fpconfig['ngkpt_fi'] = grid_sizes[j]
                    aug_fpconfig['kshift_fi'] = grid_shift[j]
                if taskname == 'gw':
                    aug_fpconfig['ngkpt'] = grid_sizes[j]
                    aug_fpconfig['kshift'] = grid_shift[j]
                aug_fpconfig['dirname'] = aug_fpconfig['dirname'] + f'-aug-{j+1:03d}-{taskname}-{labels[j]}'
                aug_fpconfig['prefix'] = org_fpconfig['prefix'] + f'-aug-{j+1:03d}-{taskname}-{labels[j]}'
                aug_fpconfigs.append(aug_fpconfig)
                bse_org_fpconfigs_of_each_aug.append(bse_org_fpconfig_of_each_aug)

        return aug_fpconfigs, bse_org_fpconfigs_of_each_aug

    @staticmethod
    def grid_size_aug_2d(num_augments:int=1, grid:list=[1,1,1]):
        """
        Only support 2D grid augmentation
        if num_augments=4,
        e.g. [6,6,1] -> [[6,6,1], [6,7,1], [7,6,1], [7,7,1]]
        1/4 * pi * r^2 = num_augments -> r is the radius of the circle
        """
        r = int( (4 * num_augments / np.pi ) ** 0.5) + 1
        x, y, z = grid
        augment_range = range(r)
        new_grids = [((x + dx)**2 + (y + dy)**2, [x + dx, y + dy, z]) for dx, dy in itertools.product(augment_range, repeat=2)]
        new_grids.sort()
        new_grids =  [(new_grid[1][0]*new_grid[1][1], new_grid[1]) for new_grid in new_grids][:num_augments]
        new_grids.sort()
        return [new_grid[1] for new_grid in new_grids]

    @staticmethod
    def grid_shift_aug_2d(num_augments:int=1):
        """
        randomly generate num_augments grid shift: [0~1, 0~1, 0]
        """
        # random seed
        np.random.seed(10)
        new_grids = [[0,0,0]]
        for i in range(num_augments-1):
            new_grids.append([round(np.random.rand(), 5), round(np.random.rand(), 5), 0])
        return new_grids[:num_augments]

    @staticmethod
    def grid_aug_2d(grid_size_augment:int=1, grid_shift_augment:int=1, org_grid=[2,2,1]) -> Tuple[List, List, List]:
        aug_grids = np.array(Mat_Flows_Augmentation.grid_size_aug_2d(num_augments=grid_size_augment, grid=org_grid))
        aug_shifts = np.array(Mat_Flows_Augmentation.grid_shift_aug_2d(num_augments=grid_shift_augment))
        if len(aug_grids) != 0 and len(aug_shifts) != 0:
            res = np.array([np.array(aug_grid) + np.array(aug_shift) for aug_grid, aug_shift in itertools.product(aug_grids, aug_shifts)])
        elif len(aug_grids) != 0:
            res = np.array(aug_grids)
        elif len(aug_shifts) != 0:
            res = np.array(aug_shifts) + np.array(org_grid)
        else:
            assert False, "No grid augmentation found"
        
        if np.allclose(res[0], np.array(org_grid)):
            res = res[1:]

        assert len(res) > 0, "No grid augmentation found"
        grid_sizes = ((res//1).astype(int)).tolist()
        grid_shift = list(map(lambda shift: [round(shifti, 5) for shifti in shift], (res%1).tolist()))
        label = []
        for i in range(len(grid_sizes)):
            if np.allclose(grid_shift[i], np.array([0,0,0])):
                label.append(f'{grid_sizes[i][0]}_{grid_sizes[i][1]}_{grid_sizes[i][2]}-unshift')
            else:
                label.append(f'{grid_sizes[i][0]}_{grid_sizes[i][1]}_{grid_sizes[i][2]}-shift')

        return grid_sizes, grid_shift, label

    @staticmethod
    def mat_statistics(flows_dir:str, dataset_type:str='BSE'):
        """
        Detect if the flow is finished for GW and BSE augmentation
        """
        assert dataset_type in ['BSE', 'GW'], "dataset_type must be either 'BSE' or 'GW'"
        folder_list = []
        flows_status = check_flows_status(flows_dir, False)

        for flow, status in flows_status.items():
            finished_tasks = set(status['Yes'].split(','))
            if dataset_type == 'BSE':
                if {"01-density", "19-absorption"} <= finished_tasks:
                    # 01-density is used to exclude all aug flows
                    folder_list.append(flow)
            elif dataset_type == 'GW':
                if {"01-density","14-inteqp"} <= finished_tasks:
                    folder_list.append(flow)
        assert len(folder_list) > 0, f"No data found under {flows_dir}"
        print(f"Found {len(folder_list)} out of {len(flows_status)} materials for {dataset_type}")
        mat_ids = np.array([os.path.basename(folder) for folder in folder_list], dtype='S')

        return folder_list, mat_ids

    def write(self):
        """
        Write the augmentation workflow
        """
        self.runscript.fname = 'run_aug.sh'
        super().write()
        with open(pjoin(self.dirname, 'augconfig.json'), 'w') as f:
            json.dump(self.config_input, f, indent=4)


class BSE_aug_Flow(Workflow):
    """
    This is modified from flow.py DFT_GW_HPRO_Flow
    """
    def __init__(self, aug_task:str='bse' ,**kwargs):
        '''
        aug_task: data augmentation for GW or BSE
        '''
        #========================================Preparation========================================#
        # write all input to a config.json file
        assert aug_task in ['bse', 'gw'], "aug_task must be either 'bse' or 'gw'"
        self.config_input = copy.deepcopy(kwargs)

        # Setup QE paths
        kwargs.update({"PW":pjoin(kwargs.get('QE_path',''),'pw.x')})
        kwargs.update({"PW2BGW":pjoin(kwargs.get('QE_path',''),'pw2bgw.x')})
        kwargs.update({"BANDS":pjoin(kwargs.get('QE_path',''),'bands.x')})

        # Setup BGW paths

        # add "structure" to kwargs (historic reason)
        kwargs.update({'structure':Structure.from_file(kwargs['stru_file'])})

        super().__init__(**kwargs)

        kwargs.pop('dirname', None)
        self.structure = kwargs['structure']
        self.atoms = ase.io.read(kwargs['stru_file'])
        self.ngkpt = kwargs.pop('ngkpt')
        self.ngkpt_fi = kwargs.pop('ngkpt_fi')
        self.kshift = kwargs.pop('kshift', [.0,.0,.0])
        self.kshift_fi = kwargs.pop('kshift_fi', [.0,.0,.0]) # for 17-wfn_fi
        self.qshift = kwargs.pop('qshift', [.0,.0,.0])
        nband_aliases = ('nbnd', 'nband')
        for key in nband_aliases:
            if key in kwargs:
                self.nbnd = kwargs.pop(key)
                break
        else:
            raise Exception(
            'Number of bands must be specified with one of these keywords: {}.'
            .format(nband_aliases))

        self.dft_flavor = check_dft_flavor(kwargs.get('dft_flavor', flavors['dft_flavor']))

        # ==== Check PseudoPotential ==== #
        # make a psuedo dir in dirname
        self.pp_dirname = pjoin(self.dirname, 'pp')
        kwargs['pseudo_dir'] = self.pp_dirname
        kwargs['pseudos'] = [str(atom_ele)+'.upf' for atom_ele in self.structure.elements]
        pseudos_z_valence = check_pseudo(pseudo_dir_src=kwargs['pseudo_dir_source'], pseudos=kwargs['pseudos'])
        print(pseudos_z_valence)
        self.n_z_valence = 0
        for atom_ele in self.atoms.get_chemical_symbols():
            pp = '.'.join([atom_ele] + (kwargs['pseudos'][0].split('.'))[1:])
            if pseudos_z_valence.get(pp, None) != None:
                self.n_z_valence += pseudos_z_valence[pp]
            else:
                print(f'warning: upf is not found for {pp}')
        if True: # consider SOC in the future
            self.n_z_valence = int(self.n_z_valence // 2)
            self.n_z_valence = None if self.n_z_valence == 0 else self.n_z_valence

        # update band_index_min & band_index_max
        assert self.n_z_valence != None, "n_z_valence is not found"
        if kwargs.get('nvbnd_sigma',None):
            kwargs.update({'ibnd_min': max(1, self.n_z_valence - kwargs.get('nvbnd_sigma',2))})
            kwargs.update({'ibnd_max': self.n_z_valence + kwargs.get('ncbnd_sigma',2)})

        assert is_dft_flavor_espresso(self.dft_flavor), "Only Quantum Espresso is supported for DFT calculations."

        if kwargs.get('SOC', False):
            kwargs['variables'] = kwargs.get('variables', {})
            kwargs['variables']['system'] = kwargs['variables'].get('system', {})
            kwargs['variables']['system'].update({'lspinorb': True, 'noncolin':True})

        if kwargs.get('smearing', False):
            kwargs['variables'] = kwargs.get('variables', {})
            kwargs['variables']['system'] = kwargs['variables'].get('system', {})
            kwargs['variables']['system'].update({'occupations': 'smearing', 'degauss': 1e-8})

        #========================================FLOW========================================#
        # ==== DFT calculations ==== #

        # ==== BSE Caculcations ==========
        if aug_task == 'bse':
            assert kwargs.get('BSE', False), "BSE is not set to True in original config"
            if kwargs.get('BSE'):
                self.make_bse_tasks_aug(**kwargs)
        
        if aug_task == 'gw':
            assert kwargs.get('GW', False), "GW is not set to True in original config"
            if kwargs.get('GW'):
                self.make_gw_tasks_aug(**kwargs)
        
    def make_bse_tasks_aug(self, **kwargs):
        if "eqp_co_corrections" in kwargs['absorption_extra_lines']:
            assert kwargs['GW'], "eqp_co_corrections is only available when GW is True"

        if kwargs.get('SOC', False):
            kwargs['variables'] = kwargs.get('variables', {})
            kwargs['variables']['system'] = kwargs['variables'].get('system', {})
            kwargs['variables']['system'].update({'lspinorb': True, 'noncolin':True}) 

        kwargs.update(dict(nbnd_cond = kwargs['nbnd_cond_co'],
                           nbnd_val = kwargs['nbnd_val_co']))

        self.wfn_fi_task_sh = QeBgwFlow(
            dirname = pjoin(self.dirname, '17-wfn_fi'),
            ngkpt = self.ngkpt_fi,
            kshift = self.kshift_fi,
            nbnd = self.n_z_valence+kwargs.get('nbnd_cond')+self.nbnd,
            rhog_flag = False,
            wfnhdf5 = True,
            symkpt =False,
            **kwargs)  
        self.add_tasks(self.wfn_fi_task_sh, merge=False)

        self.absorption_extra_lines = kwargs.pop('absorption_extra_lines', [])
        self.absorption_extra_variables = kwargs.pop('absorption_extra_variables', {})

        kwargs.update(
                wfn_fi_fname = self.wfn_fi_task_sh.wfn_fname,
                wfnq_fi_fname = self.wfn_fi_task_sh.wfn_fname,
                )

        self.absorptiontask = AbsorptionTask(
            dirname = pjoin(self.dirname, '19-absorption'),
            extra_lines = self.absorption_extra_lines,
            extra_variables = self.absorption_extra_variables,
            **kwargs
        )
        self.add_tasks([self.absorptiontask], merge=False)

    def make_gw_tasks_aug(self, **kwargs):
        self.wfntask_ksh = QeBgwFlow(
            dirname = pjoin(self.dirname, '02-wfn'),
            ngkpt = self.ngkpt,
            kshift = self.kshift,
            nbnd = self.n_z_valence+self.nbnd,
            rhog_flag = True,
            paraband_nproc = False,
            **kwargs)

        self.inteqp_task = IneqpTask_plot(
            dirname = pjoin(self.dirname, '14-inteqp'),
            eqp_co_fname = kwargs['eqp_fname'],
            wfn_fi_fname = self.wfntask_ksh.wfn_fname,
            nbnd = self.n_z_valence,
            **kwargs
        )

        self.add_tasks([self.wfntask_ksh, self.inteqp_task], merge=False)

    def write(self):
        """
        add config.json to directory
        """
        super().write()
        with open(pjoin(self.dirname, 'config.json'), 'w') as f:
            json.dump(self.config_input, f, indent=4)

        # write cif to directory
        self.structure.to(filename=pjoin(self.dirname, 'stru.cif'), fmt='cif')

        # copy pseudopotential files from source to pp
        subprocess.run(['mkdir',self.pp_dirname], capture_output=True, text=True)
        for atom_ele in self.structure.elements:
            subprocess.run(['cp', self.config_input['pseudo_dir_source']+ '/' + str(atom_ele)+'.upf', self.pp_dirname])
            subprocess.run(['cp', self.config_input['pseudo_dir_source']+ '/' + str(atom_ele)+'.psml', self.pp_dirname])


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create augmentation workflows for GW/BSE calculations.')
    parser.add_argument('-c', '--config', type=str, default="./config/fp_augconfig.json", 
                       help='input config file')
    args = parser.parse_args()
    print(f"Using config file: {args.config}")

    assert os.path.exists(args.config), f"Config file not found: {args.config}"

    augment_flows = Mat_Flows_Augmentation(flow=BSE_aug_Flow, configfname=args.config)
    augment_flows.write()

