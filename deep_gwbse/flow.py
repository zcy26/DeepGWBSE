#!/usr/bin/env python

from __future__ import print_function

from os.path import join as pjoin
from from_bgwpy.config import flavors
from from_bgwpy.config import is_dft_flavor_espresso, check_dft_flavor
from from_bgwpy.external import Structure
from from_bgwpy.core import Workflow
from from_bgwpy.BGW import EpsilonTask, SigmaTask, IneqpTask, KernelTask, AbsorptionTask
from from_bgwpy.QE import QeScfTask, QeBgwFlow, Qe2BgwTask, QeWfnTask
from from_bgwpy.DFT import WfnBgwFlow
from ase import Atoms
import ase.io
import subprocess
from fptask import AobasisTask, HPROTask, PseudoBandTask, QeBgwFlow_NNS, EpsilonTask_NNS, SigmaTask_NNS, nns_helper_epsilon, ParaBandTask, QeBgwFlow_band, IneqpTask_plot
# from config import fp_config
import re
import json
import copy
import os

# warning management
import warnings

# Suppress FutureWarning only
# Ignore FutureWarning (like from pwscfinput.py)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Ignore UserWarning (like from pymatgen CIF parser)
warnings.simplefilter(action='ignore', category=UserWarning)

"""
This file only defines workflow for single material:
Input: only one .cif structure file
Output:
"""

class DFT_GW_HPRO_Flow(Workflow):
    """
    This class is modified from BGWpy
    Features:
    - DFT calculations using Quantum Espresso
    - GW calculations using BerkeleyGW (optional)
    - Aobasis calculations using SIESTA
    - HPRO calculations using HPRO
    Input: see ./config/single_mat_config.json
    Output: workflow directory
    """

    def __init__(self, **kwargs):
        """
        Keyword arguments

        General:
        -----------------
        dirname : str
            Directory in which the files are written and the code is executed.
            Will be created if needed.
        stru_file : str
            structure file of crystal
        prefix : str
            Prefix required by QE as a rootname.

        QE:
        -----------------
        dft_flavor : 'espresso' 
            Choice of DFT code for density and wavefunctions calculations.
        pseudo_dir : str
            Directory in which pseudopotential files are found.
        pseudos : list, str
            Pseudopotential files.
        ngkpt : list(3), float
            K-points grid. Number of k-points along each primitive vector
            of the reciprocal lattice.
        kshift : list(3), float, optional
            Relative shift of the k-points grid along each direction,
            as a fraction of the smallest division along that direction.
        qshift : list(3), float
            Q-point used to treat the Gamma point.
        nbnd : int
            Number of bands to be computed.
        ecutwfc : float
            Energy cutoff for the wavefunctions
        ecuteps : float
            Energy cutoff for the dielectric function.
        ibnd_min : int
            Minimum band index for GW corrections.
        ibnd_max : int
            Maximum band index for GW corrections.

        SIESTA:
        -----------------
        basis_set_siesta : str
            Basis precision set (single or double Zeta)
        mesh_cutoff_siesta : float (Ry)
            xxx
        dm_tolerance_siesta : float           
            Self consistent calculation tolerance
        
        Optional:
        -----------------        
        truncation_flag : str, optional
            Which truncation flag to use in BerkeleyGW, e.g. "cell_slab_truncation".
        sigma_kpts : list of list(3), optional
            K-points to evaluate self-energy operator. Defaults to all
            k-points defined by the Monkhorst-Pack grid ngkpt.
        epsilon_extra_lines : list, optional
            Any other lines that should appear in the epsilon input file.
        epsilon_extra_variables : dict, optional
            Any other variables that should be declared in the epsilon input file.
        sigma_extra_lines : list, optional
            Any other lines that should appear in the sigma input file.
        sigma_extra_variables : dict, optional
            Any other variables that should be declared in the sigma input file.

        pseudobands : bool (default is True)
            do pseudobands

        max_scf_iter_siesta : int
            Max SCF Iteration steps            
        """
        #========================================Preparation========================================#
        # write all input to a config.json file
        self.config_input = copy.deepcopy(kwargs)

        # Setup QE paths
        kwargs.update({"PW":pjoin(kwargs.get('QE_path',''),'pw.x')})
        kwargs.update({"PW2BGW":pjoin(kwargs.get('QE_path',''),'pw2bgw.x')})
        kwargs.update({"BANDS":pjoin(kwargs.get('QE_path',''),'bands.x')})

        # Setup BGW paths

        # add "structure" to kwargs (historic reason)
        kwargs.update({'structure':Structure.from_file(kwargs['stru_file'])})

        super(DFT_GW_HPRO_Flow, self).__init__(**kwargs)

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

        # Quantum Espresso flavor
        if kwargs.get('GW', False):
            fnames = self.make_dft_tasks_espresso(**kwargs)
            kwargs.update(fnames)
        else:
            fnames = self.make_dft_tasks_espresso_DFTonly(**kwargs)
            kwargs.update(fnames)
        
        # ==== Aobasis(SIESTA) ==== #
        if kwargs.get('siesta_calculator', False):
            aobasis_dirname = self.make_ao_basis(**kwargs)
            kwargs.update(aobasis_dirname)

        # ==== GW calculations ==== #
        if kwargs.get('GW', False):
            self.make_gw_tasks_bgw(**kwargs)
 
        # ==== SIESTA/HPRO ==========
        if kwargs.get('hpro_calcator', False):
            self.make_hpro_task(**kwargs)

        # ==== BSE Caculcations ==========
        if kwargs.get('BSE', False):
            self.make_bse_tasks_bgw(**kwargs)

        if kwargs.get('compact_data', False):
            self.compact_data(**kwargs)
        

    def make_dft_tasks_espresso_DFTonly(self, **kwargs):
        """
        Initialize all DFT tasks using Quantum Espresso.
        Return a dictionary of file names.
        """

        if 'charge_density_fname' in kwargs:
            if 'data_file_fname' not in kwargs:
                raise Exception("Error, when providing charge_density_fname, data_file_fname is required.")

        else:
            self.scftask = QeScfTask(
                dirname = pjoin(self.dirname, '01-density'),
                ngkpt = self.ngkpt,
                kshift = self.kshift,
                **kwargs)

            self.add_task(self.scftask)

            # Add a scf2bgw task for scf (HPRO)
            self.scf2bgwtask = Qe2BgwTask(
                dirname = self.scftask.dirname,
                ngkpt = self.ngkpt,
                kshift = self.kshift,
                rhog_flag = True,
                **kwargs)
            self.add_task(self.scf2bgwtask, merge=False)
                
            kwargs.update(
                charge_density_fname = self.scftask.charge_density_fname,
                data_file_fname = self.scftask.data_file_fname,
                spin_polarization_fname = self.scftask.spin_polarization_fname)
            
        if kwargs.get('DFT_band', True):
            self.wfnband_task = QeBgwFlow_band(
                    dirname = pjoin(self.dirname, '05-band'),
                    ngkpt = self.ngkpt,
                    kshift = self.kshift,
                    nbnd = self.n_z_valence + 12,
                    **kwargs
            )
            self.add_task(self.wfnband_task)


        fnames = dict(VSC_fname = self.scftask.dirname+'/VSC')

        return fnames
    
    def make_dft_tasks_espresso(self, **kwargs):
        """
        Initialize all DFT tasks using Quantum Espresso.
        Return a dictionary of file names.
        """
        # if kwargs.get('SOC', False):
        #     kwargs['variables'] = kwargs.get('variables', {})
        #     kwargs['variables']['system'] = kwargs['variables'].get('system', {})
        #     kwargs['variables']['system'].update({'lspinorb': True, 'noncolin':True})


        if 'charge_density_fname' in kwargs:
            if 'data_file_fname' not in kwargs:
                raise Exception("Error, when providing charge_density_fname, data_file_fname is required.")

        else:
            self.scftask = QeScfTask(
                dirname = pjoin(self.dirname, '01-density'),
                ngkpt = self.ngkpt,
                kshift = self.kshift,
                **kwargs)

            self.add_task(self.scftask)

            # Add a scf2bgw task for scf (HPRO)
            self.scf2bgwtask = Qe2BgwTask(
                dirname = self.scftask.dirname,
                ngkpt = self.ngkpt,
                kshift = self.kshift,
                rhog_flag = True,
                **kwargs)
            self.add_task(self.scf2bgwtask, merge=False)
                
            kwargs.update(
                charge_density_fname = self.scftask.charge_density_fname,
                data_file_fname = self.scftask.data_file_fname,
                spin_polarization_fname = self.scftask.spin_polarization_fname)
            
        # Wavefunction tasks for Epsilon
        self.wfntask_ksh = QeBgwFlow(
            dirname = pjoin(self.dirname, '02-wfn'),
            ngkpt = self.ngkpt,
            kshift = self.kshift,
            nbnd = self.n_z_valence+self.nbnd,
            rhog_flag = True,
            paraband_nproc = True,
            **kwargs)

        if kwargs.get('paraband'):
            assert kwargs.get('pseudobands') # pseudobands are required when parabands is used

        if kwargs.get('paraband'):
            self.paraband_task = ParaBandTask(
                dirname= pjoin(self.dirname, '02-wfn'),
                **kwargs)

        else:
            if kwargs.get('pseudobands', True):
                self.paraband_task = PseudoBandTask(
                    dirname= pjoin(self.dirname, '02-wfn'),
                    wfn2hdfonly = True,
                    **kwargs)

        self.wfntask_qsh = QeBgwFlow(
            dirname = pjoin(self.dirname, '03-wfnq'),
            ngkpt = self.ngkpt,
            kshift = self.kshift,
            qshift = self.qshift,
            # nbnd = None,
            nbnd = self.n_z_valence + 4,
            **kwargs)

        if kwargs.get('pseudobands', True):
            self.pseudoband_q = PseudoBandTask(
                dirname= pjoin(self.dirname, '03-wfnq'),
                wfnq_dir = pjoin(self.dirname, '03-wfnq'),
                wfnk_dir = pjoin(self.dirname, '02-wfn'),
                wfn2hdfonly = False,
                **kwargs)

        if kwargs.get('pseudobands', True):
            self.add_tasks([self.wfntask_ksh, self.paraband_task, self.wfntask_qsh, self.pseudoband_q])
        else:
            self.add_tasks([self.wfntask_ksh, self.wfntask_qsh])

        if kwargs.get('use_NNS', True):
            self.wfntask_q_nns = QeBgwFlow_NNS(
                dirname = pjoin(self.dirname, '06-wfnq-nns'),
                ngkpt = self.ngkpt,
                kshift = self.kshift,
                # qshift = self.qshift,
                # nbnd = None,
                nbnd = self.n_z_valence + 4,
                wfn_fname_nns_input = self.wfntask_ksh.wfn_fname,
                **kwargs)
            self.add_task(self.wfntask_q_nns)


        self.wfnband_task = QeBgwFlow_band(
                dirname = pjoin(self.dirname, '05-band'),
                ngkpt = self.ngkpt,
                kshift = self.kshift,
                nbnd = self.n_z_valence + 12,
                **kwargs
        )
        self.add_task(self.wfnband_task)

        self.wfntask_ush = self.wfntask_ksh

        fnames = dict(VSC_fname = self.scftask.dirname+'/VSC',
                    vxc_fname = self.wfntask_ksh.dirname+'/VXC',
                    wfn_fname = self.wfntask_ksh.wfn_fname,
                    wfnq_fname = self.wfntask_qsh.wfn_fname,
                    wfn_co_fname = self.wfntask_ush.wfn_fname,
                    rho_fname = self.wfntask_ush.rho_fname,
                    # vxc_dat_fname = self.wfntask_ush.vxc_dat_fname,
                    wfn_band_fname = self.wfnband_task.wfn_fname,
                    )
        
        if kwargs.get('use_NNS', True):
            fnames.update(dict(wfn_nns_dir=self.wfntask_q_nns.dirname,
                    wfn_nns_fname=self.wfntask_q_nns.wfn_fname))

        return fnames
    
    def make_gw_tasks_bgw(self, **kwargs):
        # Set some common variables for Epsilon and Sigma
        self.epsilon_extra_lines = kwargs.pop('epsilon_extra_lines', [])
        self.epsilon_extra_variables = kwargs.pop('epsilon_extra_variables',{})
        
        self.sigma_extra_lines = kwargs.pop('sigma_extra_lines', [])
        self.sigma_extra_variables = kwargs.pop('sigma_extra_variables', {})
        
        # Dielectric matrix computation and inversion (epsilon)
        self.epsilontask = EpsilonTask(
            dirname = pjoin(self.dirname, '11-epsilon'),
            ngkpt = self.ngkpt,
            qshift = self.qshift,
            extra_lines = self.epsilon_extra_lines,
            extra_variables = self.epsilon_extra_variables,
            **kwargs)

        if kwargs.get('use_NNS', True):
            self.nns_helper_epsilon_task = nns_helper_epsilon(dirname=pjoin(self.dirname, '12-epsilon-nns'), **kwargs)
            self.epsilontask_nns = EpsilonTask_NNS(
                dirname = pjoin(self.dirname, '12-epsilon-nns'),
                ngkpt = self.ngkpt,
                qshift = self.qshift,
                extra_lines = self.epsilon_extra_lines,
                extra_variables = self.epsilon_extra_variables,
                **kwargs)
            kwargs.update(dict(eps0_nns_dir=pjoin(self.dirname, '12-epsilon-nns')))

        # Self-energy calculation (sigma)
        self.sigmatask = SigmaTask_NNS(
            dirname = pjoin(self.dirname, '13-sigma'),
            ngkpt = self.ngkpt,
            extra_lines = self.sigma_extra_lines,
            extra_variables = self.sigma_extra_variables,
            eps0mat_fname = self.epsilontask.eps0mat_fname,
            epsmat_fname = self.epsilontask.epsmat_fname,
            **kwargs)

        self.inteqp_task = IneqpTask_plot(
            dirname = pjoin(self.dirname, '14-inteqp'),
            eqp_co_fname = self.sigmatask.dirname+'/eqp1.dat',
            wfn_fi_fname = self.wfnband_task.wfn_fname,
            nbnd = self.n_z_valence,
            **kwargs
        )

        # Add tasks to the workflow
        if kwargs.get('use_NNS', True):
            self.add_tasks([self.epsilontask, 
                            self.nns_helper_epsilon_task,
                            self.epsilontask_nns ,
                            self.sigmatask,
                            self.inteqp_task], merge=False)
        else:
            self.add_tasks([self.epsilontask, self.sigmatask, self.inteqp_task], merge=False)

        self.truncation_flag = kwargs.get('truncation_flag')
        self.sigma_kpts = kwargs.get('sigma_kpts')

    def make_ao_basis(self, **kwargs):
        self.aobasis_task = AobasisTask(
             dirname = pjoin(self.dirname, '07-aobasis'),
             **kwargs)
        self.add_task(self.aobasis_task)
        # kwargs.update(dict(aobasis_dirname=self.aobasis_task.dirname))
        return dict(aobasis_dirname=self.aobasis_task.dirname)

    def make_hpro_task(self, **kwargs):
        self.hpro_task = HPROTask(
            dirname = pjoin(self.dirname, '16-reconstruction'),
             **kwargs)
        self.add_task(self.hpro_task)

    def make_bse_tasks_bgw(self, **kwargs):

        if "eqp_co_corrections" in kwargs['absorption_extra_lines']:
            assert kwargs['GW'], "eqp_co_corrections is only available when GW is True"

        if kwargs.get('SOC', False):
            kwargs['variables'] = kwargs.get('variables', {})
            kwargs['variables']['system'] = kwargs['variables'].get('system', {})
            kwargs['variables']['system'].update({'lspinorb': True, 'noncolin':True}) 
        kwargs.update(
                charge_density_fname = self.scftask.charge_density_fname,
                data_file_fname = self.scftask.data_file_fname,
                spin_polarization_fname = self.scftask.spin_polarization_fname,
                eps0mat_fname = self.epsilontask.eps0mat_fname,
                epsmat_fname = self.epsilontask.epsmat_fname)
        kwargs.pop('vxc_fname')
        kwargs.pop('wfn_fname')
        kwargs.pop('rho_fname')

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

        self.kernel_extra_lines = kwargs.pop('kernel_extra_lines', [])
        self.kernel_extra_variables = kwargs.pop('kernel_extra_variables',{})
        
        self.kerneltask = KernelTask(
            dirname = pjoin(self.dirname, '18-kernel'),
            extra_lines = self.kernel_extra_lines,
            extra_variables = self.kernel_extra_variables,
            **kwargs
        )

        
        self.absorption_extra_lines = kwargs.pop('absorption_extra_lines', [])
        self.absorption_extra_variables = kwargs.pop('absorption_extra_variables', {})

        kwargs.update(
                wfn_fi_fname = self.wfn_fi_task_sh.wfn_fname,
                wfnq_fi_fname = self.wfn_fi_task_sh.wfn_fname,
                bsemat_fname = self.kerneltask.bsemat_fname,
                eqp_fname = self.sigmatask.eqp1_fname
                )

        self.absorptiontask = AbsorptionTask(
            dirname = pjoin(self.dirname, '19-absorption'),
            extra_lines = self.absorption_extra_lines,
            extra_variables = self.absorption_extra_variables,
            **kwargs
        )
        self.add_tasks([self.kerneltask, self.absorptiontask], merge=False)

    def compact_data(self, **kwargs):
        if kwargs.get('unwanted_files', None):
            self.runscript.append('collect_tool.py compact -folder ./ -unwanted {}'.format(kwargs['unwanted_files']))
        else:
            self.runscript.append('collect_tool.py compact -folder ./')


    def summary(self, verbose):
        pass

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

def check_pseudo(pseudo_dir_src='./from_oncvpsp/', pseudos=['S.upf','H.upf']):
    pattern = 'z_valence'
    pseudos_z_valence = {}
    for pseudo in pseudos:
        result = subprocess.run(['grep', pattern, pseudo_dir_src+'/'+pseudo], capture_output=True, text=True)
        # print(result.stdout)
        match = re.search(r'[-+]?\d*\.?\d+', result.stdout)
        if match:
            number = float(match.group())  # Convert to float if needed
            # print(pseudo, 'z_valence:', number)
            pseudos_z_valence.update({pseudo:number})
        else:
            print('Warning: pseudo file is not found:', pseudo)
    # print(pseudos_z_valence)
    return pseudos_z_valence


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Create a workflow for a single material.')
    parser.add_help = True
    parser.add_argument('-c', '--config', type=str, default='./config/single_mat_config.json', help='input config file')
    args = parser.parse_args()
    print(args.config)

    assert os.path.exists(args.config), "config file not found"

    read_from_existing = True
    if read_from_existing: # allow to read config from existing file
        # config_path = './config/single_mat_config.json'
        config_path = args.config
        print('read from existing config file:', config_path)
        with open(config_path, 'r') as f:
            config = json.load(f)
        flow = DFT_GW_HPRO_Flow(**config)

    else: # used for debugging
        print('create a new config file')
        flow = DFT_GW_HPRO_Flow(
            mpirun='srun',
            nproc_flag = '-n',
            nproc=512,
            nproc_per_node_flag='',
            nproc_per_node='',
            Siesta = '/global/homes/b/bwhou/anaconda3/envs/siesta/bin/siesta',
            hpro = '/pscratch/sd/b/bwhou/12-deepGWBSE/Deep-GWBSE/HPRO/src/calc.py',
            PWFLAGS='-nk 16',
            PW='pw.x',
            prefix = 'hBN', ###
            dirname='flow-2', ###
            stru_file = './fp-input/mat-3/stru.cif', ###
            ecuteps = 20.0,
            ncbnd_sigma = 4,
            nvbnd_sigma = 5, 
            ngkpt = [12, 12, 1],
            qshift = [.001,.0,.0],
            nbnd = 100,
            ecutwfc = 75,
            pseudo_dir_source = './from_oncvpsp/',
            basis_set_siesta = 'DZP',
            mesh_cutoff_siesta = 320,
            dm_tolerance_siesta = 1e-6, 
            max_scf_iter_siesta = 300,
            epsilon_extra_lines=['restart','dont_check_norms','cell_slab_truncation'],
            sigma_extra_lines=['dont_check_norms','frequency_dependence 1','screening_semiconductor','cell_slab_truncation','dont_use_vxcdat'],
            use_NNS = True,
            pseudobands = True, # assert ture if parabands is ture
            N_P_cond = 10,
            N_S_cond = 40,
            N_xi_cond = 5,
            paraband = True,
            nparaband = 10000,
            kpath_band = ['0 0 0 20','0.5 0 0 20','0.33333 0.33333 0 20', '0 0 0 20', '-0.333333 -0.33333 0 20'],
            SOC = False,
            GW = True,
            DFT_band = True, # ignore this if GW is True
        )

    flow.write()


