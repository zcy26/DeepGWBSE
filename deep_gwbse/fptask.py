from os.path import join as pjoin
import ase.io
from ase.calculators.siesta import Siesta
from .from_bgwpy.core import MPITask, IOTask
import os
from .from_bgwpy.QE import QeScfTask, QeWfnTask, Qe2BgwTask, QeBgwFlow
from .from_bgwpy.DFT import WfnBgwFlow
from .from_bgwpy.BGW import EpsilonTask, SigmaTask, IneqpTask
from .from_bgwpy.BGW.bgwtask import BGWTask
from .from_bgwpy.QE.pseudobands_str import pseudoband_py
from .from_bgwpy.BGW.kgrid   import KgridTask, get_kpt_grid
from .from_bgwpy.BGW.inputs  import EpsilonInput
import os
import json
import subprocess
# import matplotlib as mpl

# with open('./from_bgwpy/QE/pseudobands.py','r') as file:
#     pseudoband_py = file.read()

class DeepTask(MPITask, IOTask):
    _TAG_JOB_COMPLETED = 'TOTAL'
    pass

def update_link_in_targe_dir(dirname, source, target):
        original_dir = os.getcwd()  # Save current directory
        files = set(os.listdir(dirname))
        try:
            os.chdir(dirname)  # Change to target directory
            if target not in files:
                os.symlink(source, target)
        finally:
            os.chdir(original_dir)  # Restore original directory

class AobasisTask(DeepTask):
    """
        Arguments
        ---------
        dirname : str
            Directory in which the files are written and the code is executed.
            Will be created if needed.

        Keyword arguments (see DFT_GW_HPRO_FLOW())
        -----------------
        (All mandatory unless specified otherwise)

        Properties
        ----------
    """
    def __init__(self, dirname, **kwargs):
        super(AobasisTask, self).__init__(dirname, **kwargs)

        self.dirname = dirname
        self.atoms = ase.io.read(kwargs['stru_file'])
        self.symbols = str(self.atoms.symbols)
        self.prefix = self.symbols if 'prefix' not in kwargs else kwargs['prefix']
        self.atom_symbols = list(set(self.atoms.get_chemical_symbols()))

        mpirun_flag = kwargs.get('mpirun', 'mpirun')
        siesta_flag = kwargs.get('Siesta','siesta')
        nproc_flag = kwargs.get('nproc_flag', ' -n ')

        self.calc = Siesta(directory=self.dirname,
                           label=self.prefix,
                           xc='PBE',
                           mesh_cutoff = kwargs['mesh_cutoff_siesta'],
                        #    basis_set=kwargs['basis_set_siesta'],
                           pseudo_path=os.path.relpath(kwargs['pseudo_dir'], self.dirname),
                        #    pseudo_qualifier = 'psf',
                           fdf_arguments={'MaxSCFIterations': kwargs['max_scf_iter_siesta'],
                                          'DM.MixingWeight':kwargs['dm_tolerance_siesta']})

        self.atoms.calc = self.calc
        self.runscript.fname = "aobasis.run"
        self.runscript.append(mpirun_flag+' '+nproc_flag+' 1 '+f'{siesta_flag} < {self.prefix}.fdf &> siesta.out')

    def write(self):
        # print('write siesta:', self.calc.getpath())
        super(AobasisTask, self).write()
        self.calc.write_input(self.atoms,'density')

        # relink .ion files for HPRO
        files = os.listdir(self.dirname) 
        # print(files)
        for f in files:
            if ('.psml' in f) or ('.psf' in f):
                update_link_in_targe_dir(self.dirname, '.'.join(f.split('.')[:2]+['ion']),
                        '.'.join(f.split('.')[:1]+['ion']))

class HPROTask(DeepTask):
    def __init__(self, dirname, **kwargs):
        super(HPROTask, self).__init__(dirname, **kwargs)
        self.dirname = dirname
        # self.link_test()
        mpirun_flag = kwargs.get('mpirun', 'mpirun')
        nproc_flag = kwargs.get('nproc_flag', ' -n ')

        self.PW2AO_kwargs = {
                'Warning': "you might modify fptask.py to change path if you change folder name of previous step",
                'lcao_interface':'siesta',
                'lcaodata_root':os.path.relpath(kwargs['aobasis_dirname'], self.dirname), 
                'hrdata_interface':'qe-bgw',
                # 'vscdir':'../01-density/VSC',
                'vscdir':os.path.relpath(kwargs['VSC_fname'], self.dirname),
                'upfdir':f"{os.path.relpath(kwargs['pseudo_dir'], self.dirname)}",
                'ecutwfn':kwargs.get('ecutwfn_hpro', 30),
                'outdir':f"./aohamiltonian"}
        self.runscript.fname = 'hpro.run'
        self.runscript.append(mpirun_flag+' '+nproc_flag+' 1 '+f"python {kwargs['hpro']} > hpro.out")

    def write(self):
        super(HPROTask, self).write()
        with open(self.dirname+'/calc.json', 'w') as file:
            json.dump(self.PW2AO_kwargs, file, indent=4)

        # update_link_in_targe_dir(self.dirname, './a', './b')

class PseudoBandTask(DeepTask):
    def __init__(self, dirname, **kwargs):
        super(PseudoBandTask, self).__init__(dirname, **kwargs)
        self.dirname = dirname
        self.wfn2hdfonly = kwargs.get('wfn2hdfonly', None)
        # wfn2hdf5
        mpirun_flag = kwargs.get('mpirun', 'mpirun')
        nproc_flag = kwargs.get('nproc_flag', ' -n ')
        self.runscript.fname = 'pseudo.sh'
        self.runscript.append(mpirun_flag+' '+nproc_flag+' 1 '+'wfn2hdf.x BIN wfn.cplx wfn.h5 &> wfn2hdf.out')

        # pseudonize
        if not kwargs.get('wfn2hdfonly'):
            cur_pat = self.dirname
            self.wfnq_fname = os.path.relpath(kwargs['wfnq_dir'] + '/wfn.h5', cur_pat)
            self.wfnk_fname = os.path.relpath(kwargs['wfnk_dir'] + '/wfn.h5', cur_pat)

            self.wfnq_fname_out = os.path.relpath(kwargs['wfnq_dir'] + '/wfn_q.h5', cur_pat)
            self.wfnk_fname_out = os.path.relpath(kwargs['wfnq_dir'] + '/wfn_k.h5', cur_pat)

            self.wfnq_fname_out_h5 = os.path.relpath(kwargs['wfnq_dir'] + '/wfn.cplx', cur_pat)
            self.wfnk_fname_out_h5 = os.path.relpath(kwargs['wfnk_dir'] + '/wfn.cplx', cur_pat)

            # TODO: add pseudobands setting; wfnq?
            self.runscript.append(mpirun_flag+' '+nproc_flag+' 1 '+f'python pseudobands.py --fname_in {self.wfnk_fname} --fname_in_q {self.wfnq_fname} --fname_out {self.wfnk_fname_out} --fname_out_q {self.wfnq_fname_out} --N_P_cond {kwargs.get("N_P_cond", 100)} --N_S_cond {kwargs.get("N_S_cond", 10)} --N_xi_cond {kwargs.get("N_xi_cond", 5)}  &> pseudo.out')
            # self.runscript.append(mpirun_flag+' '+nproc_flag+' 1 '+f'hdf2wfn.x BIN {self.wfnq_fname_out} {self.wfnq_fname_out_h5} &> wfn2hdf.out') # we don't do anything to wfnq
            self.runscript.append(mpirun_flag+' '+nproc_flag+' 1 '+f'hdf2wfn.x BIN {self.wfnk_fname_out} {self.wfnk_fname_out_h5} &> wfn2hdf.out')
            self.runscript.append(f'mv {self.wfnk_fname_out} {os.path.dirname(self.wfnk_fname_out_h5)}/wfn.h5')

            # self.runscript.append(mpirun_flag+' '+nproc_flag+' 1 '+'python pseudobands.py --fname_in WFN.h5 --fname_in_q WFNq.h5 --fname_out WFN_SPB.h5 --fname_out_q WFN_SPB_q.h5 --N_P_val 10 --N_P_cond 10 --N_S_val 10 --N_S_cond 150 --N_xi_val 2 --N_xi_cond 2')
    
    def write(self,): 
        super().write()

        if not self.wfn2hdfonly:        
            with open(self.dirname+'/pseudobands.py', 'w') as file:
                file.write(pseudoband_py)
        pass

class ParaBandTask(DeepTask):
    def __init__(self, dirname, **kwargs):
        super(ParaBandTask, self).__init__(dirname, **kwargs)

        self.input_file = ['input_wfn_file wfn.cplx\n',
                           'output_wfn_file wfn.h5\n',
                           'vsc_file VSC\n',
                           'vkb_file VKB\n',
                           f'number_bands {kwargs.get("nparaband", 1000)}\n']
        mpirun_flag = kwargs.get('mpirun', 'mpirun')
        nproc_flag = kwargs.get('nproc_flag', ' -n ')
        nproc = kwargs.get('nproc')
        self.runscript.fname = 'para.sh'
        self.runscript.append(mpirun_flag+' '+nproc_flag+' '+str(nproc)+' '+'parabands.cplx.x &> parabands.out')

    def write(self):
        super().write()
        with open(self.dirname+'/parabands.inp','w') as file:
            file.writelines(self.input_file)

class nns_helper(DeepTask):
    def __init__(self, dirname, **kwargs):
        super().__init__(**kwargs)
        self.dirname = dirname
    def write(self):
        return super().write()
    
class QeBgwFlow_NNS(WfnBgwFlow):
    _charge_density_fname = ''
    _spin_polarization_fname = ''
    _data_file_fname = ''

    def __init__(self, **kwargs):
 
        super(QeBgwFlow_NNS, self).__init__(**kwargs)

        kwargs.pop('dirname', None)

        self.wfn_fname_nns_input = kwargs['wfn_fname_nns_input']
        self.charge_density_fname = kwargs['charge_density_fname']
        self.data_file_fname = kwargs['data_file_fname']
        self.spin_polarization_fname = kwargs.get('spin_polarization_fname', 'dummy')


        # nns_helper
        self.nns_helper = nns_helper(dirname= self.dirname, **kwargs)
        self.nns_helper.runscript.fname = 'nns_helper.sh'
        self.nns_helper.runscript.append(' '.join(['cp','./wfn.head.in','wfn.in']))
        self.nns_helper.runscript.append(' '.join(['setup_subsampling_nns.x','BIN', os.path.relpath(self.wfn_fname_nns_input, self.dirname),'&> nns_kpt.out']))
        self.nns_helper.runscript.append(' '.join(['cat','kpoints_all.dat','>>','wfn.in']))
        self.add_task(self.nns_helper, merge=False)

        # Wfn task
        self.wfntask = QeWfnTask(dirname = self.dirname, **kwargs)
        self.wfntask.runscript.fname = 'wfn.run.sh'
        self.add_task(self.wfntask, merge=False)

        # Wfn 2 BGW
        self.wfnbgwntask = Qe2BgwTask(dirname = self.wfntask.dirname, **kwargs)
        self.wfnbgwntask.runscript.fname = 'pw2bgw.run.sh'
        self.add_task(self.wfnbgwntask, merge=False)


    def write(self):
        super().write()

        # rewrite NNS wfn.in
        file_path = self.dirname+"/wfn.in"
        headfile_path = self.dirname+"/wfn.head.in"
        with open(file_path, "r") as file:
            lines = file.readlines()
        K_Points_index = lines.index("K_POINTS crystal\n")
        lines = lines[:K_Points_index] + lines[K_Points_index+int(lines[K_Points_index+1])+2:]
        with open(file_path, "w") as file:
            file.writelines(lines)        
        with open(headfile_path, "w") as file:
            file.writelines(lines)      


    @property
    def charge_density_fname(self):
        """The charge density file used by QE."""
        return self._charge_density_fname

    @charge_density_fname.setter
    def charge_density_fname(self, value):
        self._charge_density_fname = value

    @property
    def spin_polarization_fname(self):
        """The spin polarization file used by QE."""
        return self._spin_polarization_fname

    @spin_polarization_fname.setter
    def spin_polarization_fname(self, value):
        self._spin_polarization_fname = value

    @property
    def data_file_fname(self):
        """The XML data file used by QE."""
        return self._data_file_fname

    @data_file_fname.setter
    def data_file_fname(self, value):
        self._data_file_fname = value


    @property
    def rho_fname(self):
        """The charge density file name for BerkeleyGW."""
        return self.wfnbgwntask.rho_fname

    @property
    def wfn_fname(self):
        """The wavefunctions file name for BerkeleyGW."""
        return self.wfnbgwntask.wfn_fname

    @property
    def vxc_fname(self):
        raise NotImplementedError(
            'Please use vxc_dat_fname instead of vxc_fname.')

    @property
    def vxc_dat_fname(self):
        """The xc potential file name for BerkeleyGW."""
        return self.wfnbgwntask.vxc_dat_fname

class SigmaTask_NNS(SigmaTask):
    def __init__(self, dirname, **kwargs):
        if kwargs.get('use_NNS', True):
            # if 'extra_lines' not in kwargs:
            if 'subsample' not in kwargs['extra_lines']:
                kwargs['extra_lines'].append('subsample')
            
        super().__init__(dirname, **kwargs)

        if kwargs.get('use_NNS', True):
            self.kwargs = kwargs
            self.eps0mat_fname = kwargs['eps0_nns_dir']+'/eps0mat.h5'
            self.subweight_fname = kwargs['wfn_nns_dir']+'/subweights.dat'

    def write(self):
        super().write()
        # subprocess.run(['cp',self.kwargs['wfn_nns_dir']+'/subweights.dat',self.dirname])

    @property
    def eps0mat_fname(self):
        return self._eps0mat_fname

    @eps0mat_fname.setter
    def eps0mat_fname(self, value):
        self._eps0mat_fname = value
        dest = 'eps0mat.h5' if self._use_hdf5 else 'eps0mat'
        self.update_link(value, dest)

class nns_helper_epsilon(DeepTask):
    def __init__(self, dirname, **kwargs):
        super().__init__(**kwargs)
        if kwargs.get('use_NNS', True):
            self.dirname = dirname
            self.runscript.fname = 'nns_helper.sh'
            self.runscript.append(' '.join(['cp','epsilon.head.in','epsilon.inp']))
            # self.runscript.append(' '.join(['setup_subsampling_nns.x','BIN', os.path.relpath(self.wfn_fname_nns_input, self.dirname)]))
            self.runscript.append(' '.join(['cat',os.path.relpath(kwargs['wfn_nns_dir']+'/epsilon_q0s.inp', self.dirname),'>>','epsilon.inp']))

    def write(self):
        return super().write()

class EpsilonTask_NNS(EpsilonTask):
    """Inverse dielectric function calculation."""
    _TASK_NAME = 'Epsilon'
    _input_fname  = 'epsilon.inp'
    _output_fname = 'epsilon.out'
    def __init__(self, dirname, **kwargs):
        self.kwargs = kwargs
        super(EpsilonTask_NNS, self).__init__(dirname, **kwargs)

        if kwargs.get('use_NNS', True):        
            self.wfnq_fname = kwargs['wfn_nns_fname']

    def write(self):
        super().write()

        # subprocess.run(['cp',self.kwargs['wfn_nns_dir']+'/epsilon_q0s.inp',self.dirname])
        if self.kwargs.get('use_NNS', True):
            # rewrite NNS epsilon
            file_path = self.dirname+f"/{EpsilonTask._input_fname}"
            headfile_path = self.dirname+"/epsilon.head.in"
            epsilon_q0s_inp = self.dirname+f'/epsilon_q0s.inp'
            with open(file_path, "r") as file:
                lines = file.readlines()
            begin_index, end_index = lines.index("begin qpoints\n"), lines.index("end\n")
            lines = lines[:begin_index] + lines[end_index+1:]

            with open(file_path, "w") as file:
                file.writelines(lines)        
            with open(headfile_path, "w") as file:
                file.writelines(lines)

    @property
    def wfnq_fname(self):
        return self._wfnq_fname

    @wfnq_fname.setter
    def wfnq_fname(self, value):
        self._wfnq_fname = value
        self.update_link(value, 'WFNq')




class QeBgwFlow_band(WfnBgwFlow):
    _charge_density_fname = ''
    _spin_polarization_fname = ''
    _data_file_fname = ''

    def __init__(self, **kwargs):
 
        super(QeBgwFlow_band, self).__init__(**kwargs)

        kwargs.pop('dirname', None)
        mpirun_flag = kwargs.get('mpirun', 'mpirun')
        nproc_flag = kwargs.get('nproc_flag', ' -n ')

        self.charge_density_fname = kwargs['charge_density_fname']
        self.data_file_fname = kwargs['data_file_fname']
        self.spin_polarization_fname = kwargs.get('spin_polarization_fname', 'dummy')
        assert kwargs['kpath_band'] # ["kx ky kz nk", ...]
        self.prefix = kwargs['prefix']

        #  2025/02/27: setup kpath_band by user
        # for kpt in kwargs['kpath_band']:
        #     assert len(kpt.split()) == 4
        # self.kpath_band = ['K_POINTS crystal_b'] + [str(len(kwargs['kpath_band']))] + kwargs['kpath_band']
        self.kpath_band = kwargs['kpath_band']

        # band_helper
        self.nns_helper = nns_helper(dirname= self.dirname, **kwargs)
        self.nns_helper.runscript.fname = 'band_helper.sh'
        self.nns_helper.runscript.append(' '.join(['cp','./wfn.head.in','wfn.in']))
        self.nns_helper.runscript.append(' '.join(['cat','kpath.txt','>>','wfn.in']))
        self.add_task(self.nns_helper, merge=False)

        # Wfn task
        self.wfntask = QeWfnTask(dirname = self.dirname, **kwargs)
        self.wfntask.runscript.fname = 'wfn.run.sh'
        self.add_task(self.wfntask, merge=False)

        # Wfn 2 BGW
        self.wfnbgwntask = Qe2BgwTask(dirname = self.wfntask.dirname, **kwargs)
        self.wfnbgwntask.runscript.fname = 'pw2bgw.run.sh'
        self.wfnbgwntask.runscript.append(f"\nBANDS='{kwargs['BANDS']}'")
        self.wfnbgwntask.runscript.append(' '.join(['$MPIRUN','$BANDS','$PWFLAGS','-in','bands.in','&>','bands.out']))
        self.wfnbgwntask.runscript.append(mpirun_flag+' '+nproc_flag+' 1 '+'wfn2hdf.x BIN wfn.cplx wfn.h5 &> wfn2hdf.out')
        self.add_task(self.wfnbgwntask, merge=False)


    def write(self):
        super().write()

        # rewrite NNS wfn.in
        file_path = self.dirname+"/wfn.in"
        headfile_path = self.dirname+"/wfn.head.in"
        with open(file_path, "r") as file:
            lines = file.readlines()
        K_Points_index = lines.index("K_POINTS crystal\n")
        lines = lines[:K_Points_index] + lines[K_Points_index+int(lines[K_Points_index+1])+2:]
        with open(file_path, "w") as file:
            file.writelines(lines)        
        with open(headfile_path, "w") as file:
            file.writelines(lines)      
        
        with open(self.dirname+"/kpath.txt",'w') as file:
            file.writelines([line + '\n' for line in self.kpath_band])
        
        bands_in_file = ['&bands\n',f'  prefix = "{self.prefix}",\n', '  outdir = "./",\n' ,'  filband = "bands.dat",\n','  lsym = .false.,\n','/\n']
        with open(self.dirname+"/bands.in",'w') as file:
            file.writelines([line  for line in bands_in_file])



    @property
    def charge_density_fname(self):
        """The charge density file used by QE."""
        return self._charge_density_fname

    @charge_density_fname.setter
    def charge_density_fname(self, value):
        self._charge_density_fname = value

    @property
    def spin_polarization_fname(self):
        """The spin polarization file used by QE."""
        return self._spin_polarization_fname

    @spin_polarization_fname.setter
    def spin_polarization_fname(self, value):
        self._spin_polarization_fname = value

    @property
    def data_file_fname(self):
        """The XML data file used by QE."""
        return self._data_file_fname

    @data_file_fname.setter
    def data_file_fname(self, value):
        self._data_file_fname = value


    @property
    def rho_fname(self):
        """The charge density file name for BerkeleyGW."""
        return self.wfnbgwntask.rho_fname

    @property
    def wfn_fname(self):
        """The wavefunctions file name for BerkeleyGW."""
        return self.wfnbgwntask.wfn_fname

    @property
    def vxc_fname(self):
        raise NotImplementedError(
            'Please use vxc_dat_fname instead of vxc_fname.')

    @property
    def vxc_dat_fname(self):
        """The xc potential file name for BerkeleyGW."""
        return self.wfnbgwntask.vxc_dat_fname


# class QeBgwFlow_hdf5(QeBgwFlow):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)




class IneqpTask_plot(IneqpTask):
    def __init__(self, dirname, **kwargs):
        self.dirname = dirname
        self.kwargs = kwargs
        super().__init__(dirname, **kwargs)
        self.runscript.append('python plot.py')
    def write(self):
        super().write()
        with open(self.dirname+'/plot.py','w') as f:
            f.write(f'''
import numpy as np
data = np.loadtxt('bandstructure.dat')
bands = data[:,1]
kpts = data[:,2:5]
emf = data[:,5]
eqp = data[:,6]
emf -= np.amax(emf[bands=={self.kwargs['nbnd']}])
eqp -= np.amax(eqp[bands=={self.kwargs['nbnd']}])
def get_x(ks):
    global dk_len
    dk_vec = np.diff(ks, axis=0)
    dk_len = np.linalg.norm(dk_vec, axis=1)
    return np.insert(np.cumsum(dk_len), 0, 0.)
xmin, xmax = np.inf, -np.inf
bands_uniq = np.unique(bands).astype(int)
f = open("band.dat","w")
for ib in bands_uniq:
    cond = bands==ib
    x = get_x(kpts[cond])
    xmin, xmax = min(xmin, x[0]), max(xmax, x[-1])
    for i_n in range(len(x)):
        f.write("%.9f %.9f %.9f \\n" % (x[i_n], emf[cond][i_n], eqp[cond][i_n]))
    f.write("\\n")
f.close()
    ''')



if __name__ == "__main__":
    aobasistask = AobasisTask(dirname='./aobasis', 
                              stru_file='./fp-input/mat-1/stru.cif',
                              mesh_cutoff_siesta=300,                      
                              basis_set_siesta='DZP',
                              pseudo_dir='./from_oncvpsp',
                              max_scf_iter_siesta=100,
                              dm_tolerance_siesta=1e-6)
    aobasistask.write()
