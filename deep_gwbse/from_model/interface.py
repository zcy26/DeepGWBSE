import numpy as np
import numpy as np
import h5py as h5
import os
import matplotlib.pyplot as plt
from scipy.io import FortranFile
from deep_gwbse.from_model.model_util import H5ls, time_watch, memory_watch, eV2Ry
from tqdm import tqdm
import logging
from deep_gwbse.from_model import wigner, wigner3d
from scipy.ndimage import zoom
import time
import matplotlib.pyplot as plt
from pathos.multiprocessing import ProcessingPool as Pool
from abc import ABC, abstractmethod
Ry2eV = 13.605693009


# Define ABC Interface
class BGWIO(ABC):
    """
    Abstract base class for BGWIO
    """
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_dataset(self):
        pass

class eqp(BGWIO):
    """
    These object decompose eqp.dat into data_DFT, data_GW, klist and spin_list
    """
    def __init__(self,fname):
        # These variables are initialized
        # (1) nbnd, nk
        # (2) data_GW, data_DFT -> (nk, nbnd)
        # (3) band_index, spin ->(nband,);
        # (4) klist -> (nk,)

        # self.write()
        logging.debug(f'Loading eqp.dat {fname}')
        self.fname = fname
        self.read_eqp()

    def read_eqp(self):
        f = open(self.fname, 'r')
        lines = f.readlines()
        f.close()

        self.nbnd = int(lines[0].split()[-1])
        self.nk = int(len(lines) / (self.nbnd + 1))

        logging.debug(f'number of bands:{self.nbnd}')
        logging.debug(f'number of kpoints:{self.nk}')

        self.data_GW = np.zeros((self.nk, self.nbnd))
        self.data_DFT = np.zeros((self.nk, self.nbnd))
        self.band_index = np.zeros((self.nk, self.nbnd),dtype=int)
        self.spin_index = np.zeros((self.nk, self.nbnd),dtype=int)
        self.klist = []

        # get
        line = 0
        for i in range(self.nk):
            temp_k = lines[line]
            self.klist.append("  ".join(temp_k.split()[:3]))
            # print(" ".join(temp_k.split()[:3]))
            line += 1
            for j in range(self.nbnd):
                self.band_index[i,j] = int(lines[line].split()[1])
                self.spin_index[i,j] = int(lines[line].split()[0])
                self.data_DFT[i,j] = lines[line].split()[-2]
                self.data_GW[i,j] = lines[line].split()[-1]
                line += 1
        self.data_DFT = np.around(self.data_DFT,9)

    def write_eqp(self):
        f_new = open('eqp_new.dat','w')
        line = 0
        for i in range(self.nk):
            f_new.write('  '+self.klist[i]+'      %s'%self.nbnd+'\n')
            line += 1
            for j in range(self.nbnd):
                f_new.write('       %s     %s    %.9f    %.9f\n' % (self.spin_index[i,j], self.band_index[i,j], self.data_DFT[i,j], self.data_GW[i,j]))
                # print('%s %s %s %s' % (self.spin_index[j], self.band_index[j], self.data_DFT[i,j], self.data_GW[i,j]))
                line += 1

    def plot_eig(self):
        f = open('band.dat', 'w')
        for i in range(self.nbnd):
            for j in range(self.nk):
                f.write("%s %s %s\n" % (j + 1, self.data_DFT[j, i], self.data_GW[j, i]))
            f.write('\n')
        f.close()

    def get_dataset(self,)->dict:
        """
        Get the dataset of the eqp.dat for ML
        Input:
            nc: number of conduction bands
            nv: number of valence bands
        Output:
            {
                "qp": (nk, nb, 1),
                "mf": (nk, nb, 1),
                "corr: (nk, nb, 1), # :qp-mf
                "band_indices_abs": (nk, nb, 1),
            }
        """
        dataset = {
            "qp": self.data_GW[:, :, None],
            "mf": self.data_DFT[:, :, None],
            "corr": (self.data_GW - self.data_DFT)[:, :, None],
            "band_indices_abs": self.band_index[:, :, None],
        }
        return dataset


class vloc:
    """
    modified from /HPRO/bgwio.py
    vscfile: VXC, VSC files
    """
    def __init__(self, vscfile):
        self.fname = vscfile
        self.file = FortranFile(vscfile, 'r')
        self._read = False
        self._set_vcsg = False
        self.read_v()


    def read_v(self):        
        assert not self._read
        rec = self.file.read_record('S1')
        self.stitle_sdate_stime = rec.copy()
        rec = rec.tobytes().decode()
        self.stitle = rec[0:32].rstrip(' ')
        self.sdate = rec[32:64].rstrip(' ')
        self.stime = rec[64:96].rstrip(' ')
        # print(self.stitle, self.sdate, self.stime)
        
        rec = self.file.read_record('i4', 'i4', 'i4', 'i4', 'i4', 'f8')
        rec = map(lambda x: x.item(), rec)
        self.nsf, self.ng_g, self.ntran, self.cell_symmetry, self.nat, self.ecutrho = rec # nk = nk_g / ns
        # print(self.nsf, self.ng_g, self.ntran, self.cell_symmetry, self.nat, self.ecutrho)
        
        rec = self.file.read_record(*(['i4']*3))
        rec = map(lambda x: x.item(), rec)
        self.nr1, self.nr2, self.nr3 = rec

        rec = self.file.read_record('f8')
        self.omega = rec[0] # cell volume, in bohr^3
        self.alat = rec[1] # in bohr
        self.at = rec[2:11].reshape(3, 3)
        self.adot = rec[11:20].reshape(3, 3)
        # print(omega, alat)
        # print(at)
        # print(adot) # ai dot aj, in bohr^2
        
        rec = self.file.read_record('f8')
        self.recvol = rec[0]
        self.tpiba = rec[1] # in bohr-1
        self.bg = rec[2:11].reshape(3, 3) # reciprocal lattice vecs
        self.bdot = rec[11:20].reshape(3, 3)
        # print(recvol, tpiba)
        # print(bg)
        # print(adot)
        
        rec = self.file.read_record('i4')
        self.rotmat = rec.reshape(self.ntran, 3, 3) # rotation matrices
        # print(rotmat)
        
        rec = self.file.read_record('f8')
        self.frac_tran = rec.reshape(self.ntran, 3) # fractional translations
        # print(frac_tran)
        
        rec = self.file.read_record(*(['f8']*3 + ['i4'])*self.nat)
        self.tau = np.empty((self.nat, 3), dtype=float) # (nat, 3) atomic positions (alat)
        self.atomic_number = np.empty(self.nat, dtype=int) # (nat) atomic numbers
        for iat in range(self.nat):
            for d in range(3):
                self.tau[iat, d] = rec[iat*4+d]
            self.atomic_number[iat] = rec[iat*4+3]
        # print(tau) # alat
        # print(atomic_number)

        
        nrecord = self.file.read_record('i4').item()
        self.ng_g = self.file.read_record('i4').item() # number of charge_density g-vecs
        self.g_g = self.file.read_record('i4').reshape(self.ng_g, 3)
        # print(nrecord, ng_g)

        nrecord = self.file.read_record('i4').item()
        self.ng_g = self.file.read_record('i4').item() # number of charge_density g-vecs
        self._vscg = self.file.read_record('c16').reshape(self.nsf, self.ng_g) * 0.5 # Ry->Ha

        self._read = True
        FFTgrid = np.array([self.nr1, self.nr2, self.nr3])
        _, self.g_g_full = np.divmod(self.g_g, FFTgrid)
        self.FFTgrid = FFTgrid
        self.file.close()

        
    def write_v(self, outfile=None):
        assert self._read
        print('Writing VXC/VSC')
        self.check_reset()
        if outfile is None:
            outfile = self.fname + '.new'
        with FortranFile(outfile, 'w') as f:
            # Write header information
            # rec = (self.stitle.ljust(32) + self.sdate.ljust(32) + self.stime.ljust(32)).encode()
            # rec = (self.stitle.ljust(32)+ self.sdate.ljust(32) + self.stime.ljust(32)).encode()
            # f.write_record(np.array(list(rec), dtype='S1'))
            f.write_record(self.stitle_sdate_stime)
            
            # Write integer and float metadata
            f.write_record(np.array([self.nsf, self.ng_g, self.ntran, self.cell_symmetry, self.nat], dtype='i4'),
                        np.array([self.ecutrho], dtype='f8'))
            f.write_record(np.array([self.nr1, self.nr2, self.nr3], dtype='i4'))
            
            # Write lattice and atomic data
            f.write_record(np.array([self.omega, self.alat] + self.at.flatten().tolist() + self.adot.flatten().tolist(), dtype='f8'))
            f.write_record(np.array([self.recvol, self.tpiba] + self.bg.flatten().tolist() + self.bdot.flatten().tolist(), dtype='f8'))
            f.write_record(self.rotmat.flatten().astype('i4'))
            f.write_record(self.frac_tran.flatten().astype('f8'))
            
            # Write atomic positions and numbers
            # TODO
            flat_data = []
            for iat in range(self.nat):
                flat_data.extend(np.array(self.tau[iat], dtype='f8'))  # Extend with 3 components of tau
                flat_data.append(np.array(self.atomic_number[iat], dtype='i4'))  # Append atomic number
            f.write_record(*flat_data)
            
            # Write g-vector information
            f.write_record(np.array([1], dtype='i4'))  # Dummy value for compatibility
            f.write_record(np.array([self.ng_g], dtype='i4'))
            f.write_record(self.g_g.astype('i4'))
            
            f.write_record(np.array([1], dtype='i4'))  # Another dummy value
            f.write_record(np.array([self.ng_g], dtype='i4'))
            # print(self.vscg) 
            f.write_record((self._vscg*2).astype('c16')) # Ha->Ry

    def get_vlocr(self, plotXY=False):
        assert self._read
        self.check_reset()
        vlocg_full = np.zeros(self.FFTgrid, dtype='c16')
        vlocg_full[self.g_g_full[:, 0], self.g_g_full[:, 1], self.g_g_full[:, 2]] = self.vscg
        vlocr = np.fft.ifftn(vlocg_full, s=self.FFTgrid, norm='forward')
        assert np.max(np.abs(vlocr.imag)) < 1e-6
        vlocr = vlocr.real
        if plotXY:
            plt.figure()
            plt.imshow(vlocr.sum(axis=2))
            plt.colorbar()
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'Vlocr Reset: {self._set_vcsg}')
            plt.show()
        # print('vlocr shape:', vlocr.shape)
        self.vlocg_full = vlocg_full
        self.vlocr = vlocr
        return vlocr

    def check_reset(self):
        if self._set_vcsg:
            if np.allclose(self.old_vcsg, self._vscg):
                print('vscg is reset: vscg is the same')
            else:
                print('vscg is reset: vscg is different')
        else:
            print('vscg is not reset')

    def IO_test(self):
        assert self._read
        self.write_v('./VXC.test')
        v_test = vloc('./VXC.test')
        for attr in dir(v_test):
            # print(attr, getattr(self, attr), getattr(v_test, attr))
            if not attr.startswith('_'):
                if type(getattr(self, attr)) == np.ndarray:
                    # dtype != s1
                    if getattr(self, attr).dtype != 'S1':
                        assert np.allclose(getattr(self, attr), getattr(v_test, attr))
                    # assert np.allclose(getattr(self, attr), getattr(v_test, attr))
                    pass
        os.remove('./VXC.test')
        print('IO test VXC/VSC: pass')

    @property
    def vscg(self):
        return self._vscg

    @vscg.setter
    def vscg(self, value):
        assert self._read # Only allow setting if the file has been read
        assert value.shape == self._vscg.shape 
        self.old_vcsg = self._vscg
        self._vscg = value
        if np.allclose(self.old_vcsg, self._vscg):
            print('resetting: vscg is the same')
        else:
            print('resetting: vscg is different')
        self._set_vcsg = True

class wfn(BGWIO):
    def __init__(self, wfn_file_h5: str):
        """
        wfn_file_h5: path of BGW wfn.h5 file
        """
        # Status flags
        self._read_header = False
        self._get_wfn_g = False

        self.wfn_file_h5 = wfn_file_h5
        self.wfn_file = h5.File(wfn_file_h5, 'r')

        # Get the names of the datasets
        h5ls = H5ls()
        self.wfn_file.visititems(h5ls)   
        self.names = h5ls.names

        # Get the header information
        self.crystal = {}
        self.gspace = {}
        self.kpoints = {}
        self.symmetry = {}
        self.wfns = {}
        self.read_header()

        self.wfn_file.close()

    def read_header(self):
        for name in self.names:
            if 'crystal' in name.split('/'):
                self.crystal[name.split('/')[-1]] = self.wfn_file[name][()]
            elif 'gspace' in name.split('/'):
                self.gspace[name.split('/')[-1]] = self.wfn_file[name][()]
            elif 'kpoints' in name.split('/'):
                self.kpoints[name.split('/')[-1]] = self.wfn_file[name][()]
            elif 'symmetry' in name.split('/'):
                self.symmetry[name.split('/')[-1]] = self.wfn_file[name][()]
            elif 'wfns/gvecs' == name:
                self.wfns[name.split('/')[-1]] = self.wfn_file[name][()]
            else:
                pass
        cum_sum = np.cumsum(np.concatenate((np.array([0]),self.kpoints['ngk'])))
        self.nkg_slice = [[cum_sum[i], cum_sum[i+1]] for i in range(self.kpoints['nrk'])]
        self._read_header = True

        self.nk = self.kpoints['occ'].shape[1]
        self.nb = self.kpoints['occ'].shape[2]
        self.g_g = self.wfns['gvecs']
        self.el = self.kpoints['el'][0] * Ry2eV # eV
        self.k_weights = self.kpoints['w']
        self.FFTgrid = self.gspace['FFTgrid']
        self.lattice = self.crystal['avec'] * self.crystal['alat'] * 0.52917721067 # lattice in Cartesian coordinates, in angstrom

        self.ifmax = self.kpoints['ifmax']
        self.nspin = self.kpoints['nspin']
        self.hovb = int(np.ceil(np.sum(self.ifmax)/self.nspin/self.nk))
    
    # @time_watch, tqdm is used instead to show the progress
    # @memory_watch()
    def get_wfn_g_in_grid(self, nband_max:int=100):
        """
        TODO: set nc, nv to get wfn
        Calculate the wavefunction in full G-grid   
        Input: 
            nband_max: maximum number of bands to read
        Output:
            create self.wfn_nk_ggrid: (nband_max, nk, FFTgrid[0], FFTgrid[1], FFTgrid[2])
        """
        assert self._read_header
        logging.debug('Loading wfn.h5...')
        # Be careful with the shape of the wavefunction coefficients, 
        f = h5.File(self.wfn_file_h5, 'r')
        self.wfns['coeffs'] = f['wfns/coeffs'][:nband_max,...]
        f.close()

        FFTgrid = self.gspace['FFTgrid']
        self.wfn_nk_ggrid = np.zeros((nband_max, self.nk, FFTgrid[0], FFTgrid[1], FFTgrid[2]), dtype='c16')
        _, g_g_full = np.divmod(self.g_g, FFTgrid)

        # print('Raw Wavefunction:', self.wfns['coeffs'].nbytes/1024/1024, 'MB')
        # for ib in tqdm(range(nband_max), desc='Building Wavefunction in Full G-grid'):
        for ib in range(min(nband_max, self.nb)):
            wfn_b = self.wfns['coeffs'][ib,0,:,0] + self.wfns['coeffs'][ib,0,:,1]*1j
            for ik in range(self.nk):
                gx = g_g_full[self.nkg_slice[ik][0]:self.nkg_slice[ik][1], 0]
                gy = g_g_full[self.nkg_slice[ik][0]:self.nkg_slice[ik][1], 1]
                gz = g_g_full[self.nkg_slice[ik][0]:self.nkg_slice[ik][1], 2]
                self.wfn_nk_ggrid[ib,ik,gx,gy,gz] = wfn_b[self.nkg_slice[ik][0]:self.nkg_slice[ik][1]]

        self._get_wfn_g = True

    def get_wfn_r_in_grid(self, nc:int=1 ,nv:int=1, **kwargs)-> np.ndarray:
        """
        Calculate the real-space wavefunction near Fermi level (all kpoints)a
        Input:
            nc: number of conduction bands
            nv: number of valence bands
            nk: number of kpoints  (default -1: all kpoints) (TODO)
        Output:
            wfn_r: (nk, nb, FFTgrid[0], FFTgrid[1], FFTgrid[2]), |phi(r)|^2
            el: (nk, nb, 1) eigenvalues corresponding to the wfn_r
        """
        if False: # add kpoints selection
            raise NotImplementedError('Only support all kpoints')

        if not self._get_wfn_g:
            logging.debug(f'Getting wfn in G-grid nband: {self.hovb + nc + 10}')
            self.get_wfn_g_in_grid(nband_max=self.hovb + nc + 10)
        else:
            if nc + nv + self.hovb + 10 > self.wfn_nk_ggrid.shape[0]:
                logging.debug(f'Getting wfn in G-grid nband: {self.hovb + nc + 10}')
                self.get_wfn_g_in_grid(nband_max=self.hovb + nc + 10)
            else:
                logging.debug('Using existing wfn in G-grid')

        wfn_r = np.zeros_like(self.wfn_nk_ggrid[self.hovb-nv:self.hovb+nc])
        el_r = np.zeros(wfn_r.shape[:2])[..., None] # (nb, nk, 1)
        occ = np.zeros_like(el_r)
        for ik in range(self.nk):
            wf_qp_g = self.wfn_nk_ggrid[self.hovb-nv:self.hovb+nc, ik] # make the order consistent with the G-grid
            # print(el_r.shape, self.el.shape)
            # print(el_r[:, ik, 0].shape, self.el[ik, self.hovb-nv:self.hovb+nc].shape)
            # print(self.hovb-nv, self.hovb+nc)
            el_r[:, ik, 0] = self.el[ik, self.hovb-nv:self.hovb+nc]
            occ[:, ik, 0] = self.kpoints['occ'][0, ik, self.hovb-nv:self.hovb+nc]
            wf_qp_r = np.fft.ifftn(wf_qp_g, s=self.FFTgrid, norm='forward', axes=(1,2,3)) / np.sqrt(np.prod(self.FFTgrid))
            wfn_r[:, ik] = wf_qp_r

        wfn_r = wfn_r.transpose(1,0,2,3,4)
        el_r = el_r.transpose(1,0,2)
        occ = occ.transpose(1,0,2)

        assert ((abs(wfn_r)**2).sum(axis=(2,3,4)) - 1 < 1e-6).all(), 'Check Norm Failed'
        return abs(wfn_r)**2, el_r, occ
    
    def get_dipole(self, nc, nv):
        with h5.File(self.wfn_file_h5, 'r') as f:
            bdot =f['/mf_header/crystal/bdot'][()]
            nk = f['mf_header/kpoints/nrk'][()]
            nb = nv+nc
            rk = f['mf_header/kpoints/rk'][()]
            ngk = f['mf_header/kpoints/ngk'][()]
            k_index =np.hstack((np.array([0]), np.cumsum(ngk)))
            hovb = self.hovb
            dipole_matrix = np.zeros([nk,nb,nb],dtype=np.complex128)
            for ik in range(nk):
                coeffs_k = f['wfns/coeffs'][hovb-nv:hovb+nc, :, k_index[ik]:k_index[ik + 1], 0]+1j*f['wfns/coeffs'][hovb-nv:hovb+nc, :, k_index[ik]:k_index[ik + 1], 1]
                gvecs_k = f['wfns/gvecs'][k_index[ik]:k_index[ik + 1],:]
                kmatrix = np.repeat(rk[ik:ik + 1, :], k_index[ik + 1] - k_index[ik], axis=0)
                Gpulsk = np.matmul(bdot, (gvecs_k + kmatrix).T).T

                dipole_b1 = np.einsum("mg,g,ng -> mn", np.conj(coeffs_k[:, 0, :]), Gpulsk[:, 0], coeffs_k[:, 0, :], optimize='optimal')
                dipole_matrix[ik,:,:] = dipole_b1/np.sqrt(bdot[0, 0])
        return dipole_matrix

    @time_watch
    def get_dataset(self, nc:int=6 ,nv:int=2, cell_slab_truncation:int=40, useWignerXY:bool=False, useWignerXYZ:bool=False,
                        AngstromPerPixel:float=0.1, operator = None, **kwargs)->dict:
        """
        Get the dataset of the wavefunction for ML
        Input:
            useWignerXY (Highly recommended for 2D materials!): 
                fold the wavefunction into the Wigner-Seitz cell for X-Y plane

            cell_slab_truncation (required if useWignerXY is True): 
                number of grids to preserve along z direction after
                truncation rule: 
                    For 2D system, we find the argmax of the sum of the wavefunction along z direction,
                    and truncate the cell centered at the argmax by cell_slab_truncation slices
                    e.g. if we set cell_slab_truncation = 40, and AngstromPerPixel_z = 0.1, then we preserve
                    40x0.1A=4A in z direction.

            AngstromPerPixel (required if useWignerXY is True): 
                pixel size in angstrom, only used when useWigner is True
                (see WignerXY __init__ for detail)

            AngstromPerPixel_z (default: AngstromPerPixel): 
                pixel size in angstrom, used to standardize the wavefunction along z direction

            other parameters (**kwargs): 
                see get_wfn_r_in_grid
                see WignerXY for detail
        Output:
            {
                "wfn": (nk, nc+nv, FFTgrid[0], FFTgrid[1], FFTgrid[2]),
                "occ": (nk, nc+nv, 1),
                "el": (nk, nc+nv, 1),
                "kpt_weights": (nk, nc+nv, 1),
                "kpt": (nk, nc+nv, 3),
                "band_indices: (nk, nc+nv, 1), # [-2,-1,1,2,3,4] (start with 1 or -1)
                "band_indices_abs": (nk, nc+nv, 1), #[3,4,5,6,7,8] (start with 1)
            }
        possible operators:
            'dipole': (nk, nc+nv, nc+nv) dipole matrix elements
        """
        if operator == 'dipole':
            dipole = self.get_dipole(nc=nc, nv=nv)
            print(f'Dipole matrix shape: {dipole.shape}')
        elif operator is not None:
            raise NotImplementedError(f'Operator {operator} is not implemented yet')

        logging.debug(f'Creating dataset for {self.wfn_file_h5}')

        wfn_r, el_r, occ = self.get_wfn_r_in_grid(nc=nc, nv=nv, **kwargs)
        band_indices = np.array([iv - nv for iv in range(nv)] + [ic + 1 for ic in range(nc)], \
                                dtype=int)[None,:,None].repeat(self.nk, axis=0)
        band_indices_abs = np.array([self.hovb-nv+iv+1 for iv in range(nv)] + [self.hovb+ic+1 for ic in range(nc)], \
                                dtype=int)[None,:,None].repeat(self.nk, axis=0)

        kpt = self.kpoints['rk'][:, None, :].repeat(nc + nv, axis=1)
        kpt_weights = self.k_weights[:, None, None].repeat(nc + nv, axis=1)

        # self.wfn_r_original = wfn_r
        # self.z_projection_original = np.sum(wfn_r, axis=(0,1,2,3))

        if cell_slab_truncation:
            assert AngstromPerPixel is not None, 'AngstromPerPixel is required when cell_slab_truncation is not None'

            AngstromPerPixel_z = kwargs.get('AngstromPerPixel_z', AngstromPerPixel)

            # standardize the wavefunction along z-wavefunction (make grid as 0.1A/pixel along z direction)
            target_z_dim = round(self.lattice[2,2] / AngstromPerPixel_z)
            # resize_wfn_r = lambda image: cv2.resize(image.reshape(1, -1), (target_z_dim,1), interpolation=cv2.INTER_CUBIC).squeeze()
            resize_wfn_r = lambda image: zoom(image, target_z_dim/self.FFTgrid[2], order=3)
            # print('\n',self.wfn_file_h5,'wfn_r.shape:',wfn_r.shape)
            # print('wfn_r.shape:',wfn_r.shape, target_z_dim, self.FFTgrid[2], wfn_r[0,0,0,0,:3])
            wfn_r = np.apply_along_axis(resize_wfn_r, axis=4, arr=wfn_r)
            logging.debug(f'ratio: {target_z_dim/self.FFTgrid[2]:.2f}, wfn_r shape: {wfn_r.shape},z_AngstromPerPixel: {self.lattice[2,2]/wfn_r.shape[4]:.2f}')

            self.z_projection = np.sum(wfn_r, axis=(0,1,2,3))

            wfn_r = wfn_r / np.sum(wfn_r, axis=(2,3,4), keepdims=True) # normalize the wavefunction
            wfn_r = np.roll(wfn_r, wfn_r.shape[-1]//2 - np.argmax(self.z_projection) , axis=4) # center the argmax of the z_projection
            wfn_r = wfn_r[:,:,:,:,max(wfn_r.shape[-1]//2-cell_slab_truncation//2, 0):min(wfn_r.shape[-1]//2+cell_slab_truncation//2, wfn_r.shape[-1]-1)]
            logging.debug(f'Average charge after truncation: {np.sum(wfn_r, axis=(2,3,4)).mean():.2f}')

        if useWignerXYZ:
            assert AngstromPerPixel is not None, 'AngstromPerPixel is required when useWignerXYZ is True'
            self.wigner3d = wigner3d.WignerXYZ(self.lattice, 
                                             self.FFTgrid, 
                                             AngstromPerPixel, 
                                             **kwargs)

            wfn_r_wigner3d = np.zeros((wfn_r.shape[0], wfn_r.shape[1], self.wigner3d.XI.shape[0], self.wigner3d.XI.shape[1], self.wigner3d.XI.shape[2]), dtype=wfn_r.dtype)

            # print('--debug:' ,wfn_r.shape)

            for k in range(wfn_r.shape[0]):
                for b in range(wfn_r.shape[1]):
                    wfn_r_wigner3d[k,b] = self.wigner3d.Wigner_fast_nearest(wfn_r[k,b], kwargs.get('max_distance', 0.2))

            wfn_r = wfn_r_wigner3d

        elif useWignerXY:
            assert cell_slab_truncation is not None, 'cell_slab_truncation is required when useWignerXY is True'
            assert AngstromPerPixel is not None, 'AngstromPerPixel is required when useWignerXY is True'
            assert np.allclose(self.lattice[2,:2], 0), 'Wigner: only support 2D system for now (a3=(0,0,c))'

            self.wigner = wigner.WignerXY(self.lattice, 
                                          self.FFTgrid, 
                                          AngstromPerPixel, 
                                          **kwargs)
            
            
            wfn_r_wigner = np.zeros((wfn_r.shape[0], wfn_r.shape[1], self.wigner.xi.shape[0], self.wigner.xi.shape[1], wfn_r.shape[-1]), dtype=wfn_r.dtype)
            
            
            for k in range(wfn_r.shape[0]):
                for b in range(wfn_r.shape[1]):
                    # Note: see wigner.py for benchmark between regular interpolation and fast one!
                    # If use fast nearest interpolation, use upsampling > 1 for finer original grid!!
                    ##################################################################################
                    # wfn_r_wigner[k,b] = self.wigner.WignerInterpolate(wfn_r[k,b])
                    wfn_r_wigner[k,b] = self.wigner.Wigner_fast_nearest_interpolation(wfn_r[k,b], max_distance=AngstromPerPixel)
                    ##################################################################################
            
            # See src/note.md for benchmark between regular interpolation and fast one!
            # interpolate_wigner = lambda args: (args[0], args[1], args[3].Wigner_fast_nearest_interpolation(args[2], max_distance=args[4]))
            # with Pool(8) as pool:
            #     tasks = [(k, b, wfn_r[k, b], self.wigner, AngstromPerPixel) 
            #             for k in range(wfn_r.shape[0]) 
            #             for b in range(wfn_r.shape[1])]
            #     results = pool.map(interpolate_wigner, tasks)
            # for k, b, value in results:
            #     wfn_r_wigner[k, b] = value


            wfn_r = wfn_r_wigner
        else:
            print('wavefunction in XY fractional coor. will be saved:', wfn_r.shape)
            pass

        dataset = {
            "wfn": wfn_r,
            "el": el_r,
            "occ": occ,
            "kpt_weights": kpt_weights,
            "kpt": kpt,
            "band_indices": band_indices.astype(int),
            "band_indices_abs": band_indices_abs.astype(int),
        }
        if operator == 'dipole':
            dataset['dipole'] = dipole

        return dataset

class AScvk(BGWIO):
    def __init__(self,fname):
        # self.write()
        logging.debug(f'Loading eigenvectors {fname}')

        self.eigenvech5_file = fname
        self.eigenvech5 = h5.File(fname, 'r')

        # Get the names of the datasets
        h5ls = H5ls()
        self.eigenvech5.visititems(h5ls)   
        self.names = h5ls.names

        # Get the header information
        self.crystal = {}
        self.gspace = {}
        self.kpoints = {}
        self.kpoints_exciton = {}
        self.symmetry = {}
        self.params = {}
        self.read_header()

        self.eigenvech5.close()

        self.eigenvectors = None
        self.eigenvalues = None

    def read_header(self):
        for name in self.names:
            if 'crystal' in name.split('/'):
                self.crystal[name.split('/')[-1]] = self.eigenvech5[name][()]
            elif 'gspace' in name.split('/'):
                self.gspace[name.split('/')[-1]] = self.eigenvech5[name][()]
            elif 'kpoints' in name.split('/'):
                if 'exciton_header' in name.split('/'):
                    self.kpoints_exciton[name.split('/')[-1]] = self.eigenvech5[name][()]
                else:
                    self.kpoints[name.split('/')[-1]] = self.eigenvech5[name][()]
            elif 'symmetry' in name.split('/'):
                self.symmetry[name.split('/')[-1]] = self.eigenvech5[name][()]
            elif 'params' in name.split('/'):
                self.params[name.split('/')[-1]] = self.eigenvech5[name][()]
            else:
                pass


    def get_acvkS(self):
        with h5.File(self.eigenvech5_file, 'r') as f:
            self.eigenvectors = f['exciton_data/eigenvectors'][0,:,:,:,:,0,:] # (S,k,c,v,2)
            self.eigenvalues = f['exciton_data/eigenvalues'][()]
            self.eigenvectors = self.eigenvectors[...,0] + 1j * self.eigenvectors[...,1]

            assert self.eigenvectors.shape == (self.params['nevecs'], self.kpoints_exciton['nk'], self.params['nc'], self.params['nv']), \
                f"Shape mismatch: {self.eigenvectors.shape} != {(self.params['nevecs'], self.kpoints_exciton['nk'], self.params['nc'], self.params['nv'])}"
            assert self.eigenvalues.shape == self.params['nevecs'], "Shape mismatch: {self.eigenvalues.shape} != {self.params['nevecs']}"

        return self.eigenvectors, self.eigenvalues

    def get_dataset(self):
        """
        Since exciton nS is a mixed up of all nc, nv and nk, it is not reasonbale to specify nc, nv, nk then extract nS=nc*nv*nk
        So, here we just return all the eigenvectors and eigenvalues in the bse eigenvector.h5
        """
        if not self.eigenvectors or not self.eigenvalues:
            self.get_acvkS()

        # get dipole strength
        abs_dirname = os.path.dirname(self.eigenvech5_file)
        dipole_sequred = np.loadtxt(os.path.join(abs_dirname, 'eigenvalues_b1.dat'))[:,1]

        assert np.allclose(np.loadtxt(os.path.join(abs_dirname, 'eigenvalues_b1.dat'))[:,0],
                           self.eigenvalues)

        dataset = {
            "eigenvectors": abs(self.eigenvectors),
            "eigenvalues": self.eigenvalues[:, None],
            "dipole_squared": dipole_sequred[:, None]
        }

        return dataset  

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # VXC
    # vsc = vloc('../../examples/flows/mat-5/02-wfn/VSC')

    # WFN 
    wf = wfn('../../examples/flows/mat-5/02-wfn/wfn.h5')
    dp_wfn = wf.get_dataset(useWignerXY=True, cell_slab_truncation=60, AngstromPerPixel=0.1, AngstromPerPixel_z=0.1,
                                upsampling_factor=3)
    assert abs(abs(dp_wfn['wfn'][0,0,  5,5,30])-0.0009519374081944384) < 1e-7 # unit test
    print("WFN: unit test passed!")

    wf = wfn('../../examples/flows/mat-5/02-wfn/wfn.h5')
    dp_wfn_3d = wf.get_dataset(useWignerXYZ=True, AngstromPerPixel=0.1, cell_slab_truncation=None)
    assert abs(abs(dp_wfn_3d['wfn'][0,0,  5,5,5])-0.00014902410416606067) < 1e-7 # unit test
    print("WFN 3D: unit test passed!")

    # EQP
    eqp = eqp('../../examples/flows/mat-5/13-sigma/eqp1.dat')
    dp_eqp = eqp.get_dataset()
    assert np.allclose(dp_eqp['mf'].sum(), -13.864995302)
    print("eqp: unit test passed!")

    # AcvkS
    acv = AScvk('../../examples/flows/mat-5/19-absorption/eigenvectors.h5')
    d_acv = acv.get_dataset()
    assert abs(d_acv['eigenvalues'][15,0] - 10.619205333508571) < 1e-7
    assert abs(abs(acv.eigenvectors[0,1,0,0]) - 1.00) < 1e-7