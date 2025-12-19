import h5py
from gpaw import GPAW, PW, FermiDirac
from gpaw.response.g0w0 import G0W0
from c2db_to_ase import *
import os


class MaterialJobController:
    r"""
    Job controller of a single material.

    Side effects:
    - GPAW intermediate files are created in the folder where the program is run.
    """

    def __init__(self,
                 library_path: str,  # The HDF library containing crystal information
                 material_idx: int,  # The nth material in `library_path` is calculated here
                 scf_log_filename=None,
                 bands_log_filename=None,
                 g0w0_log_filename=None,
                 scf_gpw_filename=None,
                 gw_filename=None,
                 ecutoff_dft=200,
                 ecutoff_gw=100,
                 occupations=FermiDirac(0.01),
                 kpts_scf={'size': (6, 6, 1), 'gamma': True},
                 kpts_bands=(10, 10, 1),
                 nbands=1000  # TODO decide nbands according to number of G vectors
                 ) -> None:
        fid = h5py.File(library_path, 'r')
        self.fid = fid
        self.atoms = atoms(fid, material_idx)
        self.unique_id = str(unique_id(fid, material_idx), encoding='utf-8')

        fid.close()  # do not for get this when you finish using h5

        self.scf_log_filename = scf_log_filename
        self.bands_log_filename = bands_log_filename
        self.g0w0_log_filename = g0w0_log_filename

        if scf_gpw_filename is None:
            self.scf_gpw_filename = self.unique_id + "-scf.gpw"
        else:
            self.scf_gpw_filename = scf_gpw_filename
        if gw_filename is None:
            self.gw_filename = self.unique_id + "_g0w0_ppa"
        else:
            self.gw_filename = gw_filename

        self.ecutoff_dft = ecutoff_dft
        self.ecutoff_gw = ecutoff_gw
        self.occupations = occupations
        self.kpts_scf = kpts_scf
        self.kpts_bands = kpts_bands
        self.nbands = nbands

    def scf(self):
        r"""
        Setting up the SCF task.
        Currently we can do an all-in-one SCF run
        instead of the standard SCF -> bands procedure,
        following the example of
        https://wiki.fysik.dtu.dk/gpaw/tutorialsexercises/electronic/gw/gw.html#ground-state-calculation.
        Thus the burden of generating empty bands is placed
        here; and the `bands` method is temporarily abandoned.
        """
        if self.scf_log_filename is None:
            calc = GPAW(
                mode=PW(self.ecutoff_dft),
                xc='PBE',
                kpts=self.kpts_scf,
                random=True,
                occupations=self.occupations,
            )
        else:
            calc = GPAW(
                mode=PW(self.ecutoff_dft),
                xc='PBE',
                kpts=self.kpts_scf,
                random=True,
                occupations=self.occupations,
                txt=self.scf_log_filename
            )
        self.atoms.calc = calc

        # Run
        # TODO: I don't know which of the following lines
        # triggers the actual DFT calculation;
        # anyway these two lines always appear in the tutorial and
        # after putting them here DFT calculation does start
        self.atoms.get_potential_energy()
        calc.get_fermi_level()

        # Generating empty bands by exact diagonalization
        # Be mindful: this is essentially parabands!!! Very slow!!!
        calc.diagonalize_full_hamiltonian()
        calc.write(self.scf_gpw_filename, 'all')

    def bands(self):
        calc = GPAW(self.scf_gpw_filename).fixed_density(
            kpts=self.kpts_bands
            # TODO: the correct parameters here
        )

    # TODO: API design: whether bands_range should be given as a field of the object
    def g0w0(self, bands_range=(-2, 2), nbands=None):
        if nbands is None:
            nbands = self.nbands

        if os.path.exists(self.gw_filename):
            os.remove(self.gw_filename)

        self.gw = G0W0(
            self.scf_gpw_filename,
            filename=self.gw_filename,
            kpts=None,
            ecut=self.ecutoff_gw,
            ppa=True,
            bands=bands_range,
            # nbands=nbands
        )

        self.result = self.gw.calculate()


if __name__ == "__main__":
    print('running')
    controller = MaterialJobController('c2db-2022-11-30.h5', 6161, ecutoff_dft=100, ecutoff_gw=80)
    controller.scf()
    controller.g0w0(bands_range=(12, 14))

