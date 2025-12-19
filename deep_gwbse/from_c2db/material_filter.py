import numpy as np
import h5py as h5
import h5py
from ase import Atoms
import os

def atom_index_range(fid: h5py.File, material_idx):
    r"""
    Find the indices of all atoms in material `material_idx` in `fid`.
    The right boundary of the return value is not included.
    """

    numbers_of_atoms = fid["numbers_of_atoms"][()] # import data to memory is safer
    # This is also the index of the first atom in the material
    num_atoms_prev_materials = sum(numbers_of_atoms[0:material_idx])
    num_atoms_this_material = numbers_of_atoms[material_idx]
    return range(num_atoms_prev_materials, num_atoms_prev_materials + num_atoms_this_material)


def atom_list(fid: h5py.File, material_idx):
    r"""
    Find the atomic numbers of the atoms in the material.
    """
    return fid["atoms"][list(atom_index_range(fid, material_idx))] # Iterator itself might not be directly used as indexing
    
def atom_positions(fid, material_idx):
    r"""
    Find the positions of the atoms. 
    The first index is the index of the atom,
    and the second index labels the coordinate (x, y, z).
    """
    return fid["positions"][list(atom_index_range(fid, material_idx)), :] # Iterator itself might not be directly used as indexing

def cell(fid, material_idx):
    r"""
    Find the primitive unit cell basis of the material.
    The first index specifies the index of the three basis vectors 
    and the second index specifies the coordinate x, y, z.
    """
    return fid["cell"][material_idx, :, :]

def atoms(fid, material_idx):
    r"""
    Convert the information about the material in `fid`
    into an `ase.Atoms` object.
    """
    return Atoms(
        numbers=atom_list(fid, material_idx),
        cell=cell(fid, material_idx),
        positions=atom_positions(fid, material_idx),
        pbc=[1, 1, 1]
    )

def unique_id(fid, material_idx):
    return fid["unique_id"][material_idx]

class crystal_system:
    # all filter function return an 0/1 array with size (n,) where n is the number of materials
    # then we can get intersection of several filter result to get what we want.
    # e.g. res1 is returned by filter for number of atoms (natom=3), res2 is returned by hexagonal filter
    # then we can use command like: (res1 & res2) to get what we request.

    def __init__(self, fname) -> None:
        self.fname = fname
        self.read_data_base()
    
    def read_data_base(self):
        fid = h5.File(self.fname,'r')
        self.cells = fid['cell'][()]
        self.atoms = fid['cell'][()]
        self.id = fid['id'][()]
        self.numbers_of_atoms = fid['numbers_of_atoms'][()]
        self.positions = fid['positions'][()]
        self.unique_ide = fid['unique_id'][()]
        fid.close()

    # def cell2abc(self):
    #     abc = np.zeros((self.cells[0],6)) # (n, 6) 6:a,b,c,alpha1, alpha2, alpha3
    #     for i in range()

    # todo: add more filters
    def filter_hexagonal(self):
        """
        cell: (n,3,3)
        """
        res = np.zeros(self.cells.shape[0])
        for n in range(len(self.cells)):
            if self.is_hexagonal_lattice(self.cells[n][0],self.cells[n][1],self.cells[n][2]):
                res[n] = 1
        
        return res.astype(int) # (n,) where 1 denotes True and 0 denotes False

    def atom_numbser_filter(self,natom=3):
        return (self.numbers_of_atoms == natom).astype(int)

    def abc_axis(self, abc_min=0, abc_max=20, axis='c'):
        if axis not in 'abc':
            raise Exception('specify axis:"a", "b" or "c" ')
        if axis == 'a': cell_length = np.linalg.norm(self.cells[:,0,:], axis=1)
        if axis == 'b': cell_length = np.linalg.norm(self.cells[:,1,:], axis=1)
        if axis == 'c': cell_length = np.linalg.norm(self.cells[:,2,:], axis=1)
        return np.where((cell_length<abc_max)&(abc_min<cell_length),1,0)
    
    
    def is_hexagonal_lattice(self,v1, v2, v3):
        # Calculate angles between vectors
        angle_v1_v2 = np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
        angle_v1_v3 = np.degrees(np.arccos(np.dot(v1, v3) / (np.linalg.norm(v1) * np.linalg.norm(v3))))
        angle_v2_v3 = np.degrees(np.arccos(np.dot(v2, v3) / (np.linalg.norm(v2) * np.linalg.norm(v3))))

        # Check if the angles are approximately equal to 60 degrees and 90 degrees
        return (
            (np.isclose(angle_v1_v2, 120, atol=1))and
            np.isclose(angle_v1_v3, 90, atol=1) and
            np.isclose(angle_v2_v3, 90, atol=1)
        )

def write_filter_result_to_cif(fid, materials_index, save_dir, z_length=12, **kwargs):

    os.makedirs(save_dir, exist_ok=True)

    c2dbh5 = h5.File(fid,'r')
    for i, material_idx in enumerate(materials_index):    
        stru = atoms(c2dbh5, material_idx)
        
        # preprocess
        atom_positions = stru.get_positions()
        cell = stru.get_cell()
        cell[2,2] = z_length
        stru.set_cell(cell)
        assert np.allclose(atom_positions, stru.get_positions()), "atom positions should not change"

        # save to cif
        os.makedirs(os.path.join(save_dir, f"{i:03d}-mat-{material_idx}"), exist_ok=True)
        stru.write(os.path.join(save_dir,f"{i:03d}-mat-{material_idx}",'stru.cif'), format='cif')
    
    # summary
    print(f'Number of materials: {len(materials_index)}')




if __name__ == '__main__':

    fid = 'c2db-2022-11-30.h5'
    c2dbh5 = h5.File(fid,'r')
    crystal = crystal_system(fid)

    #################### Dataset-NC #####################
    # this dataset is filtered as same as this paper:
    # https://www.nature.com/articles/s41467-024-53748-7
    # Number of materials:  302

    res_threeatoms = crystal.atom_numbser_filter(3)
    res_fouratoms = crystal.atom_numbser_filter(4)
    res_clengh = crystal.abc_axis(18,18.5,axis='c')
    res_alengh = crystal.abc_axis(3.25,3.75,axis='a')
    res_blengh = crystal.abc_axis(2,4,axis='b')

    # intersection of all filters:
    res3 = res_clengh  & res_alengh & res_blengh & (res_threeatoms | res_fouratoms)
    materials_index = list(np.where(res3 == 1)[0])


    write_filter_result_to_cif(fid, materials_index, 'nc300', z_length=12)
