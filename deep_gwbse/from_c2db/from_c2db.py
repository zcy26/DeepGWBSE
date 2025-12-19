from ase.db import connect
import numpy as np
import os
from tqdm import tqdm
# Two useful websites for using ASE database:
# Ref1: https://wiki.fysik.dtu.dk/ase/tutorials/tut06_database/database.html#browsing-data
# Ref2: https://cmr.fysik.dtu.dk/c2db/c2db.html#key-value-pairs

"""Basic Setup"""
save_dir = 'c2db-1000-7-0.4'
db = connect('../../examples/c2db/c2db-2022-11-30.db')
select_rule = "natoms<7,gap>0.4" # use help(db.select) to see all the keys
z_length=12

"""Filter materials"""
os.makedirs(save_dir, exist_ok=True)

rows = db.select(select_rule)
atoms = []
for i, row in tqdm(enumerate(rows)):
    stru = row.toatoms()

    # preprocess
    if z_length is not None:
        atom_positions = stru.get_positions()
        cell = stru.get_cell()
        cell[2,2] = z_length
        stru.set_cell(cell)
        assert np.allclose(atom_positions, stru.get_positions()), "atom positions should not change"

    # save to cif
    os.makedirs(os.path.join(save_dir, f"{i+1:04d}-mat-{row.id}"), exist_ok=True)
    stru.write(os.path.join(save_dir,f"{i+1:04d}-mat-{row.id}",'stru.cif'), format='cif')

    atoms.append(stru)

print(f"Number of materials: {len(atoms)}")