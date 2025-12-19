#!/usr/bin/env python
import os
import subprocess
from os.path import join as pjoin
import numpy as np
import json
import ase
from ase import Atoms
from ase.io.espresso import read_espresso_in, read_espresso_out
from ase.visualize.ngl import NGLDisplay
import matplotlib.pyplot as plt
import copy
import numpy as np
from os.path import join as pjoin
import os
import subprocess
from tqdm import tqdm
import glob
import h5py as h5
import sys
import os
from utils import jobdone, check_flows_status
from from_model.data import ManyBodyData 



def collect_from_md(md_input_fname = './flow-hBN-md/flow-hBN-AB-661/01-density/scf.in',
                    md_output_fname = './flow-hBN-md/flow-hBN-AB-661/01-density/md.out',
                    suffix = ''):

    # only md atomic positions
    # md = read_espresso_out('./flow-hBN/01-density/md.out')
    stru_dir = './collected-md-stru/'
    structure = read_espresso_in(md_input_fname)
    # NGLDisplay([structure])
    natom = len(structure)
    -3
    with open(md_output_fname,'r') as f:
        lines = f.readlines()

    temperature = []
    structures = []

    for i, line in enumerate(lines):
        if "ATOMIC_" in line:
            assert "crystal" in line, "Only crystal coordinates are supported"
            temp_structure = copy.deepcopy(structure)
            pos_frac = np.array([list(map(float, x.split()[1:])) for x in lines[i+1:i+1+natom]])
            pos_cart = pos_frac @ structure.cell.array
            temp_structure.set_positions(pos_cart)
            structures.append(temp_structure)

        if "temperature           =" in line:
            temperature.append(float(line.split()[2]))
            # print(line.split()[1])
        
    # create stru list directory
    print(f"Creating directory {stru_dir}")
    os.makedirs(stru_dir, exist_ok=True)
    for i, stru in enumerate(structures):
        dir_name = f'mat-{i+1:03d}-{temperature[i]:.2f}-{suffix}'
        os.makedirs(pjoin(stru_dir, dir_name), exist_ok=True)
        stru.write(pjoin(stru_dir, dir_name, 'stru.cif'), format='cif')

def collect_from_flows_2_deep(deeph_flows='./'):

    assert os.path.exists(deeph_flows), f"Directory {deeph_flows} does not exist"

    roots = []
    for root, dirs, files in os.walk(deeph_flows):
        if "hamiltonians.h5" in files:
            roots.append(root)
            print(root)

    os.makedirs("dataset", exist_ok=True)
    for i, root in tqdm(enumerate(roots), total=len(roots)):
        subprocess.run(["cp", "-r", root, f"dataset/ham-{i:03d}"])


def metal_seek(flows='./flows-bwhou'):
    # walk through the directory
    roots = []
    for root, dirs, files in os.walk(flows):
        if "stru.cif" in files:
            roots.append((root, root.split('/')[-1]))

    summary = {"metal":[], "semiconductor":[], "unknown":[]}
    for root, mat_id in roots:

        scf_out = pjoin(root, "01-density",'scf.out') 
        scf_in = pjoin(root, "01-density",'scf.in')
        bands_dat = pjoin(root, "05-band",'bands.dat.gnu')

        if not os.path.exists(scf_out) or not os.path.exists(scf_in) or not os.path.exists(bands_dat):
            print(f"Missing files in {root}, skipping...")
            summary['unknown'].append(mat_id)
            continue

        # grep "Fermi" of scf_out
        print(f"Material: {mat_id}")
        result = subprocess.run(f"grep 'Fermi' {scf_out}", capture_output=True, shell=True)
        if result.stdout == b'':
            print(f"Fermi level not found in {scf_out}")
            summary['unknown'].append(mat_id)
            continue
        Fermi_level = float(result.stdout.split()[-2])
        print(f"  Fermi level: {Fermi_level:.2f} eV")

        ele_res = subprocess.run(f"grep 'number of electrons' {scf_out}", capture_output=True, shell=True)
        if ele_res.stdout == b'':
            print(f"Number of electrons not found in {scf_out}")
            continue
        num_electrons = float(ele_res.stdout.split()[-1])
        print(f"  Number of electrons: {num_electrons:.2f}")
        # read the bands.dat.gnu file

        spin_orb_res = subprocess.run(f"grep 'lspinorb' {scf_in}", capture_output=True, shell=True)
        if spin_orb_res.stdout == b'':
            soc = False
            nvalence = num_electrons / 2 
        else:
            soc = True
            nvalence = num_electrons
        print(f"  Spin-orbit coupling: {'on' if soc else 'off'}")
        print(f"  Number of valence electrons: {nvalence:.2f}")

        with open(bands_dat, 'r') as f:
            lines = f.readlines()
            nk = lines.index('\n') # number of k-points
            assert len(lines) % (nk + 1) == 0, "bands.dat.gnu file is not correctly formatted"
            nb = len(lines) // (nk + 1) # +1 is from space line
            # extract the last number
            f = np.loadtxt(bands_dat)[:,1].reshape(nb, nk)

        if nvalence % 1 != 0: # half-occupation
            vbm_index = int(np.ceil(nvalence)) - 1
            cbm_index = vbm_index
        else:
            vbm_index = int(nvalence) - 1
            cbm_index = vbm_index + 1

        vbm = np.max(f[vbm_index, :])
        cbm = np.min(f[cbm_index, :])

        print(f"  VBM index (start with 0): {vbm_index}, VBM energy: {vbm:.2f} eV")
        print(f"  CBM index (start with 0): {cbm_index}, CBM energy: {cbm:.2f} eV")

        if cbm > Fermi_level and Fermi_level > vbm:
            print(f"  {mat_id} is a semiconductor")
            summary['semiconductor'].append(mat_id)
        else:
            print(f"  {mat_id} is a metal")
            summary['metal'].append(mat_id)

    print("Summary:")
    print("  Metals:", len(summary['metal']))
    print("  Semiconductors:", len(summary['semiconductor']))
    print("  Unknown:", len(summary['unknown']))
    print("  saved to metal_seek.json")
    with open("metal_seek.json", "w") as f:
        json.dump(summary, f, indent=4)


def set_sbatch(cluster, nodes, hours):
    assert cluster in ['perlmutter'], "Only perlmutter is supported"
    return f"#!/bin/bash\n#SBATCH -N {nodes}\n#SBATCH -C cpu\n#SBATCH -q regular\n#SBATCH -t {hours}:00:00\n"

def generate_sbatch_jobs(fname='./run_aug.sh', nsbatch=3, hours=4, cluster='perlmutter', nodes=4):
    """
    Parses a script file to extract tasks and generates multiple SBATCH job scripts.

    Parameters:
        fname (str): Path to the input script file.
        nsbatch (int): Number of batch jobs to create.
        hours (int): Time in hours for each job.
        cluster (str): Cluster name (only 'perlmutter' is supported).
        nodes (int): Number of nodes for the job.
    """
    
    with open(fname, 'r') as f:
        lines = f.readlines()
    
    tasks = []
    start = None
    stack = []
    for i, line in enumerate(lines):
        if "cd" in line and "cd .." not in line:
            stack.append('cd')
            if start is None:
                start = i
        if "cd .." in line and start is not None:
            stack.pop()
            if not stack:
                tasks.append(''.join(lines[start:i+1]))
                start = None
            
    
    tasks_per_job = int(np.ceil(len(tasks) / nsbatch))
    
    for i in range(nsbatch):
        start = i * tasks_per_job
        end = min((i + 1) * tasks_per_job, len(tasks))
        job_tasks = tasks[start:end]
        
        prefix = os.path.splitext(fname)[0]
        job_file = prefix+f'_sub_{i+1}.sh'
        with open(job_file, 'w') as f:
            f.write(set_sbatch(cluster, nodes, hours))
            f.write("\n")
            for task in job_tasks:
                f.write(task)
                f.write("\n")
    
    print(f"Generated {nsbatch} job scripts.")

def compact_data_folder(folder: str = '.', unwanted: dict = None):
    if unwanted is None:
        unwanted = {
        "unwanted_files": [
        "01-density/VSC",
        "01-density/VXC",
        "01-density/wfn.cplx",
        "01-density/*.wfc*",
        "01-density/*.save/wfc*.dat",
        "02-wfn/VSC",
        "02-wfn/VXC",
        "02-wfn/VKB",
        # "02-wfn/wfn.h5", # This is parabands wfn, we don't need it
        "02-wfn/*.wfc*",
        "02-wfn/*.save/wfc*.dat",
        "03-wfnq/wfn.h5",
        "03-wfnq/wfn.cplx",
        "03-wfnq/wfn_q.h5",
        "03-wfnq/*.wfc*",
        "03-wfnq/*.save/wfc*.dat",
        # "05-band/wfn.cplx", # saved for data-aug
        "05-band/*.wfc*",
        "05-band/*.save/wfc*.dat",
        "06-wfnq-nns/wfn.cplx",
        "06-wfnq-nns/*.wfc*",
        "06-wfnq-nns/*.save/wfc*.dat",
        "12-epsilon-nns/eps0mat.h5",
        # "17-wfn_fi/wfn.cplx", # saved for data-aug
        "17-wfn_fi/*.wfc*",
        "17-wfn_fi/*.save/wfc*.dat"]}

    unwanted_files = unwanted.get('unwanted_files', [])
    for file in unwanted_files:
        pattern = glob.glob(pjoin(folder, file))
        if not pattern:
            print(f"{pjoin(folder, file)} does not exist")
        else:
            for file_path in pattern:
                os.remove(file_path)
            print(f"Removed {pjoin(folder, file)}")

def compact_data_flows(flows: str = './flows', unwanted: dict = None):
    f = os.scandir(flows)
    for entry in f:
        if entry.is_dir():
            print("")
            print(f"Compacting {entry.path}")
            compact_data_folder(entry.path, unwanted)
    
    # remove wfn.h5 from all unfinished flow
    # status = check_flows_status(flows, dump=False)
    # unwanted_wfn = {"unwanted_files":['02-wfn/wfn.h5']}
    # for flow, status in status.items():
    #     if status["Yes"] and status["No"]: # unfinished
    #         compact_data_folder(flow, unwanted_wfn)


def restart_sbatch_jobs(flows):
    """
    Restart sbatch jobs for unfinished flows.
    """
    # check the status of the flows
    flows_status = check_flows_status(flows, dump=False)

    length_of_flows = len(flows_status)
    
    # find the unfinished flows
    unfinished_flows = []
    for flow, status in flows_status.items():
        if not status["Yes"]: # TODO: modify this after updating "bug_list" to check_flows_status 
            unfinished_flows.append(flow)
    length_of_unfinished_flows = len(unfinished_flows)
    print(f"Found {length_of_unfinished_flows} unfinished flows out of {length_of_flows} flows.")
    # generate sbatch jobs for the unfinished flows

    with open('restart_run.sh', 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("\n")
        for flow in unfinished_flows:
            f.write(f"cd {os.path.basename(flow)}\n")
            f.write(f'echo "{os.path.basename(flow)}"\n')
            f.write("bash run.sh\n")
            f.write("cd ..\n")
            f.write("\n")
    
    print(f"Generated restart_run.sh for {length_of_unfinished_flows} unfinished flows.")



def merge_dataset(path:str, dataset_fname:str):
    """
    Merge multiple dataset h5files into one dataset h5 file.
    Requirements: info must be the same in all h5 files.
    """
    h5_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.h5')]

    data = ManyBodyData.from_existing_dataset(h5_files[0], slice(0,0))
    info = copy.deepcopy(data.info)
    info.merged_data = True
    del data

    print("h5 files to be merged:", h5_files)

    # compare the info with the rest of the files
    for h5_file in h5_files:
        with h5.File(h5_file,'r') as f:
            assert f['info'], f"info not found in file {h5_file}"
            for key, val in f['info'].items():
                if key == 'merged_data':
                    raise ValueError(f"detected h5file created from merging, please double check the file {h5_file}")
                assert key in info.__dict__, f"key {key} not found in file {h5_file}"
                if key == 'mat_id':
                    mat_id_list = val[()]
                    info.__dict__[key] = np.concatenate([info.__dict__[key], mat_id_list])
                    continue
                v = val[()]

                if isinstance(val[()], bytes):
                    v = val[()].decode('utf-8')
                assert np.array_equal(v, info.__dict__[key]), f"key {key} not equal in file {h5_file}"
            assert len(f['info']) + 1 == len(info.__dict__), f"key {key} not found in file {h5_file}"
    # start merging
    ManyBodyData.init_dataset_h5(dataset_dir=path,
                                dataset_fname=dataset_fname,
                                info=info,
                                multiprocessing = False)

    for h5_file in h5_files:
        print(f"merging {h5_file}")
        with h5.File(h5_file, 'r') as f:
            for mat_id in tqdm(f['info']['mat_id'][()]):
                ManyBodyData.datapoint_interface_h5(os.path.join(path, dataset_fname), mat_id, f[mat_id], mode='a')
            


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Collect Tools')

    parser = argparse.ArgumentParser(
        description="""Collect Tools:
    A collection tools for different modes.
    """,
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('mode', choices=['md', 'deeph', 'metalseek', 'st', 'sub','compact', 'restart', 'merge'], help="""\
    md: collect structures from MD output.
    deeph: collect DFT-Ham from DFT/SIESTA/HPRO flows.
    metalseek: determine metallicity from DFT flows
    st: check the status of the flows
    compact: compact data. delete the unwanted files to save space
    restart: generate new sbatch jobs for all the unfinished flows
    """)
    # parser.add_argument('mode', choices=['md', 'deeph', 'metal'], help='md: collect structures from MD output. \ndeeph: collect DFT-Ham from DFT/SIESTA/HPRO flows. \nmetal:')
    parser.add_argument('-md_input', type=str, help='md: input file name')
    parser.add_argument('-md_output', type=str, help='md: output file name')
    parser.add_argument('-md_suffix', type=str, default='', help='md: suffix for MD files')
    parser.add_argument('-flows', type=str, help='deeph/metalseek/compact/restart: directory containing DFT/SIESTA/HPRO/GW/BSE flows')
    parser.add_argument('-job', type=str, help='sub: sbatch job file name')
    parser.add_argument('-nsbatch', type=int, default=3, help='sub: number of sub-sbatch jobs')
    parser.add_argument('-hours', type=int, default=4, help='sub: hours for each job')
    parser.add_argument('-nodes', type=int, default=4, help='sub: number of nodes for each job')
    parser.add_argument('-folder', type=str, help='compact: flow dir to compact; merge: folder containing h5 files to merge')
    parser.add_argument('-dataset_fname',type=str, default='merged_dataset.h5', help='merge: output dataset file name')
    parser.add_argument('-unwanted', type=str, default=None, help='compact: json includes unwanted files to delete')

    args = parser.parse_args()

    if args.mode == 'md':
        if not args.md_input or not args.md_output:
            parser.error('--md_input_fname and --md_output_fname are required in "md" mode')
        if not args.md_suffix:
            args.md_suffix = ''
        collect_from_md(args.md_input, args.md_output, args.md_suffix)
        
    elif args.mode == 'deeph':
        collect_from_flows_2_deep(args.deeph_flows)
    
    elif args.mode == 'metalseek':
        if not args.flows:
            parser.error('--flows is required in "metalseek" mode')
        metal_seek(args.flows)
    
    elif args.mode == 'st':
        if not args.flows:
            parser.error('--flows is required in "st" mode')
        check_flows_status(args.flows)
    
    elif args.mode == 'sub':
        if not args.job:
            parser.error('--job is required in "sub" mode')
        generate_sbatch_jobs(args.job, args.nsbatch, args.hours, 'perlmutter', args.nodes)

    
    elif args.mode == 'compact':
        if not args.unwanted:
            # parser.error('--unwanted is required in "compact" mode')
            print('default unwanted list is used')
            unwanted = None
        else:
            print('Using user provided unwanted list')
            with open(args.unwanted, 'r') as f:
                unwanted = json.load(f)
        if args.flows != None and args.folder == None:
            compact_data_flows(args.flows, unwanted)
        elif args.flows == None and args.folder != None:
            compact_data_folder(args.folder, unwanted)
        elif args.flows != None and args.folder != None:
            parser.error('--folder and --flows are mutually exclusive')
        elif args.flows == None and args.folder == None:
            parser.error('--folder or --flows is required in "compact" mode')

    elif args.mode == 'restart':
        if not args.flows:
            parser.error('--flows is required in "restart" mode')
        restart_sbatch_jobs(args.flows)
        
    elif args.mode == 'merge':
        if not args.folder:
            parser.error('--folder is required in "merge" mode')
        merge_dataset(args.folder, args.dataset_fname)
