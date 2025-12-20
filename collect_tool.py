#!/usr/bin/env python
import json
import os
from deep_gwbse.utils import check_flows_status
from deep_gwbse.collect_tool import collect_from_md, collect_from_flows_2_deep, metal_seek, check_flows_status, generate_sbatch_jobs, compact_data_folder, compact_data_flows, restart_sbatch_jobs, merge_dataset


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
