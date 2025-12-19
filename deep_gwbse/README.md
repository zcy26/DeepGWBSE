## General workflow
```

Molecule Dynamncse─┐
      external src───(Collect)─>fp-input ──(QE/SIESTA/BGW + HPRO)─>─┌── ml-train-set──(ML)─> model
               ...─┘                                                └── ml-test-set
```
**Path 1** Tweisted-angle study of hBN (**not working well**)
```

1. Train:
supercell.cif─(flow.py)─> MD─(md.py)─>fp-input─(flows.py)─> ml_dataset ──(deep-collect.py, deephe3-train.py)─> model

2. Use:
twist.cif─┌──(deephe3-xx.py, diag_plot.py)─> band.png 
    model─┘
```

**Path 2** MBFormer for GW-BSE
```
--Path 2--:
1. Train:
external database─>stru_input─(flows.py,flows-aug.py)─>flows─>(data.py)─>dataset.h5─(trainer.py)─> model

2. Use:
Features: G0W0, BSE (binding energy, |<cvk|S>|)
```

## Folder Structure

### 1. **stru-input** folder
The stru-input folder contains the crystal structures
```bash
stru-input
├── mat-1 # (extensible)
|   └── stru.cif
├── mat-2
|   └── stru.cif
└── ...
```
Related files on top of the folder:
- `flow.py` (**unit-test**): `-c` reads .json file, create simple material flow.
- `flows.py` (**unit-test**): `-c` reads .json file, create multiple material flows.
- `flows-augmentations.py`: `-c` reads .json file, create `GW` or `BSE` augmentation flows for finished flows.
- `fptask.py`: customized task for the `flow.py` script.
- `collect_tool.py`:
    - md: `collect_tool.py md -md_input MD_INPUT -md_output MD_OUTPUT -md_suffix MD_SUFFIX`
- `config/single_mat_config.json`: The configuration file for the `flow.py`(single material flow).
- `config/fpconfig.json`: The configuration file for the `flows.py` script(multiple material flows).

### 2. **pp** folder
The pp folder contains all .upf and .psml for QE and SIESTA
```
pseudo_src/ # (built-in)
├── ele1.upf
├── ele2.upf
├── ...
├── ele1.psf/psml
├── ele2.psf/psml
└── ...
```

### 3. **flows** folder
```bash
flows/
├── mat-1
|   ├── config.json
|   ├── stru.cif
|   ├── pp/ # (built-in)
|   |   ├── ele1.upf
|   |   ├── ele2.upf
|   |   ├── ...
|   |   ├── ele1.psf/psml
|   |   ├── ele2.psf/psml
|   |   └── ...
|   ├──01-density
|   |   ├── VSC # (DFT Ham.)
|   |   └── ...
|   ├──02-wfn
|   ├──03-wfnq
|   ├──05-band
|   ├──06-wfnq-nns
|   ├──07-aobasis
|   |   ├── ele1.ion # (LCAO basis)
|   |   ├── ele2.ion
|   |   └── ...
|   ├──11-epsilon
|   ├──11-epsilon-nns
|   ├──13-sigma
|   |   ├── eqp1.dat # (G0W0 corr.)
|   |   └── ...
|   ├──14-inteqp
|   ├──16-reconstruction
|   |   ├──aohamiltonian
|   |   |   ├── element.dat
|   |   |   ├── hamiltonians.h5
|   |   |   ├── info.json
|   |   |   ├── lat.dat
|   |   |   ├── orbital_types.dat
|   |   |   ├── overlaps.h5
|   |   |   ├── rlat.dat
|   |   └── └── site_positions.dat
|   ├──17-wfn_fi
|   ├──18-kernel
|   └──19-absorption
├── mat-2
|   └──  ...
└── ...
```

Related files on top of the folder:
- `QE, BGW, HPRO, SIESTA`: First-principle calculator
- `collect_tool.py`(see `-h`): 
    - deeph: `collect_tool.py deeph -flows FLOWS`
    - metalseek: `collect_tool.py metalseek -flows FLOWS `
    - st: `collect_tool.py st -flows FLOWS`
    - sub: `collect_tool.py sub -job JOB -hours HOURS -nodes NODES`
    - compact: `collect_tool.py compact -flows FLOWS (-folder FOLDER) (-unwanted UNWANTED)` (delete unwanted files for all flow and delete 02-wfn/wfn.h5 for all unifhished flow to save space)
    - restart: `collect_tool.py restart -flows FLOWS`
- `from_model/data.py` (**unit-test**): create for WFN, GW, BSE datatype
    - `from_model/wigner.py` (**unit-test**): create wigner cell for WFN
    - `from_model/interface.py` (**unit-test**): interface for `data.py`, including eqp, vloc, wfn, and AScvk classes



### 4. **ManyBodyData.h5** file
```
dataset.h5 (see data.py)
├── info/dict{}
├── mat-1/dict{}
├── mat-2/dict{}
├── mat-3/dict{}
```

Related files on top of the file:
- `collect_tool.py`(see `-h`): 
    - merge: `collect_tool.py merge -folder FOLDER -dataset_fname DATASET_FNAME` (merge all dataset h5 files into one)
- `from_model/data.py` (**unit-test**): load from h5 file
- `from_model/trainer.py`: train the model on the dataset
- `from_model/bsetrainer.py` (**unit-test**)
- `from_model/gwtrainer.py`
- `from_model/e2vaetrainer.py` (todo)
- `from_mode/wfnembedder.py` (**unit-test**)
  - create latent rep to manybodydata
  - create latent rep and save to manybodydata h5 file (suggested!)
  - parallel I/O

- models:
    - `from_model/transformer.py` (**unit-test**)
        - `from_model/basisassembly.py` (**unit-test**)
        - `from_model/posemb.py` (**unit-test**)
    - `from_model/e2vae.py` (**unit-test**)

### 5. **DeepH-E3** input folder

```
ml-train/test
├──graph_file (created by deep-preprocess.py)
├──ham1
|   ├── element.dat
|   ├── hamiltonians.h5
|   ├── info.json
|   ├── lat.dat
|   ├── orbital_types.dat
|   ├── overlaps.h5
|   ├── rlat.dat
|   └── site_positions.dat
├──ham2
|   └──  ...
└── ...
```
Related files on top of the folder:
see deeph3-train.py for more details.

### Benchmark

#### 1. data.py parallelization
| | interface.py  | data.py   | wall time |
|:----:|:------------:|:--------:|:-----------|
| **8 bands**          | -          | -      | <span style="color:red;">237s</span> (base line)    |
| | pool()     | -      | 232s      |
| | pool(4)    | -      | 218s      |
| | pool(8)    | -      | 217s      |
| | -          | pool() | <span style="color:green;">**30s**</span> (fast)      |
| **18 bands**| -          | pool() |  72s      |
| | pool(8)    | -      |     517s      |

