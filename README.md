# Deep-GWBSE

Deep-GWBSE is an end-to-end deep learning pipeline designed for DFT-GW-BSE calculations.

Author: Bowen Hou (bowen.hou@yale.edu)

Contributors: Xian Xu (xian.xu@yale.edu), Jinyuan Wu (jinyuan.wu@yale.edu)

## Outline
- [Deep-GWBSE](#deep-gwbse)
  - [Features](#features)
  - [Documentation](#documentation)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
  - [Scripts Usage](#scripts-usage)
  - [License](#license)
  - [Acknowledgements](#acknowledgements)


## Features
This package provides deep learning models for DFT-GW-BSE calculations from crystal structures, including the following:
- Fully-automatic GW+BSE workflow
  - Parabands + Pseudobands
  - NNS
- VAE+MBFormer: transformer-based model for many-body GW-BSE  
  - model scheme:  
    <p align="center">
      <img src="deep_gwbse/from_model/fig/01-model.png" width="100%">
    </p>

## Documentation

If you only want to use the workflow scripts to quickly setup GW-BSE calculations, you might skip this part.

For developers and advanced users, please carefully read this [**Documentation**](./deep_gwbse/note.md) for more details.


## Installation

### Prerequisites: First-principles Packages
- [Quantum ESPRESSO](https://www.quantum-espresso.org/) version 6.8
- [BerkeleyGW](https://berkeleygw.org/documentation/tutorial/) version 3

### Deep-GWBSE Package Installation

#### Option 1: Using `uv` (Recommended)

```bash
# Clone the repository
git clone https://github.com/bwhou1997/Deep-GWBSE.git
cd Deep-GWBSE

# Install using uv
uv sync
```

## Quick Start

### 0. Setup Configuration Files

First, set up your configuration files with paths to Quantum ESPRESSO, BerkeleyGW, and pseudopotentials:

```bash
# Copy example config files
cp deep_gwbse/config/single_mat_config.json ./
cp deep_gwbse/config/fpconfig.json ./
```

Edit the configuration files to set:
- `"QE_path"`: Path to Quantum ESPRESSO installation
- `"BGW_path"`: Path to BerkeleyGW installation  
- `"pseudo_dir_source"`: Path to pseudopotential directory (can use `./deep_gwbse/from_oncvpsp` for built-in pseudos)

## Scripts Usage

### DFT-GW-BSE Workflow Scripts

#### 1. Multiple Materials Workflow (`flows.py`)

Create workflows for multiple materials from a directory:

```bash
python flows.py -c fpconfig.json
cd flows
sbatch run.sh
```

#### 2. Workflow Augmentation (`flows-augmentation.py`)

Create augmentation workflows for GW or BSE calculations from existing completed flows:

```bash
python flows-augmentation.py -c augconfig.json
cd <augmentation_directory>
sbatch run_aug.sh
```

### MBFormer Training Scripts

#### 1. Data Preprocessing (`mbformer_data.py`)

Preprocess raw data from GW-BSE calculations to create training datasets:

```bash
python mbformer_data.py
```

This will create three HDF5 files:
- `dataset_WFN.h5` - For training VAE
- `dataset_GW.h5` - For training GW-MBFormer
- `dataset_BSE.h5` - For training BSE-MBFormer

**Note**: Modify the script to point to your data directory and adjust parameters as needed.

#### 2. VAE Training (`mbformer_vae.py`)

Train an E2-VAE (Equivariant Variational Autoencoder) to embed KS wavefunctions:

```bash
python mbformer_vae.py
```

The trained model will be saved as `./vae_e2_wfn.save`.

#### 3. GW Training (`mbformer_gw.py`)

Train a transformer model for GW (G0W0) energy predictions:

```bash
python mbformer_gw.py
```

**Prerequisites**: Requires a trained VAE model (`./vae_e2_wfn.save`) and GW dataset (`./dataset/dataset_GW.h5`).

#### 4. BSE Training (`mbformer_bse.py`)

Train transformer models for BSE predictions (eigenvalues, eigenvectors, dipole moments):

```bash
python mbformer_bse.py
```

**Prerequisites**: Requires a trained VAE model (`./vae_e2_wfn.save`) and BSE dataset (`./dataset/dataset_BSE.h5`).

## Detailed Workflow

### Step 1: Run DFT-GW-BSE Calculations

For a single material:
```bash
python flows.py -c single_mat_config.json
```

For multiple materials:
```bash
python flows.py -c fpconfig.json
```

### Step 2: Preprocess Data for Machine Learning

After your calculations complete, preprocess the data:

```bash
python mbformer_data.py
```

Modify the script to point to your `flows` directory and adjust dataset parameters.

### Step 3: Train MBFormer Models

Train models in sequence:

1. **VAE** (required for GW and BSE models):
   ```bash
   python mbformer_vae.py
   ```

2. **GW Model**:
   ```bash
   python mbformer_gw.py
   ```

3. **BSE Model**:
   ```bash
   python mbformer_bse.py
   ```

## Project Structure

```
Deep-GWBSE/
├── deep_gwbse/          # Main package
│   ├── __init__.py
│   ├── flow.py          # Single material workflow class
│   ├── flows.py         # Multiple materials workflow
│   ├── flows-augmentation.py  # Augmentation workflows
│   ├── from_bgwpy/      # BGWpy integration
│   ├── from_model/      # ML models and trainers
│   ├── config/          # Configuration templates
│   └── ...
├── flows.py             # Root-level script for multiple materials
├── flows-augmentation.py  # Root-level augmentation script
├── mbformer_data.py     # Data preprocessing script
├── mbformer_vae.py      # VAE training script
├── mbformer_gw.py       # GW training script
├── mbformer_bse.py      # BSE training script
├── pyproject.toml       # Package configuration
└── README.md
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


## Acknowledgements
We would like to acknowledge the following open-source projects that have made this work possible:
[Quantum ESPRESSO](https://www.quantum-espresso.org/), [BerkeleyGW](https://berkeleygw.org/), [SIESTA](https://docs.siesta-project.org/projects/siesta/en/stable/index.html), [DeepH-E3](https://github.com/Xiaoxun-Gong/DeepH-E3), [HPRO](https://github.com/Xiaoxun-Gong/HPRO), bgwpy
