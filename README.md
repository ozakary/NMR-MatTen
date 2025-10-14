# Steps to Generate the NMR-ML Model using MatTen Architecture

---
📄 Author: **Ouail Zakary**  
- 📧 Email: [Ouail.Zakary@oulu.fi](mailto:Ouail.Zakary@oulu.fi)  
- 🔗 ORCID: [0000-0002-7793-3306](https://orcid.org/0000-0002-7793-3306)  
- 🌐 Website: [Personal Webpage](https://cc.oulu.fi/~nmrwww/members/Ouail_Zakary.html)  
- 📁 Portfolio: [Academic Portfolio](https://ozakary.github.io/)
---

## Workflow Overview

The complete workflow consists of three main stages:

```
[0] DFT Calculations (TURBOMOLE)
[1] Dataset Preparation (prepare_matten_dataset.sh)
[2] Model Training (train_atomic_tensor.py)
[3] Prediction (predict_atomic_tensor.py)
```

## [1] Dataset Preparation

### Input Requirements

The preparation script `prepare_matten_dataset.sh` expects TURBOMOLE calculation outputs organized as follows:

```
./
├── config.txt                          # Configuration file
├── prepare_matten_dataset.sh
├── matten_dataset_output/ (created by script)
├── FINISHED/
│   ├── cluster_1/
│   │   ├── coord_1.xyz
│   │   ├── coord_2.xyz
│   │   ├── ...
│   │   └── mpshift.out
│   ├── cluster_2/
│   │   ├── coord_1.xyz
│   │   └── mpshift.out
│   ├── ...
│   ├── ml_nmr_schnet_dataset_oz-t.py
│   ├── alignements.py
│   ├── code_data.py
│   ├── csv_to_json_converter_enhanced.py
│   ├── grep_commands_verif.sh (optional)
│   └── info.sh (optional)
```

**Important**: All processing scripts must be placed inside the `./FINISHED/` directory.

### Configuration File (`config.txt`)

Create a `config.txt` file in the same directory as `prepare_matten_dataset.sh` with the following parameters:

```bash
# Target atom for NMR predictions
TARGET_ATOM=Xe

# Lattice parameters (9 values: 3x3 matrix flattened)
LATTICE_PARAMS=23.86466077 0.0 0.0 0.0 23.86466077 0.0 0.0 0.0 23.86466077

# Output directory name
OUTPUT_DIR=matten_dataset_output

# Dataset split ratios (must sum to 1.0)
TRAIN_RATIO=0.8
VAL_RATIO=0.1
TEST_RATIO=0.1

# Random seed for reproducibility
RANDOM_SEED=42
```

**Configuration Parameters:**

| Parameter | Description | Example |
|-----------|-------------|---------|
| `TARGET_ATOM` | Element symbol for NMR predictions | `Xe`, `C`, `H`, `N` |
| `LATTICE_PARAMS` | 9 values defining the unit cell | `23.86 0 0 0 23.86 0 0 0 23.86` |
| `OUTPUT_DIR` | Name of output directory | `matten_dataset_output` |
| `TRAIN_RATIO` | Fraction for training set (0-1) | `0.8` |
| `VAL_RATIO` | Fraction for validation set (0-1) | `0.1` |
| `TEST_RATIO` | Fraction for test set (0-1) | `0.1` |
| `RANDOM_SEED` | Seed for reproducible splits | `42` |

Make the script executable and run it from the directory containing `FINISHED/` and `config.txt`:

```bash
chmod +x prepare_matten_dataset.sh
./prepare_matten_dataset.sh config.txt
```
---
### The `prepare_matten_dataset.sh` will follow three stages:

#### **Stage 1: DFT Data Processing (this stage also applies to SchNet dataset preparation)**

1. **Data Extraction** (`ml_nmr_schnet_dataset_oz-t.py`)
   - Scans all `cluster_*/` directories in `FINISHED/`
   - Extracts atomic coordinates from `coord_*.xyz` files
   - Parses magnetic shielding tensors from `mpshift.out` files
   - Handles missing tensor data with zero-padding
   - Creates individual CSV files per molecule
   - Generates concatenated datasets

   **Output:**
   ```
   FINISHED/
   ├── dataset_schnet_shielding_tensors/
   │   ├── structure_*.csv              # Individual molecule tensors
   │   └── magnetic_shielding_tensors.csv
   │
   └── dataset_schnet_atomic_coordinates/
       ├── structure_*.csv              # Individual molecule coordinates
       └── structures.csv
   ```

2. **Tensor Alignment** (`alignements.py`)
   - Matches molecules between structure and tensor files
   - Identifies the targeted atom(s) positions from structure data
   - Places non-zero tensors at the correct targeted atom(s) positions
   - Fills remaining positions with zero tensors
   - Ensures consistent atom indexing

   **Output:**
   ```
   FINISHED/dataset_schnet_shielding_tensors/magnetic_shielding_tensors_modified.csv
   ```

3. **Sanity check** (optional: `grep_commands_verif.sh`)
   - Removes zero-tensor entries for verification
   - Extracts only structures with the targeted atom(s) for validation
   - Generates verification CSV files

   **Output:**
   ```
   FINISHED/
   ├── ms_verif.csv     # Non-zero tensors only
   └── str_verif.csv    # targeted atom(s) structures only
   ```

4. **Cluster Validation** (optional: `info.sh`)
   - Lists all processed cluster directories
   - Validates completeness of calculations

#### **Stage 2: Dataset Integration**

**Isotropic Shielding Calculation** (`code_data.py`)
- Loads structure and tensor data
- Symmetrizes tensors: `T_sym = (T + T^T) / 2`
- Computes eigenvalues via diagonalization
- Calculates isotropic shielding: `σ_iso = (λ₁ + λ₂ + λ₃) / 3`
- Merges σ_iso with structural data

**Output:**
```
FINISHED/structures_with_sigma_iso.csv
```

#### **Stage 3: JSON Conversion for MatTen**

**Format Conversion** (`csv_to_json_converter_enhanced.py`)
- Creates output directory: `matten_dataset_output/`
- Generates configuration file with lattice parameters
- Converts CSV data to PyMatGen Structure objects
- Validates tensor symmetry and corrects if needed
- Splits dataset: 80% train, 10% validation, 10% test
- Outputs MatTen-compatible JSON files

**Output:**
```
matten_dataset_output/
├── dataset_train.json              # Training set
├── dataset_val.json                # Validation set
├── dataset_test.json               # Test set
├── dataset_test_structures.xyz     # Test structures (XYZ format)
├── structures_with_sigma_iso_and_tensors.csv  # Test data (CSV)
└── config.txt                      # Configuration used
```
---

## [2] Model Training

### Installation Setup (Tested in PUHTI and Mahti in [CSC](https://docs.csc.fi/computing/), as well as [LUMI](https://docs.lumi-supercomputer.eu/) Supercomputers)

```bash
# These are the installation steps for PUHTI (if you don't have access to LUMI, it is recommended to use MatTen in PUHTI instead of MAHTI, as there are more GPU resources)
# Load Python module (it should be python --version 3.10)
module load python-data/3.10

# Create a virtual environment (recommended)
python3 -m venv matten_env

# Activate the created virtual environment
source matten_env/bin/activate

# Upgrade Pypl
pip install --upgrade pip

# Install PyTorch 2.0.1 with CUDA 11.8
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install torch-geometric and dependencies with specific versions
pip install torch-geometric==2.3.1
pip install torch-scatter==2.1.1 -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

# Install other required dependencies
pip install pytorch-lightning==2.0.7
pip install torchmetrics==0.11.4
pip install e3nn==0.5.1
pip install ase==3.22.1
pip install pymatgen==2023.8.10
pip install loguru
pip install torchtyping==0.1.4
pip install numpy==1.23.5
pip install wandb

# Clone the repository
git clone https://github.com/wengroup/matten
cd matten
pip install -e .
```

### Setup Training Environment

```bash
# Create training directory structure
mkdir -p training_run
cd training_run

mkdir -p configs datasets matten_logs
```

### Copy Datasets

```bash
# Copy prepared JSON datasets
cp ../matten_dataset_output/dataset_train.json datasets/
cp ../matten_dataset_output/dataset_val.json datasets/
cp ../matten_dataset_output/dataset_test.json datasets/
```

### Configure Model

Create `configs/atomic_tensor.yaml`.

### Training Script

Use the provided `train_atomic_tensor.py`.

### Submit Training Job (`script_train.job`)

### Training Output

The training process generates:

- **Checkpoints**: `matten_logs/checkpoints/last.ckpt`, `best.ckpt`
- **Logs**: Training metrics, loss curves, validation scores
- **Weights & Biases**: Online visualization (if configured)

---

## [3] Prediction from Trained Model

### Prepare Input Structures

Create an XYZ file with your structures (can be from MD trajectories, geometry optimizations, etc.):

```
1170
Lattice="23.86466077 0.0 0.0 0.0 23.86466077 0.0 0.0 0.0 23.86466077" Properties=species:S:1:pos:R:3
O  23.4449080892 19.9457264109 16.8624230406
C  23.6869558502 13.4755800908 15.7803088909
...
Xe  6.82709 11.3068 8.4432
...
```

### Run Predictions

Use the provided `predict_atomic_tensor.py` script:

```bash
python3 predict_atomic_tensor.py \
    input_structures.xyz \
    output_predictions.csv \
    --model-path matten_logs/checkpoints/ \
    --checkpoint last.ckpt
```

### Submit Prediction Job (`script_predict.job`)

### Prediction Output Format

Example output:

```csv
structure_id,atom_index,element,x,y,z,sigma_iso,tensor_xx,tensor_xy,tensor_xz,tensor_yx,tensor_yy,tensor_yz,tensor_zx,tensor_zy,tensor_zz
0,265,Xe,6.82709,11.3068,8.4432,5697.37,5694.35,-20.91,15.23,-20.91,5701.82,-8.45,15.23,-8.45,5695.94
1,918,Xe,15.4523,18.9876,12.7654,5682.15,5680.22,-15.67,12.89,-15.67,5685.43,-6.78,12.89,-6.78,5680.80
```

### Converting to Chemical Shifts

To convert magnetic shielding (σ) to chemical shifts (δ):

```python
# Reference: Free Xe atom shielding (this method applies to other noble gases)
sigma_ref = 5797.0  # ppm (typical value, adjust for your reference)

# Calculate chemical shift
delta = sigma_ref - sigma_iso
```

---

## Directory Structure

### Complete Project Layout

```
matten-nmr-workflow/
│
├── README.md
├── config.txt                          # Configuration file (NEW!)
├── prepare_matten_dataset.sh           # Main dataset preparation script
│
├── FINISHED/                           # DFT calculation outputs + scripts
│   ├── cluster_1/
│   │   ├── coord_1.xyz
│   │   ├── coord_2.xyz
│   │   └── mpshift.out
│   ├── cluster_2/
│   │   └── ...
│   ├── ...
│   │
│   ├── ml_nmr_schnet_dataset_oz-t.py      # Data extraction script
│   ├── alignements.py                      # Tensor alignment script
│   ├── code_data.py                        # Isotropic shielding calculation
│   ├── csv_to_json_converter_enhanced.py  # JSON conversion script
│   ├── grep_commands_verif.sh             # Quality control (optional)
│   ├── info.sh                            # Cluster validation (optional)
│   │
│   ├── dataset_schnet_shielding_tensors/  # Generated during processing
│   │   ├── Xe_TBA_*.csv
│   │   ├── magnetic_shielding_tensors.csv
│   │   └── magnetic_shielding_tensors_modified.csv
│   │
│   ├── dataset_schnet_atomic_coordinates/ # Generated during processing
│   │   ├── Xe_TBA_*.csv
│   │   └── structures.csv
│   │
│   └── structures_with_sigma_iso.csv      # Generated during processing
│
├── matten_dataset_output/             # Generated by preparation script
│   ├── dataset_train.json
│   ├── dataset_val.json
│   ├── dataset_test.json
│   ├── dataset_test_structures.xyz
│   ├── structures_with_sigma_iso_and_tensors.csv
│   └── config.txt
│
├── training_run/                      # Training directory
│   ├── configs/
│   │   └── atomic_tensor.yaml
│   ├── datasets/
│   │   ├── dataset_train.json
│   │   ├── dataset_val.json
│   │   └── dataset_test.json
│   ├── matten_logs/
│   │   └── checkpoints/
│   │       ├── last.ckpt
│   │       └── best.ckpt
│   ├── train_atomic_tensor.py
│   └── script_train.job
│
└── prediction/                        # Prediction directory
    ├── predict_atomic_tensor.py
    ├── script_predict.job
    ├── input_structures.xyz
    └── output_predictions.csv
│   │
│   └── structures_with_sigma_iso.csv      # Generated during processing
│
├── matten_dataset_output/             # Generated by preparation script
│   ├── dataset_train.json
│   ├── dataset_val.json
│   ├── dataset_test.json
│   ├── dataset_test_structures.xyz
│   ├── structures_with_sigma_iso_and_tensors.csv
│   └── config.txt
│
├── training_run/                      # Training directory
│   ├── configs/
│   │   └── atomic_tensor.yaml
│   ├── datasets/
│   │   ├── dataset_train.json
│   │   ├── dataset_val.json
│   │   └── dataset_test.json
│   ├── matten_logs/
│   │   └── checkpoints/
│   │       ├── last.ckpt
│   │       └── best.ckpt
│   ├── train_atomic_tensor.py
│   └── script_train.job
│
└── prediction/                        # Prediction directory
    ├── predict_atomic_tensor.py
    ├── script_predict.job
    ├── input_structures.xyz
    └── output_predictions.csv
```
---
For further details, please refer to the respective folders or contact the author via the provided email.
