#!/bin/bash

################################################################################
# MatTen Dataset Preparation Script
# Author: Ouail Zakary (Ouail.Zakary@oulu.fi)
# Description: Automated workflow for processing DFT-calculated NMR 
#              magnetic shielding tensors and converting them into JSON format
#              for the MatTen architecture
# 
# Usage: ./prepare_matten_dataset.sh [config_file]
#        If no config file is specified, uses config.txt in current directory
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# Default configuration file
CONFIG_FILE="config.txt"
WORK_DIR="FINISHED"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_section() {
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}========================================${NC}"
}

# Load configuration from file
load_config() {
    local config_file=$1
    
    log_section "Loading Configuration"
    
    if [[ ! -f "$config_file" ]]; then
        log_error "Configuration file not found: $config_file"
        log_info "Please create $config_file with required parameters"
        log_info "Example format:"
        echo "  TARGET_ATOM=Xe"
        echo "  LATTICE_PARAMS=23.86466077 0.0 0.0 0.0 23.86466077 0.0 0.0 0.0 23.86466077"
        echo "  OUTPUT_DIR=matten_dataset_output"
        echo "  TRAIN_RATIO=0.8"
        echo "  VAL_RATIO=0.1"
        echo "  TEST_RATIO=0.1"
        echo "  RANDOM_SEED=42"
        exit 1
    fi
    
    log_info "Reading configuration from: $config_file"
    
    # Read configuration file (skip comments and empty lines)
    while IFS='=' read -r key value; do
        # Skip comments and empty lines
        [[ "$key" =~ ^#.*$ ]] && continue
        [[ -z "$key" ]] && continue
        
        # Remove leading/trailing whitespace
        key=$(echo "$key" | xargs)
        value=$(echo "$value" | xargs)
        
        # Export variables
        case "$key" in
            TARGET_ATOM)
                TARGET_ATOM="$value"
                log_info "  Target atom: $TARGET_ATOM"
                ;;
            LATTICE_PARAMS)
                LATTICE_PARAMS="$value"
                log_info "  Lattice parameters: $LATTICE_PARAMS"
                ;;
            OUTPUT_DIR)
                OUTPUT_DIR="$value"
                log_info "  Output directory: $OUTPUT_DIR"
                ;;
            TRAIN_RATIO)
                TRAIN_RATIO="$value"
                ;;
            VAL_RATIO)
                VAL_RATIO="$value"
                ;;
            TEST_RATIO)
                TEST_RATIO="$value"
                ;;
            RANDOM_SEED)
                RANDOM_SEED="$value"
                ;;
        esac
    done < "$config_file"
    
    # Validate required parameters
    if [[ -z "${TARGET_ATOM:-}" ]]; then
        log_error "TARGET_ATOM not defined in config file"
        exit 1
    fi
    
    if [[ -z "${LATTICE_PARAMS:-}" ]]; then
        log_error "LATTICE_PARAMS not defined in config file"
        exit 1
    fi
    
    if [[ -z "${OUTPUT_DIR:-}" ]]; then
        OUTPUT_DIR="matten_dataset_output"
        log_warning "OUTPUT_DIR not defined, using default: $OUTPUT_DIR"
    fi
    
    # Set default split ratios if not defined
    TRAIN_RATIO=${TRAIN_RATIO:-0.8}
    VAL_RATIO=${VAL_RATIO:-0.1}
    TEST_RATIO=${TEST_RATIO:-0.1}
    RANDOM_SEED=${RANDOM_SEED:-42}
    
    log_info "  Dataset split: Train=${TRAIN_RATIO}, Val=${VAL_RATIO}, Test=${TEST_RATIO}"
    log_info "  Random seed: $RANDOM_SEED"
    
    # Validate split ratios sum to 1.0
    local sum=$(echo "$TRAIN_RATIO + $VAL_RATIO + $TEST_RATIO" | bc)
    if (( $(echo "$sum != 1.0" | bc -l) )); then
        log_warning "Split ratios sum to $sum (should be 1.0), proceeding anyway..."
    fi
    
    log_success "Configuration loaded successfully"
}

# Update Python scripts with TARGET_ATOM
update_python_scripts() {
    log_section "Updating Python Scripts with Target Atom"
    
    cd "$WORK_DIR" || exit 1
    
    log_info "Setting target atom to: $TARGET_ATOM"
    
    # Update csv_to_json_converter_enhanced.py
    if [[ -f "csv_to_json_converter_enhanced.py" ]]; then
        log_info "Updating csv_to_json_converter_enhanced.py..."
        # Replace "Xe" with TARGET_ATOM in the relevant lines
        sed -i.bak "s/if element == 'Xe'/if element == '$TARGET_ATOM'/g" csv_to_json_converter_enhanced.py
        sed -i.bak "s/# Only include Xe atoms/# Only include $TARGET_ATOM atoms/g" csv_to_json_converter_enhanced.py
        sed -i.bak "s/qn_values.append(0)  # Xe atoms/qn_values.append(0)  # $TARGET_ATOM atoms/g" csv_to_json_converter_enhanced.py
        log_success "Updated csv_to_json_converter_enhanced.py"
    else
        log_warning "csv_to_json_converter_enhanced.py not found"
    fi
    
    # Update alignements.py
    if [[ -f "alignements.py" ]]; then
        log_info "Updating alignements.py..."
        sed -i.bak "s/mol_struct\['atom'\] == 'Xe'/mol_struct['atom'] == '$TARGET_ATOM'/g" alignements.py
        sed -i.bak "s/# Find indices where atom is Xe/# Find indices where atom is $TARGET_ATOM/g" alignements.py
        log_success "Updated alignements.py"
    else
        log_warning "alignements.py not found"
    fi
    
    # Update grep_commands_verif.sh
    if [[ -f "grep_commands_verif.sh" ]]; then
        log_info "Updating grep_commands_verif.sh..."
        sed -i.bak "s/',Xe,'/','$TARGET_ATOM,'/g" grep_commands_verif.sh
        log_success "Updated grep_commands_verif.sh"
    else
        log_warning "grep_commands_verif.sh not found (optional)"
    fi
    
    log_success "Python scripts updated for target atom: $TARGET_ATOM"
    
    cd - > /dev/null
}

# Update csv_to_json_converter for split ratios
update_json_converter_splits() {
    log_section "Configuring Dataset Split Ratios"
    
    cd "$WORK_DIR" || exit 1
    
    if [[ -f "csv_to_json_converter_enhanced.py" ]]; then
        log_info "Updating split ratios in csv_to_json_converter_enhanced.py..."
        log_info "  Train: $TRAIN_RATIO, Val: $VAL_RATIO, Test: $TEST_RATIO"
        log_info "  Random seed: $RANDOM_SEED"
        
        # Create a Python script to update the split_dataset function
        cat > update_splits.py << EOF
import re

with open('csv_to_json_converter_enhanced.py', 'r') as f:
    content = f.read()

# Update default parameters in split_dataset function
pattern = r'def split_dataset\(json_data, train_ratio=[\d.]+, val_ratio=[\d.]+, test_ratio=[\d.]+, random_state=\d+\):'
replacement = f'def split_dataset(json_data, train_ratio=$TRAIN_RATIO, val_ratio=$VAL_RATIO, test_ratio=$TEST_RATIO, random_state=$RANDOM_SEED):'
content = re.sub(pattern, replacement, content)

with open('csv_to_json_converter_enhanced.py', 'w') as f:
    f.write(content)

print("Split ratios updated successfully")
EOF
        
        python3 update_splits.py
        rm update_splits.py
        
        log_success "Split ratios configured"
    else
        log_warning "csv_to_json_converter_enhanced.py not found"
    fi
    
    cd - > /dev/null
}

# Check if required Python scripts exist
check_dependencies() {
    log_section "Checking Dependencies"
    
    # Check if FINISHED directory exists
    if [[ ! -d "$WORK_DIR" ]]; then
        log_error "FINISHED directory not found in current location"
        log_info "Please run this script from the directory containing FINISHED/"
        exit 1
    fi
    
    log_info "Checking for required scripts in $WORK_DIR/..."
    
    local missing_deps=0
    local required_scripts=(
        "ml_nmr_schnet_dataset_oz-t.py"
        "alignements.py"
        "code_data.py"
        "csv_to_json_converter_enhanced.py"
    )
    
    local optional_scripts=(
        "grep_commands_verif.sh"
        "info.sh"
    )
    
    for script in "${required_scripts[@]}"; do
        if [[ ! -f "$WORK_DIR/$script" ]]; then
            log_error "Required script not found: $WORK_DIR/$script"
            missing_deps=1
        else
            log_success "Found: $script"
        fi
    done
    
    for script in "${optional_scripts[@]}"; do
        if [[ -f "$WORK_DIR/$script" ]]; then
            log_success "Found (optional): $script"
        else
            log_warning "Optional script not found: $script"
        fi
    done
    
    if [[ $missing_deps -eq 1 ]]; then
        log_error "Missing dependencies. Please ensure all required scripts are in $WORK_DIR/"
        exit 1
    fi
    
    log_success "All required dependencies found in $WORK_DIR/"
}

# Process DFT calculation data
process_dft_data() {
    log_section "Processing DFT Calculation Data"
    
    log_info "Working directory: $WORK_DIR/"
    log_info "Looking for TURBOMOLE calculation outputs..."
    
    # Change to FINISHED directory
    cd "$WORK_DIR" || exit 1
    
    # Check for cluster subdirectories
    local cluster_count=$(find . -maxdepth 1 -type d -name "cluster_*" | wc -l)
    log_info "Found $cluster_count cluster directories"
    
    if [[ $cluster_count -eq 0 ]]; then
        log_error "No cluster_* directories found in $WORK_DIR/"
        cd - > /dev/null
        exit 1
    fi
    
    # Step 1: Data Extraction
    log_info "Step 1: Extracting atomic coordinates and magnetic shielding tensors..."
    python3 ml_nmr_schnet_dataset_oz-t.py
    log_success "Data extraction completed"
    
    # Check extraction outputs
    if [[ -d "dataset_schnet_shielding_tensors" ]] && [[ -d "dataset_schnet_atomic_coordinates" ]]; then
        log_success "Created dataset directories"
        local tensor_files=$(find dataset_schnet_shielding_tensors -name '*.csv' | wc -l)
        local struct_files=$(find dataset_schnet_atomic_coordinates -name '*.csv' | wc -l)
        log_info "  - Tensor files: $tensor_files CSV files"
        log_info "  - Structure files: $struct_files CSV files"
    else
        log_error "Dataset directories not created as expected"
        cd - > /dev/null
        exit 1
    fi
    
    # Step 2: Tensor Alignment
    log_info "Step 2: Aligning tensor data with atomic structure data..."
    log_info "  (Focusing on $TARGET_ATOM atoms)"
    python3 alignements.py
    log_success "Tensor alignment completed"
    
    if [[ -f "dataset_schnet_shielding_tensors/magnetic_shielding_tensors_modified.csv" ]]; then
        local line_count=$(wc -l < dataset_schnet_shielding_tensors/magnetic_shielding_tensors_modified.csv)
        log_info "  - Modified tensor file contains: $line_count lines"
    else
        log_error "magnetic_shielding_tensors_modified.csv not created"
        cd - > /dev/null
        exit 1
    fi
    
    # Step 3: Data Verification (optional)
    if [[ -f "grep_commands_verif.sh" ]]; then
        log_info "Step 3: Running quality control checks..."
        chmod +x grep_commands_verif.sh
        ./grep_commands_verif.sh
        log_success "Quality control checks completed"
        
        if [[ -f "ms_verif.csv" ]] && [[ -f "str_verif.csv" ]]; then
            local nonzero_count=$(($(wc -l < ms_verif.csv) - 1))
            local target_count=$(($(wc -l < str_verif.csv) - 1))
            log_info "  - Non-zero tensor entries: $nonzero_count"
            log_info "  - $TARGET_ATOM structure entries: $target_count"
        fi
    else
        log_warning "grep_commands_verif.sh not found, skipping verification"
    fi
    
    # Step 4: Cluster Validation (optional)
    if [[ -f "info.sh" ]]; then
        log_info "Step 4: Validating cluster directories..."
        chmod +x info.sh
        ./info.sh ./
        log_success "Cluster validation completed"
    else
        log_warning "info.sh not found, skipping cluster validation"
    fi
    
    log_success "DFT data processing completed"
    
    # Return to parent directory
    cd - > /dev/null
}

# Integrate isotropic shielding
integrate_dataset() {
    log_section "Dataset Integration"
    
    log_info "Computing isotropic shielding (σ_iso) from tensor components..."
    log_info "Process: Symmetrize tensors → Compute eigenvalues → Calculate σ_iso"
    
    # Change to FINISHED directory
    cd "$WORK_DIR" || exit 1
    
    # Verify input files exist
    if [[ ! -f "dataset_schnet_atomic_coordinates/structures.csv" ]]; then
        log_error "structures.csv not found in dataset_schnet_atomic_coordinates/"
        cd - > /dev/null
        exit 1
    fi
    
    if [[ ! -f "dataset_schnet_shielding_tensors/magnetic_shielding_tensors_modified.csv" ]]; then
        log_error "magnetic_shielding_tensors_modified.csv not found"
        cd - > /dev/null
        exit 1
    fi
    
    # Update code_data.py to use correct paths
    cat > code_data_updated.py << 'EOF'
import pandas as pd
import numpy as np

# Load structures.csv
structures_df = pd.read_csv('./dataset_schnet_atomic_coordinates/structures.csv')
print(f"Loaded structures data with shape: {structures_df.shape} and columns: {structures_df.columns.tolist()}")

# Load magnetic shielding tensors
tensors_df = pd.read_csv('./dataset_schnet_shielding_tensors/magnetic_shielding_tensors_modified.csv')
print(f"Loaded magnetic shielding tensors with shape: {tensors_df.shape} and columns: {tensors_df.columns.tolist()}")

# Extract and process tensor components
tensor_components = tensors_df.iloc[:, 2:].values  # get the 9 tensor values
tensor_components = tensor_components.reshape(-1, 3, 3)
tensor_components = 0.5 * (tensor_components + np.transpose(tensor_components, (0, 2, 1)))  # symmetrize

# Compute eigenvalues and derived parameters
w, _ = np.linalg.eigh(tensor_components)
sigma_iso = np.mean(w, axis=1)

# Build DataFrame for calculated values
params_df = tensors_df[['molecule_name', 'atom_index']].copy()
params_df['sigma_iso'] = sigma_iso

# Merge sigma_iso into structures.csv based on molecule_name and atom_index
merged_df = pd.merge(structures_df, params_df[['molecule_name', 'atom_index', 'sigma_iso']],
                     on=['molecule_name', 'atom_index'], how='left')

# Save new CSV with sigma_iso appended
merged_df.to_csv('structures_with_sigma_iso.csv', index=False)
print("Saved merged data to 'structures_with_sigma_iso.csv'")
EOF
    
    python3 code_data_updated.py
    rm code_data_updated.py
    
    log_success "Dataset integration completed"
    
    if [[ -f "structures_with_sigma_iso.csv" ]]; then
        local line_count=$(wc -l < structures_with_sigma_iso.csv)
        log_info "  - Integrated dataset contains: $line_count lines"
    else
        log_error "structures_with_sigma_iso.csv not created"
        cd - > /dev/null
        exit 1
    fi
    
    # Return to parent directory
    cd - > /dev/null
}

# Convert to JSON format for MatTen
convert_to_json() {
    log_section "JSON Conversion for MatTen"
    
    log_info "Converting CSV data to MatTen-compatible JSON format..."
    log_info "Creating output directory: $OUTPUT_DIR"
    
    # Create output directory in parent folder
    mkdir -p "$OUTPUT_DIR"
    
    # Create config file
    log_info "Generating configuration file..."
    cat > "$OUTPUT_DIR/config.txt" << EOF
../$WORK_DIR/structures_with_sigma_iso.csv,../$WORK_DIR/dataset_schnet_shielding_tensors/magnetic_shielding_tensors_modified.csv,$LATTICE_PARAMS
EOF
    log_success "Created config.txt with lattice parameters"
    log_info "  Target atom: $TARGET_ATOM"
    log_info "  Lattice params: $LATTICE_PARAMS"
    
    # Copy converter script to output directory
    cp "$WORK_DIR/csv_to_json_converter_enhanced.py" "$OUTPUT_DIR/"
    
    # Run conversion
    cd "$OUTPUT_DIR" || exit 1
    
    python3 csv_to_json_converter_enhanced.py config.txt
    log_success "JSON conversion completed"
    
    # Report output files
    echo ""
    log_info "Generated MatTen dataset files:"
    
    if [[ -f "dataset_train.json" ]]; then
        local size=$(du -h dataset_train.json | cut -f1)
        local entries=$(python3 -c "import json; print(len(json.load(open('dataset_train.json'))['sigma_iso']))" 2>/dev/null || echo "N/A")
        log_info "  ✓ dataset_train.json ($size, $entries structures)"
    fi
    
    if [[ -f "dataset_val.json" ]]; then
        local size=$(du -h dataset_val.json | cut -f1)
        local entries=$(python3 -c "import json; print(len(json.load(open('dataset_val.json'))['sigma_iso']))" 2>/dev/null || echo "N/A")
        log_info "  ✓ dataset_val.json ($size, $entries structures)"
    fi
    
    if [[ -f "dataset_test.json" ]]; then
        local size=$(du -h dataset_test.json | cut -f1)
        local entries=$(python3 -c "import json; print(len(json.load(open('dataset_test.json'))['sigma_iso']))" 2>/dev/null || echo "N/A")
        log_info "  ✓ dataset_test.json ($size, $entries structures)"
    fi
    
    if [[ -f "dataset_test_structures.xyz" ]]; then
        local size=$(du -h dataset_test_structures.xyz | cut -f1)
        log_info "  ✓ dataset_test_structures.xyz ($size)"
    fi
    
    if [[ -f "structures_with_sigma_iso_and_tensors.csv" ]]; then
        local size=$(du -h structures_with_sigma_iso_and_tensors.csv | cut -f1)
        log_info "  ✓ structures_with_sigma_iso_and_tensors.csv ($size)"
    fi
    
    cd - > /dev/null
}

# Generate summary report
generate_summary() {
    log_section "Dataset Preparation Summary"
    
    echo ""
    log_info "Configuration:"
    log_info "  - Target atom: $TARGET_ATOM"
    log_info "  - Lattice parameters: $LATTICE_PARAMS"
    log_info "  - Dataset split: ${TRAIN_RATIO}/${VAL_RATIO}/${TEST_RATIO}"
    
    echo ""
    log_info "Processing Statistics:"
    
    # DFT processing outputs
    if [[ -d "$WORK_DIR" ]]; then
        local cluster_count=$(find "$WORK_DIR" -maxdepth 1 -type d -name "cluster_*" | wc -l)
        log_info "  - Processed cluster directories: $cluster_count"
    fi
    
    if [[ -f "$WORK_DIR/dataset_schnet_shielding_tensors/magnetic_shielding_tensors_modified.csv" ]]; then
        local count=$(($(wc -l < "$WORK_DIR/dataset_schnet_shielding_tensors/magnetic_shielding_tensors_modified.csv") - 1))
        log_info "  - Aligned tensor entries: $count"
    fi
    
    if [[ -f "$WORK_DIR/structures_with_sigma_iso.csv" ]]; then
        local count=$(($(wc -l < "$WORK_DIR/structures_with_sigma_iso.csv") - 1))
        log_info "  - Structures with σ_iso: $count"
        
        # Count target atoms
        if command -v awk &> /dev/null; then
            local target_atom_count=$(awk -F, -v atom="$TARGET_ATOM" '$3 == atom' "$WORK_DIR/structures_with_sigma_iso.csv" | wc -l)
            log_info "  - $TARGET_ATOM atoms with data: $target_atom_count"
        fi
    fi
    
    echo ""
    log_info "Output Directory: $OUTPUT_DIR/"
    
    if [[ -d "$OUTPUT_DIR" ]]; then
        log_info "MatTen JSON Datasets:"
        
        for dataset in dataset_train.json dataset_val.json dataset_test.json; do
            if [[ -f "$OUTPUT_DIR/$dataset" ]]; then
                local size=$(du -h "$OUTPUT_DIR/$dataset" | cut -f1)
                log_info "  ✓ $dataset ($size)"
            fi
        done
        
        if [[ -f "$OUTPUT_DIR/dataset_test_structures.xyz" ]]; then
            local size=$(du -h "$OUTPUT_DIR/dataset_test_structures.xyz" | cut -f1)
            log_info "  ✓ dataset_test_structures.xyz ($size)"
        fi
    fi
    
    echo ""
    log_success "Dataset preparation completed successfully!"
    echo ""
    log_info "Next Steps:"
    log_info "  1. Review the generated datasets in: $OUTPUT_DIR/"
    log_info "  2. Copy JSON files to your training directory:"
    log_info "     cp $OUTPUT_DIR/dataset_*.json /path/to/training/datasets/"
    log_info "  3. Configure atomic_tensor.yaml for training"
    log_info "  4. Start training: python3 train_atomic_tensor.py"
    echo ""
    log_info "For predictions on new structures:"
    log_info "  - Use dataset_test_structures.xyz as a template"
    log_info "  - Run: python3 predict_atomic_tensor.py your_structures.xyz output.csv"
}

################################################################################
# Main Execution
################################################################################

main() {
    log_section "MatTen Dataset Preparation Pipeline"
    log_info "Starting automated dataset preparation workflow"
    log_info "Author: Ouail Zakary (Ouail.Zakary@oulu.fi)"
    echo ""
    
    # Check for config file argument
    if [[ $# -gt 0 ]]; then
        CONFIG_FILE="$1"
    fi
    
    # Record start time
    START_TIME=$(date +%s)
    
    # Load configuration
    load_config "$CONFIG_FILE"
    
    # Check dependencies
    check_dependencies
    
    # Update Python scripts with configuration
    update_python_scripts
    update_json_converter_splits
    
    # Stage 1: Process DFT calculation data
    log_section "STAGE 1: DFT Data Processing"
    process_dft_data || {
        log_error "Failed to process DFT data"
        exit 1
    }
    
    # Stage 2: Dataset integration
    log_section "STAGE 2: Dataset Integration"
    integrate_dataset || {
        log_error "Failed to integrate dataset"
        exit 1
    }
    
    # Stage 3: JSON conversion
    log_section "STAGE 3: JSON Conversion for MatTen"
    convert_to_json || {
        log_error "Failed to convert to JSON format"
        exit 1
    }
    
    # Calculate execution time
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    MINUTES=$((DURATION / 60))
    SECONDS=$((DURATION % 60))
    
    # Generate summary
    generate_summary
    
    log_info "Total execution time: ${MINUTES}m ${SECONDS}s"
    
    echo ""
    log_success "All stages completed successfully!"
}

# Run main function
main "$@"
