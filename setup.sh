#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Get the script's directory
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------
# Function: Install Required Libraries
# ---------------------------------------
install_dependencies() {
    echo "Installing required Python libraries..."

    required_libraries=(
        jetnet
        "jetnet[emdloss]"
        torch_geometric
        particle
        file_read_backwards
        # torch==2.4+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
    )

    pip install "${required_libraries[@]}"
}

# ---------------------------------------
# Function: Build and Install pTokenizer
# ---------------------------------------
build_pTokenizer() {
    echo "Building and installing pTokenizer..."

    # Ensure build directory exists
    mkdir -p "$script_dir/pTokenizer/build"

    # Change to pTokenizer directory
    cd "$script_dir/pTokenizer"

    # Build python module
    python setup.py bdist_wheel

    # Install the built wheel (handles different versions dynamically)
    pip install dist/pTokenizer-*.whl --force-reinstall
}

# ---------------------------------------
# Main Logic
# ---------------------------------------
if [[ "$1" == "pTokenizer" ]]; then
    echo "Argument 'pTokenizer' provided — skipping dependency installation."
    build_pTokenizer
else
    install_dependencies
    build_pTokenizer
fi