#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Get the script's directory
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------
# Function: Install Required Python Libraries
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
# grab_ROOT() {
#     echo "Setting up ROOT..."

#     mkdir -p "$script_dir/third_party/ROOT"
#     cd "$script_dir/third_party/ROOT"

#     # wget https://root.cern/download/root_v6.24.08.Linux-centos7-x86_64-gcc4.8.tar.gz
#     # tar -xzvf root_v6.24.08.Linux-centos7-x86_64-gcc4.8.tar.gz
#     # rm root_v6.24.08.Linux-centos7-x86_64-gcc4.8.tar.gz
#     source root/bin/thisroot.sh

#     # Env variable used for CMake to locate custom ROOT installation
#     export particleGPT_ROOT="$script_dir/third_party/ROOT/root"
# }

# ---------------------------------------
# Function: Build and Install pTokenizer
# ---------------------------------------
build_pTokenizer() {
    echo "Building and installing pTokenizer..."

    mkdir -p "$script_dir/pTokenizer/build"
    cd "$script_dir/pTokenizer"

    python setup.py bdist_wheel
    pip install dist/ptokenizer-*.whl --force-reinstall
}

# ---------------------------------------
# Main Logic
# ---------------------------------------
if [[ "$1" == "pTokenizer" ]]; then
    echo "Argument 'pTokenizer' provided — skipping dependency installations."
    build_pTokenizer
else
    install_dependencies
    # grab_ROOT
    build_pTokenizer
fi