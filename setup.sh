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
    )

    pip install "${required_libraries[@]}"
}

# ---------------------------------------
# Main Logic
# ---------------------------------------
# install_dependencies

# This is more of a legacy script, and is not currently used.
# This will be removed soon.
