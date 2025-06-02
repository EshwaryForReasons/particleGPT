#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Get the script's directory
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

build_ROOT_from_source() {
    echo "Building ROOT from source (GLIBC compatible)..."

    mkdir -p "$script_dir/third_party/ROOT"
    cd "$script_dir/third_party/ROOT"

    # git clone --branch latest-stable --depth=1 https://github.com/root-project/root.git root_src

    mkdir -p builddir installdir
    cd builddir

    cmake -DCMAKE_INSTALL_PREFIX="installdir" ../root_src \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_STANDARD=23 \
        -Dminimal=ON \
        -Dbuiltin_vdt=ON
    cmake --build . --target install -- -j$(nproc)

    # make -j$(nproc)
    # make install

    echo "ROOT built and installed."
}

build_ROOT_from_source