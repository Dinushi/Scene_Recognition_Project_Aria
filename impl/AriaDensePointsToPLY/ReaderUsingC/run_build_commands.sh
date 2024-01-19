#!/bin/bash

# Navigate to the script's directory (optional, remove if not needed)
cd "$(dirname "$0")"

# Remove the build directory if it exists, then create a new one
rm -rf build/
mkdir build

# Navigate into the build directory
cd build

# Run cmake and make
cmake ..
make