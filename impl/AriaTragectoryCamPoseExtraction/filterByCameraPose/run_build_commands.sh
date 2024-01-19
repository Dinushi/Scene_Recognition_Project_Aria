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

#FOV=120

#echo "--------Running filter with vFOV=${FOV} and hFOV=${FOV}-----------"
#./filter ../../pcds $FOV $FOV 0 0 1.51215361509941 3.88807193304466 1.43420936331043 0.17498913549234 0.642008353975212 -0.291806301477169 0.6870612478548739