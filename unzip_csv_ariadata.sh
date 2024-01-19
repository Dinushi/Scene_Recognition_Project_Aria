#!/bin/bash

# Directory containing the subfolders
base_dir="projectaria_tools_ase_data"

# Loop through each subfolder
for folder in "$base_dir"/*; do
    if [ -d "$folder" ]; then
        echo "Processing folder: $folder"

        # Define the paths to the .csv.gz files
        semidense_observations="$folder/semidense_observations.csv.gz"
        semidense_points="$folder/semidense_points.csv.gz"

        # Check if the files exist and unzip them
        if [ -f "$semidense_observations" ]; then
            gzip -d "$semidense_observations"
        else
            echo "File not found: $semidense_observations"
        fi

        if [ -f "$semidense_points" ]; then
            gzip -d "$semidense_points"
        else
            echo "File not found: $semidense_points"
        fi
    fi
done
