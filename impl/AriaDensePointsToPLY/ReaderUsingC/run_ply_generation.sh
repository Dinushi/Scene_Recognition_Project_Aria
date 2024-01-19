#!/bin/bash

# Navigate into the build directory
cd build

start_scene_no=0
end_scene_no=0

aria_input_data_path="/home/tran5174/temp_env/Scene_Recognition_Project_Aria/SynEnvDataset";
output_ply_write_path="/home/tran5174/temp_env/Scene_Recognition_Project_Aria/PLYDataset";

echo "--------Running ply generator fr scenes from ID=${start_scene_no} to ID=${end_scene_no}-----------"
./ply_generator $aria_input_data_path $output_ply_write_path $start_scene_no $end_scene_no