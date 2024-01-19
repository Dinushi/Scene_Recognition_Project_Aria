import subprocess
import os
import pandas as pd
from config_parser import *

class CamaraTrajectoryBasedPLYReader():

    def __init__(self, aria_trajectory_csv_name, column_names_to_read):
        self.aria_trajectory_csv_name = aria_trajectory_csv_name
        self.column_names_to_read = column_names_to_read

    def read_data_from_csv(self, folder_path):
        usecols = self.column_names_to_read

        csv_file_path = folder_path + "/" + self.aria_trajectory_csv_name
        df = pd.read_csv(csv_file_path, usecols=usecols)
        return df[usecols[0]].tolist(), df[usecols[1]].tolist(), df[usecols[2]].tolist(), df[usecols[3]].tolist(), df[usecols[4]].tolist(), df[usecols[5]].tolist(), df[usecols[6]].tolist(), df[usecols[7]].tolist()

if __name__ == "__main__":

    config_file_path = generateAndAccessArgsForConfigFile()
    configParser = ConfigParser(config_file_path)

    aria_original_input_data_path = configParser.getConfigParam("aria_input_data_path") 
    aria_ply_data_path = configParser.getConfigParam("aria_ply_data_path") 
    scene_number_range = list(configParser.getConfigParam("scene_number_range_to_process"))

    VFOV = configParser.getConfigParam("v_fov_angle") 
    HFOV = configParser.getConfigParam("h_fov_angle") 


    aria_tragectory_csv_name = "trajectory.csv"
    column_names_to_read = ["tracking_timestamp_us", "tx_world_device", "ty_world_device", "tz_world_device", "qx_world_device", "qy_world_device", "qz_world_device", "qw_world_device"]
    
    camaraTrajectoryBasedPLYReader = CamaraTrajectoryBasedPLYReader(aria_tragectory_csv_name, column_names_to_read)

    # Iterate over the scene numbers
    for scene_number in range(scene_number_range[0], scene_number_range[1] + 1):
        original_scene_directory_path = os.path.join(aria_original_input_data_path, str(scene_number))
        ply_scene_directory_path = os.path.join(aria_ply_data_path, str(scene_number))
        
        timestamp_list, tx_list, ty_list, tz_list, qx_list, qy_list, qz_list, qw_list = camaraTrajectoryBasedPLYReader.read_data_from_csv(original_scene_directory_path)

        for trajectory_row_id in range(len(timestamp_list)):
            timestamp = timestamp_list[trajectory_row_id]
            tx = tx_list[trajectory_row_id]
            ty = ty_list[trajectory_row_id]
            tz = tz_list[trajectory_row_id]
            qx = qx_list[trajectory_row_id]
            qy = qy_list[trajectory_row_id]
            qz = qz_list[trajectory_row_id]
            qw = qw_list[trajectory_row_id]

            print(f"--------Running filter with vFOV={VFOV} and hFOV={HFOV} Scene={scene_number}  TrajectoryID={trajectory_row_id} TIMESTAMP={timestamp}-----------")

            # Navigate into the build folder of C++ filter
            os.chdir("filterByCameraPose/build")
            #command = ["./filter", "../../pcds", str(VFOV), str(HFOV)]
            command = ["./filter", ply_scene_directory_path, str(VFOV), str(HFOV), str(scene_number), str(timestamp), str(tx), str(ty), str(tz), str(qx), str(qy), str(qz), str(qw)]
            subprocess.run(command)
            os.chdir("../../")
            #break
