import subprocess
import os
import pandas as pd
import open3d as o3d
from config_parser import *

class CamaraTrajectoryBasedPLYReader():

    def __init__(self, aria_trajectory_csv_name, column_names_to_read):
        self.aria_trajectory_csv_name = aria_trajectory_csv_name
        self.column_names_to_read = column_names_to_read

    def read_data_from_csv(self, folder_path):
        usecols = self.column_names_to_read

        csv_file_path = folder_path + "/" + self.aria_trajectory_csv_name
        df = pd.read_csv(csv_file_path, usecols=usecols)
        return df[usecols[0]].tolist()

if __name__ == "__main__":

    config_file_path = generateAndAccessArgsForConfigFile()
    configParser = ConfigParser(config_file_path)

    aria_original_input_data_path = configParser.getConfigParam("aria_input_data_path") 
    aria_ply_data_path = configParser.getConfigParam("aria_ply_data_path") 
    scene_number_range = list(configParser.getConfigParam("scene_number_range_to_process"))

    aria_aggregated_ply_write_path_seconds = configParser.getConfigParam("aria_aggregated_ply_write_path_seconds") 

    aria_tragectory_csv_name = "trajectory.csv"
    column_names_to_read = ["tracking_timestamp_us"]
    
    camaraTrajectoryBasedPLYReader = CamaraTrajectoryBasedPLYReader(aria_tragectory_csv_name, column_names_to_read)

    # Iterate over the scene numbers
    for scene_number in range(scene_number_range[0], scene_number_range[1] + 1):
        original_scene_directory_path = os.path.join(aria_original_input_data_path, str(scene_number))
        ply_scene_directory_path = os.path.join(aria_ply_data_path, str(scene_number))

        # folder path to write aggregrated ply files
        agg_ply_write_dir_path = os.path.join(aria_aggregated_ply_write_path_seconds, str(scene_number))
        os.makedirs(agg_ply_write_dir_path, exist_ok=True)
        
        timestamp_list = camaraTrajectoryBasedPLYReader.read_data_from_csv(original_scene_directory_path) # time is micro_seconds in csv

        time_counter = 0 # assume unit is seconds
        pcd_accumilated = o3d.geometry.PointCloud()

        for trajectory_row_id in range(len(timestamp_list)):
            timestamp = int(timestamp_list[trajectory_row_id])
            print(f"--------Read View Port PLY Scene {scene_number} at TIMESTAMP={timestamp}-----------")

            pcd_file_name_at_timetamp = str(scene_number) + "_" + str(timestamp) + ".ply"
            pcd = o3d.io.read_point_cloud(ply_scene_directory_path + "/" + pcd_file_name_at_timetamp)
            pcd_accumilated += pcd

            # write the pcd only at each 1 second time interval otherwise loop kept accumilating points
            if ((time_counter == 0 and timestamp == 0) or (timestamp % 1000000 == 0)): # check if there is no decimal remainder when micros converted to s
                print(f"\t\t Write Scene {scene_number} at TIMESTAMP(ms)={timestamp} Time(S)= {time_counter}-----------")
                pcd_accumilated = pcd_accumilated.remove_duplicated_points()
                write_agg_pcd_file_name = str(scene_number) + "_" + str(time_counter) + ".ply"
                o3d.io.write_point_cloud(agg_ply_write_dir_path + "/" + write_agg_pcd_file_name, pcd_accumilated)
                time_counter += 1

            if (timestamp == 20000000): break
                


            

           
