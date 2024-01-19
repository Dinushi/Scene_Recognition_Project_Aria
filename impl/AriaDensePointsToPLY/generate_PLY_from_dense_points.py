import numpy as np
import pandas as pd
import sys
import os 
import glob

import open3d as o3d
from config_parser import *

class PointsToPLYConverter():

    def __init__(self, aria_dense_points_csv_name, column_names_to_read):
        self.aria_dense_points_csv_name = aria_dense_points_csv_name
        self.column_names_to_read = column_names_to_read
    
    def read_data_from_csv(self, folder_path):
        usecols = self.column_names_to_read

        csv_file_path = folder_path + "/" + self.aria_dense_points_csv_name
        df = pd.read_csv(csv_file_path, usecols=usecols)
        return df[usecols[0]].tolist(), df[usecols[1]].tolist(), df[usecols[2]].tolist()
    
    def create_point_cloud(self, x_cord_list, y_cord_list, z_cord_list):
        points = [[x, y, z] for x, y, z in zip(x_cord_list, y_cord_list, z_cord_list)]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd

if __name__ == "__main__":

        config_file_path = generateAndAccessArgsForConfigFile()
        configParser = ConfigParser(config_file_path)

        aria_input_data_path = configParser.getConfigParam("aria_input_data_path") 
        scene_num_range = list(configParser.getConfigParam("scene_number_range_to_process"))

        output_ply_write_path = configParser.getConfigParam("output_ply_write_path") 

        aria_dense_points_csv_name = "semidense_points.csv"
        column_names_to_read = ["px_world", "py_world", "pz_world"]
    
        pointsToPLYConverter = PointsToPLYConverter(aria_dense_points_csv_name, column_names_to_read)

        #scene_folder_paths = glob.glob(os.path.join(aria_input_data_path, '*/'))
        #for scene_folder_path in scene_folder_paths:
        for scene_number in range(scene_num_range[0], scene_num_range[1]+1):
            scene_number_str = str(scene_number)                                                        #scene_number = os.path.basename(os.path.dirname(scene_folder_path))
            x_cord_list, y_cord_list, z_cord_list = pointsToPLYConverter.read_data_from_csv(os.path.join(aria_input_data_path, scene_number_str))
            pcd = pointsToPLYConverter.create_point_cloud(x_cord_list, y_cord_list, z_cord_list)

            #o3d.visualization.draw_geometries([pcd])
            ply_write_folder = output_ply_write_path + "/" + scene_number_str
     
            os.makedirs(ply_write_folder , exist_ok=True)
            ply_path = ply_write_folder + "/" + scene_number_str + '_original.ply'
            o3d.io.write_point_cloud(ply_path, pcd)

        
