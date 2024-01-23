import subprocess
import os
import open3d as o3d
import numpy as np

from config_parser import *

if __name__ == "__main__":
    
    config_file_path = generateAndAccessArgsForConfigFile()
    configParser = ConfigParser(config_file_path)

    aria_ply_data_path = configParser.getConfigParam("aria_ply_data_path") 
    scene_number_range = list(configParser.getConfigParam("scene_number_range_to_process"))
    max_height_z = float(configParser.getConfigParam("max_height_z_axis"))

    # Iterate over the scene numbers
    for scene_number in range(scene_number_range[0], scene_number_range[1] + 1):
 
        ply_scene_directory_path = os.path.join(aria_ply_data_path, str(scene_number))

        original_scene_file_name = str(scene_number) + "_" + "original.ply"
        cropped_scene_file_name_to_write = str(scene_number) + "_" + "originalZCrop.ply"

        pcd_original = o3d.io.read_point_cloud(ply_scene_directory_path + "/" + original_scene_file_name)
        points_original = np.asarray(pcd_original.points)

        filtered_points = points_original[points_original[:, 2] < max_height_z]

        # Create a new point cloud with the filtered points
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

        # Save the filtered point cloud
        o3d.io.write_point_cloud(ply_scene_directory_path + "/" + cropped_scene_file_name_to_write, filtered_pcd)