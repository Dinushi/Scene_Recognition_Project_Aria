import open3d as o3d
import numpy as np
from pathlib import Path

def readPointCloud(file_path_string):
    
    input_file_path_data = Path(file_path_string)
    point_cloud = o3d.io.read_point_cloud(file_path_string)
    return point_cloud

pcd = readPointCloud("/Users/tran5174/Documents/PhD/Scene_Recognition_Project_Aria/PLYDataset/0/0_0.ply")
o3d.visualization.draw_geometries([pcd])
points = pcd.points
print("No of Points : " + str(np.asarray(points)))

