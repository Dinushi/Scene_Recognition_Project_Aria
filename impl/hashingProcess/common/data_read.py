import numpy as np
import open3d as o3d
from pathlib import Path
import os
import re

import glob

def readMeshVF(file_path_string):

    input_file_path_data = Path(file_path_string)
    textured_mesh = o3d.io.read_triangle_mesh(file_path_string)
    v = np.asarray(textured_mesh.vertices)     # Get the vertices
    f = np.asarray(textured_mesh.triangles)    # Get the faces

    print("Read " + input_file_path_data.name + " file with " + str(len(v)) + " vertices.")
    return v,f,textured_mesh

def readMesh(file_path_string):

    input_file_path_data = Path(file_path_string)
    textured_mesh = o3d.io.read_triangle_mesh(file_path_string)
    return textured_mesh

def readPointCloud(file_path_string):

    input_file_path_data = Path(file_path_string)
    point_cloud = o3d.io.read_point_cloud(file_path_string)
    return point_cloud

def readPointCloudWithDataExtracted(file_path_string):
    pcd = readPointCloud(file_path_string)
    return pcd, np.asarray(pcd.points), np.asarray(pcd.normals), np.asarray(pcd.colors)

def readMeshWithDataExtracted(file_path_string):
    mesh = readMesh(file_path_string)
    return mesh, np.asarray(mesh.vertices), np.asarray(mesh.normals), np.asarray(mesh.triangles), np.asarray(pcd.colors)

def readInputFilePathData(file_path):
    input_file_path_data = Path(file_path)
    file_stem_name = input_file_path_data.stem
    return file_stem_name

def readInputFolderPathData(folder_path):
    # extract the last part of the path
    last_folder_name = os.path.basename(folder_path) 
    return last_folder_name

# get .ply file path list in a given folder, the order will be arbitary
def readPLYFileListInGivenFolder(folder_path):
    ply_file_list = glob.glob(folder_path + "/*.ply") # => stanford data 
    #file_list = glob.glob(folder_path + "/***/*.ply", recursive = True) # => graps data
    return ply_file_list

# get .ply file path list in a given folder
def readSortedOrderPLYFileListInGivenFolder(folder_path):
    ply_file_list = glob.glob(folder_path + "/*.ply") # => stanford data 
    
    # this does not work when file name has ints at end as well eg: 36-squeezeBottle-1_10000
    # def get_number_from_filename(filename):
        #return int(''.join(filter(str.isdigit, filename)))

    def get_number_from_filename(filename):
        # Extract the first sequence of digits as a string
        base_name = os.path.basename(filename)
        match = re.search(r'\d+', base_name)
        if match:
            return int(match.group())
        else:
            return float('inf') 

    #def custom_sort(filename):
        #base_name = os.path.basename(filename)
        #parts = base_name.split('-')
        #key = (int(parts[0]), parts[-1], int(parts[-1].split('_')[0]))
        #return key
    
    # Sort the PLY files based on the numbers in their names
    sorted_ply_file_list = sorted(ply_file_list, key=get_number_from_filename)
    #print("sorted_ply_file_list" + str(sorted_ply_file_list))
    return sorted_ply_file_list
    
