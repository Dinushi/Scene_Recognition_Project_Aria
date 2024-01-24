import numpy as np
import open3d as o3d
from pathlib import Path

#import point_cloud_utils as pcu

import igl

# compute the FPFH features for the whole point cloud
def computeFPFHFeatures_2(pcd, normal_radius_max=0.1, normal_max_nn = 30, FPFH_radius_max=0.5, FPFH_max_nn=100):
    
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=float(normal_radius_max), normal_max_nn=int(normal_max_nn)))
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=float(FPFH_radius_max), FPFH_max_nn=int(FPFH_max_nn)))

    # Convert the FPFH feature vector to a numpy array
    fpfh_array = np.asarray(pcd_fpfh.data)
    print("computed FPFH feature shape: =>  " + str(fpfh_array.shape))
    #print(fpfh_array)
    #print(fpfh_array[:, 0])
    return fpfh_array

def computeFPFHFeatures(pcd, normal_radius_max=0.1, normal_max_nn = 30, FPFH_radius_max=0.5, FPFH_max_nn=100):
    radius_normal = 0.1
    radius_fpfh = 0.5
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_fpfh, max_nn=100))

    # Convert the FPFH feature vector to a numpy array
    fpfh_array = np.asarray(pcd_fpfh.data)
    print("computed FPFH feature shape: =>  " + str(fpfh_array.shape))
    #print(fpfh_array)
    #print(fpfh_array[:, 0])
    return fpfh_array

# compute using 'igl' libarary https://libigl.github.io/libigl-python-bindings/tut-chapter1/
def compute_principle_curvatures_igl_lib(v, f):
    d1, d2, k1, k2 = igl.principal_curvature(v, f)
    return k1, k2

def compute_principle_curvatures_PCUtils_lib(v, f):
    # Load vertices, faces, per-vertex normals, and per-vertex colors
    # v, f, n= pcu.load_mesh_vfn(pcd_file_path)
    k1 = 0
    k2 = 0
    #k1, k2, d1, d2  = pcu.mesh_principal_curvatures(v, f)
    #k1, k2, d1, d2 = pcu.mesh_principal_curvatures(v, f, r=0.1)
    return k1, k2

# this is used when faces are readily available in input data
# input point cloud is now read as a mesh
def computeNormalizedMeanCurvatures(mesh):
    #radius_normal = 0.1
    #pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Create a mesh using Poisson surface reconstruction
    #mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    k1, k2  = compute_principle_curvatures_igl_lib(vertices, faces)
    #k1, k2  = compute_principle_curvatures_PCUtils_lib(vertices, faces)
    mean_curvatures = (k1 + k2) / 2.0
    abs_mean_curvatures = np.abs(mean_curvatures)
    normalized_mean_curvatures = (abs_mean_curvatures - np.min(abs_mean_curvatures)) / (np.max(abs_mean_curvatures) - np.min(abs_mean_curvatures))
    return normalized_mean_curvatures

# this function is to estimate the face traingles from PCD and compute the curvature
# this is used when faces are not readily available in input data
def computeNormalizedMeanCurvaturesAfterInCodeMeshTriangulation(pcd):

    tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
    alpha = 0.1
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha, tetra_mesh, pt_map)
    mesh.compute_vertex_normals()
    #o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    k1, k2  = compute_principle_curvatures_igl_lib(vertices, faces)
    #k1, k2  = compute_principle_curvatures_PCUtils_lib(vertices, faces)
    mean_curvatures = (k1 + k2) / 2.0
    abs_mean_curvatures = np.abs(mean_curvatures)
    normalized_mean_curvatures = (abs_mean_curvatures - np.min(abs_mean_curvatures)) / (np.max(abs_mean_curvatures) - np.min(abs_mean_curvatures))
    return normalized_mean_curvatures

