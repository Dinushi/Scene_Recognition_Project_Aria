import numpy as np
import open3d as o3d
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def visualize(pcd):
    pcd.paint_uniform_color([0.784, 0.784, 0.784])

    #indices = np.where(np.all(np.array(pcd.points) == max_point, axis=1))
    #print("Indices" + str(indices))

    # Create a new point cloud with a single point colored red
    #red_point = o3d.geometry.PointCloud()
    #red_point.points = o3d.utility.Vector3dVector([pcd.points[indices[0]]])  # Extract the known point
    #red_point.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0]]))  # Red color

    # Combine the original point cloud and the red point cloud
    #pcd = pcd + red_point  

    o3d.visualization.draw_geometries([pcd])

def visualizeSelectedPoints(selected_points):
    # Optionally, visualize the selected points
    selected_pcd = o3d.geometry.PointCloud()
    selected_pcd.points = o3d.utility.Vector3dVector(selected_points)
    selected_pcd.paint_uniform_color([0.784, 0.784, 0.784])
    o3d.visualization.draw_geometries([selected_pcd])

def visualizeHeatMapsOnPointCloud(pcd, curvature_values_normalized):

    # Choose a colormap
    colormap = plt.get_cmap("viridis")

    # Map curvature values to colors
    colors = (colormap(curvature_values_normalized) * 255).astype(np.uint8)[:, :3]

    colored_pcd = o3d.geometry.PointCloud()
    colored_pcd.points = pcd.points

    colors_double = colors / 255.0 # to normalize the color values to a range between 0 and 1
    colored_pcd.colors = o3d.utility.Vector3dVector(colors_double)
    #print("Point Cloud colurs  {}".format(np.asarray(colored_pcd.colors)))
  
    o3d.visualization.draw_geometries([colored_pcd])
    
    # Create a color bar to display the mapping
    norm = Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])  # create a color bar without actual data
    plt.colorbar(sm, label="Curvature (0-1)")
    plt.show()

# points : points array belong to the spherical point cloud portion
# labels : the results from clustering which is a array which says to which cluster-number each point belongs
# n_clusters : the number of clusters
def visualizeL2Clusters(points, labels, n_clusters):

    clustered_pcd = o3d.geometry.PointCloud() # Create an Open3D PointCloud for visualization
    colors = np.zeros((len(points), 3))

    # todo : here colors are fixed to 5 assuming 5 clusters
    # Red, Green, Blue, Yellow, Cyan
    known_colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1)]

    # Assign a color to each point based on its cluster label
    for i in range(n_clusters):
        cluster_indices = np.where(labels == i)
        colors[cluster_indices] = known_colors[i]  # select the color assigned for each cluster

    clustered_pcd.points = o3d.utility.Vector3dVector(np.asarray(points))
    clustered_pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the clustered point cloud with a color legend
    o3d.visualization.draw_geometries([clustered_pcd], window_name="Clustered spherical point cloud portion")