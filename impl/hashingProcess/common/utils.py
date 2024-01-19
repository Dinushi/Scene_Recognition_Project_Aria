import numpy as np
import open3d as o3d
from pathlib import Path

def findCenterofPointCloud(pcd):
    center = pcd.get_center()
    return center

def getMaxBound(pcd):
    max_bound = pcd.get_max_bound()
    print(f'Max boundh: {max_bound}')
    return max_bound

def findTheDistancesfromcenter(points, center):

    # Calculate the Euclidean distances from the center to all points
    distances = np.linalg.norm(points - center, axis=1)
    
    furthest_distance = np.max(distances)
    # Find the index of the point farthest from the center
    furthest_point_index = np.argmax(distances)
    # Get the coordinates of the farthest point
    furthest_point = points[furthest_point_index]

    print(f"The farthst distance is {furthest_distance}")
    #print(f"The index of the farthest point is {furthest_point_index}")
    #print(f"The coordinates of the farthest point are {furthest_point}")

    return distances, furthest_distance, furthest_point

def findPointWithSpaceofGivenRadius(distances, radius_1, radius_2, points):

    # Find the indices of points within the specified radius
    indices_within_radius_1 = np.where(distances <= radius_1)

    # Find the indices of points within the second radius
    indices_within_radius_2 = np.where(distances <= radius_2)

    indices_within_range = np.setdiff1d(indices_within_radius_2, indices_within_radius_1)

    # Extract the points within the specified radius
    points_within_range = points[indices_within_range]
    return indices_within_range, points_within_range