import numpy as np

def calculate_complexity(points, curvatures, curvature_threshold=0.2, distance_threshold=0.1):

    distances = np.linalg.norm(points - np.mean(points, axis=0), axis=1)
    
    # Determine complexity based on curvature and distances
    num_high_curvature_points = np.sum(curvatures > curvature_threshold)
    curvature_std = np.std(curvatures)
    print("curvature_std :" + str(curvature_std))
    max_distance = np.max(distances)
    
    if num_high_curvature_points > len(points)/4 and curvature_std > 0.5 and max_distance > distance_threshold:
        return "Complex"
    else:
        return "Simple"
