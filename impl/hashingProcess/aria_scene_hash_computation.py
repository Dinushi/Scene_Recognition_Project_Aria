from common.data_read import *
from common.data_write import *
from common.pcd_visualizer import *
from common.utils import *
from common.feature_computation import *
from common.distance_computation import *
from common.csv_analysis import * 
from common.graph_generation import * 
from common.complexity_analysis import *

from aria_scene_robustness_analyzer import *
from aria_scene_uniqueness_analyzer import *
from config_parser import *

import numpy as np
import time
import os

import itertools

from sklearn.cluster import KMeans

dict_all_points = {}

class HashCalculator:

    def __init__(self, parameterObject):
        self.parameterObject = parameterObject

    def convertFPFHArrayOfRegionToBinary(self, columnwise_fpfh_regions_array, avg_columnwise_fpfh_entire_pcd):
        comparison_result = (columnwise_fpfh_regions_array > avg_columnwise_fpfh_entire_pcd) # an array of arrays of trues and falses [[False False False ... False False False], [False False False ... False False False],..., [False False False ... False  True False]]
        #print("comparison")
        #print(comparison_result)
        #print(comparison_result.shape)
        binary_array = comparison_result.astype(int) # this converted to 0, 1 format  [[0 0 0 ... 0 0 0], [0 0 0 ... 0 0 0], ...,[0 0 0 ... 0 0 0]]
        #print("binary_array " + str(binary_array)) 

        binary_strings = []
        for row in binary_array:
            binary_string = ''.join(map(str, row))
            binary_strings.append(binary_string)

        final_binary_string = ''.join(binary_strings)
        #print(final_binary_string)
        return final_binary_string, binary_array

    def convertFPFHArraysOfRegionToBinaryUsingWindowBasedMethod(self, columnwise_fpfh_regions_array, window_span = 1):

        binary_arrays = []
        final_binary_string = ''

        for i in range((columnwise_fpfh_regions_array.shape)[1]): # this count should be same as the the no_of_componemnets

            current_column = columnwise_fpfh_regions_array[:, i]

            # Calculate the start and end indices for the window based on window_span
            window_start = max(0, i - window_span)
            window_end = min((columnwise_fpfh_regions_array.shape)[1], i + window_span + 1)

            # Extract elements for the window columns, considering boundaries
            window_columns = columnwise_fpfh_regions_array[:, window_start:window_end]
            # Calculate the mean of the elements within the window
            window_average = np.mean(window_columns, axis=1)
            #print("window_average :" + str(window_average.shape))

            comparison_result = (current_column > window_average)

            #print("comparison_result : " + str(comparison_result))
            binary_array = comparison_result.astype(int) 
            #print("comparison_result_binary : " + str(binary_array))
            binary_arrays.append(binary_array)
            
            binary_string = ''.join(map(str, binary_array))
            #print("binary string : " + binary_string)
            final_binary_string += binary_string

        #print(final_binary_string)
        return final_binary_string, binary_arrays

    def compute_l2_clustering(self, points, normalized_curvature, n_clusters):
        data = np.column_stack((points, normalized_curvature))

        kmeans = KMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(data)

        # Create an Open3D PointCloud for visualization
        if bool(configParser.getConfigParam("display_pcd_clusters")):
            visualizeL2Clusters(points, labels, n_clusters)

        # Calculate average curvature values for each cluster
        cluster_avg_curvatures = {}
        for i in range(n_clusters):
            cluster_indices = np.where(labels == i)
            cluster_curvatures = normalized_curvature[cluster_indices]
            avg_curvature = np.mean(cluster_curvatures)

            cluster_avg_curvatures[i] = avg_curvature
            #print(f"Cluster {i} - Average Curvature: {avg_curvature:.4f}")

        sorted_cluster_avg_curvatures = dict(sorted(cluster_avg_curvatures.items(), key=lambda item: item[1]))
        #for i, avg_curvature in sorted_cluster_avg_curvatures.items():
            #print(f"Sorted - Cluster {i} - Average Curvature: {avg_curvature:.4f}")

        sorted_cluster_ids = [cluster_id for cluster_id in sorted_cluster_avg_curvatures]
        #print("Sorted Cluster IDs {}".format(sorted_cluster_ids))

        ## todo : write a graping or tabulating mechanism to see how these curcature avarges are varing when you do differnt iterations for the same model
        return labels, sorted_cluster_ids

    def select_simplified_FPFH_for_spherical_cluster(self, fpfh_array, l2_cluster_labels_of_points, sorted_cluster_ids):
        #print("fpfh_array : shape: {}".format(fpfh_array.shape))
        #print("fpfh_array : " + str(fpfh_array))
        #print("l2_cluster_labels_of_points : shape : {}".format(l2_cluster_labels_of_points.shape))
        #print("l2_cluster_labels_of_points :" + str(l2_cluster_labels_of_points))

        combined_avg_fpfh_features_list = []
        for cluster_id in sorted_cluster_ids:
            result = np.where(l2_cluster_labels_of_points == cluster_id)
            cluster_indices = result[0]
            #print("Cluster Indices : {}".format(cluster_indices))
            print("No of points belong to cluster {} : {}".format(cluster_id, len(cluster_indices)))
            cluster_fpfh_features = fpfh_array[:, cluster_indices]
            print("cluster_fpfh_features shape of cluster Id {} : {}".format(cluster_id, cluster_fpfh_features.shape))

            # get the average of the FPFH features of the points belong to this cluster
            avg_cluster_fpfh_feature = np.mean(cluster_fpfh_features, axis=1, keepdims=True)
            print("avg cluster_fpfh_features shape of cluster Id {} : {}".format(cluster_id, avg_cluster_fpfh_feature.shape))
            # append the average fpfh feature vector value for the cluster
            combined_avg_fpfh_features_list.append(avg_cluster_fpfh_feature)
    
        # append averaged fpfh vector of each cluster lineraly in the order of the provides sorted cluster order
        combined_avg_fpfh_features = np.concatenate(combined_avg_fpfh_features_list, axis=0)  # Convert the list of average FPFH features to a NumPy array. Stack along axis 1
        print("Final avg_fpfh_features shape: {}".format(combined_avg_fpfh_features.shape)) # The resulting shape of avg_fpfh_features will be (33, 5)
        return combined_avg_fpfh_features

    def convertPCDToMesh(self, pcd):

        #tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
        #alpha = 0.1
        #mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha, tetra_mesh, pt_map)
        #mesh.compute_vertex_normals()
        #o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
        #return mesh
        #("Points count in PCD" + str(np.asarray(pcd.points)))

        #print('run Poisson surface reconstruction')

        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as cm:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=9)
        #o3d.visualization.draw_geometries([mesh],zoom=0.664,front=[-0.4761, -0.4698, -0.7434],lookat=[1.8900, 3.2596, 0.9284], up=[0.2304, -0.8825, 0.4101])
        #print("Points count in mesh" + str(np.asarray(mesh.vertices)))
        return mesh

    # This function computes the binary hash for a pcd file at the specified path
    def execute_computation_on_single_pcd(self, file_path_string):

        pcd_file_name = readInputFilePathData(file_path_string)
        pcd, pcd_points, pcd_normals, pcd_colors = readPointCloudWithDataExtracted(file_path_string)
        #visualize(pcd)
        print(f"Points count - {len(np.asarray(pcd.points))}")
        point_count = len(pcd_points)
        model_complexity = None

        if(self.parameterObject.l2_cluster_enabled):
            if (self.parameterObject.in_code_triangulation_required):
                mesh = self.convertPCDToMesh(pcd)
            else:
                mesh = readMesh(file_path_string)
                mean_curv_array = computeNormalizedMeanCurvatures(mesh)

            #print("Mean_curvature_array {}".format(mean_curv_array.shape))
            #print(mean_curv_array)
            model_complexity = calculate_complexity(pcd_points, mean_curv_array)
            #visualizeHeatMapsOnPointCloud(pcd, mean_curv_array)

        center = findCenterofPointCloud(pcd)
        print("Center of the point cloud - {}".format(center))
        distances, farthest_dist, farthest_point = findTheDistancesfromcenter(pcd_points, center)
        print("The distance to furtherst point -  {}".format(farthest_dist))

        fpfh_array = computeFPFHFeatures(pcd, self.parameterObject.normal_max_radius, self.parameterObject.normal_max_nn, 
                                                    self.parameterObject.FPFH_max_radius, self.parameterObject.FPFH_max_nn)
        print("FPFH array size - {}".format(fpfh_array.shape))
        #print("FPFH array - {}".format(fpfh_array))
        
        sub_radii_size = farthest_dist / self.parameterObject.component_size
        print(f"Sub-radii size - {sub_radii_size}")

        n_clusters = 5 
        fpfh_sum_columnwise_list = []

        print(f"Summarize features in each spherical cluster ===>")

        for radii_index in range(self.parameterObject.component_size): 
        
            # find the points belong to given L1 cluster
            radius_start = sub_radii_size * radii_index
            #radius_start = sub_radii_size * 0
            radius_end = sub_radii_size * (radii_index + 1)
            indices_within_range, points_within_range = findPointWithSpaceofGivenRadius(distances, radius_start, radius_end, pcd_points)

            #visualizeSelectedPoints(points_within_range)

            dict_all_points[radii_index] = points_within_range

            #print(f"Number of points within radius: {radius_start} and {radius_end}= {len(points_within_range)}")

            if bool(configParser.getConfigParam("display_pcd_clusters")): 
                if len(indices_within_range) != 0: visualizeSelectedPoints(points_within_range)

            #print(f"The sub area {str(radii_index)} indices array size => {indices_within_range.shape}")

            # select the fpfh features of te points belong to the L1 cluster
            selected_fpfh_features = fpfh_array[:, indices_within_range]

            #print(f"Sub area {str(radii_index)} fpfh array size => {selected_fpfh_features.shape}")

            if(self.parameterObject.l2_cluster_enabled):
                if (len(indices_within_range) != 0 and len(indices_within_range) >= n_clusters) :
                    # get the L2 clusters and their order for the L1 cluster
                    l2_cluster_labels_of_points, sorted_cluster_ids = self.compute_l2_clustering(pcd_points[indices_within_range], mean_curv_array[indices_within_range], n_clusters)
                    combined_avg_fpfh_features = self.select_simplified_FPFH_for_spherical_cluster(selected_fpfh_features, l2_cluster_labels_of_points, sorted_cluster_ids)
    
                    # lineraly combine for all the L1 cluster
                    fpfh_sum_columnwise_list.append(combined_avg_fpfh_features)
                else:
                    fpfh_sum_columnwise_list.append(np.zeros((selected_fpfh_features.shape[0] * n_clusters , 1))) # append a np array of size (33*n_clusters, 1) all set to 0 for sphrical portions with no points or less than 5 points

            else:
                # get the mean of the selected fpfh features for the sub area
                if (len(indices_within_range) != 0):
                    combined_avg_fpfh_features = np.mean(selected_fpfh_features, axis=1, keepdims=True) # todo : change this to median as well and test
                    #print(f"Sub area {str(radii_index)} Averaged fpfh array size => {combined_avg_fpfh_features.shape}")
                    fpfh_sum_columnwise_list.append(combined_avg_fpfh_features) # => todo: save this in a file under the radius index and analyze
                else:
                    #print(f"Sub area {str(radii_index)} Averaged fpfh array size set with zeros => {np.zeros((selected_fpfh_features.shape[0] , 1)).shape}")
                    fpfh_sum_columnwise_list.append(np.zeros((selected_fpfh_features.shape[0] , 1)))

            
        columnwise_fpfh_regions_array = np.concatenate(fpfh_sum_columnwise_list, axis=1)
        print(f"Combined fpfh array size for the entire pcd portions => {columnwise_fpfh_regions_array.shape}")
        avg_columnwise_fpfh_entire_pcd = np.mean(columnwise_fpfh_regions_array, axis=1, keepdims=True)
        print(f"Averaged fpfh array size  => {avg_columnwise_fpfh_entire_pcd.shape}\n")

        if (bool(configParser.getConfigParam("windowed_average_for_quantization"))): 
            final_binary_string, binary_array = self.convertFPFHArraysOfRegionToBinaryUsingWindowBasedMethod(columnwise_fpfh_regions_array)
        else: 
            final_binary_string, binary_array = self.convertFPFHArrayOfRegionToBinary(columnwise_fpfh_regions_array, avg_columnwise_fpfh_entire_pcd)
        return pcd_file_name, point_count, final_binary_string, binary_array, columnwise_fpfh_regions_array, model_complexity


class HashComputationProcessor:
    
    def __init__(self, parameterObject):
        self.parameterObject = parameterObject
        self.hashCalculator = HashCalculator(parameterObject)

    # Define a custom key function to sort the test ply files as they are read unsorted by glob, ordered according to scene number, tragectory timestamp
    def custom_sorting_key(self, file_path):
        parts = os.path.basename(file_path).split('_') # ply fle name broken into parts using _ (0_0.py => 0, 0)
        
        # Extract scene number and trajectory timetamp of the scene
        scene_number = parts[0]
        tragectory_timestamp = parts[1].replace('.ply', '')
        return scene_number, tragectory_timestamp

    def computeOriginalHashes(self, input_folder_path, output_folder_path, original_scene_hash_csv_name):
        original_hash_key_dict = {}
        print("\n======================Compute original hashes==============")
        createNewCSV(output_folder_path, original_scene_hash_csv_name, ["Scene_No", "Point_Count", "Computation_Time(s)", "Hash" ])

        #for ply_file_path in pcd_file_list_original:
        for scene_number in range(self.parameterObject.scene_number_range[0], self.parameterObject.scene_number_range[1] + 1):
            scene_directory_path =  input_folder_path + "/" + str(scene_number)
            original_ply_path = scene_directory_path + "/" + str(scene_number) + self.parameterObject.original_scene_file_suffix +".ply"
            print("Compute hash for file: {} ====>\n".format(original_ply_path))

            start_time = time.time()
            pcd_file_name, point_count, final_binary_string, binary_array, columnwise_fpfh_regions_array, model_complexity = self.hashCalculator.execute_computation_on_single_pcd(original_ply_path)
            end_time = time.time()
            elapsed_time = end_time - start_time

            original_hash_key_dict[str(scene_number)] = final_binary_string
            updateExistingCSV(output_folder_path, original_scene_hash_csv_name, [str(scene_number), point_count, elapsed_time, final_binary_string])

    def readStoredOriginalHashesFromCSV(self, output_folder_path, original_scene_hash_csv_name):
        original_hash_key_dict = {}
        # read stored original hashes from csv 
        print("Read stored original hashes from csv")
        scene_numbers, point_sizes, hash_strings = readDataFromCSVGivenColumns(output_folder_path, original_scene_hash_csv_name, ["Scene_No", "Point_Count", "Hash"])

        for row_index in range(len(scene_numbers)):
            #print("Scene {} : hash {}".format(pcd_file_names[row_index], hash_strings[row_index] ))
            original_hash_key_dict[scene_numbers[row_index]] = hash_strings[row_index]

        return original_hash_key_dict
    
    def computeTestHashes(self, input_folder_path, output_folder_path, test_scene_hash_csv_name):
    
        print("\n===================Compute test hashes ==================== ")
        createNewCSV(output_folder_path,test_scene_hash_csv_name, ["Scene_No", "Point_Count", "Trajectory_Timestamp", "Hash"])

        for scene_number in range(self.parameterObject.scene_number_range[0], self.parameterObject.scene_number_range[1] + 1):
            scene_directory_path = input_folder_path + "/" + str(scene_number)
            scene_test_file_list = [file for file in glob.glob(scene_directory_path + "/*.ply", recursive=True) if "original" not in os.path.basename(file)]
            scene_test_file_list_sorted = sorted(scene_test_file_list, key=self.custom_sorting_key)
            for ply_file_test in scene_test_file_list_sorted:
                print("Compute hash for file: {} ====>\n".format(ply_file_test))
                pcd_file_name = os.path.basename(ply_file_test)
                pcd_file_name_r, point_count, final_binary_string, binary_array, columnwise_fpfh_regions_array, model_complexity = self.hashCalculator.execute_computation_on_single_pcd(ply_file_test)

                # save results to the test data csv
                trajectory_timetamp = pcd_file_name.split('_')[1].replace('.ply', '')
                updateExistingCSV(output_folder_path, test_scene_hash_csv_name, [str(scene_number), point_count, trajectory_timetamp, final_binary_string])

        return self.readTestHashes(output_folder_path, test_scene_hash_csv_name)

    def readTestHashes(self, output_folder_path, test_scene_hash_csv_name):
        test_hash_dictionary = {}
        # read stored test hashes from csv 
        print("Read stored test hashes from csv")
        #print(readDataFromCSVGivenColumns(output_folder_path, test_scene_hash_csv_name, ["Scene_Name", "Point Count", "Attack", "Final Hash"]))
        scene_numbers, attacks, hash_strings = readDataFromCSVGivenColumns(output_folder_path, test_scene_hash_csv_name, ["Scene_No", "Trajectory_Timestamp", "Hash"])

        for row_index in range(len(scene_numbers)):
            scene_name = scene_numbers[row_index]
            attack = attacks[row_index]
            test_hash_str = hash_strings[row_index]
            if scene_name in test_hash_dictionary:
                inner_dict = test_hash_dictionary[scene_name]
                inner_dict[attack] = test_hash_str

                test_hash_dictionary[scene_name] = inner_dict
            else:
                test_hash_dictionary[scene_name] = {attack: test_hash_str}
        return test_hash_dictionary
    
class ParameterSet:
    
    def __init__(self, component_size = 20, scene_number_range = [0,0],  normal_max_radius = 0.1, normal_max_nn = 30, 
                 FPFH_max_radius = 0.5, FPFH_max_nn=100, original_scene_file_suffix = "_original", 
                 l2_cluster_enabled = False, in_code_triangulation_required = False):
        # boolean params
        self.l2_cluster_enabled = l2_cluster_enabled
        self.in_code_triangulation_required = in_code_triangulation_required

        # numbered params
        self.component_size = component_size
        self.normal_max_radius = normal_max_radius
        self.normal_max_nn = normal_max_nn
        self.FPFH_max_radius = FPFH_max_radius
        self.FPFH_max_nn = FPFH_max_nn

        # list type parameters
        self.scene_number_range = scene_number_range

        # string type parameters
        self.original_scene_file_suffix = original_scene_file_suffix

def get_computed_original_and_test_dicts(parameterObject, input_folder_path, output_folder_path, original_scene_hash_csv_name, test_scene_hash_csv_name):

    hashComputationProcessor = HashComputationProcessor(parameterObject)

    original_hash_key_dict = {} #format => {Scene_1 :  hash_string,  Scene_1: hash_string,  ...}
    if (compute_original):
        original_hash_key_dict = hashComputationProcessor.computeOriginalHashes(input_folder_path, output_folder_path, original_scene_hash_csv_name)
    else: 
        original_hash_key_dict = hashComputationProcessor.readStoredOriginalHashesFromCSV(output_folder_path, original_scene_hash_csv_name)

    test_hash_dictionary = {} #format => {Scene_1 : {timestamp: hash_string,  timestamp: hash_string, ..}, ...}
    if (compute_test):
        test_hash_dictionary = hashComputationProcessor.computeTestHashes(input_folder_path, output_folder_path, test_scene_hash_csv_name)
    else:
        test_hash_dictionary = hashComputationProcessor.readTestHashes(output_folder_path, test_scene_hash_csv_name)

    return original_hash_key_dict, test_hash_dictionary


if __name__ == "__main__":

    config_file_path = generateAndAccessArgsForConfigFile()
    configParser = ConfigParser(config_file_path)

    component_sizes_list = list(configParser.getConfigParam("component_sizes")) # 100 means final hash has 100*33 bits when L2 cluster is disabled
    normal_radius_nn_list = list(configParser.getConfigParam("normal_radius_nn_list"))
    FPFH_radius_nn_list = list(configParser.getConfigParam("FPFH_radius_nn_list"))

    l2_cluster_enabled = configParser.getConfigParam("l2_cluster_enabled")
    in_code_triangulation_required = configParser.getConfigParam("in_code_triangulation_required")

    compute_original = configParser.getConfigParam("compute_original_hashes")
    compute_test = configParser.getConfigParam("compute_test_hashes")
    analyze = configParser.getConfigParam("analyze")

    input_folder_path = configParser.getConfigParam("input_folder_path")
    output_folder_path = makeAdirectoryAtGivenPath(input_folder_path, str(configParser.getConfigParam("output_folder_name")))
    scene_number_range = list(configParser.getConfigParam("scene_number_range_to_process"))

    #original_scene_hash_csv_name = "original_scene_hash_set"
    original_scene_hash_csv_name = configParser.getConfigParam("original_hash_write_csv_name")

    #test_scene_hash_csv_name = "test_scene_hash_set"
    test_scene_hash_csv_name = configParser.getConfigParam("test_hash_write_csv_name")

    threshold = int(configParser.getConfigParam("hash_comparison_threshold"))
    original_scene_file_suffix = configParser.getConfigParam("original_scene_file_suffix")

    threshold = configParser.getConfigParam("hash_comparison_threshold")

    iteration_count = -1
    accuracy_array = []
    original_test_hash_dict_collection_of_all_itertions = {}

    for component_size in component_sizes_list:

        for fpfh_comp_radius_nn_entry in FPFH_radius_nn_list:
            fpfh_radius = fpfh_comp_radius_nn_entry[0]
            fpfh_nn = fpfh_comp_radius_nn_entry[1]

            for normal_comp_radius_nn_entry in normal_radius_nn_list: 
                iteration_count+=1
                print("======================== Results for iteration_count ======={}===================>  {}".format(str(normal_comp_radius_nn_entry), str(iteration_count)))
                normal_radius = normal_comp_radius_nn_entry[0]
                normal_nn = normal_comp_radius_nn_entry[1]

                original_scene_hash_csv_name_itr = original_scene_hash_csv_name + "_" + str(iteration_count)
                test_scene_hash_csv_name_itr = test_scene_hash_csv_name + "_" + str(iteration_count)
                parameterObject = ParameterSet(component_size, scene_number_range, normal_radius, normal_nn, fpfh_radius, fpfh_nn, original_scene_file_suffix, l2_cluster_enabled, in_code_triangulation_required)
        
                original_hash_key_dict, test_hash_dictionary = get_computed_original_and_test_dicts(parameterObject, input_folder_path, output_folder_path, original_scene_hash_csv_name_itr, test_scene_hash_csv_name_itr)

                original_test_hash_dict_collection_of_all_itertions[iteration_count] = (original_hash_key_dict, test_hash_dictionary)

                if (analyze):
        
                    uAnalyzer = UniquenessAnalyzer(output_folder_path, original_hash_key_dict, test_hash_dictionary, threshold)
                    rAnalyzer = RobustnessAnalyzer(output_folder_path, original_hash_key_dict, test_hash_dictionary, threshold)
        
                    if (scene_number_range[0] != scene_number_range[1]):
                        uAnalyzer.analyze_hammingDist_between_original_scenes()

                    accuracy = rAnalyzer.analyze_robustness_of_trajectory_scene_against_original_scene()
                    accuracy_array.append(accuracy)

    robustnessAnalyzerForAllIterations = RobustnessAnalyzerForAllIterations(original_test_hash_dict_collection_of_all_itertions, threshold) 
    if (scene_number_range[0] == scene_number_range[1]):
        # these methods can be run only when a single scene is analyzed
        robustnessAnalyzerForAllIterations.plot_accuracy_against_changed_param(range(iteration_count+1), accuracy_array, normal_radius_nn_list, "Accuracy of scene robustness with normal computation neighboord and radius", "max neighboord radius and point count", "Accuracy for robustness of Views" )
        robustnessAnalyzerForAllIterations.analyze_robustness_of_trajectory_scene_againt_changedParameter_for_singleScene("Radius/NN - Normal Comp.", normal_radius_nn_list)

        #robustnessAnalyzerForAllIterations.plot_accuracy_against_changed_param(range(iteration_count+1), accuracy_array, FPFH_radius_nn_list, "Accuracy of scene robustness with FPFH computation neighboord and radius", "max neighboord radius and point count", "Accuracy for robustness of Views" )
        #robustnessAnalyzerForAllIterations.analyze_robustness_of_trajectory_scene_againt_changedParameter_for_singleScene("Radius/NN - FPFH Comp.", FPFH_radius_nn_list)

        #robustnessAnalyzerForAllIterations.plot_accuracy_against_changed_param(range(iteration_count+1), accuracy_array, component_sizes_list, "Accuracy of scene robustness with No of Clusters", "No of Clusters", "Accuracy for robustness of Views" )
        #robustnessAnalyzerForAllIterations.analyze_robustness_of_trajectory_scene_againt_changedParameter_for_singleScene("No of Clusters", component_sizes_list)
        
        #analyzer.computeTotalMetrics(threshold)
        #analyzer.computeAttackWiseMetrics("FILT", threshold) # SPD_, R_, RCrop, RNoise, HCrop 
        #analyzer.plotAttackWiseResultsAllScenesTogether("RNoise")
        #analyzer.analyzeAllAttack_RandomSelection(threshold = 30)

        #analyzer.analyzeAttack_RandomSelection(attack_name="FILT", threshold = 30)
        #analyzer.analyzeRobustness_withOriginalScene()
        #analyzer.analyze_Robustness_across_captures_of_same_scene()
        #analyzer.analyze_robustness_all_scenes_against_fov()

        














###########Notes
### existing issue 1 to be handled: (limitations in curvature computation on L2 cluster step)
# in server:  feature computation is changed to use pcd_utils library, but it does not work in mac, so igl lib is enable here
# at the same time, igl lib also gives segmentation errors for certain point clouds.

### existing issue 2 to be handled: (always faces are needed for L2 cluster step)
# all subsampled versions of grasp dataset is created by running pcd-sampling in server. 
# write now mesh version is saved during sampling. beacause the 2 curvature computation functions need faces which are a integral part of a mesh.
# so, write now the code can not be run on ply files with only points if L2_clustering is enable