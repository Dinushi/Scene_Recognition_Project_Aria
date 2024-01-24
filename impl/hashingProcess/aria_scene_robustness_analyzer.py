from common.distance_computation import *

import numpy as np
import time
import os
import matplotlib.pyplot as plt
from itertools import combinations
import random

import matplotlib.pyplot as plt
import seaborn as sns


class GraphicalVisualizer():

    def plotHistogramHamingDistanceDistribution(self, hammming_dist_array, title_name):
        # Define the bin edges based on your requirements (0-5, 5-10, ..., 95-100)
        bin_edges = np.arange(0, 105, 5)

        # Plot the histogram
        plt.hist(hammming_dist_array, bins=bin_edges, edgecolor='black', alpha=0.7)
        plt.title('Histogram of Hamming Distance score between scenes - 3D Hashing for ' + title_name)
        plt.xlabel('Hamming Distance Score')
        plt.ylabel('Frequency')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    def plotHeatMapOfOriginalSceneHammingDistances(self, original_hash_key_dict):

        sorted_original_hash_key_dict = {k: original_hash_key_dict[k] for k in sorted(original_hash_key_dict)}

        scene_names_x = []
        heatmap_data = []

        for scene_name_1, hash_string_1 in sorted_original_hash_key_dict.items():
            HD_array_for_scene = []
            for scene_name_2, hash_string_2 in sorted_original_hash_key_dict.items():
                hamming_dist_score = computeHammingDistancePercentage(hash_string_1, hash_string_2)
                HD_array_for_scene.append(hamming_dist_score)
            scene_names_x.append(scene_name_1)
            heatmap_data.append(HD_array_for_scene)

        data_array = np.array(heatmap_data)
        sns.heatmap(data_array, xticklabels=scene_names_x, yticklabels=scene_names_x, cmap='Blues', annot=True, vmin=0, vmax=100)
        plt.show()

    def plotMutipleGraphsInOnePlot(self, x_data_array, y_data_dict, scene_name = "A"):
        plt.figure(figsize=(10, 6))
        for graph_type, y_data_array in y_data_dict.items():
            plt.plot(x_data_array, y_data_array, label="FOV " +graph_type[0:int(len(graph_type)/2)])

        plt.xlabel('Angle of Rotation')
        plt.ylabel('Hamming Distance with Original Hash (%)')
        plt.title('Hash Simillarity of various FOV and rotation based extractions with original scene : ' + scene_name)
        plt.legend()
        plt.grid(True)
        plt.show()

class RobustnessAnalyzer:
    def __init__(self, output_folder_path, original_hash_key_dict, test_hash_dictionary, threshold):
        self.output_folder_path = output_folder_path
        self.original_hash_key_dict = original_hash_key_dict #format => {Scene_1 : hash_string, Scene_2 : hash_string, ...}
        self.test_hash_dictionary = test_hash_dictionary #format => {Scene_1 : {attack_1: hash_string,  atatck_2: hash_string, ..}, ...}
        self.threshold = threshold

    def calculate_metrics(self, TP, TN, FP, FN):
        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        accuracy = (TP + TN) / (TP + TN + FP + FN) if TP + TN + FP + FN > 0 else 0

        return precision, recall, accuracy


    def analyzeHDBetween_All_DifferentTestScenes(self):
        total_combinations = 0
        incorrect = 0
        correct = 0
        
        FOV_array = ["130130", "120120", "110110", "100100", "9090"]
        angle_array = ["0", "15", "30", "45", "60", "75", "90"]

        acc_result_fovs = {}
        for FOV in FOV_array:
            
            bin_edges = np.arange(0, 105, 5)
            plt.title('Histogram of Hamming Distance score between scenes - 3D Hashing for scenes with FOV ' + FOV[0:int(len(FOV)/2)])
            plt.xlabel('Hamming Distance Score')
            plt.ylabel('Frequency (No of Pairs with HD > T)')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            acc_result_angles = []

            for angle in angle_array:
                attack_name = "FILT" + FOV + "_" + angle
                accuracy, hamming_dist_score_array = self.analyzeHDBetweenDifferentTestScenes(30, attack_name)

                #plt.hist(hamming_dist_score_array, bins=bin_edges, edgecolor='black', alpha=0.3, label="Rotation Angle" + angle)
    
                hist, bins = np.histogram(hamming_dist_score_array, bins=bin_edges)
                plt.plot(bins[:-1], hist, label="Rotation Angle " + angle)

                acc_result_angles.append(accuracy)
            acc_result_fovs[FOV] = acc_result_angles
            plt.legend()
            plt.grid(True)
            plt.show()

        GraphicalVisualizer().plotMutipleGraphsInOnePlot(angle_array,acc_result_fovs)

    def analyze_robustness_of_trajectory_scene_against_original_scene(self):

        plt.figure(figsize=(10, 6))

        for scene_no, tragecTimestamp_and_hash_dict_unsorted in self.test_hash_dictionary.items():
            #if (scene_no == 5):
                total = 0
                incorrect = 0
                correct = 0
                # sort the test_dict as its unsorted in csvs
                tragecTimestamp_and_hash_dict = {k: tragecTimestamp_and_hash_dict_unsorted[k] for k in sorted(tragecTimestamp_and_hash_dict_unsorted)}

                tg_timestamp_lits_scene = []
                hdist_list_scene = []

                original_hash_string = self.original_hash_key_dict[scene_no]
                for tg_timestamp, tg_hash_string in tragecTimestamp_and_hash_dict.items ():
                    hamming_dist_score = computeHammingDistancePercentage(original_hash_string, tg_hash_string)
                    tg_timestamp_lits_scene.append(tg_timestamp)
                    hdist_list_scene.append(hamming_dist_score)

                    total+=1
                    if(hamming_dist_score < self.threshold):
                        correct+=1
                    else:
                        incorrect+=1

                print("\tResults => total views : {}, correct : {} incorrect : {} \n".format(total, correct, incorrect))
                print("\tScene no {} - Robustness Accuracy: {} % \n".format(scene_no, (correct/total)*100 ))

                plt.plot([value / 1000 for value in tg_timestamp_lits_scene], hdist_list_scene, label="scene_number: "+str(scene_no))
    
        plt.title('Hamming Distance score (Hash Difference) of each Camara View hash against the hash of entire scene')

        plt.xlabel('Timestamp at each tragectory point (milli seconds)')
        plt.ylabel('Hamming Distance (%) with original hash')
        plt.legend()
        plt.grid(True)
        #plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
        return (correct/total)*100

    def analyze_robustness_of_accumilated_trajectory_scene_against_original_scene(self):
    
        plt.figure(figsize=(10, 6))

        for scene_no, tragecTimestamp_and_hash_dict_unsorted in self.test_hash_dictionary.items():
            # sort the test_dict as its unsorted in csvs
            #if (scene_no == 2):
                tragecTimestamp_and_hash_dict = {k: tragecTimestamp_and_hash_dict_unsorted[k] for k in sorted(tragecTimestamp_and_hash_dict_unsorted)}

                tg_timestamp_lits_scene = []
                hdist_list_scene = []
                
                original_hash_string = self.original_hash_key_dict[scene_no]
                for tg_timestamp, tg_hash_string in tragecTimestamp_and_hash_dict.items ():
                    hamming_dist_score = computeHammingDistancePercentage(original_hash_string, tg_hash_string)
                    print("HM dist original-original: " + str(computeHammingDistancePercentage(original_hash_string, original_hash_string)))
                    tg_timestamp_lits_scene.append(tg_timestamp)
                    hdist_list_scene.append(hamming_dist_score)

                plt.plot(tg_timestamp_lits_scene, hdist_list_scene, label="scene_number: "+str(scene_no))
    
        plt.title('Hamming Distance score (Hash Difference) : Accumilating Camara Views vs Entire scene(CROPPED)')

        plt.xlabel('Time at each tragectory point (Seconds)')
        plt.ylabel('Hamming Distance (%)')
        plt.xticks(np.arange(0, 21, 1)) 
        plt.legend()
        plt.grid(True)
        #plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
            
class RobustnessAnalyzerForAllIterations:

    def __init__(self, original_test_hash_dict_collection_of_all_itertions, threshold):
        self.original_test_hash_dict_collection_of_all_itertions = original_test_hash_dict_collection_of_all_itertions
        self.threshold = threshold

    def plot_accuracy_against_changed_param(self, x_data_array, y_data_array, x_lables, graph_title, x_title, y_tile):
        plt.figure(figsize=(10, 6))
        plt.plot(x_data_array, y_data_array)

        plt.xlabel(x_title)
        plt.ylabel(y_tile)
        plt.xticks(x_data_array, x_lables)
        plt.title(graph_title)
        plt.grid(True)
        plt.show()

    def analyze_robustness_of_trajectory_scene_againt_changedParameter_for_singleScene(self, changed_variable_name, variable_list):
    
        plt.figure(figsize=(10, 6))

        for iteration_no, (original_hash_key_dict, test_hash_dictionary) in self.original_test_hash_dict_collection_of_all_itertions.items():
            for scene_no, tragecTimestamp_and_hash_dict_unsorted in test_hash_dictionary.items():
                #if (scene_no == 5):
                    # sort the test_dict as its unsorted in csvs
                    tragecTimestamp_and_hash_dict = {k: tragecTimestamp_and_hash_dict_unsorted[k] for k in sorted(tragecTimestamp_and_hash_dict_unsorted)}

                    tg_timestamp_lits_scene = []
                    hdist_list_scene = []

                    original_hash_string = original_hash_key_dict[scene_no]
                    for tg_timestamp, tg_hash_string in tragecTimestamp_and_hash_dict.items ():
                        hamming_dist_score = computeHammingDistancePercentage(original_hash_string, tg_hash_string)
                        tg_timestamp_lits_scene.append(tg_timestamp)
                        hdist_list_scene.append(hamming_dist_score)

                    plt.plot([value / 1000 for value in tg_timestamp_lits_scene], hdist_list_scene, label=changed_variable_name+"-"+str(variable_list[iteration_no]))
        
        plt.title('Hamming Distance score (Hash Difference) of each Camara View hash against the hash of entire scene')

        plt.xlabel('Timestamp at each tragectory point (milli seconds)')
        plt.ylabel('Hamming Distance (%) with original hash')
        plt.legend()
        plt.grid(True)
        #plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()