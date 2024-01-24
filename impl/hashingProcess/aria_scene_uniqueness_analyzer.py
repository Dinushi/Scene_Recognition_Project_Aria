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

    def plotHistogramHamingDistanceDistribution(self, output_folder_path, hammming_dist_array, title_name):
        # Define the bin edges based on your requirements (0-5, 5-10, ..., 95-100)
        plt.figure(figsize=(20, 20))
        bin_edges = np.arange(0, 105, 5)

        # Plot the histogram
        plt.hist(hammming_dist_array, bins=bin_edges, edgecolor='black', alpha=0.7)
        plt.title('Histogram of Hamming Distance score (Hash Difference) between scenes - ' + title_name)
        plt.xlabel('Hamming Distance Score (Hash Difference) %')
        plt.ylabel('Frequency')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        #plt.savefig(output_folder_path +'filename.png', bbox_inches='tight')
        plt.show()

    def plotHeatMapOfOriginalSceneHammingDistances(self, output_folder_path, original_hash_key_dict, title_name):

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
        plt.title("Heatmap Hamming Distance score (Hash Difference) between scenes - " + title_name)
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

class UniquenessAnalyzer:
    def __init__(self, output_folder_path, original_hash_key_dict, test_hash_dictionary, threshold = 30):
        self.output_folder_path = output_folder_path
        self.original_hash_key_dict = original_hash_key_dict #format => {Scene_1 : hash_string, Scene_2 : hash_string, ...}
        self.test_hash_dictionary = test_hash_dictionary #format => {Scene_1 : {attack_1: hash_string,  atatck_2: hash_string, ..}, ...}
        self.threshold = threshold

    def calculate_metrics(self, TP, TN, FP, FN):
        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        accuracy = (TP + TN) / (TP + TN + FP + FN) if TP + TN + FP + FN > 0 else 0

        return precision, recall, accuracy

    def analyze_hammingDist_between_original_scenes(self):
        total_combinations = 0
        incorrect = 0
        correct = 0
        
        hamming_dist_score_array = []
        all_original_combinations = list(combinations(self.original_hash_key_dict.items(), 2))
   
        for pair1, pair2 in all_original_combinations:
            scene_name_1, original_hash_1 = pair1
            scene_name_2, original_hash_2 = pair2

            hamming_dist_score = computeHammingDistancePercentage(original_hash_1, original_hash_2)
            hamming_dist_score_array.append(hamming_dist_score)

            total_combinations += 1
            if (hamming_dist_score <= self.threshold):
                incorrect += 1
            else:
                correct += 1 

        print("\nUniqueness Analysis ==>")
        print("\tTotal different pair combinations : {}, total_higher_than_threshold: {} , total_lesser_than_threshold : {}".format(total_combinations, correct, incorrect))
        print("\tAccuracy for different original scene pairs: " + str((correct/total_combinations)* 100))
        graphicalVisualizer = GraphicalVisualizer()
        graphicalVisualizer.plotHistogramHamingDistanceDistribution(self.output_folder_path, hamming_dist_score_array, "Entire Scenes - 10")
        graphicalVisualizer.plotHeatMapOfOriginalSceneHammingDistances(self.output_folder_path, self.original_hash_key_dict, "Entire Scenes - 10")

    def analyzeHDBetweenDifferentTestScenes(self, threshold = 30, attack_name = "FILT130130_30"):
        total_combinations = 0
        incorrect = 0
        correct = 0

        hamming_dist_score_array = []
        all_test_combinations = list(combinations(self.test_hash_dictionary.items(), 2))
        for pair1, pair2 in all_test_combinations:
            scene_name_1, attack_hash_dict_1 = pair1
            scene_name_2, attack_hash_dict_2 = pair2

            try:
                hamming_dist_score = computeHammingDistancePercentage(attack_hash_dict_1[attack_name], attack_hash_dict_2[attack_name])
            except KeyError as e:
                print(f"KeyError occurred: {e}")
                # Optionally, set hamming_dist_score to a default value or perform other actions
                hamming_dist_score = 0 
            hamming_dist_score_array.append(hamming_dist_score)
                
            total_combinations += 1
            if (hamming_dist_score <= threshold):
                incorrect += 1
            else:
                correct += 1 

        print("\nUniqueness Analysis ==> " + attack_name)
        print("\tTotal different pair combinations : {}, total_higher_than_threshold: {} , total_lesser_than_threshold : {}".format(total_combinations, correct, incorrect))
        accuracy = (correct/total_combinations)* 100
        print("\tAccuracy for different test scene pairs: " + str(accuracy))
        #GraphicalVisualizer().plotHistogramHamingDistanceDistribution(hamming_dist_score_array, attack_name)
        return accuracy, hamming_dist_score_array