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

class Analyzer:
    def __init__(self, configsMap, original_hash_key_dict, test_hash_dictionary):
        self.configsMap = configsMap
        self.original_hash_key_dict = original_hash_key_dict #format => {Scene_1 : hash_string, Scene_2 : hash_string, ...}
        self.test_hash_dictionary = test_hash_dictionary #format => {Scene_1 : {attack_1: hash_string,  atatck_2: hash_string, ..}, ...}

    def calculate_metrics(self, TP, TN, FP, FN):
        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        accuracy = (TP + TN) / (TP + TN + FP + FN) if TP + TN + FP + FN > 0 else 0

        return precision, recall, accuracy


    def analyzeHDBetween_All_DifferentTestScenes(self, threshold = 30):
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
    
        plt.title('Hamming Distance score (Hash Difference) of each Camara View hash against the hash of entire scene')
        plt.xlabel('Timestamp at each tragectory point (ms)')
        plt.ylabel('Hamming Distance (%)')
        plt.legend()
        plt.grid(True)
        #plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
            

    def analyze_Robustness_across_captures_of_same_scene(self, threshold = 30):
        rotation_angle_array = ["0", "15", "30", "45", "60", "75", "90"]

        for scene_name, attack_hash_dict in self.test_hash_dictionary.items():
            sorted_attack_hash_dict = {k: attack_hash_dict[k] for k in sorted(attack_hash_dict)}

            attack_names_x = []
            heatmap_data = []

            for attack_name_1, hash_string_1 in sorted_attack_hash_dict.items():
                HD_array_for_attack = []
                for attack_name_2, hash_string_2 in sorted_attack_hash_dict.items():
                    hamming_dist_score = computeHammingDistancePercentage(hash_string_1, hash_string_2)
                    HD_array_for_attack.append(hamming_dist_score)
                attack_names_x.append(attack_name_1)
                heatmap_data.append(HD_array_for_attack)

            data_array = np.array(heatmap_data)
            sns.heatmap(data_array, xticklabels=attack_names_x, yticklabels=attack_names_x, cmap='Blues', annot=True, vmin=0, vmax=100)
            plt.title('Heatmap of Hamming Distance score between different FOV and angle based extractions of ' + scene_name)
            plt.show()

    def analyze_robustness_all_scenes_against_fov(self, threshold = 30):

        plt.figure(figsize=(10, 6))

        rotation_angle = "30"
        fov_array = ["130", "120", "110", "100", "90"]

        for scene_name, attack_hash_dict in self.test_hash_dictionary.items():
            hd_for_each_fov_scene = []
            original_hash_string = self.original_hash_key_dict[scene_name]
            for fov in fov_array:
                attack_name = "FILT" + fov + fov + "_" + rotation_angle
                test_hash_string = attack_hash_dict[attack_name]
                hamming_dist_score = computeHammingDistancePercentage(original_hash_string, test_hash_string)
                hd_for_each_fov_scene.append(hamming_dist_score)

            plt.plot(fov_array, hd_for_each_fov_scene, label=scene_name)
 
        plt.xlabel('Angle of FOV')
        plt.ylabel('Hamming Distance with Original Hash (%)')
        plt.title('Hash Simillarity of various FOV based extractions (rotation angle-'+rotation_angle+ ') with each original scene' )
        plt.legend()
        plt.grid(True)
        plt.show()

    def computeTotalMetrics(self, threshold = 30):
        total_combination_pairs = 0
        TP = 0 
        FP = 0
        FN = 0
        TN = 0
        for scene_name, value_dict in self.test_hash_dictionary.items():
            for attack, test_hash_string in value_dict.items ():
                for original_scene_name_key, original_hash_value in self.original_hash_key_dict.items():
                    total_combination_pairs +=1
                    hamming_dist_score = computeHammingDistancePercentage(original_hash_value, test_hash_string)
                    #print("\tHamming distance score between {} original and attack {} : {}".format(original_scene_name_key, attack, hamming_dist_score))
                                
                    if (hamming_dist_score <= threshold and original_scene_name_key==scene_name): #Assume diff less than threshold => Positive Answer)
                        # has a difference score lesser than the thresold with it's RELAVANT original hash (expected behavior)  -> True Positive
                        TP += 1
                    elif (hamming_dist_score <= threshold and original_scene_name_key!=scene_name):
                        # has a difference score lesser than the thresold with an IRRELAVANT original hash -> False Positive
                        FP += 1
                    elif (hamming_dist_score > threshold and original_scene_name_key==scene_name):
                        # has a difference score higher than the thresold with it's RELAVANT original hash -> False Negative
                        FN += 1
                    elif (hamming_dist_score > threshold and original_scene_name_key!=scene_name):
                        # has a difference score higher than the thresold with an IRRELAVANT original hash (expected behavior) -> True Negative
                        TN += 1
        print("\nTotal Results ==>")
        print("\tTotal combinations : {}, TP :{} , FP : {}, FN : {}, TN : {} ".format(total_combination_pairs, TP, FP, FN, TN))
        precision, recall, accuracy = self.calculate_metrics(TP, TN, FP, FN)
        print("\tResults => Precision : {}, Recall : {} Accuracy : {} \n".format(precision, recall, accuracy))

    def computeAttackWiseMetrics(self, attack_name, threshold = 30):
        total_combination_pairs = 0
        TP = 0 
        FP = 0
        FN = 0
        TN = 0
        for scene_name, value_dict in self.test_hash_dictionary.items():
            for attack, test_hash_string in value_dict.items ():

                if (attack_name in attack):
                    for original_scene_name_key, original_hash_value in self.original_hash_key_dict.items():
                        total_combination_pairs +=1
                        hamming_dist_score = computeHammingDistancePercentage(original_hash_value, test_hash_string)
                        #print("\tHamming distance score between {} original and attack {} : {}".format(original_scene_name_key, attack, hamming_dist_score))
                            
                        if (hamming_dist_score <= threshold and original_scene_name_key==scene_name): #Assume diff less than threshold => Positive Answer)
                            # has a difference score lesser than the thresold with it's RELAVANT original hash (expected behavior)  -> True Positive
                            TP += 1
                        elif (hamming_dist_score <= threshold and original_scene_name_key!=scene_name):
                            # has a difference score lesser than the thresold with an IRRELAVANT original hash -> False Positive
                            FP += 1
                        elif (hamming_dist_score > threshold and original_scene_name_key==scene_name):
                            # has a difference score higher than the thresold with it's RELAVANT original hash -> False Negative
                            FN += 1
                        elif (hamming_dist_score > threshold and original_scene_name_key!=scene_name):
                            # has a difference score higher than the thresold with an IRRELAVANT original hash (expected behavior) -> True Negative
                            TN += 1
        print("\nRobustness against Attacks ==>")
        print("\tAttack - {} => Total combinations  : {}, TP :{} , FP : {}, FN : {}, TN : {} ".format(attack_name, total_combination_pairs, TP, FP, FN, TN))
        precision, recall, accuracy = self.calculate_metrics(TP, TN, FP, FN)
        print("\tAttack - {} => Precision : {}, Recall : {} Accuracy : {} \n".format(attack_name, precision, recall, accuracy))

    def plotAttackWiseResultsAllScenesTogether(self, attack_to_display):
        fig_spd_down, ax_spd_down = plt.subplots()
        fig_spd_up, ax_spd_up = plt.subplots()
        
        fig_rotation, ax_rotation = plt.subplots()
        fig_crop, ax_crop = plt.subplots()
        fig_random_noise, ax_random_noise = plt.subplots()
        
        for scene_name, value_dict in self.test_hash_dictionary.items():
            original_hash = self.original_hash_key_dict[scene_name]
                
            x_SPD_values_up = []
            y_SPD_values_up = []

            x_SPD_values_down = []
            y_SPD_values_down = []

            # for rotation
            x_R_values = []
            y_R_values = []

            # for crop
            x_Crop_values = []
            y_Crop_values = []

            # for random point addition
            x_RNoise_values = []
            y_RNoise_values = []
            
            custom_order_SPD = ["SPD_100000", "SPD_95000", "SPD_90000", "SPD_85000", "SPD_80000",
            "SPD_75000", "SPD_70000", "SPD_65000", "SPD_60000", "SPD_55000",
            "SPD_50000", "SPD_45000", "SPD_40000", "SPD_35000", "SPD_30000",
            "SPD_25000", "SPD_20000"]
            custom_order_R = ["R_z15", "R_z45", "R_z90", "R_z180", "R_z270"]
            custom_order_H_CROP = ["HCrop_xz0", "HCrop_xz45", "HCrop_xz90", "HCrop_xz135" "HCrop_xz180", "HCrop_xz225", "HCrop_xz270", "HCrop_xz315"]
            custom_order_R_CROP =[]
            custom_order_R_NOISE =['RNoise0_1', 'RNoise0_2', 'RNoise0_3', 'RNoise0_4', 'RNoise0_5',
                    'RNoise0_6', 'RNoise0_7', 'RNoise0_8', 'RNoise0_9', 'RNoise0_10',
                    'RNoise0_11', 'RNoise0_12', 'RNoise0_13', 'RNoise0_14', 'RNoise0_15',
                    'RNoise0_16', 'RNoise0_17', 'RNoise0_18', 'RNoise0_19', 'RNoise0_20',
                    'RNoise0_21', 'RNoise0_22', 'RNoise0_23', 'RNoise0_24', 'RNoise0_25']
            

            if ("RCrop" in attack_to_display):
                sorted_value_dict = {key: value_dict[key] for key in custom_order_R_CROP if key in value_dict}

            for attack, test_hash_string in sorted_value_dict.items ():
                # compure hamming distance of the test hash with its relavant original scene hash
                hamming_dist_score = computeHammingDistancePercentage(original_hash, test_hash_string)

                if ("SPD_" in attack):
                    sample_rate = attack.split('_')[1]
                    #if (int(sample_rate) < 50000):
                    x_SPD_values_down.append(sample_rate)
                    y_SPD_values_down.append(hamming_dist_score)
                    #else:
                        #x_SPD_values_up.append(sample_rate)
                        #y_SPD_values_up.append(hamming_dist_score)

                elif ("R_" in attack):
                    angle = attack.split('_')[1][1:]
                    x_R_values.append(angle)
                    y_R_values.append(hamming_dist_score)

                elif ("RCrop" in attack): # "RCrop" , "HCrop"
                    crop_rate = attack.split('_')[1]
                    x_Crop_values.append(crop_rate)
                    y_Crop_values.append(hamming_dist_score)

                elif ("RNoise" in attack):
                    noise_rate = attack.split('_')[1]
                    x_RNoise_values.append(noise_rate)
                    y_RNoise_values.append(hamming_dist_score)
                
            if ("SPD_" == attack_to_display):
                ax_spd_down.plot(x_SPD_values_down, y_SPD_values_down, label=scene_name)
                ax_spd_up.plot(x_SPD_values_up, y_SPD_values_up, label=scene_name)

                ax_spd_down.set_xlabel('No of Points')
                ax_spd_down.set_ylabel('Diff. Score')
                ax_spd_down.set_title("Poisson Disk Down-Sampling Attack - Hash Difference")
                ax_spd_down.legend()

                #ax_spd_up.set_xlabel('Sample Rates')
                #ax_spd_up.set_ylabel('Diff. Score')
                #ax_spd_up.set_title("Poisson Disk Up-Sampling Attack - Hash Difference")
                #ax_spd_up.legend()

            elif ("R_" == attack_to_display):
                ax_rotation.plot(x_R_values, y_R_values, label=scene_name)

                ax_rotation.set_xlabel('Rotation Angle')
                ax_rotation.set_ylabel('Diff. Score')
                ax_rotation.set_title("Rotation Attack - Hash Difference")
                ax_rotation.legend()

            elif ("RCrop" == attack_to_display): # "RCrop" , "HCrop"
                ax_crop.plot(x_Crop_values, y_Crop_values, label=scene_name)

                ax_crop.set_xlabel('Crop Rate')
                ax_crop.set_ylabel('Diff. Score')
                ax_crop.set_title("Randomly Cropping a percentage of space - Hash Difference")
                #ax_crop.set_title("Rotate by a angle and Crop half of the space - Hash Difference")
                ax_crop.legend()
                
            elif ("RNoise" == attack_to_display):
                ax_random_noise.plot(x_RNoise_values, y_RNoise_values, label=scene_name)
                ax_random_noise.set_xlabel('Random Noise Rate')
                ax_random_noise.set_ylabel('Diff. Score')
                ax_random_noise.set_title("Randomly adding a percentage of noise points - Hash Difference")
                ax_random_noise.legend()
        plt.show()


    def analyzeAttack_RandomSelection(self, attack_name, threshold = 30):

        expected_TP = 0
        expected_TN = 0

        computed_TP = 0
        computed_FN = 0
        computed_FP = 0
        computed_TN = 0

        for scene_name, value_dict in self.test_hash_dictionary.items():
            print("Evaluating scene ==> {} ".format(scene_name))
            original_hash_of_scene = self.original_hash_key_dict[scene_name]

            other_scene_hash_dict = [(scene, orginal_hash) for scene, orginal_hash in self.original_hash_key_dict.items() if scene != scene_name]
            for attack, test_hash in value_dict.items ():
                if (attack_name in attack):
                    # compare with the original hash of the test scene
                    expected_TP += 1
                    hamming_dist_score_between_attacked_and_itsoriginal = computeHammingDistancePercentage(test_hash, original_hash_of_scene)
                    print("\tAttack {} of {} with original hash = {} %".format(attack, scene_name, hamming_dist_score_between_attacked_and_itsoriginal))
                    if (hamming_dist_score_between_attacked_and_itsoriginal <= threshold):
                        computed_TP += 1
                    else: 
                        computed_FN += 1

                    # compare with a randomly selected different scene
                    expected_TN += 1
                    random_diff_scene, random_diff_hash = random.choice(other_scene_hash_dict)
                    hamming_dist_score_between_attacked_and_otherscene = computeHammingDistancePercentage(test_hash, random_diff_hash)
                    print("\tAttack {} of {} with randomly selected different scene {} = {} %".format(attack, scene_name, random_diff_scene, hamming_dist_score_between_attacked_and_otherscene))
                    if (hamming_dist_score_between_attacked_and_otherscene > threshold):
                        computed_TN += 1
                    else: 
                        computed_FP += 1

        print("\nTotal TP & TN balanced Scenario (Attack {}): Results ==>".format(attack_name))
        total = expected_TP + expected_TN
        print("\tTotal combinations : {}  Expected =>  TP : {},  FP : {}, FN : {}, TN : {} ".format(total, expected_TP, 0, 0, expected_TN))
        print("\tTotal combinations : {}, Calculated => TP :{},  FP : {},  FN : {},  TN : {} ".format(total, computed_TP, computed_FP, computed_FN, computed_TN))

        print("")

        print("\tRobustness (between original and attacked scenes) => Correctly Identified : {},   Mis-Idnetified : {},  Accuracy (computed_TP/expected_TP) = {}/{} = {}".format (computed_TP, computed_FN, computed_TP, expected_TP, computed_TP/expected_TP))
        print("\tUniqueness (different scenes) => Correctly Identified : {},   Mis-Idnetified : {},  Accuracy (computed_TN/expected_TN) = {}/{} = {}".format(computed_TN, computed_FP, computed_TN, expected_TN, computed_TN/expected_TN))
            
        print("\n\tOveral Accuracy =>")
        exp_precision, exp_recall, exp_accuracy = self.calculate_metrics(expected_TP, expected_TN, FP = 0, FN = 0)
        print("\tExpected Results => Precision : {}, Recall : {} Accuracy : {}".format(exp_precision, exp_recall, exp_accuracy))
        cal_precision, cal_recall, cal_accuracy = self.calculate_metrics(computed_TP, computed_TN, computed_FP, computed_FN)
        print("\tComputed Results => Precision : {}, Recall : {} Accuracy : {}".format(cal_precision, cal_recall, cal_accuracy))


    def analyzeAllAttack_RandomSelection(self, threshold = 30):

        expected_TP = 0
        expected_TN = 0

        computed_TP = 0
        computed_FN = 0
        computed_FP = 0
        computed_TN = 0

        for scene_name, value_dict in self.test_hash_dictionary.items():
            original_hash_of_scene = self.original_hash_key_dict[scene_name]
            print("Evaluating scene ==> {} ".format(scene_name))

            other_scene_hash_dict = [(scene, orginal_hash) for scene, orginal_hash in self.original_hash_key_dict.items() if scene != scene_name]
            for attack, test_hash in value_dict.items ():
                # compare with the original hash of the test scene
                expected_TP += 1
                hamming_dist_score_between_attacked_and_itsoriginal = computeHammingDistancePercentage(test_hash, original_hash_of_scene)
                print("\tAttack {} of {} with original hash = {} %".format(attack, scene_name, hamming_dist_score_between_attacked_and_itsoriginal))
                if (hamming_dist_score_between_attacked_and_itsoriginal <= threshold):
                    computed_TP += 1
                else: 
                    computed_FN += 1

                # compare with a randomly selected different scene
                expected_TN += 1
                random_diff_scene, random_diff_hash = random.choice(other_scene_hash_dict)
                hamming_dist_score_between_attacked_and_otherscene = computeHammingDistancePercentage(test_hash, random_diff_hash)
                print("\tAttack {} of {} with randomly selected different scene {} = {} %".format(attack, scene_name, random_diff_scene, hamming_dist_score_between_attacked_and_otherscene))
                if (hamming_dist_score_between_attacked_and_otherscene > threshold):
                    computed_TN += 1
                else: 
                    computed_FP += 1

        print("\nTotal TP & TN balanced Scenario (All Attacks): Results ==>")
        total = expected_TP + expected_TN
        print("\tTotal combinations : {}  Expected =>  TP : {},  FP : {}, FN : {}, TN : {} ".format(total, expected_TP, 0, 0, expected_TN))
        print("\tTotal combinations : {}, Calculated => TP :{},  FP : {},  FN : {},  TN : {} ".format(total, computed_TP, computed_FP, computed_FN, computed_TN))

        print("")

        print("\tRobustness (between original and attacked scenes) => Correctly Identified : {},   Mis-Idnetified : {},  Accuracy (computed_TP/expected_TP) = {}/{} = {}".format (computed_TP, computed_FN, computed_TP, expected_TP, computed_TP/expected_TP))
        print("\tUniqueness (different scenes) => Correctly Identified : {},   Mis-Idnetified : {},  Accuracy (computed_TN/expected_TN) = {}/{} = {}".format(computed_TN, computed_FP, computed_TN, expected_TN, computed_TN/expected_TN))
            
        print("\n\tOveral Accuracy =>")
        exp_precision, exp_recall, exp_accuracy = self.calculate_metrics(expected_TP, expected_TN, FP = 0, FN = 0)
        print("\tExpected Results => Precision : {}, Recall : {} Accuracy : {}".format(exp_precision, exp_recall, exp_accuracy))
        cal_precision, cal_recall, cal_accuracy = self.calculate_metrics(computed_TP, computed_TN, computed_FP, computed_FN)
        print("\tComputed Results => Precision : {}, Recall : {} Accuracy : {}".format(cal_precision, cal_recall, cal_accuracy))
        
