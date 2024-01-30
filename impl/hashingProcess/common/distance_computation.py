import numpy as np

# count as one if the bits are differnt at the comparing location of the 2 hash strings
# the count explains how the 2 hash strings differ from each other
def computeHammingDistance(bit_str_1, bit_str_2):
    count = 0
    for i in range(len(bit_str_1)):
        if(bit_str_1[i] != bit_str_2[i]):
            count += 1
    return count



def computeHammingDistancePercentage(bit_str_1, bit_str_2):
    count = computeHammingDistance(bit_str_1, bit_str_2)
    return (count/len(bit_str_1)) * 100

def computeEucliedianDistancePercentage(fpfh_array_1, fpfh_array_2):
    differences = fpfh_array_1 - fpfh_array_2   
    euclidean_distances = np.linalg.norm(differences, axis=1)
    average_distance_mean = np.mean(euclidean_distances)
    return average_distance_mean