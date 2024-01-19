import os
import csv

def makeAdirectoryAtGivenPath(base_path, new_folder_Name):
    # Use os.makedirs to create the folder if it doesn't exist
    folder_path = base_path + "/" +new_folder_Name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

def createNewCSV(folder_path, file_name, column_name_list):
    with open(folder_path + "/" + file_name + ".csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(column_name_list)

def updateExistingCSV(folder_path, file_name, value_list):
    with open(folder_path + "/" + file_name + ".csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(value_list)

    # create a csv file to store hashing results
    #with open(output_folder_path + "/hashing_results.csv", "w", newline="") as f:
        #writer = csv.writer(f)
        #writer.writerow(["PCD file", "Point Size", "Hash String"])
