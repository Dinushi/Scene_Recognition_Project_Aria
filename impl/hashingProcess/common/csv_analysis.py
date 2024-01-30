import csv

import numpy as np
import pandas as pd

def createNewCSVFile(folder_path, csv_file_name, col_names):

   file_path = folder_path + "/" + csv_file_name + ".csv"
   # Write data to CSV file
   with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(col_names)  # Write header row

   return file_path

# todo: generalize this function further
def readDataFromCSVGivenColumns(folder_path, csv_file_name, usecols):

   pcd_file_names = []
   pcd_file_point_sizes = []
   hash_strings = []

   file_path = folder_path + "/" + csv_file_name + ".csv"
   df = pd.read_csv(file_path,  usecols=usecols)

   for value_array in df.values:
        pcd_file_names.append(int(value_array[0]))
        pcd_file_point_sizes = np.append(pcd_file_point_sizes, value_array[1])
        hash_strings = np.append(hash_strings, value_array[2])
   return pcd_file_names, pcd_file_point_sizes.tolist(), hash_strings.tolist()

def compute_average_scores_of_analyzed_csvs(csv_folder_path, final_csv_name, component_size, iterations):
 
     csv_files = []
     for itr in range(iterations):
        analyzed_csv_name = "Anly_Results_"+str(component_size)+"_"+str(itr)
        csv_files.append(csv_folder_path + "/" + analyzed_csv_name + ".csv")

     result_data = []     
     df = pd.read_csv(csv_files[0])
     num_rows = df.shape[0]
     num_cols = df.shape[1]
    
     for row_index in range(0, num_rows):  # Specify the range of rows (2 to 89)
        for col_index in range(3, num_cols):  # Specify the range of columns (D to CM)
            cell_sum = 0
            cell_values = []
            for csv_file_path in csv_files:
               df = pd.read_csv(csv_file_path)
               cell_value = df.iat[row_index, col_index]
               #print("At row {}, column {} : cell value is {}".format(row_index, col_index,cell_value))
               if (isinstance(cell_value, str)):
                  cell_value_float = float(cell_value.rstrip('%'))
               else:
                  cell_value_float = cell_value
               cell_sum += cell_value_float
               cell_values.append(cell_value_float)

            cell_average = round(np.mean(cell_values), 2)
            cell_std_dev = round(np.std(cell_values), 2)
            result_data.append((row_index, col_index, (cell_average, cell_std_dev)))

     # Update the first DataFrame with the computed cell averages
     for row_index, col_index, (cell_average, cell_std_dev) in result_data:
        df.iat[row_index, col_index] = str(cell_average) + "(" + str(cell_std_dev) +")"

     #columns_to_sort = df.columns[3:num_cols]  
     #df[columns_to_sort] = df[columns_to_sort].apply(lambda col: col.astype(str).str[:2].str.zfill(2))
     #print(sorted(columns_to_sort))
     #df = df.reindex(columns=df.columns[0:3].append(sorted(columns_to_sort)))  

     #rows_to_sort = df.index[0:num_rows]
     #df.loc[rows_to_sort] = df.loc[rows_to_sort].apply(lambda row: row.astype(str).str[:2].str.zfill(2))
     #df = df.reindex(index=sorted(rows_to_sort))    

     df.to_csv(csv_folder_path + "/" + final_csv_name + ".csv", index=False)

