import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import seaborn as sns
import os

# plot mutiple scatter plots on a single graph for a combination of (x, y) data
def scatterPlotSetOfData(x_y_data_set):
    i =0
    for x, y in x_y_data_set:
        plt.scatter(x, y, c='red', label='Iteration '+ i)
        # Set plot labels and title
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Multiple Data Sets Scatter Plot')

    # Add a legend
    plt.legend()

    # Display the plot
    plt.show()

# to remove the stanard deviation values of scores
def extract_numeric_or_full(cell_value):
    if '(' in cell_value:
        return float(cell_value.split('(')[0])
    else:
        return float(cell_val)

def generateHeatMapofFinalScores(analyzed_csv_path, columns_to_keep, is_display_heatmap):

    print(analyzed_csv_path)
    data = pd.read_csv(analyzed_csv_path, index_col=0)  # Assuming the first column is the index
    data = data.drop(data.columns[[0, 1]], axis=1)  # Drop Point Count and Final Hash columns
    
    # Filter columns based on the names you want to keep
    if (columns_to_keep != []):
        data = data[columns_to_keep]
        # select all the cell values under 1st column
        first_cell_values = data.index.str.extract("^(.*)", expand=False)
        # remove rows which belong to removed columns
        data = data[first_cell_values.isin(columns_to_keep)]

    #data = data.apply(lambda x: x.str.rstrip('%').astype('float'))  # Convert percentage strings to floats
    data = data.applymap(extract_numeric_or_full)
    #data = data.apply(lambda x: x.str.rstrip('%').astype('float') / 100)  
    #data = data.round().astype(int) # Round the data values to the nearest integer

    #mask = np.triu(np.ones_like(data, dtype=bool)) ##Create a mask to hide the upper triangular part

    g = sns.heatmap(data, annot=True, cmap="YlGnBu", vmin=0, vmax=100) #fmt=".2f", mask=mask
    #g = sns.heatmap(data, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1)

    g.set_yticklabels(g.get_yticklabels(), rotation=0)
    g.set_title('Heatmap of difference score (%)')

    cbar = g.collections[0].colorbar
    cbar.set_ticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    #cbar.set_ticks([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.75, 1])

    # Save the heatmap image in the same folder as the CSV file
    output_folder = os.path.dirname(analyzed_csv_path)
    output_filename = "heatmap.png"
    output_path = os.path.join(output_folder, output_filename)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1)

    plt.tight_layout()
    if (is_display_heatmap):
        plt.show()