import glob
import logging
import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Workload:
    def __init__(self, workload_name, data_dict, category):
        self.workload_name = workload_name
        self.data_dict = data_dict
        self.category = category



def coefficients_visualization(data, features):
    print(features)
    coefficients = np.corrcoef([data["Task Period"], data["ExecutionTime"],
                                data["Task Uti"],
                                data["ExecutionStart time point"], data["ExecutionStart end point"],
                                data["L1 dcache load misses"], data["L1 dcache load"], data["L1 icache load misses"],
                                data["LLC load misses"], data["Branch misses"],
                                data["Branch instructions"], data["CPU cycles"], data["Instructions"],
                                data["Reference cycles"], data["Bus cycles"],
                                data["CPU clock"], data["Task clock"], data["Slots"]])

    xlabel = ["Task Period", "ExecutionTime", "Task Uti", "ExecutionStart time point", "ExecutionEnd time point",
              'L1 dcache load misses', 'L1 dcache load', 'L1 icache load misses', 'LLC load misses', 'Branch misses',
              'Branch instructions', 'CPU cycles', 'Instructions', 'Reference cycles', 'Bus cycles', 'CPU clock', 'Task clock', 'Slots']
    ylabel = ["Task Period", "ExecutionTime", "Task Uti", "ExecutionStart time point", "ExecutionEnd time point",
              'L1 dcache load misses', 'L1 dcache load', 'L1 icache load misses', 'LLC load misses', 'Branch misses',
              'Branch instructions', 'CPU cycles', 'Instructions', 'Reference cycles', 'Bus cycles', 'CPU clock', 'Task clock', 'Slots']

    sns.heatmap(coefficients, annot=True, xticklabels=xlabel, yticklabels=ylabel, cmap="coolwarm")

    plt.title("Correlation Coefficient Heatmap")

    # Show the plot
    plt.show()

def get_data():
    # Specify the path to your CSV files.
    # directory_path = '/home/jiezou/ros2_ws/Data/ros2/*.csv'
    # directory_path = '/home/jiezou/ros2_ws/Data/ros2/*'
    directory_path = '/home/jiezou/ros2_ws/Data/ros2_2/*'

    # Find all CSV files in the directory using glob
    csv_files = glob.glob(directory_path)

    workload_list = []
    # Iterate over each CSV file
    for csv_file_path in csv_files:
        # Open the CSV file in read mode
        file_name = os.path.basename(csv_file_path)
        with open(csv_file_path, 'r') as file:
            data = [[i.strip() for i in l.split(":")] for l in file.readlines()]
        data_dict = {}
        for d in data:
            entry = d[0]
            value = float(d[1]) if entry in ["ExecutionTime", "Task Period", "Task Uti"] else int(d[1])
            if entry not in data_dict:
                data_dict[entry] = [value]
            else:
                data_dict[entry].append(value)
        data_dict = {k: np.array(v) for k, v in data_dict.items()}

        if file_name in ['core_PMC.csv', 'sha_PMC.csv', 'zip_PMC.csv', 'parser_PMC.csv', 'core_E.csv', 'sha_E.csv', 'zip_E.csv', 'parser_E.csv']:
            # ['core_PMC.csv', 'sha_PMC.csv', 'zip_PMC.csv', 'parser_PMC.csv', 'radix_PMC.csv', 'linear_PMC.csv', 'nnest_PMC', 'loops_PMC.csv']
            category = 0  # integer workloads
        else:
            category = 1  # floating-point workloads

        workload = Workload(file_name, data_dict, category)
        workload_list.append(workload)

    return workload_list

def construct_dataframe(workload_list):
    logging.debug("Performing normalisation on the workload data (list of Workload).")

    # Filter out non-feature keys.
    # features = [i for i in workload_list[0].data_dict.keys() if i not in
    #             ["Round"]]

    features = [i for i in workload_list[0].data_dict.keys() if i not in
                ["Round", "Task Period", "Task Uti", "ExecutionTime", 'ExecutionStart time point', "ExecutionStart end point", "L1 dcache load", "Instructions",
                 "Reference cycles", "Task clock", "Reference cycles", "CPU clock", "Slots", 'CPU cycles']]

    data = []
    for workload in workload_list:
        # Create DataFrame from dict.
        data_df = pd.DataFrame(workload.data_dict)

        # Make the slicing.
        logging.debug("Data range: 300 to 3300.")
        data_df = data_df.query('300 < Round <= 3300').copy()
        assert len(data_df) == 3000, "Data length is not 3000."

        # Add the workload name and category.
        data_df["workload_name"] = workload.workload_name
        data_df["class_labels"] = workload.category

        data.append(data_df)

    data = pd.concat(data)
    data = data.reset_index()
    # coefficients_visualization(data, features)

    return data, features
