import os
import glob
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import seaborn as sns

scaler_normalize = MinMaxScaler()
scaler_standardize = StandardScaler()


def Normalization(array):
    reshaped_array = [[value] for value in array]
    normalized_data = scaler_normalize.fit_transform(reshaped_array)
    normalized_array = [value[0] for value in normalized_data]
    return normalized_array

def Standardization(array):
    reshaped_array = [[value] for value in array]
    standardized_data = scaler_standardize.fit_transform(reshaped_array)
    standardized_array = [value[0] for value in standardized_data]
    return standardized_array

# Specify the path to your CSV file
# directory_path = '/home/jiezou/ros2_ws/Data/ros2/*.csv'
directory_path = '/home/jiezou/ros2_ws/Data/ros2/*'

# Find all CSV files in the directory using glob
csv_files = glob.glob(directory_path)
# Iterate over each CSV file
# file_list = []
for csv_file_path in csv_files:
    # Open the CSV file in read mode
    file_name = os.path.basename(csv_file_path)
    # file_list.append(file_name)

    with open(csv_file_path, 'r') as file:
        data = [[i.strip() for i in l.split(":")] for l in file.readlines()]
    data_dict = {}
    for d in data:
        entry = d[0]
        value = int(d[1]) if entry != "ExecutionTime" else float(d[1])
        if entry not in data_dict:
            data_dict[entry] = [value]
        else:
            data_dict[entry].append(value)

    # print(data_dict["Round"])
    # print(data_dict["ExecutionTime"])
    # print(Normalization(data_dict["ExecutionTime"]))
    # print(data_dict["L1 dcache load misses"])
    # print(data_dict["L1 dcache load"])
    # print(data_dict["L1 icache load misses"])
    # print(data_dict["LLC load misses"])
    # print(data_dict["Branch misses"])
    # print(data_dict["Branch instructions"])
    # print(data_dict["CPU cycles"])
    # print(data_dict["Instructions"])
    # print(data_dict["Reference cycles"])
    # print(data_dict["Bus cycles"])
    # print(data_dict["CPU clock"])
    # print(data_dict["Task clock"])
    # print(data_dict["Slots"])

    coefficients = np.corrcoef([data_dict["ExecutionTime"], data_dict["L1 dcache load misses"],
                                data_dict["L1 dcache load"], data_dict["L1 icache load misses"],
                                data_dict["LLC load misses"], data_dict["Branch misses"],
                                data_dict["Branch instructions"], data_dict["Instructions"],
                                data_dict["CPU cycles"], data_dict["Reference cycles"], data_dict["Bus cycles"],
                                data_dict["CPU clock"], data_dict["Task clock"], data_dict["Slots"]])
    # coefficients = np.corrcoef([Normalization(data_dict["ExecutionTime"]), Normalization(data_dict["L1 dcache load misses"]),
    #                             Normalization(data_dict["L1 dcache load"]), Normalization(data_dict["L1 icache load misses"]),
    #                             Normalization(data_dict["LLC load misses"]), Normalization(data_dict["Branch misses"]),
    #                             Normalization(data_dict["Branch instructions"]), Normalization(data_dict["Instructions"]),
    #                             Normalization(data_dict["CPU cycles"]), Normalization(data_dict["Reference cycles"]), Normalization(data_dict["Bus cycles"]),
    #                             Normalization(data_dict["CPU clock"]), Normalization(data_dict["Task clock"]), Normalization(data_dict["Slots"])])
    # print("-------------------------------------------------")

    # print(file_name)
    # print("-------------------------------------------------")
    # print(coefficients)
    #
    # xlabel = ["ExecutionTime", "L1 dcache load misses", "L1 dcache load", "L1 icache load misses", "LLC load misses",
    #           "Branch misses", "Branch instructions", "Instructions", "CPU cycles", "Reference cycles", "Bus cycles",
    #           "CPU clock", "Task clock", "Slots"]
    # ylabel = ["ExecutionTime", "L1 dcache load misses", "L1 dcache load", "L1 icache load misses", "LLC load misses",
    #           "Branch misses", "Branch instructions", "Instructions", "CPU cycles", "Reference cycles", "Bus cycles",
    #           "CPU clock", "Task clock", "Slots"]
    #
    # sns.heatmap(coefficients, annot=True, xticklabels=xlabel, yticklabels=ylabel, cmap="coolwarm")
    #
    # plt.title(f"Coefficient Heatmap - {file_name}")
    #
    # # Show the plot
    # plt.show()

    # plt.suptitle(file_name)
    # plt.subplot(3, 5, 1)
    # plt.plot(data_dict["Round"], Normalization(data_dict["ExecutionTime"]), label= "ExecutionTime")
    # plt.xlabel('Round')
    # plt.ylabel('Normalized value')
    # plt.title('ExecutionTime')
    #
    # plt.subplot(3, 5, 2)
    # plt.plot(data_dict["Round"], Normalization(data_dict["L1 dcache load misses"]), label='L1 dcache load misses')
    # plt.xlabel('Round')
    # plt.ylabel('Normalized value')
    # plt.title('L1 dcache load misses')
    #
    # plt.subplot(3, 5, 3)
    # plt.plot(data_dict["Round"], Normalization(data_dict["L1 dcache load"]), label="L1 dcache load")
    # plt.xlabel('Round')
    # plt.ylabel('Normalized value')
    # plt.title('L1 dcache load')
    #
    # plt.subplot(3, 5, 4)
    # plt.plot(data_dict["Round"], Normalization(data_dict["L1 icache load misses"]), label= "L1 icache load misses")
    # plt.xlabel('Round')
    # plt.ylabel('Normalized value')
    # plt.title('L1 icache load misses')
    #
    # plt.subplot(3, 5, 5)
    # plt.plot(data_dict["Round"], Normalization(data_dict["LLC load misses"]), label= "LLC load misses")
    # plt.xlabel('Round')
    # plt.ylabel('Normalized value')
    # plt.title('LLC load misses')
    #
    # plt.subplot(3, 5, 6)
    # plt.plot(data_dict["Round"], Normalization(data_dict["Branch misses"]), label="Branch misses")
    # plt.xlabel('Round')
    # plt.ylabel('Normalized value')
    # plt.title('Branch misses')
    #
    # plt.subplot(3, 5, 7)
    # plt.plot(data_dict["Round"], Normalization(data_dict["Branch instructions"]), label="Branch instructions")
    # plt.xlabel('Round')
    # plt.ylabel('Normalized value')
    # plt.title('Branch instructions')
    #
    # plt.subplot(3, 5, 8)
    # plt.plot(data_dict["Round"], Normalization(data_dict["Instructions"]), label="Instructions")
    # plt.xlabel('Round')
    # plt.ylabel('Normalized value')
    # plt.title('Instructions')
    #
    # plt.subplot(3, 5, 9)
    # plt.plot(data_dict["Round"], Normalization(data_dict["Bus cycles"]), label="Bus cycles")
    # plt.xlabel('Round')
    # plt.ylabel('Normalized value')
    # plt.title('Bus cycles')
    #
    # plt.subplot(3, 5, 10)
    # plt.plot(data_dict["Round"], Normalization(data_dict["CPU cycles"]), label="CPU cycles")
    # plt.xlabel('Round')
    # plt.ylabel('Normalized value')
    # plt.title('CPU cycles')
    #
    # plt.subplot(3, 5, 11)
    # plt.plot(data_dict["Round"], Normalization(data_dict["Reference cycles"]), label="Reference cycles")
    # plt.xlabel('Round')
    # plt.ylabel('Normalized value')
    # plt.title('Reference cycles')
    #
    # plt.subplot(3, 5, 12)
    # plt.plot(data_dict["Round"], Normalization(data_dict["CPU clock"]), label="CPU clock")
    # plt.xlabel('Round')
    # plt.ylabel('Normalized value')
    # plt.title('CPU clock')
    #
    # plt.subplot(3, 5, 13)
    # plt.plot(data_dict["Round"], Normalization(data_dict["Task clock"]), label="Task clock")
    # plt.xlabel('Round')
    # plt.ylabel('Normalized value')
    # plt.title('Task clock')
    #
    # plt.subplot(3, 5, 14)
    # plt.plot(data_dict["Round"], Normalization(data_dict["Slots"]), label="Slots")
    # plt.xlabel('Round')
    # plt.ylabel('Normalized value')
    # plt.title('Slots')
    #
    # # plt.legend()
    # plt.show()


    # quit()


data = np.random.rand(2, 10, 3)

class_labels = np.random.randint(2, size=8)
temp = np.full(10, 1)

print(data)
print(class_labels)