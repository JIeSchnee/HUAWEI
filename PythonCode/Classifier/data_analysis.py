import logging
import pickle
from pprint import pformat

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE, chi2
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_curve, auc
import numpy as np
from scipy.stats import pearsonr
from scipy.signal import decimate, hilbert

from PythonCode.data_io import get_data, construct_dataframe
from PythonCode.Feature_identification import preprocess
import seaborn as sns


def main():
    ##########################################
    # Load all data.
    ##########################################
    workload_list = get_data(directory_path='/home/jiezou/ros2_ws/Data/ros2_2/*')
    data, features = construct_dataframe(workload_list)
    # data, features, normaliser = preprocess(data, features)
    feature_list = ['L1 dcache load misses', 'L1 icache load misses', 'Branch instructions', 'Bus cycles', 'Branch misses']
    new_workload_list = get_data(directory_path='/home/jiezou/ros2_ws/Data/ros2_test/*')
    new_data, new_features = construct_dataframe(new_workload_list)
    # new_data, new_features = preprocess(new_data, new_features, normaliser=normaliser)

    ##########################################
    # Information.
    ##########################################
    logger.info("Training set all classes: {}".format(data['workload_name'].unique()))
    logger.info("Test set all classes: {}".format(new_data['workload_name'].unique()))

    ##########################################
    # Analytical.
    ##########################################
    cmp1_name = "sha_E.csv"
    cmp2_name = "sha_Enemy.csv"
    cmp_ratio = 50 / 20

    cmp1 = data.query(f"workload_name == '{cmp1_name}'").copy().reset_index()
    cmp2 = new_data.query(f"workload_name == '{cmp2_name}'").copy().reset_index()
    print((cmp1.loc[:, cmp1.columns != 'workload_name'] .mean() / cmp2.loc[:, cmp2.columns != 'workload_name'].mean()) - cmp_ratio)

    ##########################################
    # Visualise the data.
    ##########################################
    data.query(f"workload_name == '{cmp1_name}'").hist()
    plt.suptitle(cmp1_name)
    new_data.query(f"workload_name == '{cmp2_name}'").hist()
    plt.suptitle(cmp2_name)
    plt.show()

    data, features, normaliser = preprocess(data, features, normaliser=None)
    new_data, new_features = preprocess(new_data, new_features, normaliser=normaliser)
    print(data.loc[:, data.columns != 'workload_name'][feature_list].std())
    print(new_data.loc[:, new_data.columns != 'workload_name'][feature_list].std())


if __name__ == "__main__":
    logger = logging.getLogger()
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(format=FORMAT)
    logger.setLevel(logging.INFO)
    main()
