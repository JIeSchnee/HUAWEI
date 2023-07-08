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
from sklearn.metrics import accuracy_score
import numpy as np
from scipy.stats import pearsonr
from scipy.signal import decimate, hilbert

from PythonCode.data_io import get_data, construct_dataframe
from PythonCode.Feature_identification import preprocess
import seaborn as sns


def get_features(data, feature_list):
    """
    Extracts the features from the data
    :param data: Dataframe containing the data
    :param feature_list: List of features to be extracted
    :return: Dataframe containing the extracted features
    """
    feature_data = []
    for feature in feature_list:
        feature_data.append(data[feature].to_numpy())
    return feature_data


class classifiers:
    def __init__(self, selected_features, class_labels):
        classifier_acc_dict = {}

        self.classifier_list = {
            "Logistic Regression": LogisticRegression(random_state=42),
            "Support Vector Machine": SVC(kernel="rbf", random_state=42),
        }

        for classifier_name, classifier in self.classifier_list.items():
            classifier.fit(selected_features, class_labels)
            pred = classifier.predict(selected_features)
            accuracy = accuracy_score(class_labels, pred)
            classifier_acc_dict[classifier_name] = accuracy
            print("Accuracy of {} is {}".format(classifier_name, accuracy))

def  workloads_classify(new_data, feature_list, new_features, cs, pca, PCA_cs, method):
    # new_data, _ = construct_dataframe(new_workload_list)
    # new_data, _ = preprocess(new_data, feature_list)
    workloadclass = {}
    if method == "selected features":
        new_predicted = cs.classifier_list["Support Vector Machine"].predict(new_data[feature_list].to_numpy())
        new_predicted_1 = cs.classifier_list["Logistic Regression"].predict(new_data[feature_list].to_numpy())
    elif method == "PCA":
        new_predicted = PCA_cs.classifier_list["Support Vector Machine"].predict(
            pca.transform(new_data[new_features].to_numpy()))
        new_predicted_1 = PCA_cs.classifier_list["Logistic Regression"].predict(
            pca.transform(new_data[new_features].to_numpy()))

    # print(f"workload_classified using {method}")
    temp = {}
    for i in list(new_data["workload_name"].unique()):
        p = new_predicted[new_data["workload_name"].to_numpy() == i]
        p_1 = new_predicted_1[new_data["workload_name"].to_numpy() == i]
        # print("=====================================")
        # print(f"According to Support Vector Machine: {np.sum(p == 0) / len(p) * 100}% of the new workloads in {i} are predicted to be 0.")
        # print(f"According to Logistic Regression: {np.sum(p_1 == 0) / len(p_1) * 100}% of the new workloads in {i} are predicted to be 0.")
        if np.sum(p == 0) / len(p) > 0.5 and np.sum(p_1 == 0) / len(p_1) > 0.5:
            # print("The workload is predicted to be 0.")
            temp[i] = 0
        else:
            # print("The workload is predicted to be 1.")
            temp[i] = 1
        workloadclass[method] = temp
    return workloadclass

def main():
    # Get the Training Data
    workload_list = get_data(directory_path='/home/jiezou/ros2_ws/Data/ros2_2/*')
    # n_workloads = len(workload_list)
    data, features = construct_dataframe(workload_list)
    data, features = preprocess(data, features)
    class_labels = data["class_labels"].to_numpy()
    # logging.info("Features: {}".format(features))
    # logging.info("Shape: {}".format(data[features].shape))
    # Train the classifiers based on the selected features
    print("Train the classifiers based on the selected features")
    feature_list = ['L1 dcache load misses', 'L1 icache load misses', 'Branch instructions', 'Branch misses', 'Bus cycles']
    feature_data = data[feature_list].to_numpy()
    cs = classifiers(feature_data, class_labels)
    # Use PCA to reduce the dimensionality of the data and train the classifiers
    print("Use PCA to reduce the dimensionality of the data and train the classifiers")
    pca = PCA(n_components=3)
    transformed_data = pca.fit_transform(data[features].to_numpy())
    PCA_cs = classifiers(transformed_data, class_labels)

    # Get the Testing Data
    new_workload_list = get_data(directory_path='/home/jiezou/ros2_ws/Data/ros2_test/*')
    new_data, new_features = construct_dataframe(new_workload_list)
    new_data, new_features = preprocess(new_data, new_features)
    # Classify the new workloads from the test data
    workload_class_selected_features = workloads_classify(new_data, feature_list, new_features, cs, pca, PCA_cs, "selected features")
    # Classify the new workloads from the test data using PCA
    # print("=====================================")
    workload_class_PCA_feature= workloads_classify(new_data, feature_list, new_features, cs, pca, PCA_cs, "PCA")

    # print("=====================================")
    print(workload_class_selected_features)
    print(workload_class_PCA_feature)



if __name__ == "__main__":
    logger = logging.getLogger()
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(format=FORMAT)
    logger.setLevel(logging.INFO)
    main()
