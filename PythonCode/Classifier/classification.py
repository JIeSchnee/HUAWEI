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
    print("##########################################")
    workloadclass = {}
    if method == "selected features":
        new_predicted = cs.classifier_list["Support Vector Machine"].predict(new_data[feature_list].to_numpy())
        new_predicted_1 = cs.classifier_list["Logistic Regression"].predict(new_data[feature_list].to_numpy())
        #  plot pr curve for SVM
        precision, recall, thresholds = precision_recall_curve(new_data["class_labels"].to_numpy(), new_predicted)
        auc_score = auc(recall, precision)
        print("Based on selected features, AUC score for SVM is {}".format(auc_score))
        # plt.plot(recall, precision, label="SVM")
        # plot pr curve for LR
        precision, recall, thresholds = precision_recall_curve(new_data["class_labels"].to_numpy(), new_predicted_1)
        auc_score = auc(recall, precision)
        print("AUC score for LR is {}".format(auc_score))
        # plt.plot(recall, precision, label="LR")
        # plt.legend()
        # plt.title(f"PR curve of the new workloads based on selected features")
        # plt.show()

    elif method == "PCA":
        new_predicted = PCA_cs.classifier_list["Support Vector Machine"].predict(
            pca.transform(new_data[new_features].to_numpy()))
        new_predicted_1 = PCA_cs.classifier_list["Logistic Regression"].predict(
            pca.transform(new_data[new_features].to_numpy()))

        # #  plot pr curve for SVM
        precision, recall, thresholds = precision_recall_curve(new_data["class_labels"].to_numpy(), new_predicted)
        auc_score = auc(recall, precision)
        print("Based on PCA, AUC score for SVM is {}".format(auc_score))
        # plt.plot(recall, precision, label="SVM")
        # # plot pr curve for LR
        precision, recall, thresholds = precision_recall_curve(new_data["class_labels"].to_numpy(), new_predicted_1)
        auc_score = auc(recall, precision)
        print("AUC score for LR is {}".format(auc_score))
        # plt.plot(recall, precision, label="LR")
        # plt.legend()
        # plt.title(f"PR curve of the new workloads based on PCA")
        # plt.show()

        # plt.scatter(range(len(new_predicted)), new_predicted, label="SVM")
        # plt.scatter(range(len(new_predicted_1)), new_predicted_1, label="LR")
        # plt.legend()
        # plt.title(f"Predicted results of each data point in the new workloads")
        # plt.show()

    print(f"workload_classified using {method}")
    temp = {}
    for i in list(new_data["workload_name"].unique()):
        p = new_predicted[new_data["workload_name"].to_numpy() == i]
        p_1 = new_predicted_1[new_data["workload_name"].to_numpy() == i]

        print("=====================================")
        print(f"According to Support Vector Machine: {np.sum(p == 0) / len(p) * 100}% of the new workloads in {i} are predicted to be 0.")
        print(f"According to Logistic Regression: {np.sum(p_1 == 0) / len(p_1) * 100}% of the new workloads in {i} are predicted to be 0.")


        # plt.scatter(range(len(p)), p, label="SVM")
        # # plt.scatter(range(len(p_1)), p_1, label="LR")
        # plt.legend()
        # plt.title(f"Predicted results of each data point in the new workloads in {i}")
        # plt.show()
        if np.sum(p == 0) / len(p) > 0.80 and np.sum(p_1 == 0) / len(p_1) > 0.80:
            # print("The workload is predicted to be 0.")
            temp[i] = 0
        else:
            # print("The workload is predicted to be 1.")
            temp[i] = 1
        workloadclass[method] = temp
    return workloadclass

def main():
    # Get the Training Data
    workload_list = get_data(directory_path='/home/jiezou/ros2_ws/Data/ros2_train/*')
    # n_workloads = len(workload_list)
    data, features = construct_dataframe(workload_list)
    data, features, normaliser = preprocess(data, features)
    class_labels = data["class_labels"].to_numpy()
    # logging.info("Features: {}".format(features))
    # logging.info("Shape: {}".format(data[features].shape))
    # Train the classifiers based on the selected features
    print("Train the classifiers based on the selected features")
    # feature_list = ['L1 dcache load misses', 'L1 icache load misses', 'Branch instructions', 'Bus cycles', 'Branch misses']
    feature_list = ['L1 dcache load misses', 'Branch instructions', 'L1 icache load misses', 'Bus cycles', 'Branch misses']
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
    new_data, new_features = preprocess(new_data, new_features, normaliser=normaliser)
    # Classify the new workloads from the test data
    workload_class_selected_features = workloads_classify(new_data, feature_list, new_features, cs, pca, PCA_cs, "selected features")
    # Classify the new workloads from the test data using PCA
    # print("=====================================")
    workload_class_PCA_feature= workloads_classify(new_data, feature_list, new_features, cs, pca, PCA_cs, "PCA")

    print("==============================================\n")
    print("================ Final results ===============")
    print("==============================================\n")

    print("Ground True integer workloads Class 0: core, sha, zip, parser")
    print("Ground True floating-point workloads Class 1: radix, linear, nnest, loops")
    print(workload_class_selected_features)
    print(workload_class_PCA_feature)



if __name__ == "__main__":
    logger = logging.getLogger()
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(format=FORMAT)
    logger.setLevel(logging.INFO)
    main()
