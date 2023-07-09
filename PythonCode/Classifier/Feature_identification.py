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

import seaborn as sns


def coefficients_visualization(data, features):

    coefficients = np.corrcoef([data["Task Period"],
                                data["ExecutionStart time point"],
                                data["L1 dcache load misses"], data["L1 icache load misses"],
                                data["LLC load misses"], data["Branch misses"],
                                data["Branch instructions"], data["Bus cycles"]])

    xlabel = ['Task Period', 'ExecutionStart time point', 'L1 dcache load misses', 'L1 icache load misses',
              'LLC load misses', 'Branch misses', 'Branch instructions', 'Bus cycles']
    ylabel = ['Task Period', 'ExecutionStart time point', 'L1 dcache load misses', 'L1 icache load misses',
              'LLC load misses', 'Branch misses', 'Branch instructions', 'Bus cycles']


    sns.heatmap(coefficients, annot=True, xticklabels=xlabel, yticklabels=ylabel, cmap="coolwarm")

    plt.title("Correlation Coefficient Heatmap")

    # Show the plot
    plt.show()

def preprocess(data, features, normaliser=None):
    # Normalise w.r.t. the CPU cycles.
    cpu_cycle_norm_features = ["L1 dcache load misses", "L1 icache load misses", "LLC load misses",
                               "Branch misses", "Branch instructions", "Bus cycles", "CPU cycles"]
    data[cpu_cycle_norm_features] = data[cpu_cycle_norm_features].div(data["Instructions"], axis=0)

    # Normalise all features to 0-1.
    if normaliser is None:
        normaliser = MinMaxScaler()
        data[features] = normaliser.fit_transform(data[features])

        return data, features, normaliser
    else:
        data[features] = normaliser.transform(data[features])

        return data, features
    # coefficients_visualization(data, features)

#
# def logistic_regression_classifier(selected_features_2, class_labels, method):
#     x_train, x_test, y_train, y_test = train_test_split(selected_features_2, class_labels, test_size=0.2,
#                                                         random_state=42)
#     model = LogisticRegression()
#     model.fit(x_train, y_train)
#     y_pred = model.predict(x_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     print(method, "Accuracy:", accuracy)
#
#     return accuracy
#

def classifiers(selected_features, class_labels):
    classifier_acc_dict = {}
    x_train, x_test, y_train, y_test = train_test_split(selected_features, class_labels, test_size=1/8,
                                                        random_state=42, shuffle=True)

    classifier_list = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Support Vector Machine": SVC(kernel="rbf", random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42)
    }

    for classifier_name, classifier in classifier_list.items():
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        classifier_acc_dict[classifier_name] = accuracy

    return classifier_acc_dict


def univariate_feature_selection(methods_dict, reshaped_data, class_labels, features, k):
    logging.debug("Feature selection based on Univariate Feature Selection")
    selector = SelectKBest(score_func=chi2, k=k)
    selected_features = selector.fit_transform(reshaped_data, class_labels)
    mask = selector.get_support()
    selected_feature_indices = np.where(mask)[0]
    selected_features_names = [features[i] for i in selected_feature_indices]
    # print("features in the function", Features)
    logging.info("Univariate Feature Selection Selected Features: {}".format(selected_features_names))
    methods_dict["Univariate Feature Selection"] = [selected_features_names]

    return methods_dict, selected_features


def recursive_feature_elimination(Methods_dict, reshaped_data, class_labels, features, k):
    logging.debug("Feature selection based on Recursive Feature Elimination (RFE)")
    estimator = SVC(kernel="linear")
    rfe = RFE(estimator=estimator, n_features_to_select=k)
    selected_features = rfe.fit_transform(reshaped_data, class_labels)
    selected_feature_indices = rfe.get_support(indices=True)
    selected_features_names = [features[i] for i in selected_feature_indices]
    logging.info("Recursive Feature Elimination Selected Features: {}".format(selected_features_names))
    Methods_dict["Recursive Feature Elimination"] = [selected_features_names]
    return Methods_dict, selected_features


def feature_data_visualization(selected_features, method, class_labels):
    # print("@@@@@", selected_features.shape)
    if selected_features.shape[1] < 3:
        logging.warning("Visualization is only possible for at least 3 features.")
        return
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(selected_features[:, 0], selected_features[:, 1], selected_features[:, 2], c=class_labels)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    ax.set_title(f'Visualization (3D) {method}')
    plt.show()


def fft_data_analysis(data_array):
    # Step 1: Compute FFT for each workload and feature
    fft_data = np.fft.fft(data_array, axis=1)
    logging.info("fft_data Shape: {}".format(fft_data.shape))

    # Take the absolute values of the FFT results to obtain the magnitudes
    fft_magnitudes = np.abs(fft_data)
    logging.info("fft_magnitudes Shape: {}".format(fft_magnitudes.shape))

    # Determine the number of workloads, features, and data points
    n_workloads, n_data_points, n_features = data_array.shape

    # Normalize the FFT magnitudes for each workload and feature
    normalized_data = np.zeros((n_workloads, n_data_points, n_features))
    for workload_idx in range(n_workloads):
        for feature_idx in range(n_features):
            magnitudes = fft_magnitudes[workload_idx, :, feature_idx]

            # Normalize the magnitudes
            scaler = MinMaxScaler()
            normalized_magnitudes = scaler.fit_transform(magnitudes.reshape(-1, 1)).flatten()

            # Store the normalized magnitudes in the normalized data array
            normalized_data[workload_idx, :, feature_idx] = normalized_magnitudes

    # Define the downsampling factor
    downsampling_factor = 5
    # Downsampling the FFT magnitudes for each workload and feature
    downsampled_data = np.zeros((n_workloads, n_data_points // downsampling_factor, n_features))
    for workload_idx in range(n_workloads):
        for feature_idx in range(n_features):
            magnitudes = normalized_data[workload_idx, :, feature_idx]
            # Apply decimation to downsample the magnitudes
            downsampled_magnitudes = decimate(magnitudes, downsampling_factor)
            # Store the downsampled magnitudes in the downsampled data array
            downsampled_data[workload_idx, :, feature_idx] = downsampled_magnitudes

    # logging.info("downsampled_magnitudes Shape: {}".format(downsampled_data.shape))

    # Plot the FFT magnitudes for each workload and feature
    for workload_idx in range(n_workloads):
        for feature_idx in range(n_features):
            magnitudes = normalized_data[workload_idx, :, feature_idx]
            # Frequency axis
            sampling_rate = 1  # Modify this if your data has a specific sampling rate
            freq_axis = np.fft.fftfreq(n_data_points, d=1 / sampling_rate)

            # # Create a new plot for each combination
            # plt.figure()
            # plt.plot(freq_axis[:n_data_points // 2], magnitudes[:n_data_points // 2])
            # plt.xlabel('Frequency')
            # plt.ylabel('Magnitude')
            # plt.title(f'Workload {workload_idx + 1}, Feature {feature_idx + 1}')
            # plt.show()

            downsampled_magnitudes = downsampled_data[workload_idx, :, feature_idx]
            # Frequency axis for downsampled data
            sampling_rate = 1  # Modify this if your data has a specific sampling rate
            downsampled_freq_axis = np.fft.fftfreq(n_data_points // downsampling_factor,
                                                   d=1 / (sampling_rate * downsampling_factor))

            # # Create a new plot for each combination
            # plt.figure()
            # plt.plot(downsampled_freq_axis[:len(downsampled_magnitudes) // 2],
            #          downsampled_magnitudes[:len(downsampled_magnitudes) // 2])
            # plt.xlabel('Frequency')
            # plt.ylabel('Magnitude')
            # plt.title(f'Workload {workload_idx + 1}, Feature {feature_idx + 1} (Downsampled)')
            # plt.show()

    return normalized_data


def spectral_enveloping_analysis(fft_magnitudes):
    # Calculate the envelope for each FFT magnitude spectrum
    envelope_data = np.zeros_like(fft_magnitudes)
    for workload_idx in range(fft_magnitudes.shape[0]):
        for feature_idx in range(fft_magnitudes.shape[2]):
            magnitudes = fft_magnitudes[workload_idx, :, feature_idx]

            # Compute the analytical signal using Hilbert transform
            analytic_signal = hilbert(magnitudes)

            # Calculate the envelope as the magnitude of the analytic signal
            envelope = np.abs(analytic_signal)

            # Store the envelope in the envelope data array
            envelope_data[workload_idx, :, feature_idx] = envelope

    # Normalize the spectral enveloping
    n_workloads, n_data_points, n_features = fft_magnitudes.shape
    normalized_envelope_data = np.zeros_like(envelope_data)
    for workload_idx in range(n_workloads):
        for feature_idx in range(n_features):
            envelope = envelope_data[workload_idx, :, feature_idx]

            # Normalize the envelope using Min-Max scaling
            scaler = MinMaxScaler()
            normalized_envelope = scaler.fit_transform(envelope.reshape(-1, 1)).flatten()

            # Store the normalized envelope in the normalized envelope data array
            normalized_envelope_data[workload_idx, :, feature_idx] = normalized_envelope

    # Plot the spectral enveloping for each workload and feature
    for workload_idx in range(n_workloads):
        for feature_idx in range(n_features):
            envelope = normalized_envelope_data[workload_idx, :, feature_idx]

            # Create a new plot for each combination
            # plt.figure()
            # plt.plot(envelope)
            # plt.xlabel('Data Point')
            # plt.ylabel('Envelope')
            # plt.title(f'Workload {workload_idx + 1}, Feature {feature_idx + 1} - Spectral Enveloping')
            # plt.show()
    return normalized_envelope_data


def features_analysis_based_selection(data, features, n_workloads):
    logging.info("=== Features Analysis ===")

    d_flat = data[features].to_numpy()
    classes_reshaped = data["class_labels"].to_numpy()

    d = d_flat.reshape(n_workloads, -1, len(features))
    n_workloads, n_data_points, n_features = d.shape

    # Calculate the F-ratio score for each feature
    f_ratio_scores, _ = f_classif(d_flat, classes_reshaped)
    # print("f_ratio_scores", f_ratio_scores, len(f_ratio_scores))

    # Calculate the Trace ratio score for each feature
    within_class_scatter = np.zeros(n_features)
    between_class_scatter = np.zeros(n_features)
    for feature_idx in range(n_features):
        for workload_idx in range(n_workloads):
            class_data = d[workload_idx, :, feature_idx]
            class_mean = np.mean(class_data)
            overall_mean = np.mean(d[:, :, feature_idx])
            class_scatter = np.sum((class_data - class_mean) ** 2)
            within_class_scatter[feature_idx] += class_scatter
            between_class_scatter[feature_idx] += len(class_data) * ((class_mean - overall_mean) ** 2)

    trace_ratio_scores = between_class_scatter / within_class_scatter

    # # Print the separability scores for each feature
    # for feature_idx in range(n_features):
    #     print(
    #         f"Feature {Features[feature_idx]}: F-Ratio Score = {f_ratio_scores[feature_idx]:.4f}, Trace Ratio Score = {trace_ratio_scores[feature_idx]:.4f}")

    # TODO: Standardize the data or not can be tested
    # scaler = StandardScaler()
    # data_standardized = scaler.fit_transform(data_reshaped)

    k = int(np.ceil(n_features/2))  # Number of features to keep

    # Select features based on F-ratio scores and remove 10 features with the lowest scores.
    selector = SelectKBest(f_classif, k=k)
    F_selected_data = selector.fit_transform(d_flat, classes_reshaped)
    F_selected_feature_indices = selector.get_support(indices=True)
    F_selected_features_names = [features[i] for i in F_selected_feature_indices]
    logging.info("F Selected Features: {}".format(F_selected_features_names))

    # Select features based on Trace Ratio scores and remove 10 features with the lowest scores
    T_selected_feature_indices = np.argsort(trace_ratio_scores)[-k:]
    T_selected_features_names = [features[i] for i in T_selected_feature_indices]
    logging.info("T Selected Features: {}".format(T_selected_features_names))

    # Select the indices appeared in both F and T list.
    common_elements = np.intersect1d(F_selected_feature_indices, T_selected_feature_indices)
    selected_features_names = [features[i] for i in common_elements]
    # print("Feature indices: ", common_elements)
    logging.info("Consistent Features' names: {}".format(selected_features_names))
    logging.info("=" * 50)

    return selected_features_names


def add_fft_data_as_feature(data, features, n_workloads):
    logging.info("=== FFT data analysis and Add new feature ===")

    fft_magnitudes = fft_data_analysis(data[features].to_numpy().reshape(n_workloads, -1, len(features), order="F"))

    # Append the magnitudes as a new feature
    new_features = [f + "FFT" for f in features]
    fft_df = pd.DataFrame(fft_magnitudes.reshape(-1, len(features)), columns=new_features)
    fft_df = fft_df.reset_index()
    data[new_features] = fft_df[new_features]
    assert np.all(data[new_features].to_numpy() == fft_df[new_features].to_numpy()), "FFT data is not added correctly."
    assert np.all(data["class_labels"].to_numpy()[:8] == data["class_labels"][0]), "Class labels are messed up."

    features += new_features

    logging.info("New Data Shape: {}".format(data[features].shape))
    logging.info("New feature list: {}".format(features))

    return data, fft_magnitudes, features


def add_spectral_enveloping_data_as_feature(data, fft_magnitudes, features):
    logging.info("=== Spectral Enveloping data analysis ===")
    envelope_data = spectral_enveloping_analysis(fft_magnitudes)

    # Concatenate the envelope data as a new feature.
    new_features = [f + "_SE" for f in features if f.endswith("FFT")]
    fft_se_df = pd.DataFrame(envelope_data.reshape(-1, len(new_features)), columns=new_features)
    fft_se_df = fft_se_df.reset_index()
    data[new_features] = fft_se_df[new_features]
    features += new_features

    logging.info("New Data with Envelope Shape: {}".format(data[features].shape))

    logging.info("new feature list: {}".format(features))

    return data, features


def key_features_identification(data, features, k, visualisation=True):
    logging.info("=== Key Features Identification ===")

    methods_dict = {}
    methods_acc_dict = {}

    class_labels = data["class_labels"].to_numpy()
    reshaped_data = data[features].to_numpy()

    # # print("_class_labels Data Shape:", _class_labels.shape)
    # print(Features)
    # print(len(Features))
    # print("=====================================")
    # print(reshaped_data.shape)

    # Algorithm 1: Univariate Feature Selection
    methods_dict, selected_features = univariate_feature_selection(methods_dict, reshaped_data, class_labels, features,
                                                                   k)
    # print(":::::", selected_features.shape)

    # accuracy = LogisticRegression_classifier(selected_features_1, _class_labels, "Univariate Feature Selection")
    if visualisation:
        feature_data_visualization(selected_features, "Univariate Feature Selection", class_labels)
    classifier_acc_dic = classifiers(selected_features, class_labels)
    methods_acc_dict["Univariate Feature Selection"] = classifier_acc_dic

    # Algorithm 2: Recursive Feature Elimination (RFE)
    methods_dict, selected_features = recursive_feature_elimination(methods_dict, reshaped_data, class_labels, features,
                                                                    k)
    if visualisation:
        feature_data_visualization(selected_features, "Recursive Feature Elimination", class_labels)
    classifier_acc_dic = classifiers(selected_features, class_labels)
    methods_acc_dict["Recursive Feature Elimination"] = classifier_acc_dic

    # Algorithm 3: Tree-based Feature Selection
    # print("Feature selection based on Tree-based Methods")
    # method 1: RandomForest
    # print("RandomForest")
    model = RandomForestClassifier()
    model.fit(reshaped_data, class_labels)
    feature_importances = model.feature_importances_
    top_feature_indices = np.argsort(feature_importances)[::-1][:k]
    selected_features_names_3 = [features[i] for i in top_feature_indices]
    logging.info("RandomForest Selected Features: {}".format(selected_features_names_3))
    methods_dict["RandomForest"] = [selected_features_names_3]

    selected_data = reshaped_data[:, top_feature_indices]
    if visualisation:
        feature_data_visualization(selected_data, "RandomForest", class_labels)
    classifier_acc_dic = classifiers(selected_data, class_labels)
    methods_acc_dict["RandomForest"] = classifier_acc_dic

    # method 2: DecisionTree
    # print("DecisionTree")
    model = DecisionTreeClassifier(random_state=42)
    model.fit(reshaped_data, class_labels)
    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    key_features = sorted_indices[:k]
    selected_features_names = [features[i] for i in key_features]
    logging.info("DecisionTree Selected Features: {}".format(selected_features_names))
    methods_dict["DecisionTree"] = [selected_features_names]

    selected_data = reshaped_data[:, key_features]
    if visualisation:
        feature_data_visualization(selected_data, "DecisionTree", class_labels)
    classifier_acc_dic = classifiers(selected_data, class_labels)
    methods_acc_dict["DecisionTree"] = classifier_acc_dic

    # method 3: Gradient Boosting
    # print("Gradient Boosting")
    model = GradientBoostingClassifier(random_state=42)
    model.fit(reshaped_data, class_labels)
    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    key_features = sorted_indices[:k]
    selected_features_names = [features[i] for i in key_features]
    logging.info("Gradient Boosting Selected Features: {}".format(selected_features_names))
    methods_dict["Gradient Boosting"] = [selected_features_names]

    selected_data = reshaped_data[:, key_features]
    if visualisation:
        feature_data_visualization(selected_data, "Gradient Boosting", class_labels)
    classifier_acc_dic = classifiers(selected_data, class_labels)
    methods_acc_dict["Gradient Boosting"] = classifier_acc_dic

    # Algorithm 4: Regularization Methods
    # print("Feature selection based on Regularization Methods")
    model = LogisticRegression(penalty='l1', solver='liblinear')
    model.fit(reshaped_data, class_labels)
    feature_coefficients = model.coef_[0]
    top_feature_indices = np.argsort(np.abs(feature_coefficients))[::-1][:k]
    selected_features_names_4 = [features[i] for i in top_feature_indices]
    logging.info("l1 Regularization Selected Features: {}".format(selected_features_names_4))
    methods_dict["l1 Regularization"] = [selected_features_names_4]

    selected_data = reshaped_data[:, top_feature_indices]
    if visualisation:
        feature_data_visualization(selected_data, "l1 Regularization", class_labels)
    classifier_acc_dic = classifiers(selected_data, class_labels)
    methods_acc_dict["l1 Regularization"] = classifier_acc_dic

    model = LogisticRegression(penalty='l2', solver='liblinear')
    model.fit(reshaped_data, class_labels)
    feature_coefficients = model.coef_[0]
    top_feature_indices = np.argsort(np.abs(feature_coefficients))[::-1][:k]
    selected_features_names_4 = [features[i] for i in top_feature_indices]
    logging.info("l2 Regularization Selected Features: {}".format(selected_features_names_4))
    methods_dict["l2 Regularization"] = [selected_features_names_4]

    selected_data = reshaped_data[:, top_feature_indices]
    if visualisation:
        feature_data_visualization(selected_data, "l2 Regularization", class_labels)
    classifier_acc_dic = classifiers(selected_data, class_labels)
    methods_acc_dict["l2 Regularization"] = classifier_acc_dic

    # Algorithm 5: Correlation Analysis
    # print("Correlation Analysis")
    correlations = []
    for feature in reshaped_data.T:
        feature_labels = class_labels
        correlation, _ = pearsonr(feature, feature_labels)
        correlations.append(correlation)
    sorted_features = np.argsort(np.abs(correlations))[::-1]
    key_features = sorted_features[:k]
    selected_features_names_5 = [features[i] for i in key_features]
    logging.info("Correlation Analysis Selected Features: {}".format(selected_features_names_5))
    methods_dict["Correlation Analysis"] = [selected_features_names_5]

    selected_data = reshaped_data[:, top_feature_indices]
    if visualisation:
        feature_data_visualization(selected_data, "Correlation Analysis", class_labels)
    classifier_acc_dic = classifiers(selected_data, class_labels)
    methods_acc_dict["Correlation Analysis"] = classifier_acc_dic

    # Algorithm 6: Dimensionality Reduction Techniques
    # method 1: Principal Component Analysis (PCA)
    # print("Feature selection based on Principal Component Analysis (PCA)")
    pca = PCA(n_components=k)
    transformed_data = pca.fit_transform(reshaped_data)
    if visualisation:
        feature_data_visualization(transformed_data, "Principal Component Analysis", class_labels)
    classifier_acc_dic = classifiers(transformed_data, class_labels)
    methods_acc_dict["Principal Component Analysis"] = classifier_acc_dic

    # print("Feature selection based on Principal TSNE")
    # tsne = TSNE(n_components=3, random_state=42)
    # embedded_data = tsne.fit_transform(reshaped_data)
    # classifier_acc_dic = classifiers(embedded_data, _class_labels, "TSNE")
    # methods_acc_dict["TSNE"] = classifier_acc_dic

    feature_dict = {}
    for feature in features:
        count = 0
        for method, data in methods_dict.items():
            if feature in methods_dict[method][0]:
                count = count + 1
        feature_dict[feature] = count
    top_feature = [_k[0] for _k in sorted(feature_dict.items(), key=lambda item: item[1], reverse=True)][:k]

    logging.info("Final Selected Selected features: {}".format(top_feature))

    # top_feature_indices = [features.index(_f) for _f in top_feature]
    # selected_data = reshaped_data[:, top_feature_indices]
    # if visualisation:
    #     feature_data_visualization(selected_data, "Final Selected", class_labels)
    # classifier_acc_dic = classifiers(selected_data, class_labels)
    # methods_acc_dict["Final Selected"] = classifier_acc_dic

    # # Algorithm 7: Dimensionality Reduction Techniques
    # # method 1: Principal Component Analysis (PCA)
    # # print(f"Feature selection based on Principal PCA Based on Top {k} Features")
    # pca = PCA(n_components=k)
    # transformed_data = pca.fit_transform(selected_data)
    # if visualisation:
    #     feature_data_visualization(transformed_data, "PCA-from-K", class_labels)
    # classifier_acc_dic = classifiers(transformed_data, class_labels)
    # methods_acc_dict["PCA-from-K"] = classifier_acc_dic

    # print("Feature selection based on Principal TSNE after")
    # tsne = TSNE(n_components=3, random_state=42)
    # embedded_data = tsne.fit_transform(selected_data)
    # classifier_acc_dic = classifiers(embedded_data, _class_labels, "TSNE after")
    # methods_acc_dict["TSNE after"] = classifier_acc_dic

    logging.info("*" * 50)
    logging.info("The accuracy of different classifiers")
    for ind in methods_acc_dict:
        logging.info("{}: {}".format(ind, methods_acc_dict[ind]))
        logging.info("*" * 50)

    return methods_dict, methods_acc_dict


def Visualisation(data, name):
    Klist = list(data.keys())

    key_list = list(data[1][1].keys())
    d_dict = {}

    for k in Klist:
        for key in key_list:
            d = data[k][1][key]
            if key in d_dict:
                d_dict[key].append(d)
            else:
                d_dict[key] = [d]

    plt.suptitle(name)
    f = 0
    for m in ["Logistic Regression", 'Support Vector Machine', 'Random Forest']:
        # plt.figure()
        # print(f)
        if f < 3:
            f = f + 1
        plt.subplot(1, 3, f)
        for i in d_dict:
            # if i == "l2 Regularization":
            #     plt.figure()
            #     y_list = [v[m] for v in d_dict[i]]
            #     plt.plot(Klist, y_list, label=i + " " + m)
            #     plt.show()
            y_list = [v[m] for v in d_dict[i]]
            plt.plot(Klist, y_list, label=i + " " + m)
            plt.xlabel('Number of Features')
            plt.ylabel('The accuracy classifier')
            plt.title(f'The accuracy of {m} classifier')
            plt.legend()
    plt.show()


def main():
    workload_list = get_data()
    n_workloads = len(workload_list)
    data, features = construct_dataframe(workload_list)
    data, features, _ = preprocess(data, features)

    logging.info("Features: {}".format(features))
    logging.info("Shape: {}".format(data[features].shape))



    features_consistency_check = (features + ["class_labels"]).copy()
    data_consistency_check = data[features_consistency_check].to_numpy().copy()

    data, fft_magnitudes, features = add_fft_data_as_feature(data, features, n_workloads)
    assert np.all(data_consistency_check == data[features_consistency_check].to_numpy()), (
        "Data consistency check failed after add_fft_data_as_feature().")

    data, features = add_spectral_enveloping_data_as_feature(data, fft_magnitudes, features)
    assert np.all(data_consistency_check == data[features_consistency_check].to_numpy()), (
        "Data consistency check failed after add_spectral_enveloping_data_as_feature().")

    features = features_analysis_based_selection(data, features, n_workloads)
    assert np.all(data_consistency_check == data[features_consistency_check].to_numpy()), (
        "Data consistency check failed after features_analysis_based_selection().")

    overall_result = {}
    # for i in range(int(np.ceil(len(features) / 2))):
    # for i in range(len(features)):
    for i in range(5):
        k = i+5
        logging.info("Feature Number: {}".format(k))
        methods_dict, methods_acc_dict = key_features_identification(data, features, k)
        overall_result[k] = [methods_dict, methods_acc_dict]

    print(overall_result[5])
    # Visualisation(overall_result, "Feature Selection")

    # with open("Data_keep_all.pkl", "wb") as f:
    # pickle.dump(overall_result, f)


if __name__ == "__main__":
    logger = logging.getLogger()
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(format=FORMAT)
    logger.setLevel(logging.INFO)
    main()
