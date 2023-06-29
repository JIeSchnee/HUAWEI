import os
import glob
import pickle

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE, chi2
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from scipy.stats import pearsonr
from scipy.signal import decimate, hilbert

from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

scaler_normalize = MinMaxScaler()
scaler_standardize = StandardScaler()


class Workload:
    def __init__(self, workload_name, data_dict, category):
        self.workload_name = workload_name
        self.data_dict = data_dict
        self.category = category


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


def getData():
    # Specify the path to your CSV file
    # directory_path = '/home/jiezou/ros2_ws/Data/ros2/*.csv'
    # directory_path = '/home/jiezou/ros2_ws/Data/ros2/*'
    directory_path = '/home/jiezou/ros2_ws/Data/ros2_1/*'

    # Find all CSV files in the directory using glob
    csv_files = glob.glob(directory_path)

    Workload_data = []
    # Iterate over each CSV file
    for csv_file_path in csv_files:
        # Open the CSV file in read mode
        file_name = os.path.basename(csv_file_path)
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

        normalized_dict = {}
        for key, array in data_dict.items():
            if key != "Round":
                normalized_array = Normalization(data_dict[key])
                normalized_dict[key] = normalized_array
            else:
                normalized_dict[key] = data_dict[key]
        if file_name in ['core_PMC.csv', 'sha_PMC.csv', 'zip_PMC.csv', 'parser_PMC.csv']:
            # ['core_PMC.csv', 'sha_PMC.csv', 'zip_PMC.csv', 'parser_PMC.csv', 'radix_PMC.csv', 'linear_PMC.csv', 'nnest_PMC', 'loops_PMC.csv']
            category = 0  # integer workloads
        else:
            category = 1  # floating-point workloads

        workload = Workload(file_name, normalized_dict, category)
        Workload_data.append(workload)

        # print(data_dict["ExecutionTime"])
        # print(normalized_dict["ExecutionTime"])
        # quit()
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

    return Workload_data


def DataReconstruction(Workload_data):
    data = []
    class_labels = []
    workload_name = []
    Features = []
    for workload in Workload_data:
        # print("---------")
        # print(workload.workload_name)
        # print(workload.category)
        workload_name.append(workload.workload_name)
        class_labels.append(workload.category)
        sample = []
        i = 300
        # while i < len(workload.data_dict["Round"]):
        while i < 3300:
            sample_point = []
            for key, array in workload.data_dict.items():
                if key not in Features and key != "Round":
                    Features.append(key)
                if key != "Round":
                    point = workload.data_dict[key][i]
                    sample_point.append(point)
            i = i + 1
            sample.append(sample_point)
        # print(len(sample[0]), len(sample))
        data.append(sample)
    # print(len(data))
    return data, class_labels, workload_name, Features


def LogisticRegression_classifier(selected_features_2, class_labels, method):
    X_train, X_test, y_train, y_test = train_test_split(selected_features_2, class_labels, test_size=0.2,
                                                        random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(method, "Accuracy:", accuracy)

    return accuracy


def classifiers(selected_features, class_labels, method):
    classifier_acc_dict = {}
    X_train, X_test, y_train, y_test = train_test_split(selected_features, class_labels, test_size=0.2,
                                                        random_state=42)

    classifiers = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Support Vector Machine": SVC(kernel="linear", random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42)
    }

    for classifier_name, classifier in classifiers.items():
        # print("@:@@", len(X_train), len(X_test), len(X_train) + len(X_test))
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        # print(f"{classifier_name} Accuracy: {accuracy}")
        classifier_acc_dict[classifier_name] = accuracy

    return classifier_acc_dict


def UnivariateFeatureSelection(Methods_dict, reshaped_data, class_labels, k):
    # print("Feature selection based on Univariate Feature Selection")
    selector = SelectKBest(score_func=chi2, k=k)
    selected_features = selector.fit_transform(reshaped_data, class_labels)
    mask = selector.get_support()
    selected_feature_indices = np.where(mask)[0]
    selected_features_names = [Features[i] for i in selected_feature_indices]
    # print("features in the function", Features)
    # print("Selected Features:", selected_features_names)
    Methods_dict["Univariate Feature Selection"] = [selected_features_names]

    return Methods_dict, selected_features


def RecursiveFeatureElimination(Methods_dict, reshaped_data, class_labels, k):
    # print("Feature selection based on Recursive Feature Elimination (RFE)")
    estimator = SVC(kernel="linear")
    rfe = RFE(estimator=estimator, n_features_to_select=k)
    selected_features = rfe.fit_transform(reshaped_data, class_labels)
    selected_feature_indices = rfe.get_support(indices=True)
    selected_features_names = [Features[i] for i in selected_feature_indices]
    # print("Selected Features:", selected_features_names)
    Methods_dict["Recursive Feature Elimination"] = [selected_features_names]
    return Methods_dict, selected_features


def FeatureDataVisualization(selected_features, method, class_labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(selected_features[:, 0], selected_features[:, 1], selected_features[:, 2], c=class_labels)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    ax.set_title(f'Visualization (3D) {method}')
    plt.show()


def FFTDataAnalysis(data_array):
    # Step 1: Compute FFT for each workload and feature
    fft_data = np.fft.fft(data_array, axis=1)
    print("fft_data Shape:", fft_data.shape)

    # Take the absolute values of the FFT results to obtain the magnitudes
    fft_magnitudes = np.abs(fft_data)
    print("fft_magnitudes Shape:", fft_magnitudes.shape)

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

    print("downsampled_magnitudes Shape:", downsampled_data.shape)

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


def SpectralEnvelopingAnalysis(fft_magnitudes):
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


def FeaturesAnalysisBasedSelection(data_with_envelope, Features):
    print("###########################################")
    print("      Features Analysis          ")
    print("###########################################")
    # print("Features in the function", Features)

    n_workloads, n_data_points, n_features = data_with_envelope.shape
    data_reshaped = np.reshape(data_with_envelope, (n_workloads * n_data_points, n_features))
    # print("the original data for check")
    # print(data_reshaped[0])

    classes = np.array(class_labels)

    # Reshape the classes to match the number of samples
    classes_reshaped = np.repeat(classes, n_data_points)
    # Calculate the F-ratio score for each feature
    f_ratio_scores, _ = f_classif(data_reshaped, classes_reshaped)
    # print("f_ratio_scores", f_ratio_scores, len(f_ratio_scores))

    # Calculate the Trace ratio score for each feature
    within_class_scatter = np.zeros(n_features)
    between_class_scatter = np.zeros(n_features)
    for feature_idx in range(n_features):
        for workload_idx in range(n_workloads):
            class_data = data_with_envelope[workload_idx, :, feature_idx]
            class_mean = np.mean(class_data)
            overall_mean = np.mean(data_with_envelope[:, :, feature_idx])
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

    k = n_features - 10  # Number of features to keep

    # Select features based on F-ratio scores and remove 10 features with the lowest scores

    selector = SelectKBest(f_classif, k=k)
    F_selected_data = selector.fit_transform(data_reshaped, classes_reshaped)
    # print("Date selected according to F-ratio", F_selected_data.shape)
    F_selected_feature_indices = selector.get_support(indices=True)
    # print("\nSelected Feature Indices:", F_selected_feature_indices)
    F_selected_features_names = [Features[i] for i in F_selected_feature_indices]
    print("F Selected Features:", F_selected_features_names)
    # print("check shape consistency", F_selected_data.shape)
    # print("check consistency F term")
    # print(F_selected_data[0])

    # Select features based on Trace Ratio scores and remove 10 features with the lowest scores
    T_selected_feature_indices = np.argsort(trace_ratio_scores)[-k:]
    # T_selected_data = data_with_envelope[:, :, T_selected_feature_indices]
    # n_workloads, n_data_points, n_features = T_selected_data.shape
    # T_selected_data_reshaped = np.reshape(T_selected_data, (n_workloads * n_data_points, n_features))
    # print("\nSelected Feature Indices:", T_selected_feature_indices)
    T_selected_features_names = [Features[i] for i in T_selected_feature_indices]
    print("T Selected Features:", T_selected_features_names)
    # print("check consistency T term")
    # print(T_selected_data_reshaped[0])

    # Select the indices appeared in both F and T list
    common_elements = np.intersect1d(F_selected_feature_indices, T_selected_feature_indices)
    selected_data = data_with_envelope[:, :, common_elements]
    selected_features_names = [Features[i] for i in common_elements]
    # print("Feature indices: ", common_elements)
    print("Consistent Features' names: ", selected_features_names)
    print("Shape of selected data: ", selected_data.shape)
    print("=====================================================================")

    # n_workloads, n_data_points, n_features = selected_data.shape
    # reshaped = np.reshape(selected_data, (n_workloads * n_data_points, n_features))
    # print("Check consistency: ", reshaped[0])

    return selected_data, selected_features_names


def AddFFTDataAsFeature(data_array, InitFeatureNum, Features):
    print("###########################################")
    print("  FFT data analysis and Add new feature    ")
    print("###########################################")

    fft_magnitudes = FFTDataAnalysis(data_array)

    # Append the magnitudes as a new feature
    new_data = np.concatenate((data_array, fft_magnitudes), axis=2)
    # Reshape the new data
    # n_workloads, n_data_points, n_features = new_data.shape
    print("New Data Shape:", new_data.shape)
    # reshaped_data = new_data.reshape((n_workloads * n_data_points, n_features))

    for i in range(InitFeatureNum):
        Features.append(f"{Features[i]}FFT")

    print("new feature list", Features)
    return new_data, fft_magnitudes, Features


def AddSpectralEnvelopingDataAsFeature(new_data, fft_magnitudes, Features):
    print("###########################################")
    print("  Spectral Enveloping data analysis ")
    print("###########################################")
    envelope_data = SpectralEnvelopingAnalysis(fft_magnitudes)

    # Concatenate the envelope data as a new feature
    data_with_envelope = np.concatenate((new_data, envelope_data), axis=2)
    # Reshape the new data
    n_workloads, n_data_points, n_features = data_with_envelope.shape
    print("New Data with Envelope Shape:", data_with_envelope.shape)
    reshaped_data = data_with_envelope.reshape((n_workloads * n_data_points, n_features))
    for i in range(InitFeatureNum):
        Features.append(f"{Features[i]}FFT_SE")
    print("new feature list", Features)
    print(reshaped_data.shape)
    return data_with_envelope, Features


def KeyFeaturesIdentification(_UpdatedData, _class_labels, Features, k):
    # print("###########################################")
    # print("      Key Features Identification          ")
    # print("###########################################")
    n_workloads, n_data_points, n_features = _UpdatedData.shape
    # k = 14  # Number of key features to select

    Methods_dict = {}
    Methods_acc_dict = {}
    _class_labels = np.repeat(_class_labels, n_data_points)
    reshaped_data = _UpdatedData.reshape((n_workloads * n_data_points, n_features))

    # # print("_class_labels Data Shape:", _class_labels.shape)
    # print(Features)
    # print(len(Features))
    # print("=====================================")
    # print(reshaped_data.shape)

    # Algorithm 1: Univariate Feature Selection
    Methods_dict, selected_features = UnivariateFeatureSelection(Methods_dict, reshaped_data, _class_labels, k)
    # accuracy = LogisticRegression_classifier(selected_features_1, _class_labels, "Univariate Feature Selection")
    # FeatureDataVisualization(selected_features, "Univariate Feature Selection", _class_labels)
    classifier_acc_dic = classifiers(selected_features, _class_labels, "Univariate Feature Selection")
    Methods_acc_dict["Univariate Feature Selection"] = classifier_acc_dic
    # print("=====================================")

    # Algorithm 2: Recursive Feature Elimination (RFE)
    Methods_dict, selected_features = RecursiveFeatureElimination(Methods_dict, reshaped_data, _class_labels, k)
    # FeatureDataVisualization(selected_features, "Recursive Feature Elimination", _class_labels)
    classifier_acc_dic = classifiers(selected_features, _class_labels, "Recursive Feature Elimination")
    Methods_acc_dict["Recursive Feature Elimination"] = classifier_acc_dic
    # print("=====================================")

    # Algorithm 3: Tree-based Feature Selection
    # print("Feature selection based on Tree-based Methods")
    # method 1: RandomForest
    # print("RandomForest")
    model = RandomForestClassifier()
    model.fit(reshaped_data, _class_labels)
    feature_importances = model.feature_importances_
    top_feature_indices = np.argsort(feature_importances)[::-1][:k]
    selected_features_names_3 = [Features[i] for i in top_feature_indices]
    # print("Selected Features:", selected_features_names_3)
    Methods_dict["RandomForest"] = [selected_features_names_3]

    selected_data = reshaped_data[:, top_feature_indices]
    # FeatureDataVisualization(selected_data, "RandomForest", _class_labels)
    classifier_acc_dic = classifiers(selected_data, _class_labels, "RandomForest")
    Methods_acc_dict["RandomForest"] = classifier_acc_dic

    # print('-------------------------')

    # method 2: DecisionTree
    # print("DecisionTree")
    model = DecisionTreeClassifier(random_state=42)
    model.fit(reshaped_data, _class_labels)
    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    key_features = sorted_indices[:k]
    selected_features_names = [Features[i] for i in key_features]
    # print("Selected Features:", selected_features_names)
    Methods_dict["DecisionTree"] = [selected_features_names]

    selected_data = reshaped_data[:, key_features]
    # FeatureDataVisualization(selected_data, "DecisionTree", _class_labels)
    classifier_acc_dic = classifiers(selected_data, _class_labels, "DecisionTree")
    Methods_acc_dict["DecisionTree"] = classifier_acc_dic
    # print('-------------------------')

    # method 3: Gradient Boosting
    # print("Gradient Boosting")
    model = GradientBoostingClassifier(random_state=42)
    model.fit(reshaped_data, _class_labels)
    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    key_features = sorted_indices[:k]
    selected_features_names = [Features[i] for i in key_features]
    # print("Selected Features:", selected_features_names)
    Methods_dict["Gradient Boosting"] = [selected_features_names]

    selected_data = reshaped_data[:, key_features]
    # FeatureDataVisualization(selected_data, "Gradient Boosting", _class_labels)
    classifier_acc_dic = classifiers(selected_data, _class_labels, "Gradient Boosting")
    Methods_acc_dict["Gradient Boosting"] = classifier_acc_dic
    # print("=====================================")

    # Algorithm 4: Regularization Methods
    # print("Feature selection based on Regularization Methods")
    # print('l1')
    model = LogisticRegression(penalty='l1', solver='liblinear')
    model.fit(reshaped_data, _class_labels)
    feature_coefficients = model.coef_[0]
    top_feature_indices = np.argsort(np.abs(feature_coefficients))[::-1][:k]
    selected_features_names_4 = [Features[i] for i in top_feature_indices]
    # print("Selected Features:", selected_features_names_4)
    Methods_dict["l1 Regularization"] = [selected_features_names_4]

    selected_data = reshaped_data[:, top_feature_indices]
    # FeatureDataVisualization(selected_data, "l1 Regularization", _class_labels)
    classifier_acc_dic = classifiers(selected_data, _class_labels, "l1 Regularization")
    Methods_acc_dict["l1 Regularization"] = classifier_acc_dic

    # print("------------------------")

    # print('l2')
    model = LogisticRegression(penalty='l2', solver='liblinear')
    model.fit(reshaped_data, _class_labels)
    feature_coefficients = model.coef_[0]
    top_feature_indices = np.argsort(np.abs(feature_coefficients))[::-1][:k]
    selected_features_names_4 = [Features[i] for i in top_feature_indices]
    # print("Selected Features:", selected_features_names_4)
    Methods_dict["l2 Regularization"] = [selected_features_names_4]

    selected_data = reshaped_data[:, top_feature_indices]
    # FeatureDataVisualization(selected_data, "l2 Regularization", _class_labels)
    classifier_acc_dic = classifiers(selected_data, _class_labels, "l2 Regularization")
    Methods_acc_dict["l2 Regularization"] = classifier_acc_dic

    # print("=====================================")

    # Algorithm 5: Correlation Analysis
    # print("Correlation Analysis")
    correlations = []
    for feature in reshaped_data.T:
        feature_labels = _class_labels
        correlation, _ = pearsonr(feature, feature_labels)
        correlations.append(correlation)
    sorted_features = np.argsort(np.abs(correlations))[::-1]
    key_features = sorted_features[:k]
    selected_features_names_5 = [Features[i] for i in key_features]
    # print("Selected Features:", selected_features_names_5)
    Methods_dict["Correlation Analysis"] = [selected_features_names_5]

    selected_data = reshaped_data[:, top_feature_indices]
    # FeatureDataVisualization(selected_data, "Correlation Analysis", _class_labels)
    classifier_acc_dic = classifiers(selected_data, _class_labels, "Correlation Analysis")
    Methods_acc_dict["Correlation Analysis"] = classifier_acc_dic

    # print("=====================================")
    # Algorithm 4: Dimensionality Reduction Techniques
    # method 1: Principal Component Analysis (PCA)
    # print("Feature selection based on Principal Component Analysis (PCA)")
    pca = PCA(n_components=k)
    transformed_data = pca.fit_transform(reshaped_data)
    # FeatureDataVisualization(transformed_data, "Principal Component Analysis", _class_labels)
    classifier_acc_dic = classifiers(transformed_data, _class_labels, "Principal Component Analysis")
    Methods_acc_dict["Principal Component Analysis"] = classifier_acc_dic

    # print("------------------------")
    #
    # print("Feature selection based on Principal TSNE")
    # tsne = TSNE(n_components=3, random_state=42)
    # embedded_data = tsne.fit_transform(reshaped_data)
    # classifier_acc_dic = classifiers(embedded_data, _class_labels, "TSNE")
    # Methods_acc_dict["TSNE"] = classifier_acc_dic

    # print("=====================================")
    feature_dict = {}
    for feature in Features:
        count = 0
        for method, data in Methods_dict.items():
            # print(method,Methods_dict[method])
            if feature in Methods_dict[method][0]:
                count = count + 1
        feature_dict[feature] = count
    top_feature = [_k[0] for _k in sorted(feature_dict.items(), key=lambda item: item[1], reverse=True)][:k]

    # print("Top features:", top_feature)

    top_feature_indices = [Features.index(_f) for _f in top_feature]
    selected_data = reshaped_data[:, top_feature_indices]
    # FeatureDataVisualization(selected_data, "Final Selected", _class_labels)
    classifier_acc_dic = classifiers(selected_data, _class_labels, "Final Selected")
    Methods_acc_dict["Final Selected"] = classifier_acc_dic

    # print("=====================================")

    # Algorithm 4: Dimensionality Reduction Techniques
    # method 1: Principal Component Analysis (PCA)
    # print(f"Feature selection based on Principal PCA Based on Top {k} Features")
    pca = PCA(n_components=k)
    transformed_data = pca.fit_transform(selected_data)
    # FeatureDataVisualization(transformed_data, "PCA-from-K", _class_labels)
    classifier_acc_dic = classifiers(transformed_data, _class_labels, "PCA-from-K")
    Methods_acc_dict["PCA-from-K"] = classifier_acc_dic

    # print("------------------------")
    #
    # print("Feature selection based on Principal TSNE after")
    # tsne = TSNE(n_components=3, random_state=42)
    # embedded_data = tsne.fit_transform(selected_data)
    # classifier_acc_dic = classifiers(embedded_data, _class_labels, "TSNE after")
    # Methods_acc_dict["TSNE after"] = classifier_acc_dic
    #
    print("******************************")
    print("The accuracy of different classifiers")
    print("******************************")
    for ind in Methods_acc_dict:
        print(ind)
        print(Methods_acc_dict[ind])
        print("******************************")

    return Methods_dict, Methods_acc_dict


if __name__ == "__main__":
    Workload_data = getData()
    data, class_labels, workload_name, Features = DataReconstruction(Workload_data)
    # print(len(data))
    # print(class_labels)
    # print(workload_name)
    print("Features", Features)
    data_array = np.array(data)
    print("Shape:", data_array.shape)
    InitFeatureNum = len(Features)

    # n_workloads, n_data_points, n_features = data_array.shape
    # reshaped_data = data_array.reshape((n_workloads * n_data_points, n_features))

    # DataWithFFT, fft_magnitudes, Features = AddFFTDataAsFeature(data_array, InitFeatureNum, Features)
    #
    # DataWithFFTAndSE, Features = AddSpectralEnvelopingDataAsFeature(DataWithFFT, fft_magnitudes, Features)
    #
    # UpdatedData, UpdatedFeatures = FeaturesAnalysisBasedSelection(DataWithFFTAndSE, Features)

    OverallResult = {}
    for i in range(int(np.ceil(len(Features) / 2))):
        k = i + 1
        print("Feature Number", k)
        # Methods_dict, Methods_acc_dict = KeyFeaturesIdentification(UpdatedData, class_labels, UpdatedFeatures, k)
        Methods_dict, Methods_acc_dict = KeyFeaturesIdentification(data_array, class_labels, Features, k)
        OverallResult[k] = [Methods_dict, Methods_acc_dict]

    with open("original.pkl", "wb") as f:
        pickle.dump(OverallResult, f)
