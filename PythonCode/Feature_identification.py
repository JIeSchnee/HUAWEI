import os
import glob
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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
    directory_path = '/home/jiezou/ros2_ws/Data/ros2/*'

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
            category = 0 # integer workloads
        else:
            category = 1 # floating-point workloads

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
        i = 100
        # while i < len(workload.data_dict["Round"]):
        while i < 3100:
            sample_point = []
            for key,array in workload.data_dict.items():
                if key not in Features and key != "Round":
                    Features.append(key)
                if key != "Round":
                    point = workload.data_dict[key][i]
                    sample_point.append(point)
            i=i+1
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
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{classifier_name} Accuracy: {accuracy}")
        classifier_acc_dict[classifier_name] = accuracy

    return classifier_acc_dict

def UnivariateFeatureSelection(Methods_dict, reshaped_data, class_labels, k):

    print("Feature selection based on Univariate Feature Selection")
    selector = SelectKBest(score_func=f_classif, k=k)
    selected_features = selector.fit_transform(reshaped_data, class_labels)
    mask = selector.get_support()
    selected_feature_indices = np.where(mask)[0]
    selected_features_names = [Features[i] for i in selected_feature_indices]
    print("Selected Features:", selected_features_names)
    Methods_dict["Univariate Feature Selection"] = [selected_features_names]

    return Methods_dict, selected_features

def RecursiveFeatureElimination(Methods_dict, reshaped_data, class_labels, k):
    print("Feature selection based on Recursive Feature Elimination (RFE)")
    estimator = SVC(kernel="linear")
    rfe = RFE(estimator=estimator, n_features_to_select=k)
    selected_features = rfe.fit_transform(reshaped_data, class_labels)
    selected_feature_indices = rfe.get_support(indices=True)
    selected_features_names = [Features[i] for i in selected_feature_indices]
    print("Selected Features:", selected_features_names)
    Methods_dict["Recursive Feature Elimination"] = [selected_features_names]
    return Methods_dict, selected_features


if __name__ == "__main__":
    Workload_data = getData()
    data, class_labels, workload_name, Features = DataReconstruction(Workload_data)
    # print(len(data))
    # print(class_labels)
    print(workload_name)
    print("Features", Features)
    data_array = np.array(data)
    n_workloads, n_data_points, n_features = data_array.shape
    reshaped_data = data_array.reshape((n_workloads * n_data_points, n_features))
    # Print the shape attribute
    print("Shape:", data_array.shape)

    print("###########################################")
    print("      Key Features Identification          ")
    print("###########################################")

    k = 7  # Number of key features to select
    Methods_dict = {}
    Methods_acc_dict = {}
    class_labels = np.repeat(class_labels, n_data_points)


    print("=====================================")

    # Algorithm 1: Univariate Feature Selection
    Methods_dict, selected_features = UnivariateFeatureSelection(Methods_dict, reshaped_data, class_labels, k)
    # accuracy = LogisticRegression_classifier(selected_features_1, class_labels, "Univariate Feature Selection")
    classifier_acc_dic = classifiers(selected_features, class_labels, "Univariate Feature Selection")
    Methods_acc_dict["Univariate Feature Selection"] = classifier_acc_dic
    print("=====================================")

    # Algorithm 2: Recursive Feature Elimination (RFE)
    Methods_dict, selected_features = RecursiveFeatureElimination(Methods_dict, reshaped_data, class_labels, k)
    classifier_acc_dic = classifiers(selected_features, class_labels, "Recursive Feature Elimination")
    Methods_acc_dict["Recursive Feature Elimination"] = classifier_acc_dic
    print("=====================================")

    # Algorithm 3: Tree-based Feature Selection
    print("Feature selection based on Tree-based Methods")
    # method 1: RandomForest
    print("RandomForest")
    model = RandomForestClassifier()
    model.fit(reshaped_data, class_labels)
    feature_importances = model.feature_importances_
    top_feature_indices = np.argsort(feature_importances)[::-1][:k]
    selected_features_names_3 = [Features[i] for i in top_feature_indices]
    print("Selected Features:", selected_features_names_3)
    Methods_dict["RandomForest"] = [selected_features_names_3]

    selected_data = reshaped_data[:, top_feature_indices]
    classifier_acc_dic = classifiers(selected_data, class_labels, "RandomForest")
    Methods_acc_dict["RandomForest"] = classifier_acc_dic

    print('-------------------------')

    # method 2: DecisionTree
    print("DecisionTree")
    model = DecisionTreeClassifier(random_state=42)
    model.fit(reshaped_data, class_labels)
    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    key_features = sorted_indices[:k]
    selected_features_names = [Features[i] for i in key_features]
    print("Selected Features:", selected_features_names)
    Methods_dict["DecisionTree"] = [selected_features_names]

    selected_data = reshaped_data[:, key_features]
    classifier_acc_dic = classifiers(selected_data, class_labels, "DecisionTree")
    Methods_acc_dict["DecisionTree"] = classifier_acc_dic
    print('-------------------------')

    # method 3: Gradient Boosting
    print("Gradient Boosting")
    model = GradientBoostingClassifier(random_state=42)
    model.fit(reshaped_data, class_labels)
    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    key_features = sorted_indices[:k]
    selected_features_names = [Features[i] for i in key_features]
    print("Selected Features:", selected_features_names)
    Methods_dict["Gradient Boosting"] = [selected_features_names]

    selected_data = reshaped_data[:, key_features]
    classifier_acc_dic = classifiers(selected_data, class_labels, "Gradient Boosting")
    Methods_acc_dict["Gradient Boosting"] = classifier_acc_dic
    print("=====================================")

    # Algorithm 4: Regularization Methods
    print("Feature selection based on Regularization Methods")
    print('l1')
    model = LogisticRegression(penalty='l1', solver='liblinear')
    model.fit(reshaped_data, class_labels)
    feature_coefficients = model.coef_[0]
    top_feature_indices = np.argsort(np.abs(feature_coefficients))[::-1][:k]
    selected_features_names_4 = [Features[i] for i in top_feature_indices]
    print("Selected Features:", selected_features_names_4)
    Methods_dict["l1 Regularization"] = [selected_features_names_4]

    selected_data = reshaped_data[:, top_feature_indices]
    classifier_acc_dic = classifiers(selected_data, class_labels, "l1 Regularization")
    Methods_acc_dict["l1 Regularization"] = classifier_acc_dic

    print("------------------------")

    print('l2')
    model = LogisticRegression(penalty='l2', solver='liblinear')
    model.fit(reshaped_data, class_labels)
    feature_coefficients = model.coef_[0]
    top_feature_indices = np.argsort(np.abs(feature_coefficients))[::-1][:k]
    selected_features_names_4 = [Features[i] for i in top_feature_indices]
    print("Selected Features:", selected_features_names_4)
    Methods_dict["l2 Regularization"] = [selected_features_names_4]

    selected_data = reshaped_data[:, top_feature_indices]
    classifier_acc_dic = classifiers(selected_data, class_labels, "l2 Regularization")
    Methods_acc_dict["l2 Regularization"] = classifier_acc_dic

    print("=====================================")

    # Algorithm 5: Correlation Analysis
    print("Correlation Analysis")
    correlations = []
    for feature in reshaped_data.T:
        feature_labels = class_labels
        correlation, _ = pearsonr(feature, feature_labels)
        correlations.append(correlation)
    sorted_features = np.argsort(np.abs(correlations))[::-1]
    key_features = sorted_features[:k]
    selected_features_names_5 = [Features[i] for i in key_features]
    print("Selected Features:", selected_features_names_5)
    Methods_dict["Correlation Analysis"] = [selected_features_names_5]

    selected_data = reshaped_data[:, top_feature_indices]
    classifier_acc_dic = classifiers(selected_data, class_labels, "Correlation Analysis")
    Methods_acc_dict["Correlation Analysis"] = classifier_acc_dic


    # Algorithm 4: Dimensionality Reduction Techniques
    # method 1: Principal Component Analysis (PCA)
    print("Feature selection based on Principal Component Analysis (PCA)")
    pca = PCA(n_components=k)
    transformed_data = pca.fit_transform(reshaped_data)
    classifier_acc_dic = classifiers(transformed_data, class_labels, "Principal Component Analysis")
    Methods_acc_dict["Principal Component Analysis"] = classifier_acc_dic

    feature_dict = {}
    for feature in Features:
        count = 0
        for method, data in Methods_dict.items():
            # print(method,Methods_dict[method])
            if feature in Methods_dict[method][0]:
                count = count +1
        feature_dict[feature] = count
    top_feature = [_k[0] for _k in sorted(feature_dict.items(), key=lambda item: item[1], reverse=True)][:k]
    top_feature_indices = [Features.index(_f) for _f in top_feature]
    selected_data = reshaped_data[:, top_feature_indices]
    classifier_acc_dic = classifiers(selected_data, class_labels, "Final Selected")
    Methods_acc_dict["Final Selected"] = classifier_acc_dic

    print("Top features:", top_feature)

    # Algorithm 4: Dimensionality Reduction Techniques
    # method 1: Principal Component Analysis (PCA)
    print("Feature selection based on Principal PCA after")
    pca = PCA(n_components=k)
    transformed_data = pca.fit_transform(selected_data)
    classifier_acc_dic = classifiers(transformed_data, class_labels, "PCA after")
    Methods_acc_dict["PCA after"] = classifier_acc_dic


    print("******************************")
    print("The accuracy of different classifiers")
    print("******************************")
    for ind in Methods_acc_dict:
        print(ind)
        print(Methods_acc_dict[ind])
        print("******************************")







