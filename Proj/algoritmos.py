import sys
import argparse
from netaddr import IPNetwork, IPAddress, IPSet
import pyshark
import numpy as np
import os
from os.path import exists
from pyclbr import Class
import numpy as np
import scipy.stats as stats
import scipy.signal as signal
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.svm import OneClassSVM
from scipy.stats import multivariate_normal
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn import svm
import seaborn as sns
import time
import sys
import warnings
import pandas as pd

def calculate_metrics(tp, tn, fp, fn, nObsTest):
    accuracy = ((tp + tn) / nObsTest) * 100
    precision = (tp / (tp + fp)) * 100 if tp + fp > 0 else 0
    recall = (tp / (tp + fn)) * 100 if tp + fn > 0 else 0
    f1_score = (2 * (precision * recall) / (precision + recall)) if (precision + recall) != 0 else 0
    return accuracy, precision, recall, f1_score
############################################# -- 7 -- Centroids Distances SEM PCA #########################
def centroids_distances(trainFeatures_browsing, o2train, i3test,   o3test):
    centroid = np.mean(trainFeatures_browsing[(o2train == 0).flatten()], axis=0)
    actual_labels = o3test.flatten()
    nObsTest = len(actual_labels)

    threshold_values = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.2, 1.1, 1.5, 2.0, 3, 5, 6, 10]
    results = {'Method': [],'Threshold': [],'TP': [],'FP': [],'TN': [],'FN': [],'Accuracy': [],'Precision': [],'Recall': [],'F1 Score': [],'ConfusionMatrix': []}

    for threshold in threshold_values:
        distances = np.linalg.norm(i3test - centroid, axis=1)
        predicted_labels = (distances > threshold).astype(float) * 2.0  # Anomalies are labeled as 2.0

        tp = np.sum((predicted_labels == 2.0) & (actual_labels == 2.0))
        tn = np.sum((predicted_labels == 0.0) & (actual_labels == 0.0))
        fp = np.sum((predicted_labels == 2.0) & (actual_labels == 0.0))
        fn = np.sum((predicted_labels == 0.0) & (actual_labels == 2.0))

        accuracy, precision, recall, f1_score = calculate_metrics(tp, fp, tn, fn, nObsTest)
        cm = confusion_matrix(actual_labels, predicted_labels)

        results['Method'].append('Centroid-Based')
        results['Threshold'].append(threshold)
        results['TP'].append(tp)
        results['FP'].append(fp)
        results['TN'].append(tn)
        results['FN'].append(fn)
        results['Accuracy'].append(accuracy)
        results['Precision'].append(precision)
        results['Recall'].append(recall)
        results['F1 Score'].append(f1_score)
        results['ConfusionMatrix'].append(cm)

    # Convert results to DataFrame and save to Excel
    df = pd.DataFrame(results)
    # Find the best threshold based on F1 score
    best_result_index = df['F1 Score'].idxmax()
    best_result = df.iloc[best_result_index]
    plt.figure(figsize=(8, 6))
    sns.heatmap(best_result['ConfusionMatrix'], annot=True, cmap='Blues', fmt='d', xticklabels=['Human', 'Smart Bot'], yticklabels=['Human', 'Smart Bot'])
    plt.xlabel('Predicted')
    plt.ylabel('Real')
    plt.title(f"Best Confusion Matrix (Method: {best_result['Method']}, Threshold: {best_result['Threshold']})")
    plt.show()

####################################### -- 7.2 -- Centroids Distances Com PCA ######################### ##
def centroids_distances_with_pca(trainFeatures, o2trainClass, testFeatures_normal, testFeatures_dns, o3testClass):
    components_to_test = [5, 7, 9, 11] # não dá mais alto tem que ser menor que o min de trainFeatures
    results = {'Method': [],'Components': [],'Threshold': [],'TP': [],'FP': [],'TN': [],'FN': [],'Accuracy': [],'Precision': [],'Recall': [],'F1 Score': [],'ConfusionMatrix': []}

    for n_components in components_to_test:
        # Skipping components that exceed limit
        if n_components > min(trainFeatures.shape):
            print(f"Skipping n_components={n_components} as it exceeds limit.")
            continue

        # Apply PCA
        pca = PCA(n_components=n_components)

        print(" min trainFeatures).shape ",min (np.vstack(trainFeatures).shape))
        if n_components > min(np.vstack(trainFeatures).shape):
            print(f"Skipping n_components={n_components} as it exceeds limit.")
            continue

        try:
            pca = PCA(n_components=n_components)
            i2train_pca = pca.fit_transform(np.vstack(trainFeatures))

            # Your existing code follows...

        except ValueError as e:
            print(f"Error fitting PCA with n_components={n_components}: {e}")
            continue

        i2train_pca = pca.fit_transform(np.vstack(trainFeatures))
        centroids = np.mean(i2train_pca[(o2trainClass == 0).flatten(), :], axis=0)

        i3Atest_pca = pca.transform(np.vstack((testFeatures_normal, testFeatures_dns)))
        actual_labels = o3testClass.flatten()
        nObsTest = len(actual_labels)

        threshold_values = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.2, 1.1, 1.5, 2.0, 3, 5, 6, 10]

        for threshold in threshold_values:
            distances = np.linalg.norm(i3Atest_pca - centroids, axis=1)
            predicted_labels = (distances > threshold).astype(float) * 2.0  # Anomalies are labeled as 2.0

            tp = np.sum((predicted_labels == 2.0) & (actual_labels == 2.0))
            tn = np.sum((predicted_labels == 0.0) & (actual_labels == 0.0))
            fp = np.sum((predicted_labels == 2.0) & (actual_labels == 0.0))
            fn = np.sum((predicted_labels == 0.0) & (actual_labels == 2.0))

            accuracy, precision, recall, f1_score = calculate_metrics(tp, fp, tn, fn, nObsTest)
            cm = confusion_matrix(actual_labels, predicted_labels)

            results['Method'].append('PCA Centroid-Based')
            results['Components'].append(n_components)
            results['Threshold'].append(threshold)
            results['TP'].append(tp)
            results['FP'].append(fp)
            results['TN'].append(tn)
            results['FN'].append(fn)
            results['Accuracy'].append(accuracy)
            results['Precision'].append(precision)
            results['Recall'].append(recall)
            results['F1 Score'].append(f1_score)
            results['ConfusionMatrix'].append(cm)

    # Convert results to DataFrame and save to Excel
    df = pd.DataFrame(results)

    # Find the best result based on F1 score
    best_result = df.loc[df['F1 Score'].idxmax()]
    plt.figure(figsize=(8, 6))
    sns.heatmap(best_result['ConfusionMatrix'], annot=True, cmap='Blues', fmt='d', xticklabels=['Human', 'Smart Bot'], yticklabels=['Human', 'Smart Bot'])
    plt.title(f"Best Confusion Matrix (Method: {best_result['Method']}, Components: {best_result['Components']}, Threshold: {best_result['Threshold']})")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

######################################### -- 8 -- Anomaly Detection based on One Class Support Vector Machines WITHOUT PCA ###############################
def one_class_svm(i2train,   i3test,   o3test):
    print("i2train:\n", i2train)
    print("i3test:\n", i3test)
    print("o3test:\n", o3test)

    kernels = ['linear', 'rbf', 'poly']
    results = {'Method': [], 'TP': [], 'FP': [], 'TN': [], 'FN': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': [], 'ConfusionMatrix': []}

    actual_labels = [class_label[0] for class_label in o3test]
    nObsTest = i3test.shape[0]

    nu = 0.5

    for kernel in kernels:
        ocsvm = svm.OneClassSVM(gamma='scale', kernel=kernel, nu=nu, degree=2 if kernel == 'poly' else 3).fit(i2train)
        predictions = ocsvm.predict(i3test)
        predicted_labels = [2.0 if pred == -1 else 0.0 for pred in predictions]

        tp = sum(1 for actual, predicted in zip(actual_labels, predicted_labels) if actual == predicted == 2)
        tn = sum(1 for actual, predicted in zip(actual_labels, predicted_labels) if actual == predicted == 0)
        fp = sum(1 for actual, predicted in zip(actual_labels, predicted_labels) if actual != predicted and predicted == 2)
        fn = sum(1 for actual, predicted in zip(actual_labels, predicted_labels) if actual != predicted and predicted == 0)

        accuracy, precision, recall, f1_score = calculate_metrics(tp, tn, fp, fn, nObsTest)
        print("actual_label\n", actual_labels, "\npredicted_label\n", predicted_labels)
        cm = confusion_matrix(actual_labels, predicted_labels)
        print(cm.shape)
        results['Method'].append(kernel.capitalize())
        results['TP'].append(tp)
        results['FP'].append(fp)
        results['TN'].append(tn)
        results['FN'].append(fn)
        results['Accuracy'].append(accuracy)
        results['Precision'].append(precision)
        results['Recall'].append(recall)
        results['F1 Score'].append(f1_score)
        results['ConfusionMatrix'].append(cm)
    df = pd.DataFrame(results)
    # Find best F1 score
    best_f1_index = df['F1 Score'].idxmax()
    best_confusion_matrix = df.loc[best_f1_index, 'ConfusionMatrix']
    best_kernel = df.loc[best_f1_index, 'Method']

    # Plot best confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(best_confusion_matrix, annot=True, cmap='Blues', fmt='d',
                xticklabels=['Human', 'Smart Bot'], yticklabels=['Human', 'Smart Bot'])
    plt.xlabel('Predicted label')
    plt.ylabel('Actual label')
    plt.title(f'Best Confusion Matrix One Class Support\n Best Kernel: {best_kernel}')
    plt.show()

##################################################################################### -- 8.2 -- Anomaly Detection based on One Class Support Vector Machines with pca###############################
def one_class_svm_with_pca( trainFeatures_browsing, testFeatures_browsing, testFeatures_atck, o3test):
    n_components_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    results = {'Method': [],'Number components': [],'TP': [],'FP': [],'TN': [],'FN': [],'Accuracy': [],'Precision': [],'Recall': [],'F1 Score': [],'ConfusionMatrix': []}

    for n_components in n_components_list:
        # Skipping components that exceed limit
        if n_components > min(trainFeatures_browsing.shape):
            print(f"Skipping n_components={n_components} as it exceeds limit.")
            continue
        print(" min trainFeatures).shape ",min (np.vstack(trainFeatures_browsing).shape))
        if n_components > min(np.vstack(trainFeatures_browsing).shape):
            print(f"Skipping n_components={n_components} as it exceeds limit.")
            continue
        try:
            pca = PCA(n_components=n_components)
            i2train_pca = pca.fit_transform(np.vstack(trainFeatures_browsing))
        except ValueError as e:
            print(f"Error fitting PCA with n_components={n_components}: {e}")
            continue
        # Apply PCA
        pca = PCA(n_components=n_components)
        i2train_pca = pca.fit_transform(trainFeatures_browsing)
        i3Atest_pca = pca.transform(np.vstack((testFeatures_browsing, testFeatures_atck)))
        actual_labels = [class_label[0] for class_label in o3test]
        nObsTest = i3Atest_pca.shape[0]

        # Define kernels for One-Class SVM
        kernels = {'linear': svm.OneClassSVM(gamma='scale', kernel='linear', nu=0.5),
                   'rbf': svm.OneClassSVM(gamma='scale', kernel='rbf', nu=0.5),
                   'poly': svm.OneClassSVM(gamma='scale', kernel='poly', nu=0.5, degree=2)}

        # Fit and predict for each kernel
        for kernel_name, ocsvm_model in kernels.items():
            ocsvm_model.fit(i2train_pca)
            predictions = ocsvm_model.predict(i3Atest_pca)
            predicted_labels = [2.0 if pred == -1 else 0.0 for pred in predictions]

            # Calculate metrics
            tp = sum((predicted_label == actual_label == 2.0) for predicted_label, actual_label in zip(predicted_labels, actual_labels))
            tn = sum((predicted_label == actual_label == 0.0) for predicted_label, actual_label in zip(predicted_labels, actual_labels))
            fp = sum((predicted_label == 2.0) and (actual_label == 0.0) for predicted_label, actual_label in zip(predicted_labels, actual_labels))
            fn = sum((predicted_label == 0.0) and (actual_label == 2.0) for predicted_label, actual_label in zip(predicted_labels, actual_labels))

            accuracy, precision, recall, f1_score = calculate_metrics(tp, fp, tn, fn, nObsTest)
            cm = confusion_matrix(actual_labels, predicted_labels)

            # Append to results
            results['Method'].append(kernel_name)
            results['Number components'].append(n_components)
            results['TP'].append(tp)
            results['FP'].append(fp)
            results['TN'].append(tn)
            results['FN'].append(fn)
            results['Accuracy'].append(accuracy)
            results['Precision'].append(precision)
            results['Recall'].append(recall)
            results['F1 Score'].append(f1_score)
            results['ConfusionMatrix'].append(cm)

    # Convert results to DataFrame and save to Excel
    df = pd.DataFrame(results)

    # Find the best result based on F1 score
    best_result = df.loc[df['F1 Score'].idxmax()]
    plt.figure(figsize=(8, 6))
    sns.heatmap(best_result['ConfusionMatrix'], annot=True, cmap='Blues', fmt='d', xticklabels=['Human', 'Smart Bot'], yticklabels=['Human', 'Smart Bot'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f"Best Confusion Matrix (Method: {best_result['Method']}, Number of Components: {best_result['Number components']})")
    plt.show()


################################################################## -- 10 Classification based on Support Vector Machines without PCA -- #####################################################################################
# svm_classification(   trainFeatures_browsing,     testFeatures_browsing,      trainFeatures_attack,       testFeatures_atck,  i3train,    i3test,        o3train,        o3test)
def svm_classification( trainFeatures_browsing,     testFeatures_browsing,      trainFeatures_attack,       testFeatures_atck,  i3train,    i3test,        o3train,        o3test):

    i3train = np.vstack((trainFeatures_browsing, trainFeatures_attack))
    i3Ctest = np.vstack((testFeatures_browsing, testFeatures_atck))
    actual_labels = o3test.flatten()
    nObsTest = len(actual_labels)

    kernels = ['linear', 'rbf', 'poly']
    results = {'Method': [],'TP': [],'FP': [],'TN': [],'FN': [],'Accuracy': [],'Precision': [],'Recall': [],'F1 Score': [],'ConfusionMatrix': []}

    for kernel in kernels:
        model = svm.SVC(kernel=kernel, degree=2 if kernel == 'poly' else 3)
        model.fit(i3train, o3train)
        predictions = model.predict(i3Ctest)

        tp = np.sum((predictions == 2) & (actual_labels == 2))
        tn = np.sum((predictions == 0) & (actual_labels == 0))
        fp = np.sum((predictions == 2) & (actual_labels == 0))
        fn = np.sum((predictions == 0) & (actual_labels == 2))

        accuracy, precision, recall, f1_score = calculate_metrics(tp, fp, tn, fn, nObsTest)
        cm = confusion_matrix(actual_labels, predictions)

        results['Method'].append(kernel.capitalize())
        results['TP'].append(tp)
        results['FP'].append(fp)
        results['TN'].append(tn)
        results['FN'].append(fn)
        results['Accuracy'].append(accuracy)
        results['Precision'].append(precision)
        results['Recall'].append(recall)
        results['F1 Score'].append(f1_score)
        results['ConfusionMatrix'].append(cm)

    # Convert results to DataFrame and save to Excel
    df = pd.DataFrame(results)

    # Find the best result based on F1 score
    best_result = df.loc[df['F1 Score'].idxmax()]
    plt.figure(figsize=(8, 6))
    sns.heatmap(best_result['ConfusionMatrix'], annot=True, cmap='Blues', fmt='d', xticklabels=['Human', 'Bot'], yticklabels=['Human', 'Bot'])
    plt.title(f"Best Confusion Matrix (SVM Kernel: {best_result['Method']})")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

######################################### -- 10.2 Classification based on Support Vector Machines with PCA -- #####################################################################################
# svm_classification_with_pca(  trainFeatures_browsing,     testFeatures_browsing,      trainFeatures_attack,       testFeatures_atck,   o3train,        o3test)
def svm_classification_with_pca(trainFeatures_browsing,     testFeatures_browsing,      trainFeatures_attack,       testFeatures_atck,   o3train,        o3test):
    i3train = np.vstack((trainFeatures_browsing, trainFeatures_attack))
    i3Ctest = np.vstack((testFeatures_browsing, testFeatures_atck))

    components_to_test = [1, 3, 5, 7, 9, 11]
    
    results = {
        'Method': [],
        'Number of Components': [],
        'TP': [],
        'FP': [],
        'TN': [],
        'FN': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1 Score': [],
        'ConfusionMatrix': []
    }

    for n_components in components_to_test:
        pca = PCA(n_components=n_components)
        i3train_pca = pca.fit_transform(i3train)
        i3Ctest_pca = pca.transform(i3Ctest)

        # Define kernels for SVM
        kernels = {'linear': svm.SVC(kernel='linear'),
                   'rbf': svm.SVC(kernel='rbf'),
                   'poly': svm.SVC(kernel='poly', degree=2)}

        for kernel_name, svc_model in kernels.items():
            svc_model.fit(i3train_pca, o3train.ravel())
            predictions = svc_model.predict(i3Ctest_pca)

            tp = np.sum((predictions == 2) & (o3test == 2))
            tn = np.sum((predictions == 0) & (o3test == 0))
            fp = np.sum((predictions == 2) & (o3test == 0))
            fn = np.sum((predictions == 0) & (o3test == 2))

            accuracy, precision, recall, f1_score = calculate_metrics(tp, fp, tn, fn, len(o3test))
            cm = confusion_matrix(o3test, predictions)

            results['Method'].append(kernel_name.capitalize())
            results['Number of Components'].append(n_components)
            results['TP'].append(tp)
            results['FP'].append(fp)
            results['TN'].append(tn)
            results['FN'].append(fn)
            results['Accuracy'].append(accuracy)
            results['Precision'].append(precision)
            results['Recall'].append(recall)
            results['F1 Score'].append(f1_score)
            results['ConfusionMatrix'].append(cm)

    # Convert results to DataFrame and save to Excel
    df = pd.DataFrame(results)

    # Find the best result based on F1 score
    best_result = df.loc[df['F1 Score'].idxmax()]
    plt.figure(figsize=(8, 6))
    sns.heatmap(best_result['ConfusionMatrix'], annot=True, cmap='Blues', fmt='d',
                xticklabels=['Normal', 'DNS'], yticklabels=['Normal', 'DNS'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f"Best Confusion Matrix (SVM {best_result['Method']} with {best_result['Number of Components']} PCA Components)")
    plt.show()

################################### -- 12 Classification based on Neural Networks without pca -- #########################################################################################################
# neural_network_classification(    trainFeatures_browsing, testFeatures_browsing,  trainFeatures_attack,   testFeatures_atck,  o3train,        o3test)
def neural_network_classification(  trainFeatures_normal,   testFeatures_normal,    trainFeatures_dns,      testFeatures_dns,   o3trainClass,   o3testClass):
    i3train = np.vstack((trainFeatures_normal, trainFeatures_dns))
    i3Ctest = np.vstack((testFeatures_normal, testFeatures_dns))

    scaler = MaxAbsScaler().fit(i3train)
    i3trainN = scaler.transform(i3train)
    i3CtestN = scaler.transform(i3Ctest)

    alpha = 1
    max_iter = 100000
    clf = MLPClassifier(solver='lbfgs', alpha=alpha, hidden_layer_sizes=(20,), max_iter=max_iter)
    clf.fit(i3trainN, o3trainClass)
    LT = clf.predict(i3CtestN)

    tp_nn, fn_nn, tn_nn, fp_nn = 0, 0, 0, 0
    actual_labels = []
    predicted_labels = []
    results = []
    nObsTest, nFea = i3CtestN.shape

    for i in range(nObsTest):
        actual_labels.append(o3testClass[i][0])
        if LT[i] == o3testClass[i][0]:
                if LT[i] == 2.0:  # Comparando com o valor numérico correspondente à classe 'DNS'
                    predicted_labels.append(2.0)  # Predicted as DNS (anomaly)
                    tp_nn += 1
                else:
                    predicted_labels.append(0.0)  # Predicted as Normal
                    tn_nn += 1
        else:
            if LT[i] == 2.0:  # Comparando com o valor numérico correspondente à classe 'DNS'
                predicted_labels.append(2.0)  # Predicted as DNS (anomaly)
                fp_nn += 1
            else:
                predicted_labels.append(0.0)  # Predicted as Normal
                fn_nn += 1


    accuracy_nn = ((tp_nn + tn_nn) / (tp_nn + tn_nn + fp_nn + fn_nn)) * 100
    precision_nn = (tp_nn / (tp_nn + fp_nn)) * 100 if (tp_nn + fp_nn) != 0 else 0
    recall_nn = (tp_nn / (tp_nn + fn_nn)) * 100 if (tp_nn + fn_nn) != 0 else 0
    f1_score_nn = (2 * (precision_nn * recall_nn)) / (precision_nn + recall_nn) if (
            precision_nn + recall_nn) != 0 else 0
    
    confusionMatrix = confusion_matrix(actual_labels, predicted_labels)

    results.append({
        'TP': tp_nn,
        'FP': fp_nn,
        'TN': tn_nn,
        'FN': fn_nn,
        'Recall': recall_nn,
        'Accuracy': accuracy_nn,
        'Precision': precision_nn,
        'F1 Score': f1_score_nn,
        'Confusion Matrix': confusionMatrix,
    })

    df = pd.DataFrame(results)
    # df.to_excel(name_excel+'resultados_redes_neurais.xlsx', index=False)

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusionMatrix, annot=True, cmap='Blues', fmt='d',
                xticklabels=['Human', 'Bot'], yticklabels=['Human', 'Bot'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Best Confusion Matrix based on Neural Networks without PCA')
    plt.show()

################################### -- 12 Classification based on Neural Networks with pca -- ##################################################
# neural_network_classification_with_pca(   trainFeatures_browsing, testFeatures_browsing,  trainFeatures_attack,   testFeatures_atck,  o3train,        o3test)
def neural_network_classification_with_pca( trainFeatures_normal,   testFeatures_normal,    trainFeatures_dns,      testFeatures_dns,   o3trainClass,   o3testClass):
    components_to_test = [1, 5, 10, 15, 20]

    results = []
    i3train = np.vstack((trainFeatures_normal, trainFeatures_dns))
    i3Ctest = np.vstack((testFeatures_normal, testFeatures_dns))

    for n_components in components_to_test:
        pca = PCA(n_components=n_components)
        i3train_pca = pca.fit_transform(i3train)
        i3Ctest_pca = pca.transform(i3Ctest)

        scaler = MaxAbsScaler().fit(i3train_pca)
        i3trainN_pca = scaler.transform(i3train_pca)
        i3CtestN_pca = scaler.transform(i3Ctest_pca)

        alpha = 1
        max_iter = 100000
        clf = MLPClassifier(solver='lbfgs', alpha=alpha, hidden_layer_sizes=(20,), max_iter=max_iter)
        clf.fit(i3trainN_pca, o3trainClass)
        LT = clf.predict(i3CtestN_pca)

        tp_nn, fn_nn, tn_nn, fp_nn = 0, 0, 0, 0
        actual_labels = []
        predicted_labels = []

        nObsTest, nFea = i3CtestN_pca.shape

        for i in range(nObsTest):
            actual_labels.append(o3testClass[i][0])
            if LT[i] == o3testClass[i][0]:
                if LT[i] == 2.0:  # Comparando com o valor numérico correspondente à classe 'DNS'
                    predicted_labels.append(2.0)  # Predicted as DNS (anomaly)
                    tp_nn += 1
                else:
                    predicted_labels.append(0.0)  # Predicted as Normal
                    tn_nn += 1
            else:
                if LT[i] == 2.0:  # Comparando com o valor numérico correspondente à classe 'DNS'
                    predicted_labels.append(2.0)  # Predicted as DNS (anomaly)
                    fp_nn += 1
                else:
                    predicted_labels.append(0.0)  # Predicted as Normal
                    fn_nn += 1

        accuracy_nn = ((tp_nn + tn_nn) / (tp_nn + tn_nn + fp_nn + fn_nn)) * 100
        precision_nn = (tp_nn / (tp_nn + fp_nn)) * 100 if (tp_nn + fp_nn) != 0 else 0
        recall_nn = (tp_nn / (tp_nn + fn_nn)) * 100 if (tp_nn + fn_nn) != 0 else 0
        f1_score_nn = (2 * (precision_nn * recall_nn)) / (precision_nn + recall_nn) if (
                precision_nn + recall_nn) != 0 else 0

        confusionMatrix = confusion_matrix(actual_labels, predicted_labels)

        results.append({
            'Components': n_components,
            'TP': tp_nn,
            'FP': fp_nn,
            'TN': tn_nn,
            'FN': fn_nn,
            'Recall': recall_nn,
            'Accuracy': accuracy_nn,
            'Precision': precision_nn,
            'F1 Score': f1_score_nn,
            'Confusion Matrix': confusionMatrix,
        })

    df = pd.DataFrame(results)

    # df.to_excel(name_excel+'_redes_neurais_pca.xlsx', index=False)
    
    best_f1_index = df['F1 Score'].idxmax()

    best_number_components=df.loc[best_f1_index,'Components']


    plt.figure(figsize=(8, 6))
    sns.heatmap(confusionMatrix, annot=True, cmap='Blues', fmt='d',
                xticklabels=['Human', 'Bot'], yticklabels=['Human', 'Bot'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Best Confusion Matrix based on Neural Networks with pca {best_number_components}')
    plt.show()