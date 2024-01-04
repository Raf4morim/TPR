import sys
import argparse
from netaddr import IPNetwork, IPAddress, IPSet
import pyshark
import numpy as np
import os
from os.path import exists
import warnings
import scipy.stats as stats
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.neural_network import MLPClassifier
from scipy.stats import multivariate_normal
from sklearn import svm
import time
import warnings
import seaborn as sns
import pandas as pd
warnings.filterwarnings('ignore')

# up_count      up_payload      down_count      down_payload

sampDelta = 5   # seconds
widths = 2      # sliding window width
slide = 1       # sliding window slide
#############################################################
# file = 'test.pcap'
# file = 'big.pcap'
# NETClient = ['172.20.10.0/25']
# NETClient = ['192.168.24.0/24']
#############################################################
NETClient = ['192.168.1.107/32'] 
file = 'Captures/test2.pcap'
#############################################################
# NETClient = ['192.168.0.163'] 
# file = 'Captures/attackSmartWind.pcap'
profileClassFile = "Captures/attackSeqWind.pcap"
# file = "Captures/attackSeqWind.pcap"
# file = 'Captures/brwsg2Wind.pcap'
#############################################################
# file = 'Captures/attackSeqVM.pcap'
# file = 'Captures/brwsg1VM.pcap'
# NETClient = ['10.0.2.15']   # file = 'Captures/browsingAmorimVM.pcap'
#############################################################
# NETServer = ['157.240.212.60']  # Apenas para o do Wpp por agora
NETServer = ['0.0.0.0/0']

samplesMatrices = []


###########################################################################################
############################################# -- 7 -- Centroids Distances SEM PCA #########################
def centroids_distances(trainFeatures, o2trainClass, testFeatures_normal, testFeatures_dns, o3testClass,name_excel):
    i2train = np.vstack((trainFeatures))
    
    results = []
    actual_labels = []
    predicted_labels = []
    centroids = {}
    best_f1_index = 0  # Initializing the index of the best F1 score

    for c in range(1):  # Only the first class (client class)
        pClass = (o2trainClass == c).flatten()
        centroids.update({c: np.mean(i2train[pClass, :], axis=0)})

    i3Atest = np.vstack((testFeatures_normal, testFeatures_dns))

    print('\n-- Anomaly Detection based on Centroids Distances without PCA --')
    nObsTest, nFea = i3Atest.shape

    threshold_values = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.2, 1.1, 1.5, 2.0, 3, 5, 6, 10]

    threshold_metrics = []
    best_confusion_matrix = None  # Initializing outside the conditional block

    for AnomalyThreshold in threshold_values:
        tp_centroids = 0
        fp_centroids = 0
        tn_centroids = 0
        fn_centroids = 0
        actual_labels = []
        predicted_labels = []

        for i in range(nObsTest):
            actual_labels.append(o3testClass[i][0])
            x = i3Atest[i]
            dists = [np.linalg.norm(x - centroids[0])]
            if min(dists) > AnomalyThreshold:
                result = "Anomaly"
                predicted_labels.append(2.0)
                if o3testClass[i][0] == 2.0:
                    tp_centroids += 1
                else:
                    fp_centroids += 1
            else:
                result = "OK"
                predicted_labels.append(0.0)
                if o3testClass[i][0] == 2.0:
                    fn_centroids += 1
                else:
                    tn_centroids += 1

        accuracy_centroids = ((tp_centroids + tn_centroids) / nObsTest) * 100
        precision_centroids = (tp_centroids / (tp_centroids + fp_centroids)) * 100 if (tp_centroids + fp_centroids) != 0 else 0
        recall_centroids = (tp_centroids / (tp_centroids + fn_centroids)) * 100 if (tp_centroids + fn_centroids) != 0 else 0
        f1_score_centroids = (2 * (precision_centroids * recall_centroids) / (precision_centroids + recall_centroids)) if (precision_centroids + recall_centroids) != 0 else 0

        results.append({
            'AnomalyThreshold': AnomalyThreshold,
            'TP': tp_centroids,
            'FP': fp_centroids,
            'TN': tn_centroids,
            'FN': fn_centroids,
            'Accuracy': accuracy_centroids,
            'Precision': precision_centroids,
            'Recall': recall_centroids,
            'F1 Score': f1_score_centroids,
            'ConfusionMatrix': confusion_matrix(actual_labels, predicted_labels)
        })

    df = pd.DataFrame(results)
    df.to_excel(name_excel+'resultados_centroid.xlsx', index=False)

    best_f1_index = df['F1 Score'].idxmax()
    best_threshold = df.loc[best_f1_index, 'AnomalyThreshold']
    best_confusion_matrix = df.loc[best_f1_index, 'ConfusionMatrix']

    plt.figure(figsize=(8, 6))
    sns.heatmap(best_confusion_matrix, annot=True, cmap='Blues', fmt='d',
                xticklabels=['DNS', 'Normal'], yticklabels=['Normal', 'DNS'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Best Confusion Matrix (Threshold: {best_threshold})')
    plt.show()


####################################### -- 7.2 -- Centroids Distances Com PCA ######################### ##
def centroids_distances_with_pca(trainFeatures, o2trainClass, testFeatures_normal, testFeatures_dns, o3testClass,name_excel):
    components_to_test = [10, 15, 20, 25]
    results = []
    best_confusion_matrix = None  # Initializing outside the conditional block

    for n_components in components_to_test:
        i2train = np.vstack(trainFeatures)

        actual_labels = []
        predicted_labels = []

        pca = PCA(n_components=n_components)
        i2train_pca = pca.fit_transform(i2train)

        centroids = {}
        for c in range(1):
            pClass = (o2trainClass == c).flatten()
            centroids.update({c: np.mean(i2train_pca[pClass, :], axis=0)})

        i3Atest = np.vstack((testFeatures_normal, testFeatures_dns))
        i3Atest_pca = pca.transform(i3Atest)

        print(f'\n-- Anomaly Detection based on Centroids Distances (Components: {n_components}) --')
        nObsTest, nFea = i3Atest_pca.shape

        threshold_values = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.2, 1.1, 1.5, 2.0, 3, 5, 6, 10]

        threshold_metrics = []

        for AnomalyThreshold in threshold_values:
            tp_centroids = 0
            fp_centroids = 0
            tn_centroids = 0
            fn_centroids = 0
            actual_labels = []
            predicted_labels = []

            for i in range(nObsTest):
                actual_labels.append(o3testClass[i][0])
                x = i3Atest_pca[i]
                dists = [np.linalg.norm(x - centroids[0])]
                if min(dists) > AnomalyThreshold:
                    result = "Anomaly"
                    predicted_labels.append(2.0)
                    if o3testClass[i][0] == 2.0:
                        tp_centroids += 1
                    else:
                        fp_centroids += 1
                else:
                    result = "OK"
                    predicted_labels.append(0.0)
                    if o3testClass[i][0] == 2.0:
                        fn_centroids += 1
                    else:
                        tn_centroids += 1

            accuracy_centroids = ((tp_centroids + tn_centroids) / nObsTest) * 100
            precision_centroids = (tp_centroids / (tp_centroids + fp_centroids)) * 100 if (tp_centroids + fp_centroids) != 0 else 0
            recall_centroids = (tp_centroids / (tp_centroids + fn_centroids)) * 100 if (tp_centroids + fn_centroids) != 0 else 0
            f1_score_centroids = (2 * (precision_centroids * recall_centroids) / (precision_centroids + recall_centroids)) if (precision_centroids + recall_centroids) != 0 else 0

            results.append({
                'AnomalyThreshold': AnomalyThreshold,
                'Número de Componentes': n_components,
                'TP': tp_centroids,
                'FP': fp_centroids,
                'TN': tn_centroids,
                'FN': fn_centroids,
                'Accuracy': accuracy_centroids,
                'Precision': precision_centroids,
                'Recall': recall_centroids,
                'F1 Score': f1_score_centroids,
                'ConfusionMatrix': confusion_matrix(actual_labels, predicted_labels)
            })

    df = pd.DataFrame(results)
    df.to_excel(name_excel+'resultados_centroid_pca.xlsx', index=False)

    best_f1_index = df['F1 Score'].idxmax()
    best_threshold = df.loc[best_f1_index, 'AnomalyThreshold']
    best_number_components = df.loc[best_f1_index, 'Número de Componentes']
    best_confusion_matrix = df.loc[best_f1_index, 'ConfusionMatrix']

    plt.figure(figsize=(8, 6))
    sns.heatmap(best_confusion_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=['DNS', 'Normal'], yticklabels=['Normal', 'DNS'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Best Confusion Matrix: Threshold: {best_threshold} Number of components: {best_number_components}')
    plt.show()



######################################### -- 8 -- Anomaly Detection based on One Class Support Vector Machines WITHOUT PCA ###############################
def one_class_svm(trainFeatures, testFeatures_normal, testFeatures_dns, o3testClass,name_excel):
    tp_linear, fn_linear, tn_linear, fp_linear = 0, 0, 0, 0
    tp_rbf, fn_rbf, tn_rbf, fp_rbf = 0, 0, 0, 0
    tp_poly, fn_poly, tn_poly, fp_poly = 0, 0, 0, 0
    
    i2train = np.vstack(trainFeatures)
    i3Atest = np.vstack((testFeatures_normal, testFeatures_dns))
    
    results = []
    AnomResults = {-1: "Anomaly", 1: "OK"}
    nObsTest, nFea = i3Atest.shape

    nu = 0.5
    ocsvm = svm.OneClassSVM(gamma='scale', kernel='linear', nu=nu).fit(i2train)
    rbf_ocsvm = svm.OneClassSVM(gamma='scale', kernel='rbf', nu=nu).fit(i2train)
    poly_ocsvm = svm.OneClassSVM(gamma='scale', kernel='poly', nu=nu, degree=2).fit(i2train)

    L1 = ocsvm.predict(i3Atest)
    L2 = rbf_ocsvm.predict(i3Atest)
    L3 = poly_ocsvm.predict(i3Atest)

    actual_labels_linear = []
    predicted_labels_linear = []

    actual_labels_rbf = []
    predicted_labels_rbf = []

    actual_labels_poly = []
    predicted_labels_poly = []

    for i in range(nObsTest):
        # print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[o3testClass[i][0]],AnomResults[L1[i]],AnomResults[L2[i]],AnomResults[L3[i]]))
        actual_labels_linear.append(o3testClass[i][0])
        actual_labels_rbf.append(o3testClass[i][0])
        actual_labels_poly.append(o3testClass[i][0])

        # Linear
        if AnomResults[L1[i]] == "Anomaly":
            predicted_labels_linear.append(2.0)
            if o3testClass[i][0] == 2:
                tp_linear += 1
            else:
                fp_linear += 1
        else:
            predicted_labels_linear.append(0.0)
            if o3testClass[i][0] == 2:
                fn_linear += 1
            else:
                tn_linear += 1

        # RBF
        if AnomResults[L2[i]] == "Anomaly":
            predicted_labels_rbf.append(2.0)
            if o3testClass[i][0] == 2:
                tp_rbf += 1
            else:
                fp_rbf += 1
        else:
            predicted_labels_rbf.append(0.0)
            if o3testClass[i][0] == 2:
                fn_rbf += 1
            else:
                tn_rbf += 1

        # Poly
        if AnomResults[L3[i]] == "Anomaly":
            predicted_labels_poly.append(2.0)
            if o3testClass[i][0] == 2:
                tp_poly += 1
            else:
                fp_poly += 1
        else:
            predicted_labels_poly.append(0.0)
            if o3testClass[i][0] == 2:
                fn_poly += 1
            else:
                tn_poly += 1

    accuracy_linear = ((tp_linear + tn_linear) / nObsTest) * 100
    precision_linear = (tp_linear / (tp_linear + fp_linear)) * 100 if tp_linear + fp_linear > 0 else 0
    recall_linear = (tp_linear / (tp_linear + fn_linear)) * 100 if tp_linear + fn_linear > 0 else 0
    f1_score_linear = (2 * (precision_linear * recall_linear) / (precision_linear + recall_linear))  if (precision_linear + recall_linear) != 0 else 0

    accuracy_rbf = ((tp_rbf + tn_rbf) / nObsTest) * 100
    precision_rbf = (tp_rbf / (tp_rbf + fp_rbf)) * 100 if tp_rbf + fp_rbf > 0 else 0
    recall_rbf = (tp_rbf / (tp_rbf + fn_rbf)) * 100 if tp_rbf + fn_rbf > 0 else 0
    f1_score_rbf = (2 * (precision_rbf * recall_rbf) / (precision_rbf + recall_rbf))  if (precision_rbf + recall_rbf) != 0 else 0

    accuracy_poly = ((tp_poly + tn_poly) / nObsTest) * 100
    precision_poly = (tp_poly / (tp_poly + fp_poly)) * 100 if tp_poly + fp_poly > 0 else 0
    recall_poly = (tp_poly / (tp_poly + fn_poly)) * 100 if tp_poly + fn_poly > 0 else 0
    f1_score_poly = (2 * (precision_poly * recall_poly) / (precision_poly + recall_poly))  if (precision_poly + recall_poly) != 0 else 0

    results = {
        'Method': ['Linear', 'RBF', 'Poly'],
        'TP': [tp_linear, tp_rbf, tp_poly],
        'FP': [fp_linear, fp_rbf, fp_poly],
        'TN': [tn_linear, tn_rbf, tn_poly],
        'FN': [fn_linear, fn_rbf, fn_poly],
        'Accuracy': [accuracy_linear, accuracy_rbf, accuracy_poly],
        'Precision': [precision_linear, precision_rbf, precision_poly],
        'Recall': [recall_linear, recall_rbf, recall_poly],
        'F1 Score': [f1_score_linear, f1_score_rbf, f1_score_poly],
        'ConfusionMatrix': [
            confusion_matrix(actual_labels_linear, predicted_labels_linear),
            confusion_matrix(actual_labels_rbf, predicted_labels_rbf),
            confusion_matrix(actual_labels_poly, predicted_labels_poly)
        ]
    }

    # Create a DataFrame from the results list
    df = pd.DataFrame(results)

    # Save the DataFrame to an Excel file
    df.to_excel(name_excel+'resultados_OneClassSVM.xlsx', index=False)

    # Find the index of the row with the best F1 score
    best_f1_index = df['F1 Score'].idxmax()

    best_confusion_matrix = df.loc[best_f1_index, 'ConfusionMatrix']
    best_method = df.loc[best_f1_index, 'Method']

    # Plot the best confusion matrix if it exists
    plt.figure(figsize=(8, 6))
    sns.heatmap(best_confusion_matrix, annot=True, cmap='Oranges', fmt='d',
                xticklabels=['Negative', 'Positive'], yticklabels=['False', 'True'])
    plt.xlabel('Predicted label')
    plt.ylabel('Actual label')
    plt.title(f'Best Confusion Matrix'+ name_excel+'(Kernel:{best_method} )')
    plt.show()



##################################################################################### -- 8.2 -- Anomaly Detection based on One Class Support Vector Machines with pca###############################
def one_class_svm_with_pca(trainFeatures, testFeatures_normal, testFeatures_dns, o3testClass,name_excel):
    n_components_list = [1, 5, 10, 15, 16, 17, 18, 19, 20, 21]

    results = []
    all_results = []

    for n_components in n_components_list:
        pca = PCA(n_components=n_components)
        i2train_pca = pca.fit_transform(trainFeatures)
        i3Atest_pca = pca.transform(np.vstack((testFeatures_normal, testFeatures_dns)))

        nu = 0.5
        ocsvm = svm.OneClassSVM(gamma='scale', kernel='linear', nu=nu).fit(i2train_pca)
        rbf_ocsvm = svm.OneClassSVM(gamma='scale', kernel='rbf', nu=nu).fit(i2train_pca)
        poly_ocsvm = svm.OneClassSVM(gamma='scale', kernel='poly', nu=nu, degree=2).fit(i2train_pca)

        L1 = ocsvm.predict(i3Atest_pca)
        L2 = rbf_ocsvm.predict(i3Atest_pca)
        L3 = poly_ocsvm.predict(i3Atest_pca)

        tp_linear, fn_linear, tn_linear, fp_linear = 0, 0, 0, 0
        tp_rbf, fn_rbf, tn_rbf, fp_rbf = 0, 0, 0, 0
        tp_poly, fn_poly, tn_poly, fp_poly = 0, 0, 0, 0

        AnomResults = {-1: "Anomaly", 1: "OK"}
        actual_labels_linear = []
        predicted_labels_linear = []

        actual_labels_rbf = []
        predicted_labels_rbf = []

        actual_labels_poly = []
        predicted_labels_poly = []

        nObsTest, nFea = i3Atest_pca.shape
        for i in range(nObsTest):
            actual_labels_linear.append(o3testClass[i][0])
            actual_labels_rbf.append(o3testClass[i][0])
            actual_labels_poly.append(o3testClass[i][0])

            # Linear
            if AnomResults[L1[i]] == "Anomaly":
                predicted_labels_linear.append(2.0)
                if o3testClass[i][0] == 2:
                    tp_linear += 1
                else:
                    fp_linear += 1
            else:
                predicted_labels_linear.append(0.0)
                if o3testClass[i][0] == 2:
                    fn_linear += 1
                else:
                    tn_linear += 1

            # RBF
            if AnomResults[L2[i]] == "Anomaly":
                predicted_labels_rbf.append(2.0)
                if o3testClass[i][0] == 2:
                    tp_rbf += 1
                else:
                    fp_rbf += 1
            else:
                predicted_labels_rbf.append(0.0)
                if o3testClass[i][0] == 2:
                    fn_rbf += 1
                else:
                    tn_rbf += 1

            # Poly
            if AnomResults[L3[i]] == "Anomaly":
                predicted_labels_poly.append(2.0)
                if o3testClass[i][0] == 2:
                    tp_poly += 1
                else:
                    fp_poly += 1
            else:
                predicted_labels_poly.append(0.0)
                if o3testClass[i][0] == 2:
                    fn_poly += 1
                else:
                    tn_poly += 1

        accuracy_linear = ((tp_linear + tn_linear) / nObsTest) * 100
        precision_linear = (tp_linear / (tp_linear + fp_linear)) * 100 if tp_linear + fp_linear > 0 else 0

        accuracy_rbf = ((tp_rbf + tn_rbf) / nObsTest) * 100
        precision_rbf = (tp_rbf / (tp_rbf + fp_rbf)) * 100 if tp_rbf + fp_rbf > 0 else 0

        accuracy_poly = ((tp_poly + tn_poly) / nObsTest) * 100
        precision_poly = (tp_poly / (tp_poly + fp_poly)) * 100 if tp_poly + fp_poly > 0 else 0

        recall_linear = (tp_linear / (tp_linear + fn_linear)) * 100 if tp_linear + fn_linear > 0 else 0
        recall_rbf = (tp_rbf / (tp_rbf + fn_rbf)) * 100 if tp_rbf + fn_rbf > 0 else 0
        recall_poly = (tp_poly / (tp_poly + fn_poly)) * 100 if tp_poly + fn_poly > 0 else 0

        f1_score_linear = (2 * (precision_linear * recall_linear) / (precision_linear + recall_linear)) if (precision_linear + recall_linear) != 0 else 0
        f1_score_rbf = (2 * (precision_rbf * recall_rbf) / (precision_rbf + recall_rbf)) if (precision_rbf + recall_rbf) != 0 else 0
        f1_score_poly = (2 * (precision_poly * recall_poly) / (precision_poly + recall_poly))  if (precision_poly + recall_poly) != 0 else 0

        results = {
            'Method': ['Linear', 'RBF', 'Poly'],
            'Number components': n_components,
            'TP': [tp_linear, tp_rbf, tp_poly],
            'FP': [fp_linear, fp_rbf, fp_poly],
            'TN': [tn_linear, tn_rbf, tn_poly],
            'FN': [fn_linear, fn_rbf, fn_poly],
            'Accuracy': [accuracy_linear, accuracy_rbf, accuracy_poly],
            'Precision': [precision_linear, precision_rbf, precision_poly],
            'Recall': [recall_linear, recall_rbf, recall_poly],
            'F1 Score': [f1_score_linear, f1_score_rbf, f1_score_poly],
            'ConfusionMatrix': [
                confusion_matrix(actual_labels_linear, predicted_labels_linear),
                confusion_matrix(actual_labels_rbf, predicted_labels_rbf),
                confusion_matrix(actual_labels_poly, predicted_labels_poly)
            ]
        }
        all_results.append(results)

    # DataFrame from the results
    df = pd.concat([pd.DataFrame(res) for res in all_results], ignore_index=True)

    # DataFrame to an Excel file
    df.to_excel(name_excel+'resultados_OneClassSVM_pca.xlsx', index=False)

    # Find the index of the row with the best F1 score
    best_f1_index = df['F1 Score'].idxmax()

    best_confusion_matrix = df.loc[best_f1_index, 'ConfusionMatrix']

    # Plot the best confusion matrix if it exists
    plt.figure(figsize=(8, 6))
    sns.heatmap(best_confusion_matrix, annot=True, cmap='Blues', fmt='d',
                xticklabels=['DNS', 'Normal'], yticklabels=['Normal', 'DNS'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Best Confusion Matrix (Components: {n_components})')
    plt.show()

################################################################## -- 10 Classification based on Support Vector Machines without PCA -- #####################################################################################
def svm_classification(trainFeatures_normal, testFeatures_normal, trainFeatures_dns, testFeatures_dns, o3trainClass, o3testClass,name_excel):
    i3train = np.vstack((trainFeatures_normal, trainFeatures_dns))
    i3Ctest = np.vstack((testFeatures_normal, testFeatures_dns))

    svc = svm.SVC(kernel='linear').fit(i3train, o3trainClass)
    rbf_svc = svm.SVC(kernel='rbf').fit(i3train, o3trainClass)
    poly_svc = svm.SVC(kernel='poly', degree=2).fit(i3train, o3trainClass)

    L1 = svc.predict(i3Ctest)
    L2 = rbf_svc.predict(i3Ctest)
    L3 = poly_svc.predict(i3Ctest)

    tp_linear, fn_linear, tn_linear, fp_linear = 0, 0, 0, 0
    actual_labels_linear = []
    predicted_labels_linear = []

    tp_rbf, fn_rbf, tn_rbf, fp_rbf = 0, 0, 0, 0
    actual_labels_rbf = []
    predicted_labels_rbf = []

    tp_poly, fn_poly, tn_poly, fp_poly = 0, 0, 0, 0
    actual_labels_poly = []
    predicted_labels_poly = []

    nObsTest, nFea = i3Ctest.shape

    AnomResults = {2.0: "Anomaly", 0: "OK"}  

    for i in range(nObsTest):
        actual_labels_linear.append(o3testClass[i][0])
        actual_labels_rbf.append(o3testClass[i][0])
        actual_labels_poly.append(o3testClass[i][0])
        # Linear
        if AnomResults[L1[i]] == "Anomaly":
            predicted_labels_linear.append(2.0)
            if o3testClass[i][0] == 2:
                tp_linear += 1
            else:
                fp_linear += 1
        else:
            predicted_labels_linear.append(0.0)
            if o3testClass[i][0] == 2:
                fn_linear += 1
            else:
                tn_linear += 1

        # RBF
        if AnomResults[L2[i]] == "Anomaly":
            predicted_labels_rbf.append(2.0)
            if o3testClass[i][0] == 2:
                tp_rbf += 1
            else:
                fp_rbf += 1
        else:
            predicted_labels_rbf.append(0.0)
            if o3testClass[i][0] == 2:
                fn_rbf += 1
            else:
                tn_rbf += 1

        # Poly
        if AnomResults[L3[i]] == "Anomaly":
            predicted_labels_poly.append(2.0)
            if o3testClass[i][0] == 2:
                tp_poly += 1
            else:
                fp_poly += 1
        else:
            predicted_labels_poly.append(0.0)
            if o3testClass[i][0] == 2:
                fn_poly += 1
            else:
                tn_poly += 1

    accuracy_linear = ((tp_linear + tn_linear) / nObsTest) * 100
    precision_linear = (tp_linear / (tp_linear + fp_linear)) * 100 if tp_linear + fp_linear > 0 else 0

    accuracy_rbf = ((tp_rbf + tn_rbf) / nObsTest) * 100
    precision_rbf = (tp_rbf / (tp_rbf + fp_rbf)) * 100 if tp_rbf + fp_rbf > 0 else 0

    accuracy_poly = ((tp_poly + tn_poly) / nObsTest) * 100
    precision_poly = (tp_poly / (tp_poly + fp_poly)) * 100 if tp_poly + fp_poly > 0 else 0

    recall_linear = (tp_linear / (tp_linear + fn_linear)) * 100 if tp_linear + fn_linear > 0 else 0
    recall_rbf = (tp_rbf / (tp_rbf + fn_rbf)) * 100 if tp_rbf + fn_rbf > 0 else 0
    recall_poly = (tp_poly / (tp_poly + fn_poly)) * 100 if tp_poly + fn_poly > 0 else 0

    f1_score_linear = (2 * (precision_linear * recall_linear) / (precision_linear + recall_linear))  if (
            precision_linear + recall_linear) != 0 else 0
    f1_score_rbf = (2 * (precision_rbf * recall_rbf) / (precision_rbf + recall_rbf))  if (
            precision_rbf + recall_rbf) != 0 else 0
    f1_score_poly = (2 * (precision_poly * recall_poly) / (precision_poly + recall_poly))  if (
            precision_poly + recall_poly) != 0 else 0

    results = {
        'Method': ['Linear', 'RBF', 'Poly'],
        'TP': [tp_linear, tp_rbf, tp_poly],
        'FP': [fp_linear, fp_rbf, fp_poly],
        'TN': [tn_linear, tn_rbf, tn_poly],
        'FN': [fn_linear, fn_rbf, fn_poly],
        'Accuracy': [accuracy_linear, accuracy_rbf, accuracy_poly],
        'Precision': [precision_linear, precision_rbf, precision_poly],
        'Recall': [recall_linear, recall_rbf, recall_poly],
        'F1 Score': [f1_score_linear, f1_score_rbf, f1_score_poly],
        'ConfusionMatrix': [
            confusion_matrix(actual_labels_linear, predicted_labels_linear),
            confusion_matrix(actual_labels_rbf, predicted_labels_rbf),
            confusion_matrix(actual_labels_poly, predicted_labels_poly)
        ]
    }

    df = pd.DataFrame(results)

    df.to_excel(name_excel+'resultados_SVM.xlsx', index=False)

    best_f1_index = df['F1 Score'].idxmax()

    best_confusion_matrix = df.loc[best_f1_index, 'ConfusionMatrix']

    plt.figure(figsize=(8, 6))
    sns.heatmap(best_confusion_matrix, annot=True, cmap='Blues', fmt='d',
                xticklabels=['DNS', 'Normal'], yticklabels=['Normal', 'DNS'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Best Confusion Matrix')
    plt.show()

######################################### -- 10.2 Classification based on Support Vector Machines with PCA -- #####################################################################################
def svm_classification_with_pca(trainFeatures_normal, testFeatures_normal, trainFeatures_dns, testFeatures_dns, o3trainClass, o3testClass,name_excel):
    i3train = np.vstack((trainFeatures_normal, trainFeatures_dns))
    i3Ctest = np.vstack((testFeatures_normal, testFeatures_dns))

    # Define a range of components to test
    components_to_test = [5, 10, 15, 20, 30, 40]
    results = []
    all_results = []

    for n_components in components_to_test:
        # Initialize PCA and fit-transform the data
        pca = PCA(n_components=n_components)
        i3train_pca = pca.fit_transform(i3train)
        i3Ctest_pca = pca.transform(i3Ctest)
        svc = svm.SVC(kernel='linear').fit(i3train_pca, o3trainClass)
        rbf_svc = svm.SVC(kernel='rbf').fit(i3train_pca, o3trainClass)
        poly_svc = svm.SVC(kernel='poly', degree=2).fit(i3train_pca, o3trainClass)

        L1 = svc.predict(i3Ctest_pca)
        L2 = rbf_svc.predict(i3Ctest_pca)
        L3 = poly_svc.predict(i3Ctest_pca)

        tp_linear, fn_linear, tn_linear, fp_linear = 0, 0, 0, 0
        actual_labels_linear = []
        predicted_labels_linear = []

        tp_rbf, fn_rbf, tn_rbf, fp_rbf = 0, 0, 0, 0
        actual_labels_rbf = []
        predicted_labels_rbf = []

        tp_poly, fn_poly, tn_poly, fp_poly = 0, 0, 0, 0
        actual_labels_poly = []
        predicted_labels_poly = []

        nObsTest, nFea = i3Ctest.shape

        AnomResults = {2.0: "Anomaly", 0: "OK", 1.0:"OK"}  # Bruno is 0 and DNS is 2 and Marta "1.0"

        for i in range(nObsTest):
            # print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i, Classes[o3testClass[i][0]],Classes[L1[i]],Classes[L2[i]],Classes[L3[i]]))
            actual_labels_linear.append(o3testClass[i][0])
            actual_labels_rbf.append(o3testClass[i][0])
            actual_labels_poly.append(o3testClass[i][0])
            # Linear
            if AnomResults[L1[i]] == "Anomaly":
                predicted_labels_linear.append(2.0)  # Predicted as DNS (anomaly)
                if o3testClass[i][0] == 2:  # DNS class
                    tp_linear += 1  # True Positive
                else:  # Marta/Bruno class
                    fp_linear += 1  # False Positive
            else:  # OK
                predicted_labels_linear.append(0.0)
                if o3testClass[i][0] == 2:
                    fn_linear += 1  # False Negative
                else:
                    tn_linear += 1  # True Negative

            # RBF
            if AnomResults[L2[i]] == "Anomaly":
                predicted_labels_rbf.append(2.0)  # Predicted as DNS (anomaly)
                if o3testClass[i][0] == 2:  # DNS class
                    tp_rbf += 1  # True Positive
                else:  # Marta/Bruno class
                    fp_rbf += 1  # False Positive
            else:  # OK
                predicted_labels_rbf.append(0.0)
                if o3testClass[i][0] == 2:
                    fn_rbf += 1  # False Negative
                else:
                    tn_rbf += 1  # True Negative
            # Poly
            if AnomResults[L3[i]] == "Anomaly":
                predicted_labels_poly.append(2.0)  # Predicted as DNS (anomaly)
                if o3testClass[i][0] == 2:  # DNS class
                    tp_poly += 1  # True Positive
                else:  # Marta/Bruno class
                    fp_poly += 1  # False Positive
            else:  # OK
                predicted_labels_poly.append(0.0)
                if o3testClass[i][0] == 2:
                    fn_poly += 1  # False Negative
                else:
                    tn_poly += 1  # True Negative

        accuracy_linear = ((tp_linear + tn_linear) / nObsTest) * 100
        precision_linear = (tp_linear / (tp_linear + fp_linear)) * 100 if tp_linear + fp_linear > 0 else 0

        accuracy_rbf = ((tp_rbf + tn_rbf) / nObsTest) * 100
        precision_rbf = (tp_rbf / (tp_rbf + fp_rbf)) * 100 if tp_rbf + fp_rbf > 0 else 0

        accuracy_poly = ((tp_poly + tn_poly) / nObsTest) * 100
        precision_poly = (tp_poly / (tp_poly + fp_poly)) * 100 if tp_poly + fp_poly > 0 else 0

        recall_linear = (tp_linear / (tp_linear + fn_linear)) * 100 if tp_linear + fn_linear > 0 else 0
        recall_rbf = (tp_rbf / (tp_rbf + fn_rbf)) * 100 if tp_rbf + fn_rbf > 0 else 0
        recall_poly = (tp_poly / (tp_poly + fn_poly)) * 100 if tp_poly + fn_poly > 0 else 0

        f1_score_linear = (2 * (precision_linear * recall_linear) / (
                precision_linear + recall_linear))  if (precision_linear + recall_linear) != 0 else 0
        f1_score_rbf = (2 * (precision_rbf * recall_rbf) / (precision_rbf + recall_rbf))  if (
                precision_rbf + recall_rbf) != 0 else 0
        f1_score_poly = (2 * (precision_poly * recall_poly) / (precision_poly + recall_poly))  if (
                precision_poly + recall_poly) != 0 else 0

        results = {
            'Method': ['Linear', 'RBF', 'Poly'],
            'Number components': n_components,
            'TP': [tp_linear, tp_rbf, tp_poly],
            'FP': [fp_linear, fp_rbf, fp_poly],
            'TN': [tn_linear, tn_rbf, tn_poly],
            'FN': [fn_linear, fn_rbf, fn_poly],
            'Accuracy': [accuracy_linear, accuracy_rbf, accuracy_poly],
            'Precision': [precision_linear, precision_rbf, precision_poly],
            'Recall': [recall_linear, recall_rbf, recall_poly],
            'F1 Score': [f1_score_linear, f1_score_rbf, f1_score_poly],
            'ConfusionMatrix': [
                confusion_matrix(actual_labels_linear, predicted_labels_linear),
                confusion_matrix(actual_labels_rbf, predicted_labels_rbf),
                confusion_matrix(actual_labels_poly, predicted_labels_poly)
            ]
        }
        all_results.append(results)

    df = pd.concat([pd.DataFrame(res) for res in all_results], ignore_index=True)

    df.to_excel(name_excel+'resultados_SVM_PCA.xlsx', index=False)

    # Find the index of the row with the best F1 score
    best_f1_index = df['F1 Score'].idxmax()

    best_confusion_matrix = df.loc[best_f1_index, 'ConfusionMatrix']

    # Plot the best confusion matrix if it exists
    plt.figure(figsize=(8, 6))
    sns.heatmap(best_confusion_matrix, annot=True, cmap='Blues', fmt='d',
                xticklabels=['DNS', 'Normal'], yticklabels=['Normal', 'DNS'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Best Confusion Matrix (Components: {n_components})')
    plt.show()

################################### -- 12 Classification based on Neural Networks without pca -- #########################################################################################################
def neural_network_classification(trainFeatures_normal, testFeatures_normal, trainFeatures_dns, testFeatures_dns, o3trainClass, o3testClass,name_excel):
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
    acc_nn = []
    pre_nn = []
    actual_labels = []
    predicted_labels = []
    results = []
    nObsTest, nFea = i3CtestN.shape

    for i in range(nObsTest):
        actual_labels.append(o3testClass[i][0])
        # print(len(actual_labels))
        # print('Obs: {:2} ({:<8}): Classification->{}'.format(i,Classes[o3testClass[i][0]],Classes[LT[i]]))
        if LT[i] == o3testClass[i][0]:
            if LT[i] == 2.0:  # Comparando com o valor numérico correspondente à classe 'DNS'
                predicted_labels.append(2.0)  # Predicted as DNS (anomaly)
                tp_nn += 1
            else:
                predicted_labels.append(0.0)  # Predicted as Normal
                fp_nn += 1
        else:
            if LT[i] == 2.0:  # Comparando com o valor numérico correspondente à classe 'DNS'
                predicted_labels.append(2.0)  # Predicted as DNS (anomaly)
                fn_nn += 1
            else:
                predicted_labels.append(0.0)  # Predicted as Normal
                tn_nn += 1

    accuracy_nn = ((tp_nn + tn_nn) / (tp_nn + tn_nn + fp_nn + fn_nn)) * 100
    precision_nn = (tp_nn / (tp_nn + fp_nn)) * 100 if (tp_nn + fp_nn) != 0 else 0
    recall_nn = (tp_nn / (tp_nn + fn_nn)) * 100 if (tp_nn + fn_nn) != 0 else 0
    f1_score_nn = (2 * (precision_nn * recall_nn)) / (precision_nn + recall_nn) if (
            precision_nn + recall_nn) != 0 else 0
    
    confusionMatrix = confusion_matrix(actual_labels, predicted_labels)

    results.append({
        'Accuracy Neural Network': accuracy_nn,
        'Precision Neural Network': precision_nn,
        'Recall Neural Network': recall_nn,
        'F1 Score': f1_score_nn,
        'TP': tp_nn,
        'FP': fp_nn,
        'TN': tn_nn,
        'FN': fn_nn,
        'Accuracy': accuracy_nn,
        'Precision': precision_nn,
        'Confusion Matrix': confusionMatrix,
    })

    df = pd.DataFrame(results)
    df.to_excel(name_excel+'resultados_redes_neurais.xlsx', index=False)

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusionMatrix, annot=True, cmap='Blues', fmt='d',
                xticklabels=['DNS', 'Normal'], yticklabels=['Normal', 'DNS'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Best Confusion Matrix')
    plt.show()


################################### -- 12 Classification based on Neural Networks with pca -- ##################################################
def neural_network_classification_with_pca(trainFeatures_normal, testFeatures_normal, trainFeatures_dns, testFeatures_dns, o3trainClass, o3testClass,name_excel):
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
        acc_nn = []
        pre_nn = []
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
                    fp_nn += 1
            else:
                if LT[i] == 2.0:  # Comparando com o valor numérico correspondente à classe 'DNS'
                    predicted_labels.append(2.0)  # Predicted as DNS (anomaly)
                    fn_nn += 1
                else:
                    predicted_labels.append(0.0)  # Predicted as Normal
                    tn_nn += 1

        accuracy_nn = ((tp_nn + tn_nn) / (tp_nn + tn_nn + fp_nn + fn_nn)) * 100
        precision_nn = (tp_nn / (tp_nn + fp_nn)) * 100 if (tp_nn + fp_nn) != 0 else 0
        recall_nn = (tp_nn / (tp_nn + fn_nn)) * 100 if (tp_nn + fn_nn) != 0 else 0
        f1_score_nn = (2 * (precision_nn * recall_nn)) / (precision_nn + recall_nn) if (
                precision_nn + recall_nn) != 0 else 0

        results.append({
            'Components': n_components,
            'Accuracy Neural Network': accuracy_nn,
            'Precision Neural Network': precision_nn,
            'Recall Neural Network': recall_nn,
            'F1 Score Neural Network': f1_score_nn
        })

    df_nn = pd.DataFrame(results)

    df_nn.to_excel(name_excel+'_redes_neurais_pca.xlsx', index=False)

    confusionMatrix = confusion_matrix(actual_labels, predicted_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusionMatrix, annot=True, cmap='Blues', fmt='d',
                xticklabels=['DNS', 'Normal'], yticklabels=['Normal', 'DNS'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Best Confusion Matrix')
    plt.show()

    ###############################################################################
    ###############################################################################


def pktHandler(timestamp, srcIP, dstIP, lengthIP, sampDelta, outfile):
    global scnets
    global ssnets
    global npkts
    global T0
    global outc
    global last_ks
    global ipList

    if (IPAddress(srcIP) in scnets and IPAddress(dstIP) in ssnets) or (IPAddress(srcIP) in ssnets and IPAddress(dstIP) in scnets):

        if npkts == 0:
            T0 = float(timestamp)
            last_ks = 0
            
        ks = int((float(timestamp)-T0) / sampDelta)
        
        if ks > last_ks:
            print()
            for i in outc: 
                outfile.write(str(i[0]) + ' ' + str(i[1]) + ' ' + str(i[2]) + ' ' + str(i[3]) + ' ' + str(i[4]) + '\n')
                print('{:21s} {:10d} {:10d} {:10d} {:10d}'.format(str(ipList[i[0]]), int(i[1]), int(i[2]), int(i[3]), int(i[4])))
            outfile.write('\n')
            # print(outc)
            outc = []

        if IPAddress(srcIP) in scnets: # Upload
            try:
                ipIndex = ipList.index(dstIP)
            except:
                ipList.append(dstIP)
                ipIndex = ipList.index(dstIP)
            inOutc = False
            outCount = 0
            for iterOutc in outc:
                if iterOutc[0] == ipIndex: 
                    inOutc = True
                    outIdx = outCount
                else:
                    outCount += 1
            if not inOutc: 
                outc.append([ipIndex,0,0,0,0])
                outIdx = len(outc)-1
            outc[outIdx][1] = outc[outIdx][1] + 1
            outc[outIdx][2] = outc[outIdx][2] + int(lengthIP)

        if IPAddress(dstIP) in scnets: # Download
            try:
                ipIndex = ipList.index(srcIP)
            except:
                ipList.append(srcIP)
                ipIndex = ipList.index(srcIP)
            inOutc = False
            outCount = 0
            for iterOutc in outc:
                if iterOutc[0] == ipIndex: 
                    inOutc = True
                    outIdx = outCount
                else:
                    outCount += 1
            if not inOutc: 
                outc.append([ipIndex,0,0,0,0])
                outIdx = len(outc)-1
            outc[outIdx][3] = outc[outIdx][3] + 1
            outc[outIdx][4] = outc[outIdx][4] + int(lengthIP)
            
        # print('= ' + str(srcIP) + ' / ' + str(dstIP) + ' / ' + str(lengthIP))
        # print(outc)
        # print(ipList)
        # print()
        last_ks = ks
        npkts = npkts + 1


def dataIntoMatrices(dataFile):
    global ipList, samplesMatrices
    matrixDataFileName = dataFile + '_matrix'
    matrixDataFile = open(matrixDataFileName, 'w')
    with open(dataFile, 'r') as file:
        tmpMatrix = np.zeros((len(ipList), 4), dtype=int)
        for line in file:
            if line != '\n':
                lineArray = line.split(' ')
                tmpMatrix[int(lineArray[0])] = [int(lineArray[1]), int(lineArray[2]), int(lineArray[3]), int(lineArray[4])]
            else:
                for tmpIter in tmpMatrix:
                    matrixDataFile.write(str(tmpIter[0]) + ' ' + str(tmpIter[1]) + ' ' + str(tmpIter[2]) + ' ' + str(tmpIter[3]) + '\n')
                matrixDataFile.write('\n')
                samplesMatrices.append(tmpMatrix)
                tmpMatrix = np.zeros((len(ipList), 4), dtype=int)
    return matrixDataFileName


    ######################################################################################
    #                                      FEATURES                                      #
    ######################################################################################

def sumMatrices(matrices):
    sum = np.copy(matrices[0])
    for i in range(1, len(matrices)):
        sum = np.add(sum, np.copy(matrices[i]))
    return sum


def trafficMatrices(matrices):
    rows, columns = matrices[0].shape
    trafficList = []
    tmpMatrix = []
    for j in range(0,rows):
        for i in range(0,len(matrices)):
            tmpMatrix.append(np.copy(matrices[i][j]))
        npMatrix = np.array(tmpMatrix,dtype=int)
        trafficList.append(np.copy(npMatrix))
        tmpMatrix = []
    return trafficList


def maxMin(data):
    maxMatrix = np.copy(data[0])
    # print("\nMIIIIIIIIIIIIIIIIIN------",data[:])
    minMatrix = np.copy(data[0])
    for matrix in data[:]:
        for line in range(0,len(matrix)):
            for column in range(0,len(matrix[0])):
                if matrix[line][column] > maxMatrix[line][column]:
                    maxMatrix[line][column] = np.copy(matrix[line][column])
                if matrix[line][column] < minMatrix[line][column]:
                    minMatrix[line][column] = np.copy(matrix[line][column])
    # print("\nMIIIIIIIIIIIIIIIIIN------",maxMatrix)
    return (maxMatrix, minMatrix)


def sumColumns(matrix):
    sum = np.copy(matrix[0])
    for i in range(1,len(matrix)):
        for j in range(0,len(sum)):
            sum[j] += matrix[i][j]
    return sum


def getPercentages(matrix, sum):
    tmpMatrix = np.zeros((len(matrix),len(matrix[0])), dtype=float)
    for i in range (0,len(matrix)):
        for j in range(0,len(matrix[0])):
            try:
                tmpMatrix[i][j] = float(float(matrix[i][j]) / float(sum[j]))
            except:
                pass
    return tmpMatrix

def extractSilenceActivity(data, i, threshold=0):
    matriz = data
    # print("i[0] -> ", i[0])
    # print("i[1] -> ", i[1])
    # print("len(i[0] -> ", len(i[0]))

    save_silence_npkt_payload_ul_dl = []
        # up_count      up_payload      down_count      down_payload
    for j in range(len(i[0])):
        if(i[0][j]<=threshold):
            s=[1]
            a=[]
        else:
            s=[]
            a=[1]
        # print(f'i[0][{j}] = {i[0][j]}')
        # print(f'i[1][{j}] = {i[1][j]}')
        if(i[0][j]>threshold and i[1][j]<=threshold):
            s.append(1)
        elif(i[0][j]<=threshold and i[1][j]>threshold):
            a.append(1)
        elif (i[0][j]<=threshold and i[1][j]<=threshold):
            s[-1]+=1
        else:
            a[-1]+=1
        save_silence_npkt_payload_ul_dl.append([s,a])
        # save_silence_npkt_payload_ul_dl.append(a)
        # print('ss ', s)        
        # print('aa ', a)
    # print('save_silence_npkt_payload_ul_dl -> ', save_silence_npkt_payload_ul_dl)
    # up_count_S up_count_A     up_payload_S up_payload_A    down_count_S down_count_A   down_payload_S down_payload_A
    return save_silence_npkt_payload_ul_dl

def extractStatsAdv(data, i, threshold=0):
    nSamp=data.shape
    M1=np.mean(data,axis=0)
    Md1=np.median(data,axis=0)
    Std1=np.std(data,axis=0)

    extractSil = []
    extractAct = []
    silence_faux = [] 
    activity_faux = []

    for coluna in range(len(i[0])):
        # print("\nUOOOOOOOOOOOOOOOOOOOOOOOOOI -> ",extractSilenceActivity(data, i, threshold)[coluna]) # ---------[0, 2]
        extractSil,extractAct =extractSilenceActivity(data, i, threshold)[coluna]
        # print("\n1UOOOOOOOOOOOOOOOOOOOOOOOOOI -> ", extractSil) # -------- 0
        # print("2UOOOOOOOOOOOOOOOOOOOOOOOOOI -> ",extractAct) # -------- 2
        if len(extractSil) > 0:
            silence_faux.append([np.sum(extractSil), np.mean(extractSil), np.std(extractSil), np.median(extractSil), np.max(extractSil), np.min(extractSil)])
        else:
            silence_faux.append([0,0,0,0,0,0])
        if len(extractAct) > 0:
            activity_faux.append([np.sum(extractAct), np.mean(extractAct), np.std(extractAct), np.median(extractAct), np.max(extractAct), np.min(extractAct)])
        else:
            activity_faux.append([0,0,0,0,0,0])
        
    # print('silence_faux -> ', silence_faux)
    # print('activity_faux -> ', activity_faux)
        # i ->  [[  2 482   0   0] [  1 241   0   0]]
        # npku(sum-media-desvio-mediana-max-min)    nbytesu(sum-media-desvio-mediana-max-min)  npkd(sum-media-desvio-mediana-max-min)  nbytesd(sum-media-desvio-mediana-max-min) 
        # silence_faux ->  [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [2, 2.0, 0.0, 2.0, 2, 2], [2, 2.0, 0.0, 2.0, 2, 2]]
        # activity_faux ->  [[2, 2.0, 0.0, 2.0, 2, 2], [2, 2.0, 0.0, 2.0, 2, 2], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
    return [M1, Md1, Std1, silence_faux, activity_faux]
    

def extractStats(data):
    M1 = np.mean(data, axis=0)
    Md1 = np.median(data, axis=0)
    Std1 = np.std(data, axis=0)
    # features = np.hstack((M1, Md1, Std1))
    return [M1, Md1, Std1]


def extractFeatures(dataFile):
    global ipList, samplesMatrices
    lengthObsWindow = widths
    slidingValue = slide
    # data = np.loadtxt(dataFile, dtype=int)
    data = np.copy(samplesMatrices)
    _fname = str(dataFile.split('.')[0])

    fname = ''.join(_fname.split('_')[0])+"_features_w{}_s{}".format(lengthObsWindow,slidingValue)
    
    directory, filename = os.path.split(fname)
    directory = directory.replace('Captures', 'Features')   # Store in Features to use in Profiles
    fname = directory + '/' + filename.split('_')[0] + "_features_w{}_s{}".format(lengthObsWindow,slidingValue)
    
    sumOutFile = open(fname+'_sum', 'w')
    totalOutFile = open(fname+'_total', 'w')
    percOutFile = open(fname+'_percentages', 'w')
    maxOutFile = open(fname+'_max', 'w')
    minOutFile = open(fname+'_min', 'w')
    avgOutFile = open(fname+'_avg', 'w')
    medianOutFile = open(fname+'_median', 'w')
    stdOutFile = open(fname+'_std', 'w')
    print("\n\n### SLIDING Observation Windows with Length {} and Sliding {} ###".format(lengthObsWindow,slidingValue))

    iobs = 0
    nSamples = len(data)
    nMetrics = len(data[0])
    avgMatrix = np.array([])
    medianMatrix = np.array([])
    stdMatrix = np.array([])

    # print('================================================================== ' + str((data[0][0])))
    # ================================================================== [  2 482   0   0]
    while iobs*slidingValue <= nSamples-lengthObsWindow:
        currentData = np.copy(data[iobs*slidingValue:iobs*slidingValue+lengthObsWindow])
        # print('==================================================================\n'+str(currentData))
        sumMatrix = sumMatrices(currentData)
        sumCol = sumColumns(sumMatrix)
        maxMatrix, minMatrix = maxMin(currentData)
        currentFlows = trafficMatrices(currentData)
        percentageMatrix = getPercentages(sumMatrix, sumCol)
        silSumMatrix = []
        silAvgMatrix = []
        silStdMatrix = []
        silMedMatrix = []
        silMaxMatrix = []
        silMinMatrix = []
        n = 0
        for i in currentFlows:
            # print('==================================================================',n) 
            # print("i -> ", str(i))
            stats = extractStatsAdv(np.copy(currentData),i)
            avgMatrix = stats[0]
            medianMatrix = stats[1]
            stdMatrix = stats[2]


            tempSum = []
            tempAvg = []
            tempStd = []
            tempMed = []
            tempMax = []
            tempMin = []
            
            for metricas in range(0,4):
                tempSum.append(stats[3][metricas][0])
                tempAvg.append(stats[3][metricas][1])
                tempStd.append(stats[3][metricas][2])
                tempMed.append(stats[3][metricas][3])
                tempMax.append(stats[3][metricas][4])
                tempMin.append(stats[3][metricas][5])

            silSumMatrix.append(tempSum)
            silAvgMatrix.append(tempAvg)
            silStdMatrix.append(tempStd)
            silMedMatrix.append(tempMed)
            silMaxMatrix.append(tempMax)
            silMinMatrix.append(tempMin)

            silSumCol = sumColumns(silSumMatrix)
            silPercentageMatrix = getPercentages(silSumMatrix, silSumCol)
                    
            n += 1
        # print("\nSUMMMMMM  ", silSumMatrix)
        # print("AVGGGGGG  ", silAvgMatrix)
        # print("STDDDDDD  ", silStdMatrix)
        # print("MEDIANNN  ", silMedMatrix)

        # for s_metric in range(0, len(silAvgMatrix)):    
        #     print(float(silAvgMatrix[s_metric]))

        print('\n-------------------------')
        print('   ' + str(iobs+1))
        print('--------- Total ---------')
        print('{:6s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s}'.format('IP','\t\t\t  npktUp','     payUp','  npktDown','  payDown', ' s_npktUp',' s_payUp',' s_npktDown',' s_payDown'))
        for i in range(0,len(sumMatrix)):
            print('{:21s} {:10d} {:10d} {:10d} {:10d}  {:10d} {:10d} {:10d} {:10d}'.format(str(ipList[i]), int(sumMatrix[i][0]), int(sumMatrix[i][1]), int(sumMatrix[i][2]), int(sumMatrix[i][3]),int(silSumMatrix[i][0]), int(silSumMatrix[i][1]), int(silSumMatrix[i][2]), int(silSumMatrix[i][3])))
            sumOutFile.write(str(sumMatrix[i][0]) + ' ' + str(sumMatrix[i][1]) + ' ' + str(sumMatrix[i][2]) + ' ' + str(sumMatrix[i][3]) + ' ' + str(silSumMatrix[i][0]) + ' ' + str(silSumMatrix[i][1]) + ' ' + str(silSumMatrix[i][2]) + ' ' + str(silSumMatrix[i][3]) + '\n')
        sumOutFile.write('\n')

        print('{:21s} {:10d} {:10d} {:10d} {:10d} {:10d} {:10d} {:10d} {:10d}'.format('TOTAL: ', int(sumCol[0]), int(sumCol[1]), int(sumCol[2]), int(sumCol[3]), int(silSumCol[0]), int(silSumCol[1]), int(silSumCol[2]), int(silSumCol[3])))
        totalOutFile.write(str(sumCol[0]) + ' ' + str(sumCol[1]) + ' ' + str(sumCol[2]) + ' ' + str(sumCol[3]) +' ' + str(silSumCol[0]) + ' ' + str(silSumCol[1]) + ' ' + str(silSumCol[2]) + ' ' + str(silSumCol[3])  + '\n\n')

        print('\n-------- Perc % ---------')
        print('{:6s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s}'.format('IP','\t\t\t  npktUp','     payUp','  npktDown','  payDown', ' s_npktUp',' s_payUp',' s_npktDown',' s_payDown'))
        for i in range(0,len(percentageMatrix)):
            print('{:21s} {:10.2f} {:10.2f} {:10.2f} {:10.2f} {:10.2f} {:10.2f} {:10.2f} {:10.2f}'.format(str(ipList[i]), float(percentageMatrix[i][0]), float(percentageMatrix[i][1]), float(percentageMatrix[i][2]), float(percentageMatrix[i][3]), float(silPercentageMatrix[i][0]), float(silPercentageMatrix[i][1]), float(silPercentageMatrix[i][2]), float(silPercentageMatrix[i][3])))
            percOutFile.write(str(percentageMatrix[i][0]) + ' ' + str(percentageMatrix[i][1]) + ' ' + str(percentageMatrix[i][2]) + ' ' + str(percentageMatrix[i][3]) + ' ' + str(silPercentageMatrix[i][0]) + ' ' + str(silPercentageMatrix[i][1]) + ' ' + str(silPercentageMatrix[i][2]) + ' ' + str(silPercentageMatrix[i][3]) + '\n')
        percOutFile.write('\n')

        print('\n---------- Max ----------')
        print('{:6s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s}'.format('IP','\t\t\t  npktUp','     payUp','  npktDown','  payDown', ' s_npktUp',' s_payUp',' s_npktDown',' s_payDown'))
        for i in range(0,len(maxMatrix)):
            print('{:21s} {:10d} {:10d} {:10d} {:10d} {:10d} {:10d} {:10d} {:10d}'.format(str(ipList[i]), int(maxMatrix[i][0]), int(maxMatrix[i][1]), int(maxMatrix[i][2]), int(maxMatrix[i][3]), int(silMaxMatrix[i][0]), int(silMaxMatrix[i][1]), int(silMaxMatrix[i][2]), int(silMaxMatrix[i][3])))
            maxOutFile.write(str(maxMatrix[i][0]) + ' ' + str(maxMatrix[i][1]) + ' ' + str(maxMatrix[i][2]) + ' ' + str(maxMatrix[i][3]) +' ' +  str(silMaxMatrix[i][0]) + ' ' + str(silMaxMatrix[i][1]) + ' ' + str(silMaxMatrix[i][2]) + ' ' + str(silMaxMatrix[i][3])  + '\n')
        maxOutFile.write('\n')

        print('\n---------- Min ----------')
        print('{:6s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s}'.format('IP','\t\t\t  npktUp','     payUp','  npktDown','  payDown', ' s_npktUp',' s_payUp',' s_npktDown',' s_payDown'))
        for i in range(0,len(minMatrix)):
            print('{:21s} {:10d} {:10d} {:10d} {:10d} {:10d} {:10d} {:10d} {:10d}'.format(str(ipList[i]), int(minMatrix[i][0]), int(minMatrix[i][1]), int(minMatrix[i][2]), int(minMatrix[i][3]), int(silMinMatrix[i][0]), int(silMinMatrix[i][1]), int(silMinMatrix[i][2]), int(silMinMatrix[i][3])))
            minOutFile.write(str(minMatrix[i][0]) + ' ' + str(minMatrix[i][1]) + ' ' + str(minMatrix[i][2]) + ' ' + str(minMatrix[i][3]) + ' ' + str(silMinMatrix[i][0]) + ' ' + str(silMinMatrix[i][1]) + ' ' + str(silMinMatrix[i][2]) + ' ' + str(silMinMatrix[i][3])  + '\n')
        minOutFile.write('\n')

        print('\n---------- Avg ----------')
        print('{:6s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s}'.format('IP','\t\t\t  npktUp','     payUp','  npktDown','  payDown', ' s_npktUp',' s_payUp',' s_npktDown',' s_payDown'))
        for i in range(0,len(avgMatrix)):
            print('{:21s} {:10.2f} {:10.2f} {:10.2f} {:10.2f} {:10.2f} {:10.2f} {:10.2f} {:10.2f}'.format(str(ipList[i]), float(avgMatrix[i][0]), float(avgMatrix[i][1]), float(avgMatrix[i][2]), float(avgMatrix[i][3]), float(silAvgMatrix[i][0]), float(silAvgMatrix[i][1]), float(silAvgMatrix[i][2]), float(silAvgMatrix[i][3])))
            avgOutFile.write(str(avgMatrix[i][0]) + ' ' + str(avgMatrix[i][1]) + ' ' + str(avgMatrix[i][2]) + ' ' + str(avgMatrix[i][3]) + ' ' + str(silAvgMatrix[i][0]) + ' ' + str(silAvgMatrix[i][1]) + ' ' + str(silAvgMatrix[i][2]) + ' ' + str(silAvgMatrix[i][3]) + '\n')
        avgOutFile.write('\n')

        print('\n-------- Median ---------')
        print('{:6s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s}'.format('IP','\t\t\t  npktUp','     payUp','  npktDown','  payDown', ' s_npktUp',' s_payUp',' s_npktDown',' s_payDown'))
        for i in range(0,len(medianMatrix)):
            print('{:21s} {:10.2f} {:10.2f} {:10.2f} {:10.2f} {:10.2f} {:10.2f} {:10.2f} {:10.2f}'.format(str(ipList[i]), float(medianMatrix[i][0]), float(medianMatrix[i][1]), float(medianMatrix[i][2]), float(medianMatrix[i][3]), float(silMedMatrix[i][0]), float(silMedMatrix[i][1]), float(silMedMatrix[i][2]), float(silMedMatrix[i][3])))
            medianOutFile.write(str(medianMatrix[i][0]) + ' ' + str(medianMatrix[i][1]) + ' ' + str(medianMatrix[i][2]) + ' ' + str(medianMatrix[i][3]) + ' ' + str(silMedMatrix[i][0]) + ' ' + str(silMedMatrix[i][1]) + ' ' + str(silMedMatrix[i][2]) + ' ' + str(silMedMatrix[i][3]) + '\n')
        medianOutFile.write('\n')

        print('\n---------- Std ----------')
        print('{:6s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s}'.format('IP','\t\t\t  npktUp','     payUp','  npktDown','  payDown', ' s_npktUp',' s_payUp',' s_npktDown',' s_payDown'))
        for i in range(0,len(stdMatrix)):
            print('{:21s} {:10.2f} {:10.2f} {:10.2f} {:10.2f} {:10.2f} {:10.2f} {:10.2f} {:10.2f}'.format(str(ipList[i]), float(stdMatrix[i][0]), float(stdMatrix[i][1]), float(stdMatrix[i][2]), float(stdMatrix[i][3]), float(silStdMatrix[i][0]), float(silStdMatrix[i][1]), float(silStdMatrix[i][2]), float(silStdMatrix[i][3])))
            stdOutFile.write(str(stdMatrix[i][0]) + ' ' + str(stdMatrix[i][1]) + ' ' + str(stdMatrix[i][2]) + ' ' + str(stdMatrix[i][3]) + ' ' + str(silStdMatrix[i][0]) + ' ' + str(silStdMatrix[i][1]) + ' ' + str(silStdMatrix[i][2]) + ' ' + str(silStdMatrix[i][3]) + '\n')
        stdOutFile.write('\n')

        print('-------------------------\n\n')
        iobs += 1


    file_vars = [sumOutFile, totalOutFile, percOutFile, maxOutFile, minOutFile, avgOutFile, medianOutFile, stdOutFile]
    namesOfFeaturesFileBrowsing = []
    for file_var in file_vars:
        file_var.close()
        namesOfFeaturesFileBrowsing.append(file_var.name)

    # print("ttttttttttttttttttt -> ",namesOfFeaturesFileBrowsing)

    return namesOfFeaturesFileBrowsing

    ######################################################################################
    #                                      PROFILE                                       #
    ######################################################################################
def waitforEnter(fstop=False):
    if fstop:
        if sys.version_info[0] == 2:
            raw_input("Press ENTER to continue.")
        else:
            input("Press ENTER to continue.")

## -- 3 -- ##
def plotFeatures(features,oClass,f1index=0,f2index=1):
    nObs,nFea=features.shape
    colors=['b','g','r']
    for i in range(nObs):
        plt.plot(features[i,f1index],features[i,f2index],'o'+colors[int(oClass[i])])

    plt.show()
    waitforEnter()
    
def logplotFeatures(features,oClass,f1index=0,f2index=1):
    nObs,nFea=features.shape
    colors=['b','g','r']
    for i in range(nObs):
        plt.loglog(features[i,f1index],features[i,f2index],'o'+colors[int(oClass[i])])

    plt.show()
    waitforEnter()
    
## -- 11 -- ##
def distance(c,p):
    s=0
    n=0
    for i in range(len(c)):
        if c[i]>0:
            s+=np.square((p[i]-c[i])/c[i])
            n+=1
    
    return(np.sqrt(s/n))
        
    #return(np.sqrt(np.sum(np.square((p-c)/c))))

Classes = {0: 'Browsing', 1: 'Attack'}
plt.ion()
nfig = 1


def plotFeatures(features,oClass,f1index=0,f2index=1):
    nObs,nFea=features.shape
    colors=['b','r']
    #blue BROWSING
    #RED for Mining

    for i in range(nObs):
        plt.plot(features[i,f1index],features[i,f2index],'o'+colors[int(oClass[i])])

    # Adicionar nomes aos eixos e título
    plt.xlabel(f'Feature {f1index}')
    plt.ylabel(f'Feature {f2index}')
    plt.title(f'Gráfico de Features {f1index} vs {f2index}')

    plt.show()
    waitforEnter()

def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.readlines()
    
def convert_to_array(combined_content_str):
    lines = combined_content_str.split('\n')
    return np.array([line.split() for line in lines if line.strip()], dtype=float)

def clustering_with_kmeans(features, oClass):
    print('\n-- Clustering with K-Means --')
    kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto")
    labels = kmeans.fit_predict(features)

    for i in range(len(labels)):
        print('Obs: {:2} ({}): K-Means Cluster Label: -> {}'.format(i, Classes[oClass[i][0]], labels[i]))


def clustering_with_dbscan(features, oClass):
    print('\n-- Clustering with DBSCAN --')
    features = StandardScaler().fit_transform(features)
    db = DBSCAN(eps=0.5, min_samples=10).fit(features)
    labels = db.labels_

    for i in range(len(labels)):
        print('Obs: {:2} ({}): DBSCAN Cluster Label: -> {}'.format(i, Classes[oClass[i][0]], labels[i]))


# def anomaly_detection_with_centroids(features_train, o2trainClass, features_test, oClass_test): 
#     # i2train, o2train, i3Atest, o3testClass

#     print('\n-- Anomaly Detection based on Centroids Distances --')
#     centroids = {}
#     for c in range(2):  # Only the first two classes
#         pClass = (o2trainClass == c).flatten()
#         centroids.update({c: np.mean(features_train[pClass, :], axis=0)})
#     print('All Features Centroids:\n', centroids)

#     AnomalyThreshold = 10

#     nObsTest, nFea = features_test.shape
#     for i in range(nObsTest):
#         x = features_test[i]
#         dists = [distance(x, centroids[0]), distance(x, centroids[1])]
#         if min(dists) > AnomalyThreshold:
#             result = "Anomaly"
#         else:
#             result = "OK"

#         print(
#             'Obs: {:2} ({}): Normalized Distances to Centroids: [{:.4f},{:.4f}] -> Result -> {}'.format(
#                 i, Classes[oClass_test[i][0]], *dists, result)
#         )


# def anomaly_detection_with_ocsvm(features_train, features_test, oClass_test):
#     print('\n-- Anomaly Detection based on One Class Support Vector Machines --')
#     nu = 0.1
#     ocsvm = svm.OneClassSVM(gamma='scale', kernel='linear', nu=nu).fit(features_train)
#     rbf_ocsvm = svm.OneClassSVM(gamma='scale', kernel='rbf', nu=nu).fit(features_train)
#     poly_ocsvm = svm.OneClassSVM(gamma='scale', kernel='poly', nu=nu, degree=2).fit(features_train)

#     L1 = ocsvm.predict(features_test)
#     L2 = rbf_ocsvm.predict(features_test)
#     L3 = poly_ocsvm.predict(features_test)

#     AnomResults = {-1: "Anomaly", 1: "OK"}

#     nObsTest, nFea = features_test.shape
#     for i in range(nObsTest):
#         print(
#             'Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(
#                 i, Classes[oClass_test[i][0]], AnomResults[L1[i]], AnomResults[L2[i]], AnomResults[L3[i]]
#             )
#         )


# def classification_with_svm(features_train, features_test, oClass_train, oClass_test):
#     print('\n-- Classification based on Support Vector Machines --')
#     svc = svm.SVC(kernel='linear').fit(features_train, oClass_train)
#     rbf_svc = svm.SVC(kernel='rbf').fit(features_train, oClass_train)
#     poly_svc = svm.SVC(kernel='poly', degree=2).fit(features_train, oClass_train)

#     L1 = svc.predict(features_test)
#     L2 = rbf_svc.predict(features_test)
#     L3 = poly_svc.predict(features_test)
#     print('\n')

#     nObsTest, nFea = features_test.shape
#     for i in range(nObsTest):
#         print(
#             'Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(
#                 i, Classes[oClass_test[i][0]], Classes[L1[i]], Classes[L2[i]], Classes[L3[i]]
#             )
#         )


# def classification_with_neural_networks(features_train, features_test, oClass_train, oClass_test):
#     print('\n-- Classification based on Neural Networks --')
#     scaler = MaxAbsScaler().fit(features_train)
#     features_train_normalized = scaler.transform(features_train)
#     features_test_normalized = scaler.transform(features_test)

#     alpha = 1
#     max_iter = 100000
#     clf = MLPClassifier(solver='lbfgs', alpha=alpha, hidden_layer_sizes=(20,), max_iter=max_iter)
#     clf.fit(features_train_normalized, oClass_train)
#     LT = clf.predict(features_test_normalized)

#     nObsTest, nFea = features_test_normalized.shape
#     for i in range(nObsTest):
#         print('Obs: {:2} ({:<8}): Classification->{}'.format(i, Classes[oClass_test[i][0]], Classes[LT[i]]))

def profileClass(AllFeaturesBrowsing, profileClassFile):
    ############################## LOAD FILE BROWSING##############################
    #############################################################################

    file_paths_Brsg = [AllFeaturesBrowsing[0], AllFeaturesBrowsing[2], AllFeaturesBrowsing[3], AllFeaturesBrowsing[4], AllFeaturesBrowsing[5], AllFeaturesBrowsing[6], AllFeaturesBrowsing[7]] 
    #, AllFeaturesBrowsing[1] não entra pq total n tem numero suficiente de linhas
    all_features_Brsg = [read_file(path) for path in file_paths_Brsg]
    num_lines_brsg = len(all_features_Brsg[0])
    assert all(len(feature_brsg) == num_lines_brsg for feature_brsg in all_features_Brsg), "Files don't have the same number of lines"

    combined_content_brsg = []
    for i in range(num_lines_brsg):
        combined_line = " ".join(feature[i].strip() for feature in all_features_Brsg)
        combined_content_brsg.append(combined_line)
    combined_content_str_brsg = "\n".join(combined_content_brsg)

    # print("Combined Content brsg:\n", combined_content_str_brsg)
    

    non_empty_lines_brsg = [line for line in combined_content_str_brsg.splitlines() if line.strip()]
    # print("non_empty_lines_brsg:\n", len(non_empty_lines_brsg))
    oClass_brsg= np.ones((len(non_empty_lines_brsg),1))*0
    # print("oClass_brsg_sum---------->\n",oClass_brsg)

    ############################## LOAD FILE ATTACK##############################
    #############################################################################
    profileClassFile = profileClassFile.split('.')[0]
    directory, filename = os.path.split(profileClassFile)
    directory = directory.replace('Captures', 'Features')
    profileClassFile = directory + '/' + filename

    attackFileSum = f'{profileClassFile}_features_w2_s1_sum'
    attackFileTotal = f'{profileClassFile}_features_w2_s1_total'
    attackFilePerc = f'{profileClassFile}_features_w2_s1_percentages'
    attackFileMax = f'{profileClassFile}_features_w2_s1_max'
    attackFileMin = f'{profileClassFile}_features_w2_s1_min'
    attackFileAvg = f'{profileClassFile}_features_w2_s1_avg'
    attackFileMedian = f'{profileClassFile}_features_w2_s1_median'
    attackFileStd = f'{profileClassFile}_features_w2_s1_std'

    AllFeaturesAttack = [attackFileSum,   attackFileTotal,   attackFilePerc,   attackFileMax,   
                         attackFileMin,   attackFileAvg,   attackFileMedian,   attackFileStd]

    for fileFeatureAttack in AllFeaturesAttack:
        if not exists(fileFeatureAttack):  
            print(f'No file named {fileFeatureAttack} founded.')
            exit(0)

    file_paths_atck = [attackFileSum, attackFilePerc, attackFileMax, attackFileMin, attackFileAvg, attackFileMedian, attackFileStd] #, attackFileTotal não entra pq total n tem numero suficiente de linhas 
    all_features_atck = [read_file(path) for path in file_paths_atck]
    num_lines_atck = len(all_features_atck[0])
    assert all(len(feature_atck) == num_lines_atck for feature_atck in all_features_atck), "Files don't have the same number of lines"

    combined_content_atck = []
    for i in range(num_lines_atck):
        combined_line = " ".join(feature[i].strip() for feature in all_features_atck)
        combined_content_atck.append(combined_line)
    combined_content_str_atck = "\n".join(combined_content_atck)
    # print("####################################################################v v v v atck\n")
    # print("Combined Content atck:\n", combined_content_str_atck)

    non_empty_lines_atck = [line for line in combined_content_str_atck.splitlines() if line.strip()]
    # print("non_empty_lines_atck:\n", len(non_empty_lines_atck))
    oClass_atck= np.ones((len(non_empty_lines_atck),1))*1
    # print("oClass_brsg_sum---------->\n",oClass_atck)
    


    ##########################JOIN FEATURES ATCK& BRSG###########################
    #############################################################################
    combined_content_arr_brsg = convert_to_array(combined_content_str_brsg)
    combined_content_arr_atck = convert_to_array(combined_content_str_atck)

    
    features = np.vstack((combined_content_arr_brsg, combined_content_arr_atck))
    oClass = np.vstack(( oClass_brsg, oClass_atck))

    # print("combined_content_arr_brsg\n", features)
    # print("oclaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaass", oClass)

    print("Entrou")

    percentage = 0.5
    pB = int(len(combined_content_arr_brsg)*percentage) # 85 para test2.pcap
    pA = int(len(combined_content_arr_atck)*percentage) # 323 para attack sequencial

    trainFeatures_browsing = combined_content_arr_brsg[:pB, :]  #1ª metade das features brsg
    trainFeatures_attack = combined_content_arr_atck[:pA, :]  #1ª metade das features atck

    testFeatures_browsing = combined_content_arr_brsg[pB:,:] #2ª metade das features brsg
    testFeatures_atck = combined_content_arr_atck[pA:,:] #2ª metade das features brsg


    i3Ctrain = np.vstack((trainFeatures_browsing, trainFeatures_attack)) # junta attack 
    i3Ctest = np.vstack((testFeatures_browsing, testFeatures_atck)) # junta attack

    i2train = np.vstack((trainFeatures_browsing)) # users bons ---> testar com outro browsing
    i2test = np.vstack((testFeatures_browsing)) # users bons ---> testar com outro browsing
    o2train = np.vstack((oClass_brsg[:pB])) 
    o2test = np.vstack((oClass_brsg[pB:]))

    i3train = np.vstack((trainFeatures_browsing, trainFeatures_attack))
    i3Atest = np.vstack((testFeatures_browsing, testFeatures_atck)) # junta attack
    
    o3trainClass = np.vstack((oClass_brsg[:pB], oClass_atck[:pA]))
    o3testClass = np.vstack((oClass_brsg[pB:], oClass_atck[pA:]))

    # clustering_with_kmeans(i3Ctrain, o3trainClass)
    # clustering_with_dbscan(i3Ctrain, o3trainClass)
    # anomaly_detection_with_centroids(i2train, o2train, i3Atest, o3testClass)
    # anomaly_detection_with_ocsvm(i2train, i3Atest, o3testClass)
    # classification_with_svm(i3train, i3Ctest, o3trainClass, o3testClass)
    # classification_with_neural_networks(i3train, i3Ctest, o3trainClass, o3testClass)


    name_excel="rafa"

    centroids_distances(trainFeatures_browsing, o2train, i2test, i3Atest, o3testClass,name_excel)
    centroids_distances_with_pca(trainFeatures_browsing, o2train, i2test, i3Atest, o3testClass,name_excel)
    one_class_svm(trainFeatures_browsing, i2test, i3Atest, o3testClass,name_excel)
    one_class_svm_with_pca(trainFeatures_browsing, i2test, i3Atest, o3testClass,name_excel)
    svm_classification(trainFeatures_browsing, i2test, trainFeatures_attack, i3Atest, o3trainClass, o3testClass,name_excel)
    svm_classification_with_pca(trainFeatures_browsing, i2test, trainFeatures_attack, i3Atest, o3trainClass, o3testClass,name_excel)
    neural_network_classification(trainFeatures_browsing, i2test, trainFeatures_attack, i3Atest, o3trainClass, o3testClass,name_excel)
    neural_network_classification_with_pca(trainFeatures_browsing, i2test, trainFeatures_attack, i3Atest, o3trainClass, o3testClass,name_excel)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', nargs='?', required=False, help='input pcap file', default=file)
    # parser.add_argument('-c', '--cnet', nargs='+', required=True, help='client network(s)')
    # parser.add_argument('-s', '--snet', nargs='+', required=True, help='service network(s)')
    parser.add_argument('-c', '--cnet', nargs='+', required=False, help='client network(s)', default=NETClient)
    parser.add_argument('-s', '--snet', nargs='+', required=False, help='service network(s)', default=NETServer)
    args = parser.parse_args()

    cnets = []
    for n in args.cnet:
        try:
            nn = IPNetwork(n)
            cnets.append(nn)
        except:
            print('{} is not a network prefix'.format(n))
    if len(cnets) == 0:
        print('Not valid client network prefixes.')
        sys.exit()
    global scnets
    scnets = IPSet(cnets)

    snets = []
    for n in args.snet:
        try:
            nn = IPNetwork(n)
            snets.append(nn)
        except:
            print('{} is not a network prefix'.format(n))
    if len(snets) == 0:
        print("No valid service network prefixes.")
        sys.exit()
    global ssnets
    ssnets = IPSet(snets)

    fileInput = args.input
    global fileOutput
    fileOutput = ''.join(fileInput.split('.')[:-1])+'_samples'

    global npkts
    global T0
    global outc
    global last_ks
    global ipList
    global samplesMatrices
    

    npkts = 0
    outc = []
    count = 0
    ipList = []
    
    outfile = open(fileOutput,'w')

    print('{:6s} {:10s} {:10s} {:10s} {:10s}'.format('IP','\t\t\t  npktUp','     payUp','  npktDown','   payDown'))

    capture = pyshark.FileCapture(fileInput, display_filter='ip')
    q = 0
    for pkt in capture:
        timestamp, srcIP, dstIP, lengthIP = pkt.sniff_timestamp, pkt.ip.src, pkt.ip.dst, pkt.ip.len
        pktHandler(timestamp, srcIP, dstIP, lengthIP, sampDelta, outfile)
        print(q, end='\r')
        q += 1

        # Enquanto n falarmos com o prof geramos assim as features
        # if q >= 84920:
        if q >= 6562:
            break
        # seqFile 5850 - 6756
        # brwsg2Wind 6563

    outfile.close()

    ipOutFile = open(''.join(fileInput.split('.')[:-1])+'_ipList', 'w')
    for ip in ipList:
        # print(str(ipList.index(ip)) + ': ' + str(ip))
        ipOutFile.write(str(ipList.index(ip)) + ' ' + str(ip) + '\n')
    ipOutFile.close()

    print()

    matrixSamplesFile = dataIntoMatrices(fileOutput)
    # for i in samplesMatrices:
    #     print(str(i) + '\n')
    # print('-----------------------\n')

    # print(sumMatrices(samplesMatrices))
    
    namesOfFeaturesFileBrowsing = extractFeatures(matrixSamplesFile)

    ans=input('Check if the features to both files are created!\nYou want to stop here?\n> ')
     
    if  ans.lower() in ['yes', 'y']:
        print('Ok, bye!')
        sys.exit()
    else:   
        profileClass(namesOfFeaturesFileBrowsing, profileClassFile)


if __name__ == '__main__':
    main()

