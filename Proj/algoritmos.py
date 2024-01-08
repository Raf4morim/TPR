import numpy as np
from os.path import exists
from pyclbr import Class
import numpy as np
import scipy.stats as stats
import scipy.signal as signal
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.svm import OneClassSVM
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn import svm
import seaborn as sns
import sys
import pandas as pd

def getMatchPredict(actual_labels, predicted_labels):
    tp = sum(1 for actual, predicted in zip(actual_labels, predicted_labels) if actual == predicted == 1) # A previsão foi de que um evento ocorreria e ele realmente aconteceu.
    tn = sum(1 for actual, predicted in zip(actual_labels, predicted_labels) if actual == predicted == 0) # A previsão foi de que um evento não ocorreria e ele realmente não aconteceu.
    fp = sum(1 for actual, predicted in zip(actual_labels, predicted_labels) if actual != predicted and predicted == 1) # A previsão foi de que um evento ocorreria, mas ele não aconteceu
    fn = sum(1 for actual, predicted in zip(actual_labels, predicted_labels) if actual != predicted and predicted == 0) # A previsão foi de que um evento não ocorreria, mas ele aconteceu.
    return tp, tn, fp, fn 

def printMetrics(tp, tn, fp, fn, accuracy, precision, recall, f1_score):
    print("\nTrue Negatives: {}, False Positives: {} ".format(tn,fp))
    print("False Negatives: {}, True Positives: {}".format(fn,tp))
    print("Accuracy: {}%".format(accuracy))
    print("Precision: {}%".format(precision))
    print("Recall: {}%".format(recall))
    print("F1-Score: {}\n".format(f1_score))

def calculate_metrics(tp, tn, fp, fn):
    accuracy = (tp + tn) / (tp + tn + fp + fn) * 100
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return accuracy, precision, recall, f1_score

def resultsConfusionMatrix(actual_labels, predicted_labels, results, n_components=None, threshold=None, kernel=None):
    tp, tn, fp, fn = getMatchPredict(actual_labels, predicted_labels)
    accuracy, precision, recall, f1_score = calculate_metrics(tp, tn, fp, fn)
    cm = confusion_matrix(actual_labels, predicted_labels)

    if kernel is not None:
        results['Kernel'].append(kernel.capitalize())
    if n_components is not None:
        results['Components'].append(n_components)
    if threshold is not None:
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

    return results

def centroids_distances(sil, trainFeatures_browsing, o2train, i3test,   o3test, bot):
    print("----------------centroids_distances----------------")
    if sil:
        silence = 'Silence'
    else:
        silence = 'No Silence'
    centroid = np.mean(trainFeatures_browsing[(o2train == 0).flatten()], axis=0)
    actual_labels = o3test.flatten()

    threshold_values = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.2, 1.1, 1.5, 2.0, 3, 5, 6, 10]
    results = {'Threshold': [],'TP': [],'FP': [],'TN': [],'FN': [],'Accuracy': [],'Precision': [],'Recall': [],'F1 Score': [],'ConfusionMatrix': []}

    for threshold in threshold_values:
        distances = np.linalg.norm(i3test - centroid, axis=1)
        predicted_labels = (distances > threshold).astype(float) * 1.0  # Anomalies are labeled as 1.0
        results = resultsConfusionMatrix(actual_labels, predicted_labels, results, threshold=threshold)

    df = pd.DataFrame(results)
    best_result = df.loc[df['F1 Score'].idxmax()]
    printMetrics(best_result['TP'],  best_result['TN'], best_result['FP'], best_result['FN'], best_result['Accuracy'], best_result['Precision'], best_result['Recall'], best_result['F1 Score'])
    plt.figure(figsize=(8, 4))
    sns.heatmap(best_result['ConfusionMatrix'], annot=True, cmap='Blues', fmt='d', xticklabels=['Human',bot], yticklabels=['Human', bot])
    plt.xlabel('Predicted')
    plt.ylabel('Real')
    plt.title(f"({silence}) Best Confusion Matrix (Centroid-Based, Threshold: {best_result['Threshold']})")
    plt.show()
    return results

def centroids_distances_pca(sil, components_to_test, trainFeatures, o2trainClass, testFeatures_normal, testFeatures_dns, o3testClass, bot):
    print("----------------centroids_distances_pca----------------")
    if sil:
        silence = 'Silence'
    else:
        silence = 'No Silence'
    results = {'Components': [],'Threshold': [],'TP': [],'FP': [],'TN': [],'FN': [],'Accuracy': [],'Precision': [],'Recall': [],'F1 Score': [],'ConfusionMatrix': []}

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

        threshold_values = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.2, 1.1, 1.5, 2.0, 3, 5, 6, 10]

        for threshold in threshold_values:
            distances = np.linalg.norm(i3Atest_pca - centroids, axis=1)
            predicted_labels = (distances > threshold).astype(float) * 1.0  # Anomalies are labeled as 1.0
            results = resultsConfusionMatrix(actual_labels, predicted_labels, results, n_components=n_components, threshold=threshold, kernel=None)


    df = pd.DataFrame(results)
    best_result = df.loc[df['F1 Score'].idxmax()]
    printMetrics(best_result['TP'],  best_result['TN'], best_result['FP'], best_result['FN'], best_result['Accuracy'], best_result['Precision'], best_result['Recall'], best_result['F1 Score'])
    plt.figure(figsize=(8, 4))
    sns.heatmap(best_result['ConfusionMatrix'], annot=True, cmap='Blues', fmt='d', xticklabels=['Human', bot], yticklabels=['Human',bot])
    plt.title(f"({silence}) Best on PCA Centroid-Based is Components: {best_result['Components']}, Threshold: {best_result['Threshold']})")
    plt.xlabel('Predicted')
    plt.ylabel('Real')
    plt.show()
    return results

def oc_svm(sil, i2train,   i3test,   o3test, bot):
    print("----------------oc_svm----------------")
    if sil:
        silence = 'Silence'
    else:
        silence = 'No Silence'
    kernels = ['linear', 'rbf', 'poly']
    results = {'Kernel': [], 'TP': [], 'FP': [], 'TN': [], 'FN': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': [], 'ConfusionMatrix': []}

    actual_labels = [class_label[0] for class_label in o3test]

    for kernel in kernels:
        ocsvm = svm.OneClassSVM(gamma='scale', kernel=kernel, nu=0.5, degree=2 if kernel == 'poly' else 3).fit(i2train)
        predictions = ocsvm.predict(i3test)
        predicted_labels = [1.0 if pred == -1 else 0.0 for pred in predictions]
        results = resultsConfusionMatrix(actual_labels, predicted_labels, results, n_components=None, threshold=None, kernel=kernel)
        
    df = pd.DataFrame(results)
    # Find best F1 score
    best_f1_score = df['F1 Score'].idxmax()
    best_result = df.loc[df['F1 Score'].idxmax()]
    best_confusion_matrix = df.loc[best_f1_score, 'ConfusionMatrix']
    best_kernel = df.loc[best_f1_score, 'Kernel']
    printMetrics(best_result['TP'],  best_result['TN'], best_result['FP'], best_result['FN'], best_result['Accuracy'], best_result['Precision'], best_result['Recall'], best_result['F1 Score'])
    
    # Plot best confusion matrix
    plt.figure(figsize=(8, 4))
    sns.heatmap(best_confusion_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=['Human', bot], yticklabels=['Human', bot])
    plt.xlabel('Predicted label')
    plt.ylabel('Actual label')
    plt.title(f'({silence}) Best Confusion Matrix OCSVM\n Best Kernel: {best_kernel}')
    plt.show()
    return results

def oc_svm_pca(sil, n_components_list, trainFeatures_browsing, testFeatures_browsing, testFeatures_atck, o3test, bot):
    print("----------------oc_svm_pca----------------")
    if sil:
        silence = 'Silence'
    else:
        silence = 'No Silence'
    kernels = ['linear', 'rbf', 'poly']
    results = {'Kernel': [] ,'Components': [],'TP': [],'FP': [],'TN': [],'FN': [],'Accuracy': [],'Precision': [],'Recall': [],'F1 Score': [],'ConfusionMatrix': []}

    for n_components in n_components_list:
        
        print("n_components: ",n_components)
        # Apply PCA
        pca = PCA(n_components=n_components)
        i2train_pca = pca.fit_transform(trainFeatures_browsing)
        i3Atest_pca = pca.transform(np.vstack((testFeatures_browsing, testFeatures_atck)))
        print("i2train_pca", i2train_pca)
        print("i3Atest_pca", i3Atest_pca)

        # Fit and predict for each kernel
        for kernel in kernels:
            actual_labels = [class_label[0] for class_label in o3test]
            print(kernel,"\n")
            ocsvm_model = svm.OneClassSVM(gamma='scale', kernel=kernel, nu=0.5)
            if kernel == 'poly':
                ocsvm_model.degree = 2  # Apenas para o kernel polinomial
            ocsvm_model.fit(i2train_pca)
            print("Passou fit?")
            predictions = ocsvm_model.predict(i3Atest_pca)
            predicted_labels = [1.0 if pred == -1 else 0.0 for pred in predictions]
            results = resultsConfusionMatrix(actual_labels, predicted_labels, results, n_components=n_components, threshold=None, kernel=kernel)

    df = pd.DataFrame(results)
    # Find the best result based on F1 score
    best_result = df.loc[df['F1 Score'].idxmax()]
    printMetrics(best_result['TP'],  best_result['TN'], best_result['FP'], best_result['FN'], best_result['Accuracy'], best_result['Precision'], best_result['Recall'], best_result['F1 Score'])
    plt.figure(figsize=(8, 4))
    sns.heatmap(best_result['ConfusionMatrix'], annot=True, cmap='Blues', fmt='d', xticklabels=['Human', bot], yticklabels=['Human', bot])
    plt.xlabel('Predicted')
    plt.ylabel('Real')
    plt.title(f"({silence}) Best Confusion Matrix (PCA OC SVM {best_result['Kernel']}, Nº Components: {best_result['Components']})")
    plt.show()
    return results

def svm_classification(sil,  trainFeatures_browsing,     testFeatures_browsing,      trainFeatures_attack,       testFeatures_atck,  i3train,    i3test,        o3train,        o3test, bot):
    print("----------------svm_classification----------------")
    if sil:
        silence = 'Silence'
    else:
        silence = 'No Silence'
    i3train = np.vstack((trainFeatures_browsing, trainFeatures_attack))
    i3Ctest = np.vstack((testFeatures_browsing, testFeatures_atck))
    actual_labels = o3test.flatten()

    kernels = ['linear', 'rbf', 'poly']
    results = {'Kernel': [],'TP': [],'FP': [],'TN': [],'FN': [],'Accuracy': [],'Precision': [],'Recall': [],'F1 Score': [],'ConfusionMatrix': []}

    for kernel in kernels:
        print("Entrou no for")
        model = svm.SVC(kernel=kernel, degree=2 if kernel == 'poly' else 3)
        print("Entrou no for")
        model.fit(i3train, o3train)
        predicted_labels = model.predict(i3Ctest)
        print("PREDICT LABELSS: ",predicted_labels)
        results = resultsConfusionMatrix(actual_labels, predicted_labels, results, n_components=None, threshold=None, kernel=kernel)

    df = pd.DataFrame(results)
    # Find the best result based on F1 score
    best_result = df.loc[df['F1 Score'].idxmax()]
    printMetrics(best_result['TP'],  best_result['TN'], best_result['FP'], best_result['FN'], best_result['Accuracy'], best_result['Precision'], best_result['Recall'], best_result['F1 Score'])
    plt.figure(figsize=(8, 4))
    sns.heatmap(best_result['ConfusionMatrix'], annot=True, cmap='Blues', fmt='d', xticklabels=['Human', bot], yticklabels=['Human', bot])
    plt.title(f"({silence}) Best Confusion Matrix SVM Kernel: {best_result['Kernel']}")
    plt.xlabel('Predicted')
    plt.ylabel('Real')
    plt.show()
    return results

def svm_classification_pca(sil, components_to_test, trainFeatures_browsing,     testFeatures_browsing,      trainFeatures_attack,       testFeatures_atck,   o3train,        o3test, bot):
    print("----------------svm_classification_pca----------------")
    if sil:
        silence = 'Silence'
    else:
        silence = 'No Silence'    
    i3train = np.vstack((trainFeatures_browsing, trainFeatures_attack))
    i3test = np.vstack((testFeatures_browsing, testFeatures_atck))
    results = {'Kernel': [],'Components': [],'TP': [],'FP': [],'TN': [],'FN': [],'Accuracy': [],'Precision': [],'Recall': [],'F1 Score': [],'ConfusionMatrix': []}
    for n_components in components_to_test:
        pca = PCA(n_components=n_components)
        i3train_pca = pca.fit_transform(i3train)
        i3test_pca = pca.transform(i3test)

        # Define kernels for SVM
        kernels = {'linear': svm.SVC(kernel='linear'),'rbf': svm.SVC(kernel='rbf'),'poly': svm.SVC(kernel='poly', degree=2)}

        for kernel_name, svc_model in kernels.items():
            svc_model.fit(i3train_pca, o3train)
            predictions = svc_model.predict(i3test_pca)
            resultsConfusionMatrix(o3test, predictions, results, n_components=n_components, threshold=None, kernel=kernel_name)

    df = pd.DataFrame(results)
    best_result = df.loc[df['F1 Score'].idxmax()]
    printMetrics(best_result['TP'],  best_result['TN'], best_result['FP'], best_result['FN'], best_result['Accuracy'], best_result['Precision'], best_result['Recall'], best_result['F1 Score'])
    plt.figure(figsize=(8, 4))
    sns.heatmap(best_result['ConfusionMatrix'], annot=True, cmap='Blues', fmt='d', xticklabels=['Human', bot], yticklabels=['Human', bot])
    plt.xlabel('Predicted')
    plt.ylabel('Real')
    plt.title(f"({silence}) Best Confusion Matrix (SVM PCA {best_result['Kernel']} with {best_result['Components']} PCA Components)")
    plt.show()
    return results

def nn_classification(sil, trainFeatures_browsing, testFeatures_browsing, trainFeatures_attack, testFeatures_atck, o3train, o3test, bot):
    print("----------------nn_classification----------------")
    if sil:
        silence = 'Silence'
    else:
        silence = 'No Silence'    
    # Prepare the training and testing data
    i3train = np.vstack((trainFeatures_browsing, trainFeatures_attack))
    i3test = np.vstack((testFeatures_browsing, testFeatures_atck))

    # Normalize the data
    scaler = MaxAbsScaler().fit(i3train)
    i3trainN = scaler.transform(i3train)
    i3testN = scaler.transform(i3test)

    # Initialize and train the neural network classifier
    alpha = 1
    max_iter = 100000
    clf = MLPClassifier(solver='lbfgs', alpha=alpha, hidden_layer_sizes=(20,), max_iter=max_iter)
    clf.fit(i3trainN, o3train)
    predictions = clf.predict(i3testN)

    # Initialize results dictionary
    results = {'TP': [],'FP': [],'TN': [],'FN': [],'Accuracy': [],'Precision': [],'Recall': [],'F1 Score': [],'ConfusionMatrix': []}
    results = resultsConfusionMatrix(o3test, predictions, results, n_components=None, threshold=None, kernel=None)
    df = pd.DataFrame(results)
    best_f1_score = df['F1 Score'].idxmax()
    best_result = df.loc[best_f1_score]
    printMetrics(best_result['TP'],  best_result['TN'], best_result['FP'], best_result['FN'], best_result['Accuracy'], best_result['Precision'], best_result['Recall'], best_result['F1 Score'])
    best_cm = df.loc[best_f1_score, 'ConfusionMatrix']

    plt.figure(figsize=(8, 4))
    sns.heatmap(best_cm, annot=True, cmap='Blues', fmt='d',xticklabels=['Human', bot], yticklabels=['Human', bot])
    plt.xlabel('Predicted')
    plt.ylabel('Real')
    plt.title(f'({silence}) Best on Neural Networks without PCA')
    plt.show()
    return results

def nn_classification_pca(sil, pcaComponents, trainFeatures_browsing, testFeatures_browsing, trainFeatures_attack, testFeatures_atck, o3train, o3test, bot):
    print("----------------nn_classification_pca----------------")
    if sil:
        silence = 'Silence'
    else:
        silence = 'No Silence'    
    results = {'Components': [],'TP': [],'FP': [],'TN': [],'FN': [],'Accuracy': [],'Precision': [],'Recall': [],'F1 Score': [],'ConfusionMatrix': []}

    i3train = np.vstack((trainFeatures_browsing, trainFeatures_attack))
    i3Ctest = np.vstack((testFeatures_browsing, testFeatures_atck))

    for n_components in pcaComponents:
        pca = PCA(n_components=n_components)
        i3train_pca = pca.fit_transform(i3train)
        i3Ctest_pca = pca.transform(i3Ctest)

        scaler = MaxAbsScaler().fit(i3train_pca)
        i3trainN_pca = scaler.transform(i3train_pca)
        i3CtestN_pca = scaler.transform(i3Ctest_pca)

        alpha = 1
        max_iter = 100000
        clf = MLPClassifier(solver='lbfgs', alpha=alpha, hidden_layer_sizes=(20,), max_iter=max_iter)
        clf.fit(i3trainN_pca, o3train)
        predictions = clf.predict(i3CtestN_pca)
        results = resultsConfusionMatrix(o3test, predictions, results, n_components=n_components, threshold=None, kernel=None)

    df = pd.DataFrame(results)
    best_f1_score = df['F1 Score'].idxmax()
    best_result = df.loc[best_f1_score]
    printMetrics(best_result['TP'],  best_result['TN'], best_result['FP'], best_result['FN'], best_result['Accuracy'], best_result['Precision'], best_result['Recall'], best_result['F1 Score'])
    best_cm = df.loc[best_f1_score, 'ConfusionMatrix']
    best_components = df.loc[best_f1_score, 'Components']

    plt.figure(figsize=(8, 4))
    sns.heatmap(best_cm, annot=True, cmap='Blues', fmt='d',xticklabels=['Human', bot], yticklabels=['Human', bot])
    plt.xlabel('Predicted')
    plt.ylabel('Real')
    plt.title(f'({silence}) Best Confusion Matrix: Neural Networks with {best_components} PCA Components')
    plt.show()
    return results