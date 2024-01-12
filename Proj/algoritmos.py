import numpy as np
from os.path import exists
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn import svm
import seaborn as sns
import sys
import pandas as pd

# def getMatchPredict(actual_labels, predicted_labels):
#     tp = sum(1 for actual, predicted in zip(actual_labels, predicted_labels) if actual == predicted == 1) # A previsão foi de que um evento ocorreria e ele realmente aconteceu.
#     tn = sum(1 for actual, predicted in zip(actual_labels, predicted_labels) if actual == predicted == 0) # A previsão foi de que um evento não ocorreria e ele realmente não aconteceu.
#     fp = sum(1 for actual, predicted in zip(actual_labels, predicted_labels) if actual != predicted and predicted == 1) # A previsão foi de que um evento ocorreria, mas ele não aconteceu
#     fn = sum(1 for actual, predicted in zip(actual_labels, predicted_labels) if actual != predicted and predicted == 0) # A previsão foi de que um evento não ocorreria, mas ele aconteceu.
#     return tp, tn, fp, fn 

def printMetrics(tp, tn, fp, fn, accuracy, precision, recall, f1_score):
    print(f'\nTrue Positives: {tp}, False Negatives: {fn}')
    print(f'False Positives: {fp}, True Negatives: {tn}')
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1-Score: {f1_score}\n')

def calculate_metrics(tp, tn, fp, fn):
    accuracy = (tp + tn) / (tp + tn + fp + fn) * 100
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return accuracy, precision, recall, f1_score

def resultsConfusionMatrix(actual_labels, predicted_labels, results, n_components=None, threshold=None, kernel=None):
    tn, fp, fn, tp = confusion_matrix(actual_labels, predicted_labels).ravel()
    # print("CM ULALALAALALALALALA: ",CM)
    accuracy, precision, recall, f1_score = calculate_metrics(tp, tn, fp, fn)

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
    results['ConfusionMatrix'].append((tp, fn, fp, tn))

    # print("\nresults['TP']:\n", results['TP'])
    # print("results['FP']:\n", results['FP'])
    # print("results['TN']:\n", results['TN'])
    # print("results['FN']:\n", results['FN'])
    # print("results['ConfusionMatrix']:\n\n", results['ConfusionMatrix'])

    return results

def centroids_distances(sil, i2train, o2train, i3test,   o3test, bot):
    print("----------------centroids_distances----------------")
    if sil:
        silence = 'Silence'
    else:
        silence = 'No Silence'
    scaler = StandardScaler()
    i2train_scaled = scaler.fit_transform(i2train)
    i3test_scaled = scaler.transform(i3test)

    centroid = np.mean(i2train_scaled[(o2train == 0).flatten()], axis=0)
    actual_labels = o3test.flatten()

    threshold_values = [0.1, 0.2, 0.3, 0.4]
    results = {'Threshold': [],'TP': [],'FP': [],'TN': [],'FN': [],'Accuracy': [],'Precision': [],'Recall': [],'F1 Score': [],'ConfusionMatrix': []}

    for threshold in threshold_values:
        distances = np.linalg.norm(i3test_scaled - centroid, axis=1)
        predicted_labels = (distances > threshold).astype(float) * 1.0  # Anomalies are labeled as 1.0
        results = resultsConfusionMatrix(actual_labels, predicted_labels, results, threshold=threshold)

    df = pd.DataFrame(results)
    best_result = df.loc[df['F1 Score'].idxmax()]
    printMetrics(best_result['TP'],  best_result['TN'], best_result['FP'], best_result['FN'], best_result['Accuracy'], best_result['Precision'], best_result['Recall'], best_result['F1 Score'])
    cm_2x2 = np.array(best_result['ConfusionMatrix']).reshape(2, 2)
    plt.figure(figsize=(8, 4))
    sns.heatmap(cm_2x2, annot=True, cmap='Blues', fmt='d', xticklabels=['Human', bot], yticklabels=['Human', bot])
    plt.xlabel('Predicted')
    plt.ylabel('Real')
    plt.title(f"({silence}) Best Confusion Matrix (Centroid-Based, Threshold: {best_result['Threshold']})")
    plt.show()
    return results

def centroids_distances_pca(sil, components_to_test, trainFeatures, o2trainClass, testFeatures_normal, testFeatures_dns, o3testClass, bot):
    print("----------------centroids_distances_pca----------------")
    silence = 'Silence' if sil else 'No Silence'
    results = {'Components': [],'Threshold': [],'TP': [],'FP': [],'TN': [],'FN': [],'Accuracy': [],'Precision': [],'Recall': [],'F1 Score': [],'ConfusionMatrix': []}

    for n_components in components_to_test:
        pca = PCA(n_components=n_components)
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
    cm_2x2 = np.array(best_result['ConfusionMatrix']).reshape(2, 2)
    plt.figure(figsize=(8, 4))
    sns.heatmap(cm_2x2, annot=True, cmap='Blues', fmt='d', xticklabels=['Human', bot], yticklabels=['Human', bot])
    plt.title(f"({silence}) Best on PCA Centroid-Based is Components: {best_result['Components']}, Threshold: {best_result['Threshold']})")
    plt.xlabel('Predicted')
    plt.ylabel('Real')
    plt.show()
    return results

def oc_svm(sil, trainFeatures, testFeatures_normal, testFeatures_dns, o3testClass, bot):
    print("----------------oc_svm----------------")
    silence = 'Silence' if sil else 'No Silence'
    results = {'Kernel': [], 'TP': [], 'FP': [], 'TN': [], 'FN': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': [], 'ConfusionMatrix': []}

    i2train = np.vstack(trainFeatures)
    i3Atest = np.vstack((testFeatures_normal, testFeatures_dns))

    kernels = ['linear', 'rbf', 'poly']
    svm_models = [svm.OneClassSVM(gamma='scale', kernel=k, nu=0.5) for k in kernels]

    for kernel, model in zip(kernels, svm_models):
        model.fit(i2train)
        predictions = model.predict(i3Atest)

        # Convert predictions from -1 (anomaly) and 1 (normal) to 0 (anomaly) and 1 (normal)
        predictions = np.where(predictions == -1, 0, 1)

        # Use the resultsConfusionMatrix function to calculate and store results
        # print("kernel: ", kernel)
        results = resultsConfusionMatrix(o3testClass.flatten(), predictions, results, kernel=kernel)

    # Find the index of the row with the best F1 score
    df = pd.DataFrame(results)
    best_f1_index = df['F1 Score'].idxmax()

    # Print the best results
    best_result = df.iloc[best_f1_index]
    printMetrics(best_result['TP'], best_result['TN'], best_result['FP'], best_result['FN'], best_result['Accuracy'], best_result['Precision'], best_result['Recall'], best_result['F1 Score'])

    # Plot the best confusion matrix
    best_confusion_matrix = np.array(df.loc[best_f1_index, 'ConfusionMatrix']).reshape(2, 2)
    plt.figure(figsize=(8, 6))
    sns.heatmap(best_confusion_matrix,annot=True, cmap='Blues', fmt='d', xticklabels=['Human', bot], yticklabels=['Human', bot])
    plt.xlabel('Predicted label')
    plt.ylabel('Real label')
    plt.title(f'({silence}) Best Confusion Matrix OC SVM - {df.loc[best_f1_index, "Kernel"].capitalize()} Kernel')
    plt.show()

    return results

def oc_svm_pca(sil, max_pca_components, trainFeatures, testFeatures_normal, testFeatures_dns, o3testClass, bot):
    print("----------------oc_svm_pca----------------")
    silence = 'Silence' if sil else 'No Silence'
    results = {'Components': [], 'Kernel': [], 'TP': [], 'FP': [], 'TN': [], 'FN': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': [], 'ConfusionMatrix': []}

    i2train = np.vstack(trainFeatures)
    i3Atest = np.vstack((testFeatures_normal, testFeatures_dns))

    kernels = ['linear', 'rbf', 'poly']
    

    # Define the range of PCA components
    for n_components in max_pca_components:
        # print("n_components: ", n_components)
        i2train_pca = i2train
        i3Atest_pca = i3Atest
        svm_models = [svm.OneClassSVM(gamma='scale', kernel=k, nu=0.5) for k in kernels]

        for kernel, model in zip(kernels, svm_models):
            model.fit(i2train_pca)
            predictions = model.predict(i3Atest_pca)

            # Convert predictions from -1 (anomaly) and 1 (normal) to 0 (anomaly) and 1 (normal)
            predictions = np.where(predictions == -1, 0, 1)

            # Use the resultsConfusionMatrix function to calculate and store results
            # print("kernel: ", kernel)
            results = resultsConfusionMatrix(o3testClass.flatten(), predictions, results, kernel=kernel, n_components=n_components)

        
    # Find the index of the row with the best F1 score
    df = pd.DataFrame(results)
    best_f1_index = df['F1 Score'].idxmax()
    best_components = df.loc[best_f1_index, 'Components']

    # Print the best results
    best_result = df.iloc[best_f1_index]
    printMetrics(best_result['TP'], best_result['TN'], best_result['FP'], best_result['FN'], best_result['Accuracy'], best_result['Precision'], best_result['Recall'], best_result['F1 Score'])

    # Plot the best confusion matrix
    best_confusion_matrix = np.array(df.loc[best_f1_index, 'ConfusionMatrix']).reshape(2, 2)
    plt.figure(figsize=(8, 6))
    sns.heatmap(best_confusion_matrix,annot=True, cmap='Blues', fmt='d', xticklabels=['Human', bot], yticklabels=['Human', bot])
    plt.xlabel('Predicted label')
    plt.ylabel('Real label')
    plt.title(f'({silence}) Best Confusion Matrix OC SVM PCA- {df.loc[best_f1_index, "Kernel"].capitalize()} Kernel\nWith {best_components} PCA Components')
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
    cm_2x2 = np.array(df.loc[best_f1_score, 'ConfusionMatrix']).reshape(2, 2)
    plt.figure(figsize=(8, 4))
    sns.heatmap(cm_2x2, annot=True, cmap='Blues', fmt='d',xticklabels=['Human', bot], yticklabels=['Human', bot])
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
    best_components = df.loc[best_f1_score, 'Components']
    printMetrics(best_result['TP'],  best_result['TN'], best_result['FP'], best_result['FN'], best_result['Accuracy'], best_result['Precision'], best_result['Recall'], best_result['F1 Score'])
    cm_2x2 = np.array(df.loc[best_f1_score, 'ConfusionMatrix']).reshape(2, 2)
    plt.figure(figsize=(8, 4))
    sns.heatmap(cm_2x2, annot=True, cmap='Blues', fmt='d',xticklabels=['Human', bot], yticklabels=['Human', bot])
    plt.xlabel('Predicted')
    plt.ylabel('Real')
    plt.title(f'({silence}) Best Confusion Matrix: Neural Networks with {best_components} PCA Components')
    plt.show()
    return results