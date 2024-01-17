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
import os

# def getMatchPredict(actual_labels, predicted_labels):
    # tp = sum(1 for actual, predicted in zip(actual_labels, predicted_labels) if actual == predicted == 1) # A previsão foi de que um evento ocorreria e ele realmente aconteceu.
    # tn = sum(1 for actual, predicted in zip(actual_labels, predicted_labels) if actual == predicted == 0) # A previsão foi de que um evento não ocorreria e ele realmente não aconteceu.
    # fp = sum(1 for actual, predicted in zip(actual_labels, predicted_labels) if actual != predicted and predicted == 1) # A previsão foi de que um evento ocorreria, mas ele não aconteceu
    # fn = sum(1 for actual, predicted in zip(actual_labels, predicted_labels) if actual != predicted and predicted == 0) # A previsão foi de que um evento não ocorreria, mas ele aconteceu.
    # return tn, fp, fn, tp 

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
    # print("actual_labels: ", actual_labels)
    # print("predicted_labels: ", predicted_labels)
    
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
    # print("\nAAAAAAAAAAAAH\n",results)

    return results
def distance(c,p):
    s=0
    n=0
    for i in range(len(c)):
        if c[i]>0:
            s+=np.square((p[i]-c[i])/c[i])
            n+=1
    
    return(np.sqrt(s/n))
def centroids_distances(sil, i2train, o2train, i3test, o3test, bot):
    print("----------------centroids_distances----------------")
    silence = 'Silence' if sil else 'No Silence'
    
    # Scale the training and testing data
    trainScaler = MaxAbsScaler().fit(i2train)
    i2train_scaled = trainScaler.transform(i2train)
    i3test_scaled = trainScaler.transform(i3test)

    # Calculate the centroid for the client class (class = 0)
    centroids = {0: np.mean(i2train_scaled[(o2train == 0).flatten(), :], axis=0)}

    # Actual labels from the test set
    actual_labels = o3test.flatten()

    # Initialize results dictionary
    results = {'Threshold': [], 'TP': [], 'FP': [], 'TN': [], 'FN': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': [], 'ConfusionMatrix': []}

    # Threshold values to evaluate
    threshold_values = [0.1, 0.2, 0.3, 0.4]

    for threshold in threshold_values:
        # Compute distances for all test samples at once
        distances = np.array([distance(i3test_scaled[i], centroids[0]) for i in range(i3test_scaled.shape[0])])
        # Predict labels based on the threshold
        predicted_labels = (distances > threshold).astype(float)
        # Update results
        results = resultsConfusionMatrix(actual_labels, predicted_labels, results, threshold=threshold)
        print(results)

    df = pd.DataFrame(results)
    best_result = df.loc[df['F1 Score'].idxmax()]
    printMetrics(best_result['TP'],  best_result['TN'], best_result['FP'], best_result['FN'], best_result['Accuracy'], best_result['Precision'], best_result['Recall'], best_result['F1 Score'])
    cm_2x2 = np.array(best_result['ConfusionMatrix']).reshape(2, 2)
    plt.figure(figsize=(8, 4))
    sns.heatmap(cm_2x2, annot=True, cmap='Blues', fmt='d', xticklabels=['Human', bot], yticklabels=['Human', bot])
    plt.xlabel('Predicted')
    plt.ylabel('Real')
    plt.title(f"({silence}) Best Confusion Matrix (Centroid-Based, Threshold: {best_result['Threshold']})")
    # plt.show()
   
    if bot == 'Smart Bot':
        namePlot = f"ResultadosPlotSmart/({silence})BestCentroidsWithoutPCA.png"
        os.makedirs(os.path.dirname(namePlot), exist_ok=True)
        plt.savefig(namePlot)
    elif bot == 'Sequential Bot':
        namePlot = f"ResultadosPlotSequential/({silence})BestCentroidsWithoutPCA.png"
        os.makedirs(os.path.dirname(namePlot), exist_ok=True)
        plt.savefig(namePlot)

    return results

def centroids_distances_pca(sil, components_to_test, trainFeatures_browsing, o2train, testFeatures_browsing, testFeatures_atck, o3test, bot):
    print("----------------centroids_distances_pca----------------")
    silence = 'Silence' if sil else 'No Silence'
    results = {'Components': [],'Threshold': [],'TP': [],'FP': [],'TN': [],'FN': [],'Accuracy': [],'Precision': [],'Recall': [],'F1 Score': [],'ConfusionMatrix': []}

    for n_components in components_to_test:
        scaler = StandardScaler()
        scaled_train = scaler.fit_transform(trainFeatures_browsing)
        scaled_test = scaler.transform(np.vstack((testFeatures_browsing, testFeatures_atck)))

        pca = PCA(n_components=n_components)
        i2train_pca = pca.fit_transform(np.vstack(scaled_train))
        centroids = np.mean(i2train_pca[(o2train == 0).flatten(), :], axis=0)

        i3Atest_pca = pca.transform(scaled_test)
        actual_labels = o3test.flatten()

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
    # plt.show()
    
    if bot == 'Smart Bot':
        namePlot = f"ResultadosPlotSmart/({silence})BestCentroidsWith{best_result['Components']}PCAComponentes.png"
        os.makedirs(os.path.dirname(namePlot), exist_ok=True)
        plt.savefig(namePlot)
    elif bot == 'Sequential Bot':
        namePlot = f"ResultadosPlotSequential/({silence})BestCentroidsWith{best_result['Components']}PCAComponentes.png"
        os.makedirs(os.path.dirname(namePlot), exist_ok=True)
        plt.savefig(namePlot)
    return results

def oc_svm(data2ensemble_pred, data2ensemble_actual, sil, trainFeatures_browsing, testFeatures_browsing, testFeatures_atck, o3test, bot):
    print("----------------oc_svm----------------")
    silence = 'Silence' if sil else 'No Silence'
    results = {'Kernel': [], 'TP': [], 'FP': [], 'TN': [], 'FN': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': [], 'ConfusionMatrix': []}
    
    # scaler = MaxAbsScaler()
    i2train = trainFeatures_browsing
    i3test = np.vstack((testFeatures_browsing, testFeatures_atck))

    kernels = ['linear', 'rbf', 'poly']
    svm_models = [svm.OneClassSVM(gamma='scale', kernel=k, nu=0.5, degree=2 if k == 'poly' else 3).fit(i2train) for k in kernels]
    for kernel, model in zip(kernels, svm_models):
        # model.fit(i2train)
        predictions = model.predict(i3test)
        # Convert predictions from -1 (anomaly) and 1 (normal) to 0 (anomaly) and 1 (normal)
        predictions = np.where(predictions == -1, 0, 1)
        data2ensemble_pred.append(predictions.tolist())
        data2ensemble_actual.append(o3test.flatten().tolist())

        # Use the resultsConfusionMatrix function to calculate and store results
        # print("kernel: ", kernel)
        results = resultsConfusionMatrix(o3test.flatten(), predictions, results, kernel=kernel)

    # print("AAAAAAAAAAAAAAAH",len(data2ensemble_actual[0]))
    # Find the index of the row with the best F1 score
    df = pd.DataFrame(results)
    best_f1_index = df['F1 Score'].idxmax()

    # print("data2ensemble: \n", data2ensemble)
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
    # plt.show()

    if bot == 'Smart Bot':
        namePlot = f"ResultadosPlotSmart/({silence})BestOCSVMWithoutPCA.png"
        os.makedirs(os.path.dirname(namePlot), exist_ok=True)
        plt.savefig(namePlot)
    elif bot == 'Sequential Bot':
        namePlot = f"ResultadosPlotSequential/({silence})BestOCSVMWithoutPCA.png"
        os.makedirs(os.path.dirname(namePlot), exist_ok=True)
        plt.savefig(namePlot)

    return results, data2ensemble_pred, data2ensemble_actual

def oc_svm_pca(data2ensemble_pred, data2ensemble_actual, sil, max_pca_components, trainFeatures_browsing, testFeatures_browsing, testFeatures_atck, o3test, bot):
    print("----------------oc_svm_pca----------------")
    silence = 'Silence' if sil else 'No Silence'
    results = {'Components': [], 'Kernel': [], 'TP': [], 'FP': [], 'TN': [], 'FN': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': [], 'ConfusionMatrix': []}
    kernels = ['linear', 'rbf', 'poly']
    # Define the range of PCA components
    for n_components in max_pca_components:
        scaler = MaxAbsScaler()
        scaled_train = scaler.fit_transform(trainFeatures_browsing)
        scaled_test = scaler.transform(np.vstack((testFeatures_browsing, testFeatures_atck)))
        # Aplicação da PCA
        pca = PCA(n_components=n_components)
        i2train_pca = pca.fit_transform(scaled_train)
        i3Atest_pca = pca.transform(scaled_test)
        svm_models = [svm.OneClassSVM(gamma='scale', kernel=k, nu=0.5, degree=2 if k == 'poly' else 3).fit(i2train_pca) for k in kernels]
        for kernel, model in zip(kernels, svm_models):
            # model.fit(i2train_pca)
            predictions = model.predict(i3Atest_pca)
            # Convert predictions from -1 (anomaly) and 1 (normal) to 0 (anomaly) and 1 (normal)
            predictions = np.where(predictions == -1, 0, 1)
            data2ensemble_pred.append(predictions.tolist())
            data2ensemble_actual.append(o3test.flatten().tolist())

            # Use the resultsConfusionMatrix function to calculate and store results
            # print("kernel: ", kernel)
            results = resultsConfusionMatrix(o3test.flatten(), predictions, results, kernel=kernel, n_components=n_components)
            # print(results)

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
    plt.title(f'({silence}) Best Confusion Matrix OC SVM - {df.loc[best_f1_index, "Kernel"].capitalize()} Kernel\nWith {best_components} PCA Components')
    # plt.show()
    if bot == 'Smart Bot':
        namePlot = f"ResultadosPlotSmart/({silence})BestOCSVMWith{best_components}PCAComponents.png"
        os.makedirs(os.path.dirname(namePlot), exist_ok=True)
        plt.savefig(namePlot)
    elif bot == 'Sequential Bot':
        namePlot = f"ResultadosPlotSequential/({silence})BestOCSVMWith{best_components}PCAComponents.png"
        os.makedirs(os.path.dirname(namePlot), exist_ok=True)
        plt.savefig(namePlot)
    # print(" data2ensemble:\n",  data2ensemble)
    return results, data2ensemble_pred, data2ensemble_actual

def nn_classification(data2ensemble_pred, data2ensemble_actual, sil, trainFeatures_browsing, testFeatures_browsing, trainFeatures_attack, testFeatures_atck, o3train, o3test, bot):
    print("----------------nn_classification----------------")
    silence = 'Silence' if sil else 'No Silence'    
    # Prepare the training and testing data
    i3train = np.vstack((trainFeatures_browsing, trainFeatures_attack))
    i3test = np.vstack((testFeatures_browsing, testFeatures_atck))

    scaler = MaxAbsScaler().fit(i3train)
    i3train = scaler.transform(i3train)
    i3test = scaler.transform(i3test)

    # Initialize and train the neural network classifier
    alpha = 1
    max_iter = 100000
    clf = MLPClassifier(solver='lbfgs', alpha=alpha, hidden_layer_sizes=(20,), max_iter=max_iter)
    clf.fit(i3train, o3train)
    predictions = clf.predict(i3test)
    # print(predictions)
    data2ensemble_pred.append(predictions.tolist()) #
    data2ensemble_actual.append(o3test.flatten().tolist()) #
    # Initialize results dictionary
    results = {'TP': [],'FP': [],'TN': [],'FN': [],'Accuracy': [],'Precision': [],'Recall': [],'F1 Score': [],'ConfusionMatrix': []}
    results = resultsConfusionMatrix(o3test, predictions, results, n_components=None, threshold=None, kernel=None)
    df = pd.DataFrame(results)
    best_f1_score = df['F1 Score'].idxmax()
    best_result = df.loc[best_f1_score]
    printMetrics(best_result['TP'],  best_result['TN'], best_result['FP'], best_result['FN'], best_result['Accuracy'], best_result['Precision'], best_result['Recall'], best_result['F1 Score'])
    cm_2x2 = np.array(df.loc[best_f1_score, 'ConfusionMatrix']).reshape(2, 2)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_2x2, annot=True, cmap='Blues', fmt='d',xticklabels=['Human', bot], yticklabels=['Human', bot])
    plt.xlabel('Predicted')
    plt.ylabel('Real')
    plt.title(f'({silence}) Best on Neural Networks without PCA')
    # plt.show()

    if bot == 'Smart Bot':
        namePlot = f"ResultadosPlotSmart/({silence})BestNeuralNetworksWithoutPCA.png"
        os.makedirs(os.path.dirname(namePlot), exist_ok=True)
        plt.savefig(namePlot)
    elif bot == 'Sequential Bot':
        namePlot = f"ResultadosPlotSequential/({silence})BestNeuralNetworksWithoutPCA.png"
        os.makedirs(os.path.dirname(namePlot), exist_ok=True)
        plt.savefig(namePlot)
    
    return results, data2ensemble_pred, data2ensemble_actual

def nn_classification_pca(data2ensemble_pred, data2ensemble_actual, sil, pcaComponents, trainFeatures_browsing, testFeatures_browsing, trainFeatures_attack, testFeatures_atck, o3train, o3test, bot):
    print("----------------nn_classification_pca----------------")
    silence = 'Silence' if sil else 'No Silence'   
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
        data2ensemble_pred.append(predictions.tolist()) #
        data2ensemble_actual.append(o3test.flatten().tolist()) #
        results = resultsConfusionMatrix(o3test, predictions, results, n_components=n_components, threshold=None, kernel=None)
        # print(results)
    df = pd.DataFrame(results)
    best_f1_score = df['F1 Score'].idxmax()
    best_result = df.loc[best_f1_score]
    best_components = df.loc[best_f1_score, 'Components']
    printMetrics(best_result['TP'],  best_result['TN'], best_result['FP'], best_result['FN'], best_result['Accuracy'], best_result['Precision'], best_result['Recall'], best_result['F1 Score'])
    cm_2x2 = np.array(df.loc[best_f1_score, 'ConfusionMatrix']).reshape(2, 2)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_2x2, annot=True, cmap='Blues', fmt='d',xticklabels=['Human', bot], yticklabels=['Human', bot])
    plt.xlabel('Predicted')
    plt.ylabel('Real')
    plt.title(f'({silence}) Best Confusion Matrix: Neural Networks with {best_components} PCA Components')
    # plt.show()

    if bot == 'Smart Bot':
        namePlot = f"ResultadosPlotSmart/({silence})BestNeuralNetworkswith{best_components}PCA Components.png"
        os.makedirs(os.path.dirname(namePlot), exist_ok=True)
        plt.savefig(namePlot)
    elif bot == 'Sequential Bot':
        namePlot = f"ResultadosPlotSequential/({silence})BestNeuralNetworkswith{best_components}PCA Components.png"
        os.makedirs(os.path.dirname(namePlot), exist_ok=True)
        plt.savefig(namePlot)
    return results, data2ensemble_pred, data2ensemble_actual

def ensemble(sil, all_data2ensemble_pred, o3test, bot):
    print("----------------ensemble----------------")
    silence = 'Silence' if sil else 'No Silence'   
    final_result = {'TP': [], 'FP': [], 'TN': [], 'FN': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': [], 'ConfusionMatrix': []}

    results = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}

    print("ALLo3test\n")#40

    if all_data2ensemble_pred and isinstance(all_data2ensemble_pred[0], list):
        for i in range(len(all_data2ensemble_pred[0])):
            final_pred = [sublist[i] for sublist in all_data2ensemble_pred]

            actual = int(o3test[i])
  
            # imaginando temos um array de 10 -> int(len(final_pred)/2) = 5
            # 0: browsing 1: anomalia
            print("final_pred.count(1): ",final_pred.count(1))    
            print("final_pred.count(0): ",final_pred.count(0))    
            print("actual: ",actual)
            maioria_pred = int(len(final_pred)/2)
            # maioria_actual = int(len(final_actual)/2)
            print("maioria_pred", maioria_pred)
            # print("maioria_actual", maioria_actual)

            # anomalia prevista e é normal
            if int(final_pred.count(0)) >= maioria_pred and actual==0:
                results['TN']+=1
            # normal previsto e é normal
            if int(final_pred.count(1)) > maioria_pred and actual==0:
                results['FN']+=1
            # anomalia prevista e ataque
            if int(final_pred.count(1)) >= maioria_pred and actual==1:
                results['TP']+=1
            # normal previsto e é ataque
            if int(final_pred.count(0)) > maioria_pred and actual==1:
                results['FP']+=1
        
    else:
        print("allData2Ensemble is empty or not formatted correctly.")

    accuracy, precision, recall, f1_score= calculate_metrics(results['TP'], results['TN'],  results['FP'], results['FN'])
    # results = resultsConfusionMatrix(actual_array, final_pred, results)
    # print("results:\n",results)

    final_result['TP'].append(results['TP'])
    final_result['FP'].append(results['FP'])
    final_result['TN'].append(results['TN'])
    final_result['FN'].append(results['FN'])
    final_result['Accuracy'].append(accuracy)
    final_result['Precision'].append(precision)
    final_result['Recall'].append(recall)
    final_result['F1 Score'].append(f1_score)
    final_result['ConfusionMatrix'].append((results['TP'], results['FN'], results['FP'], results['TN']))

    df = pd.DataFrame(final_result)
    best_f1_score = df['F1 Score'].idxmax()
    best_result = df.loc[best_f1_score]
    printMetrics(best_result['TP'],  best_result['TN'], best_result['FP'], best_result['FN'], best_result['Accuracy'], best_result['Precision'], best_result['Recall'], best_result['F1 Score'])
    cm_2x2 = np.array(df.loc[best_f1_score, 'ConfusionMatrix']).reshape(2, 2)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_2x2, annot=True, cmap='Blues', fmt='d',xticklabels=['Human', bot], yticklabels=['Human', bot])
    plt.xlabel('Predicted')
    plt.ylabel('Real')
    plt.title(f'({silence}) Best Confusion Matrix Ensemble')
    # plt.show()

    if bot == 'Smart Bot':
        namePlot = f"ResultadosPlotSmart/({silence})Ensemble.png"
        os.makedirs(os.path.dirname(namePlot), exist_ok=True)
        plt.savefig(namePlot)
    elif bot == 'Sequential Bot':
        namePlot = f"ResultadosPlotSequential/({silence})Ensemble.png"
        os.makedirs(os.path.dirname(namePlot), exist_ok=True)
        plt.savefig(namePlot)

    return final_result