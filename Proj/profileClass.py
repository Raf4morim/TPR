import sys
import tkinter as tk
from tkinter import ttk
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
from algoritmos import *
warnings.filterwarnings('ignore')
######################################################################################
#                                      PROFILE                                       #
######################################################################################
def viewerAllPlotsMixed(features, features_s, oClass):
    fIdxName = [
        'Npkt Up Sum',    'Nbyte Up Sum',    'Npkt Down Sum',    'Nbyte Down Sum', 
        'Npkt Up %',      'Nbyte Up %',      'Npkt Down %',      'Nbyte Down %', 
        'Npkt Up Max',    'Nbyte Up Max',    'Npkt Down Max',    'Nbyte Down Max', 
        'Npkt Up Min',    'Nbyte Up Min',    'Npkt Down Min',    'Nbyte Down Min', 
        'Npkt Up Average','Nbyte Up Average','Npkt Down Average','Nbyte Down Average', 
        'Npkt Up Median', 'Nbyte Up Median', 'Npkt Down Median', 'Nbyte Down Median', 
        'Npkt Up Std',    'Nbyte Up Std',    'Npkt Down Std',    'Nbyte Down Std']

    

    def create_gui():
        # plotFeatures_mix(fIdxName, oClass, features, features_s, f1index=1, f2index=None,f1index_s=None, f2index_s=2, label="(x=Normal vs y=Sil)")
        def plot_and_close():
            # Determina os índices e se eles são silêncios
            f1index = fIdxName.index(combo_f1.get()) if var_f1.get() == 'Normal' else None
            f2index = fIdxName.index(combo_f2.get()) if var_f2.get() == 'Normal' else None
            f1index_s = fIdxName.index(combo_f1.get()) if var_f1.get() == 'Silence' else None
            f2index_s = fIdxName.index(combo_f2.get()) if var_f2.get() == 'Silence' else None
            label1 = var_f1.get()
            label2 =var_f2.get()

            plotFeatures_mix(fIdxName, oClass, features, features_s, f1index, f2index, f1index_s, f2index_s, label1, label2)
            # root.destroy()
        root = tk.Tk()
        root.title("Feature Selection")

        # Criação dos menus suspensos e seletores para o índice x
        label_f1 = ttk.Label(root, text="Índice x:")
        label_f1.pack()
        combo_f1 = ttk.Combobox(root, values=fIdxName)
        combo_f1.pack()
        var_f1 = tk.StringVar(value='Normal')
        radio_f1_normal = ttk.Radiobutton(root, text="Normal", variable=var_f1, value='Normal')
        radio_f1_silencio = ttk.Radiobutton(root, text="Silence", variable=var_f1, value='Silence')
        radio_f1_normal.pack()
        radio_f1_silencio.pack()

        # Criação dos menus suspensos e seletores para o índice y
        label_f2 = ttk.Label(root, text="Índice y:")
        label_f2.pack()
        combo_f2 = ttk.Combobox(root, values=fIdxName)
        combo_f2.pack()
        var_f2 = tk.StringVar(value='Normal')
        radio_f2_normal = ttk.Radiobutton(root, text="Normal", variable=var_f2, value='Normal')
        radio_f2_silencio = ttk.Radiobutton(root, text="Silence", variable=var_f2, value='Silence')
        radio_f2_normal.pack()
        radio_f2_silencio.pack()

        # Botão para plotar
        plot_button = ttk.Button(root, text="Plotar", command=plot_and_close)
        plot_button.pack()

        return root

    # Executar a GUI
    root = create_gui()
    root.mainloop()

def waitforEnter(fstop=False):
    if fstop:
        if sys.version_info[0] == 2:
            raw_input("Press ENTER to continue.")
        else:
            input("Press ENTER to continue.")

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

def plotFeatures_mix(findex_name, oClass, features, features_s, f1index=None, f2index=None, f1index_s=None, f2index_s=None, label1=None, label2=None):
    nObs, _ = features.shape # Num de obs é igual para os 2
    colors = ['b', 'r']  # blue for BROWSING, RED for Attack
    plt.figure()
    if f1index_s is None and f2index_s is None: # se não houver silencios
        for i in range(nObs):
            plt.plot(features[i, f1index], features[i, f2index], 'o' + colors[int(oClass[i])])
    if f1index_s is not None and f2index_s is None: # se x é silencio e y é normal
        for i in range(nObs):
            plt.plot(features_s[i, f1index_s], features[i, f2index], 'o' + colors[int(oClass[i])])
    if f1index_s is None and f2index_s is not None: # se x é normal e y é silencio
        for i in range(nObs):
            plt.plot(features[i, f1index], features_s[i, f2index_s], 'o' + colors[int(oClass[i])])
    if f1index_s is not None and f2index_s is not None: # se só houver silencios
        for i in range(nObs):
            plt.plot(features_s[i, f1index_s], features_s[i, f2index_s], 'o' + colors[int(oClass[i])])
    if label1 is None or label2 is None:
        print("You should specify labels, p.e.: \"(x=Sil y=Normal)\"")
        exit(0)
    else: 
        xlabel_text = findex_name[f1index] if f1index is not None else findex_name[f1index_s] # Obter os x' e y' ou do normal ou do silencio
        ylabel_text = findex_name[f2index] if f2index is not None else findex_name[f2index_s]
        plt.title(f'{label1} {xlabel_text} vs {label2} {ylabel_text}')    
        plt.xlabel(f'{label1} {xlabel_text}')
        plt.ylabel(f'{label2} {ylabel_text}')

    plt.show()
    waitforEnter()

def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.readlines()

def convert_to_array(combined_content_str):
    lines = combined_content_str.split('\n')
    return np.array([line.split() for line in lines if line.strip()], dtype=float)

def main():
    width=20
    slide=4
    ############################## LOAD FILE BROWSING##############################
    #############################################################################
    profilebrsFile = "Captures/brwsg1VM.pcap"
    profilebrsFile = profilebrsFile.split('.')[0]
    directory, filename = os.path.split(profilebrsFile)
    directory = directory.replace('Captures', 'Features_brwsg1VM')
    profilebrsFile = directory + '/' + filename

    file_suffixes = ['sum', 'total', 'percentages', 'max', 'min', 'avg', 'median', 'std']
    file_vars = [f'{profilebrsFile}_features_w{width}_s{slide}_{suffix}' for suffix in file_suffixes]

    AllFeaturesBrowsing = []
    for file_path in file_vars:
        with open(file_path, 'r') as file:
            pass

        AllFeaturesBrowsing.append(file_path)

#    print(AllFeaturesBrowsing)
        
    file_paths_Brsg = [AllFeaturesBrowsing[0], AllFeaturesBrowsing[2], AllFeaturesBrowsing[3], AllFeaturesBrowsing[4], AllFeaturesBrowsing[5], AllFeaturesBrowsing[6], AllFeaturesBrowsing[7]]
    for fileFeaturebrg in AllFeaturesBrowsing:
        if not exists(fileFeaturebrg):
            print(f'No file named {fileFeaturebrg} founded.')
            exit(0)
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
    # profileClassFile = 'Captures/attackSmartWind.pcap'
    # bot = 'Bot Smart'
    bot = 'Sequential Bot'
    profileClassFile = "Captures/attackSeqWind.pcap"
    profileClassFile = profileClassFile.split('.')[0]
    directory, filename = os.path.split(profileClassFile)
    directory = directory.replace('Captures', 'Features_attackSeqWind')
    profileClassFile = directory + '/' + filename

    file_suffixes = ['sum', 'total', 'percentages', 'max', 'min', 'avg', 'median', 'std']
    file_vars = [f'{profileClassFile}_features_w{width}_s{slide}_{suffix}' for suffix in file_suffixes]

    AllFeaturesAttack = []
    for file_path in file_vars:
        with open(file_path, 'r') as file:
            pass
        AllFeaturesAttack.append(file_path)

    file_paths_atck = [AllFeaturesAttack[0], AllFeaturesAttack[2], AllFeaturesAttack[3], AllFeaturesAttack[4], AllFeaturesAttack[5], AllFeaturesAttack[6], AllFeaturesAttack[7]]
    
    for fileFeatureAttack in AllFeaturesAttack:
        if not exists(fileFeatureAttack):
            print(f'No file named {fileFeatureAttack} founded.')
            exit(0)
    
    all_features_atck = [read_file(path) for path in file_paths_atck]
    num_lines_atck = len(all_features_atck[0])
    assert all(len(feature_atck) == num_lines_atck for feature_atck in all_features_atck), "Files don't have the same number of lines"

    combined_content_atck = []
    for i in range(num_lines_atck):
        combined_line = " ".join(feature[i].strip() for feature in all_features_atck)
        combined_content_atck.append(combined_line)
    combined_content_str_atck = "\n".join(combined_content_atck)
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

    print('Train Stats Features Size:',features.shape)
    print('Classes Size: ', oClass.shape)

    # print("len(combined_content_arr_brsg)", len(combined_content_arr_brsg))
    # print("len(combined_content_arr_atck)", len(combined_content_arr_atck))
    # print("len(oClass_brsg)", len(oClass_brsg))
    # print("len(oClass_atck)", len(oClass_atck))

    # print("combined_content_arr_brsg\n", features)
    # print("oclaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaass", oClass)

    percentage = 0.5

    pB = int(len(combined_content_arr_brsg)*percentage) # 85 para test2.pcap
    pA = int(len(combined_content_arr_atck)*percentage) # 323 para attack sequencial

    trainFeatures_browsing = combined_content_arr_brsg[:pB, :]  #1ª metade das features brsg
    trainFeatures_attack = combined_content_arr_atck[:pA, :]  #1ª metade das features atck

    testFeatures_browsing = combined_content_arr_brsg[pB:,:] #2ª metade das features brsg
    testFeatures_atck = combined_content_arr_atck[pA:,:] #2ª metade das features brsg
    
    i2train = np.vstack((trainFeatures_browsing)) # users bons ---> 1º metade features browsing
    i2test = np.vstack((testFeatures_browsing)) # users bons ---> 2º metade features browsing
    o2train = np.vstack((oClass_brsg[:pB])) # users bons ---> 1º metade oClass browsing
    o2test = np.vstack((oClass_brsg[pB:]))  # users bons ---> 2º metade oClass browsing

    i3train = np.vstack((trainFeatures_browsing, trainFeatures_attack)) # junta attack
    i3test = np.vstack((testFeatures_browsing, testFeatures_atck)) # junta attack
    o3train = np.vstack((oClass_brsg[:pB], oClass_atck[:pA]))
    o3test = np.vstack((oClass_brsg[pB:], oClass_atck[pA:]))

    print("\n#########no_silence#########")
    pcaComponents = [5, 10, 15, 20]
    sil = False
    bestF1Scores = []
    
    
    # results_cd = centroids_distances(sil, trainFeatures_browsing, o2train, i3test, o3test, bot)
    # df = pd.DataFrame(results_cd)
    # bestF1Scores.append(df.iloc[df['F1 Score'].idxmax()]['F1 Score'])
    # results_cd_pca = centroids_distances_pca(sil, pcaComponents, trainFeatures_browsing, o2train, testFeatures_browsing,            testFeatures_atck,                                 o3test, bot)
    # df = pd.DataFrame(results_cd_pca)
    # bestF1Scores.append(df.iloc[df['F1 Score'].idxmax()]['F1 Score'])

    # results_ocsvm = oc_svm(sil, i2train, i3test, o3test, bot)
    # df = pd.DataFrame(results_ocsvm)
    # bestF1Scores.append(df.iloc[df['F1 Score'].idxmax()]['F1 Score'])
    # results_ocsvm_pca = oc_svm_pca(sil, pcaComponents, trainFeatures_browsing,            testFeatures_browsing,                       testFeatures_atck,                                 o3test, bot)
    # df = pd.DataFrame(results_ocsvm_pca)
    # bestF1Scores.append(df.iloc[df['F1 Score'].idxmax()]['F1 Score'])

    # results_svm = svm_classification(sil, trainFeatures_browsing,    testFeatures_browsing, trainFeatures_attack, testFeatures_atck, i3train, i3test, o3train, o3test, bot)
    # df = pd.DataFrame(results_svm)
    # bestF1Scores.append(df.iloc[df['F1 Score'].idxmax()]['F1 Score'])
    # results_svm_pca = svm_classification_pca(sil, pcaComponents, trainFeatures_browsing,testFeatures_browsing, trainFeatures_attack, testFeatures_atck,    o3train, o3test, bot)
    # df = pd.DataFrame(results_svm_pca)
    # bestF1Scores.append(df.iloc[df['F1 Score'].idxmax()]['F1 Score'])
    
    # results_nn = nn_classification(sil, trainFeatures_browsing,     testFeatures_browsing, trainFeatures_attack, testFeatures_atck,    o3train, o3test, bot)
    # df = pd.DataFrame(results_nn)
    # bestF1Scores.append(df.iloc[df['F1 Score'].idxmax()]['F1 Score'])
    # results_nn_pca = nn_classification_pca(sil, pcaComponents, trainFeatures_browsing, testFeatures_browsing, trainFeatures_attack, testFeatures_atck,    o3train, o3test, bot)
    # df = pd.DataFrame(results_nn_pca)
    # bestF1Scores.append(df.iloc[df['F1 Score'].idxmax()]['F1 Score'])

    bestF1Score_weigth = []
    print("\nINITIAL bestF1Scores: ", bestF1Scores)
    print("\nEnsemble (no silence): ")
    for idx, bestF1Score in enumerate(bestF1Scores):
        # print("bestF1Score: ", bestF1Score)
        # Se o índice for 0 ou 1 (centroid distance W/ and without pca)
        if idx < 2:
            bestF1Score_weigth.append((0.30*bestF1Score))
        elif idx < 4: # Se o índice for 2 (neural network without pca)
            bestF1Score_weigth.append((0.20*bestF1Score))
        # elif idx < 4: # Se o índice for 3 (neural network with pca)
        #     bestF1Score_weigth.append((0.20*bestF1Score))
    print("\nApos dar pesos bestF1Scores: ", bestF1Score_weigth)
    # Soma de todas as pontuações ponderadas
    sum_of_weighted_scores = sum(bestF1Score_weigth)
    # Soma dos pesos aplicados (0.2 para os dois primeiros, 0.3 para os dois últimos)
    total_weights = (0.3 + 0.3 + 0.20 + 0.20)
    # Cálculo da média ponderada
    weighted_average = sum_of_weighted_scores / total_weights
    print("Média Ponderada: ", weighted_average)


    #######################################################################
    #######################################################################
    #######################################################################
    #                               SILENCE                               #
    #######################################################################
    #######################################################################
    #######################################################################

    ############################## LOAD FILE BROWSING##############################
    #############################################################################

    file_suffixes_s = ['sum_s', 'total_s', 'percentages_s', 'max_s', 'min_s', 'avg_s', 'median_s', 'std_s']
    file_vars_s = [f'{profilebrsFile}_features_w{width}_s{slide}_{suffix}' for suffix in file_suffixes_s]

    # print("file_vars_s: ", file_vars_s)
    AllFeaturesBrowsing_s = []
    for file_path in file_vars_s:
        with open(file_path, 'r') as file:
            pass

        AllFeaturesBrowsing_s.append(file_path)

#    print(AllFeaturesBrowsing)

    file_paths_Brsg_s = [AllFeaturesBrowsing_s[0], AllFeaturesBrowsing_s[2], AllFeaturesBrowsing_s[3], AllFeaturesBrowsing_s[4], AllFeaturesBrowsing_s[5], AllFeaturesBrowsing_s[6], AllFeaturesBrowsing_s[7]]
    
    for fileFeaturebrg_s in AllFeaturesBrowsing_s:
        if not exists(fileFeaturebrg_s):
            print(f'No file named {fileFeaturebrg_s} founded.')
            exit(0)

    #, AllFeaturesBrowsing[1] não entra pq total n tem numero suficiente de linhas
    all_features_Brsg_s = [read_file(path) for path in file_paths_Brsg_s]
    num_lines_brsg_s = len(all_features_Brsg_s[0])
    assert all(len(feature_brsg_s) == num_lines_brsg_s for feature_brsg_s in all_features_Brsg_s), "Files don't have the same number of lines"

    combined_content_brsg_s = []
    for i in range(num_lines_brsg_s):
        combined_line_s = " ".join(feature[i].strip() for feature in all_features_Brsg_s)
        combined_content_brsg_s.append(combined_line_s)
    combined_content_str_brsg_s = "\n".join(combined_content_brsg_s)

    # print("Combined Content brsg:\n", combined_content_str_brsg)


    non_empty_lines_brsg_s = [line for line in combined_content_str_brsg_s.splitlines() if line.strip()]
    # print("non_empty_lines_brsg:\n", len(non_empty_lines_brsg))
    oClass_brsg_s= np.ones((len(non_empty_lines_brsg_s),1))*0
    # print("oClass_brsg_sum---------->\n",oClass_brsg)

    ############################## LOAD FILE ATTACK##############################
    #############################################################################
    file_suffixes_s = ['sum_s', 'total_s', 'percentages_s', 'max_s', 'min_s', 'avg_s', 'median_s', 'std_s']
    file_vars = [f'{profileClassFile}_features_w{width}_s{slide}_{suffix}' for suffix in file_suffixes_s]

    AllFeaturesAttack_s = []
    for file_path in file_vars:
        with open(file_path, 'r') as file:
            pass
        AllFeaturesAttack_s.append(file_path)

    file_paths_atck_s = [AllFeaturesAttack_s[0], AllFeaturesAttack_s[2], AllFeaturesAttack_s[3], AllFeaturesAttack_s[4], AllFeaturesAttack_s[5], AllFeaturesAttack_s[6], AllFeaturesAttack_s[7]]
    
    for fileFeatureAttack_s in AllFeaturesAttack_s:
        if not exists(fileFeatureAttack_s):
            print(f'No file named {fileFeatureAttack_s} founded.')
            exit(0)

    all_features_atck_s = [read_file(path) for path in file_paths_atck_s]
    num_lines_atck_s = len(all_features_atck_s[0])
    assert all(len(feature_atck_s) == num_lines_atck_s for feature_atck_s in all_features_atck_s), "Files don't have the same number of lines"

    combined_content_atck_s = []
    for i in range(num_lines_atck_s):
        combined_line_s = " ".join(feature_s[i].strip() for feature_s in all_features_atck_s)
        combined_content_atck_s.append(combined_line_s)
    combined_content_str_atck_s = "\n".join(combined_content_atck_s)
    # print("Combined Content atck:\n", combined_content_str_atck)

    non_empty_lines_atck_s = [line for line in combined_content_str_atck_s.splitlines() if line.strip()]
    # print("non_empty_lines_atck:\n", len(non_empty_lines_atck))
    oClass_atck_s= np.ones((len(non_empty_lines_atck_s),1))*1
    # print("oClass_brsg_sum---------->\n",oClass_atck)



    ##########################JOIN FEATURES ATCK& BRSG###########################
    #############################################################################
    combined_content_arr_brsg_s = convert_to_array(combined_content_str_brsg_s)
    combined_content_arr_atck_s = convert_to_array(combined_content_str_atck_s)
    
    features_s = np.vstack((combined_content_arr_brsg_s, combined_content_arr_atck_s))
    oClass_s = np.vstack(( oClass_brsg_s, oClass_atck_s))

    print('Train Stats Features Size Silence:',features_s.shape)
    print('Classes Size Silence: ', oClass_s.shape)

    viewerAllPlotsMixed(features, features_s, oClass)






    percentage = 0.5

    pB = int(len(combined_content_arr_brsg_s)*percentage) # 85 para test2.pcap
    pA = int(len(combined_content_arr_atck_s)*percentage) # 323 para attack sequencial

    trainFeatures_browsing = combined_content_arr_brsg_s[:pB, :]  #1ª metade das features brsg
    trainFeatures_attack = combined_content_arr_atck_s[:pA, :]  #1ª metade das features atck

    testFeatures_browsing = combined_content_arr_brsg_s[pB:,:] #2ª metade das features brsg
    testFeatures_atck = combined_content_arr_atck_s[pA:,:] #2ª metade das features brsg
    
    i2train = np.vstack((trainFeatures_browsing)) # users bons ---> 1º metade features browsing
    i2test = np.vstack((testFeatures_browsing)) # users bons ---> 2º metade features browsing
    o2train = np.vstack((oClass_brsg_s[:pB])) # users bons ---> 1º metade oClass browsing
    o2test = np.vstack((oClass_brsg_s[pB:]))  # users bons ---> 2º metade oClass browsing

    i3train = np.vstack((trainFeatures_browsing, trainFeatures_attack)) # junta attack
    i3test = np.vstack((testFeatures_browsing, testFeatures_atck)) # junta attack
    o3train = np.vstack((oClass_brsg_s[:pB], oClass_atck_s[:pA]))
    o3test = np.vstack((oClass_brsg_s[pB:], oClass_atck_s[pA:]))

    print("\n#########silence#########")
    pcaComponents_s = [1, 5, 10, 15, 20]
    sil = True

    # results_cd_s = centroids_distances(sil, trainFeatures_browsing, o2train, i3test, o3test, bot)
    # results_cd_pca_s = centroids_distances_pca(sil, pcaComponents_s, trainFeatures_browsing, o2train, testFeatures_browsing,            testFeatures_atck,                                 o3test, bot)
    # results_ocsvm_s = oc_svm(sil, i2train, i3test, o3test, bot)
    # results_ocsvm_pca_s = oc_svm_pca(sil, pcaComponents_s, trainFeatures_browsing,            testFeatures_browsing,                       testFeatures_atck,                                 o3test, bot)
    
    # results_svm_s = svm_classification(sil, trainFeatures_browsing,    testFeatures_browsing, trainFeatures_attack, testFeatures_atck, i3train, i3test, o3train, o3test, bot)
    # results_svm_pca_s = svm_classification_pca(sil, pcaComponents_s, trainFeatures_browsing,testFeatures_browsing, trainFeatures_attack, testFeatures_atck,    o3train, o3test, bot)
    # results_nn_s = nn_classification(sil, trainFeatures_browsing,     testFeatures_browsing, trainFeatures_attack, testFeatures_atck,    o3train, o3test, bot)
    # results_nn_pca_s = nn_classification_pca(sil, pcaComponents_s, trainFeatures_browsing, testFeatures_browsing, trainFeatures_attack, testFeatures_atck,    o3train, o3test, bot)
    
    bestF1Score_weigth = []
    print("\nINITIAL bestF1Scores: ", bestF1Scores)
    print("\nEnsemble (silence): ")
    for idx, bestF1Score in enumerate(bestF1Scores):
        # print("bestF1Score: ", bestF1Score)
        # Se o índice for 0 ou 1 (centroid distance W/ and without pca)
        if idx < 2:
            bestF1Score_weigth.append((0.30*bestF1Score))
        elif idx < 4: # Se o índice for 2 (neural network without pca)
            bestF1Score_weigth.append((0.20*bestF1Score))
        # elif idx < 4: # Se o índice for 3 (neural network with pca)
        #     bestF1Score_weigth.append((0.20*bestF1Score))
    print("\nApos dar pesos bestF1Scores: ", bestF1Score_weigth)
    # Soma de todas as pontuações ponderadas
    sum_of_weighted_scores = sum(bestF1Score_weigth)
    # Soma dos pesos aplicados (0.2 para os dois primeiros, 0.3 para os dois últimos)
    total_weights = (0.3 + 0.3 + 0.20 + 0.20)
    # Cálculo da média ponderada
    weighted_average = sum_of_weighted_scores / total_weights
    print("Média Ponderada Silencios: ", weighted_average)

    waitforEnter(fstop=True)

if __name__ == '__main__':
    main()