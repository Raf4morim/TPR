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
from algoritmos import *
warnings.filterwarnings('ignore')
######################################################################################
#                                      PROFILE                                       #
######################################################################################
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


def plotFeatures(features,oClass,f1index=0,f2index=1, label=" No silence: "):
    nObs,nFea=features.shape
    colors=['b','r']
    #blue BROWSING
    #RED for Attack

    for i in range(nObs):
        plt.plot(features[i,f1index],features[i,f2index],'o'+colors[int(oClass[i])])

    findex_name = [
        'Npkt Upload Sum','Nbyte Upload Sum','Npkt Download Sum','Nbyte Download Sum', 
        'Npkt Upload Percentage','Nbyte Upload Percentage','Npkt Download Percentage','Nbyte Download Percentage', 
        'Npkt Upload Max','Nbyte Upload Max','Npkt Download Max','Nbyte Download Max', 
        'Npkt Upload Min','Nbyte Upload Min','Npkt Download Min','Nbyte Download Min', 
        'Npkt Upload Average','Nbyte Upload Average','Npkt Download Average','Nbyte Download Average', 
        'Npkt Upload Median','Nbyte Upload Median','Npkt Download Median','Nbyte Download Median', 
        'Npkt Upload Std','Nbyte Upload Std','Npkt Download Std','Nbyte Download Std'] 

    # Adicionar nomes aos eixos e título
    plt.xlabel(f'{findex_name[f1index]}')
    plt.ylabel(f'{findex_name[f2index]}')
    plt.title(f'{label} {findex_name[f1index]} vs {findex_name[f2index]}')

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

def main():
    width=120
    slide=12
    ############################## LOAD FILE BROWSING##############################
    #############################################################################
    profilebrsFile = "Captures/brwsg1VM.pcap"
    profilebrsFile = profilebrsFile.split('.')[0]
    directory, filename = os.path.split(profilebrsFile)
    directory = directory.replace('Captures', 'Features_Varios_IPs')
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
    profileClassFile = 'Captures/attackSmartWind.pcap'
    bot = 'Bot Smart'
    # bot = 'Sequential Bot'
    # profileClassFile = "Captures/attackSeqWind.pcap"
    profileClassFile = profileClassFile.split('.')[0]
    directory, filename = os.path.split(profileClassFile)
    directory = directory.replace('Captures', 'Features_Varios_IPs')
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

    #Plot features
    # plt.figure(1)
    # plotFeatures(features,oClass,19, 27) # media download bytes vs std dowload bytes
    plt.figure(2)
    plotFeatures(features,oClass,16, 8) # media upload pkts vs max Upload bytes
    plt.figure(3)
    plotFeatures(features,oClass,16, 12) # media upload pkts vs min Upload bytes
    plt.figure(4)
    plotFeatures(features,oClass,16, 20) # media upload pkts vs mediana Upload bytes
    plt.figure(5)
    plotFeatures(features,oClass,16, 24) # media download bytes vs std dowload bytes
    # plt.figure(3)
    # plotFeatures(features,oClass,16, 18) # std download bytes vs std dowload bytes

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
    pcaComponents = [1, 5, 10, 15, 20]
    sil = False
    bestF1Scores = []
    
    
    results_cd = centroids_distances(sil, trainFeatures_browsing, o2train, i3test, o3test, bot)
    df = pd.DataFrame(results_cd)
    bestF1Scores.append(df.iloc[df['F1 Score'].idxmax()]['F1 Score'])
    results_cd_pca = centroids_distances_pca(sil, pcaComponents, trainFeatures_browsing, o2train, testFeatures_browsing,            testFeatures_atck,                                 o3test, bot)
    df = pd.DataFrame(results_cd_pca)
    bestF1Scores.append(df.iloc[df['F1 Score'].idxmax()]['F1 Score'])
    
    
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
    
    
    results_nn = nn_classification(sil, trainFeatures_browsing,     testFeatures_browsing, trainFeatures_attack, testFeatures_atck,    o3train, o3test, bot)
    df = pd.DataFrame(results_nn)
    print(df)
    bestF1Scores.append(df.iloc[df['F1 Score'].idxmax()]['F1 Score'])
    results_nn_pca = nn_classification_pca(sil, pcaComponents, trainFeatures_browsing, testFeatures_browsing, trainFeatures_attack, testFeatures_atck,    o3train, o3test, bot)
    df = pd.DataFrame(results_nn_pca)
    bestF1Scores.append(df.iloc[df['F1 Score'].idxmax()]['F1 Score'])
    
    mean = sum(bestF1Scores)/len(bestF1Scores)
    print ("mean: ",mean)

    #######################################################################
    #######################################################################
    #######################################################################
    #                               SILENCE                               #
    #######################################################################
    #######################################################################
    #######################################################################

    ############################## LOAD FILE BROWSING##############################
    #############################################################################
    # profilebrsFile = "Captures/brwsg1VM.pcap"
    # profilebrsFile = profilebrsFile.split('.')[0]
    # directory, filename = os.path.split(profilebrsFile)
    # directory = directory.replace('Captures', 'Features_Varios_IPs')
    # profilebrsFile = directory + '/' + filename

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

    #Plot features
    # plt.figure(11)
    # plotFeatures(features_s,oClass_s,19, 27, 'Silence: ') # media download bytes vs std dowload bytes
    # plt.figure(22)
    # plotFeatures(features_s,oClass_s,16, 19, 'Silence: ') # media upload pkts vs media dowload bytes
    # plt.figure(3)
    # plotFeatures(features,oClass,16, 18) # std download bytes vs std dowload bytes

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
    
    waitforEnter(fstop=True)




    # res = detect_anomaly(pcaComponents, 1, False)


    # print("\n--------------------Ensemble Stats--------------------")
    # print("True positives: {}".format(res[0]))
    # print("False positives: {}".format(res[1]))
    # print("Accuracy: {}".format(res[2]))
    # print("Precision: {}".format(res[3]))
    # print("Recall: {}".format(res[4]))
    # print("F1-score: {}".format(res[5]))


    # resSilence = detect_anomaly(pcaComponentsSilence, 1, True)


    # print("\n--------------------Ensemble Stats w/Silence--------------------")
    # print("True positives: {}".format(resSilence[0]))
    # print("False positives: {}".format(resSilence[1]))
    # print("Accuracy: {}".format(resSilence[2]))
    # print("Precision: {}".format(resSilence[3]))
    # print("Recall: {}".format(resSilence[4]))
    # print("F1-score: {}".format(resSilence[5]))


if __name__ == '__main__':
    main()