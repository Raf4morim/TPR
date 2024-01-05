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
# NETClient = ['192.168.1.107/32']
# file = 'Captures/test2.pcap'
#############################################################
NETClient = ['192.168.0.163']
# file = 'Captures/attackSmartWind.pcap'
profileClassFile = 'Captures/attackSmartWind.pcap'
# profileClassFile = "Captures/attackSeqWind.pcap"
# file = "Captures/attackSeqWind.pcap"
file = 'Captures/brwsg2Wind.pcap'
#############################################################
# file = 'Captures/attackSeqVM.pcap'
# file = 'Captures/brwsg1VM.pcap'
# NETClient = ['10.0.2.15']   # file = 'Captures/browsingAmorimVM.pcap'
#############################################################
# NETServer = ['157.240.212.60']  # Apenas para o do Wpp por agora
NETServer = ['0.0.0.0/0']

samplesMatrices = []


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
    # features = np.vstack((combined_content_arr_brsg, combined_content_arr_atck))
    # oClass = np.vstack(( oClass_brsg, oClass_atck))

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

    # centroids_distances(                    trainFeatures_browsing, o2train,                                                                                  i3test,           o3test)
    # centroids_distances_with_pca(           trainFeatures_browsing, o2train,    testFeatures_browsing,                       testFeatures_atck,                                 o3test)
    # one_class_svm(                          i2train,                                                                                                          i3test,           o3test)
    # one_class_svm_with_pca(                 trainFeatures_browsing,             testFeatures_browsing,                       testFeatures_atck,                                 o3test)
    
    # svm_classification(                     trainFeatures_browsing,             testFeatures_browsing, trainFeatures_attack, testFeatures_atck,     i3train,    i3test,  o3train, o3test)
    svm_classification_with_pca(            trainFeatures_browsing,             testFeatures_browsing, trainFeatures_attack, testFeatures_atck,                        o3train, o3test)
    # neural_network_classification(          trainFeatures_browsing,             testFeatures_browsing, trainFeatures_attack, testFeatures_atck,                        o3train, o3test)
    # neural_network_classification_with_pca( trainFeatures_browsing,             testFeatures_browsing, trainFeatures_attack, testFeatures_atck,                        o3train, o3test)
    waitforEnter(fstop=True)

def main():
    ans = input("Already created features?\n> ")
    if ans == 'y':
        profilebrsFile = "Captures/brwsg2Wind.pcap"
        profilebrsFile = profilebrsFile.split('.')[0]
        directory, filename = os.path.split(profilebrsFile)
        directory = directory.replace('Captures', 'Features')
        profilebrsFile = os.path.join(directory, filename)

        file_suffixes = ['sum', 'total', 'percentages', 'max', 'min', 'avg', 'median', 'std']
        file_vars = [f'{profilebrsFile}_features_w2_s1_{suffix}' for suffix in file_suffixes]

        namesOfFeaturesFileBrowsing = []
        for file_path in file_vars:
            # Open and close the file here if required
            # For example:
            # with open(file_path, 'r') as file:
            #     # Perform any file operations here
            #     pass

            # If you just need the file names, append the file path to the list
            namesOfFeaturesFileBrowsing.append(file_path)

        # Assuming extractFeatures is a function you have defined earlier
        profileClass(namesOfFeaturesFileBrowsing, profileClassFile)
        print("Profile created")
        exit(0)
    
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
