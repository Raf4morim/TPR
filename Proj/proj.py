import sys
import argparse
from netaddr import IPNetwork, IPAddress, IPSet
import pyshark
import numpy as np

# up_count      up_payload      down_count      down_payload

sampDelta = 5   # seconds
widths = 120      # sliding window width
slide = 12       # sliding window slide
# file = 'test.pcap'
# file = 'big.pcap'
file = 'browsingAmorim.pcap'
# NETClient = ['172.20.10.0/25']
# NETClient = ['192.168.24.0/24']
# NETClient = ['192.168.1.107/32']
NETClient = ['10.0.2.15']
NETServer = ['0.0.0.0/0']

# samplesMatrices = np.array([])
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
    # matrixDataFile = open(matrixDataFileName, 'w')
    with open(dataFile, 'r') as file:
        tmpMatrix = np.zeros((len(ipList), 4), dtype=int)
        for line in file:
            if line != '\n':
                lineArray = line.split(' ')
                tmpMatrix[int(lineArray[0])] = [int(lineArray[1]), int(lineArray[2]), int(lineArray[3]), int(lineArray[4])]
            else:
                # for tmpIter in tmpMatrix:
                    # matrixDataFile.write(str(tmpIter[0]) + ' ' + str(tmpIter[1]) + ' ' + str(tmpIter[2]) + ' ' + str(tmpIter[3]) + '\n')
                # matrixDataFile.write('\n')
                # samplesMatrices = np.append(samplesMatrices, tmpMatrix)
                samplesMatrices.append(tmpMatrix)
                tmpMatrix = np.zeros((len(ipList), 4), dtype=int)
    return matrixDataFileName


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
    minMatrix = np.copy(data[0])
    for matrix in data[1:]:
        for line in range(0,len(matrix)):
            for column in range(0,len(matrix[0])):
                if matrix[line][column] > maxMatrix[line][column]:
                    maxMatrix[line][column] = np.copy(matrix[line][column])
                if matrix[line][column] < minMatrix[line][column]:
                    minMatrix[line][column] = np.copy(matrix[line][column])
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
    # print('================================================================================================== ' + str(len(data[0][0])))
    while iobs*slidingValue <= nSamples-lengthObsWindow:
        currentData = np.copy(data[iobs*slidingValue:iobs*slidingValue+lengthObsWindow])
        sumMatrix = sumMatrices(currentData)
        sumCol = sumColumns(sumMatrix)
        maxMatrix, minMatrix = maxMin(currentData)
        currentFlows = trafficMatrices(currentData)
        percentageMatrix = getPercentages(sumMatrix, sumCol)
        n = 0
        for i in currentFlows:
            stats = extractStats(np.copy(currentData))
            avgMatrix = stats[0]
            medianMatrix = stats[1]
            stdMatrix = stats[2]
            n += 1

        print('\n-------------------------')
        print('   ' + str(iobs+1))
        print('--------- Total ---------')
        print('{:6s} {:10s} {:10s} {:10s} {:10s}'.format('IP','\t\t\t  npktUp','     payUp','  npktDown','   payDown'))
        for i in range(0,len(sumMatrix)):
            print('{:21s} {:10d} {:10d} {:10d} {:10d}'.format(str(ipList[i]), int(sumMatrix[i][0]), int(sumMatrix[i][1]), int(sumMatrix[i][2]), int(sumMatrix[i][3])))
            sumOutFile.write(str(sumMatrix[i][0]) + ' ' + str(sumMatrix[i][1]) + ' ' + str(sumMatrix[i][2]) + ' ' + str(sumMatrix[i][3]) + '\n')
        sumOutFile.write('\n')

        print('{:21s} {:10d} {:10d} {:10d} {:10d}'.format('TOTAL: ', int(sumCol[0]), int(sumCol[1]), int(sumCol[2]), int(sumCol[3])))
        totalOutFile.write(str(sumCol[0]) + ' ' + str(sumCol[1]) + ' ' + str(sumCol[2]) + ' ' + str(sumCol[3]) + '\n\n')

        print('\n-------- Perc % ---------')
        print('{:6s} {:10s} {:10s} {:10s} {:10s}'.format('IP','\t\t\t  npktUp','     payUp','  npktDown','   payDown'))
        for i in range(0,len(percentageMatrix)):
            print('{:21s} {:10.2f} {:10.2f} {:10.2f} {:10.2f}'.format(str(ipList[i]), float(percentageMatrix[i][0]), float(percentageMatrix[i][1]), float(percentageMatrix[i][2]), float(percentageMatrix[i][3])))
            percOutFile.write(str(percentageMatrix[i][0]) + ' ' + str(percentageMatrix[i][1]) + ' ' + str(percentageMatrix[i][2]) + ' ' + str(percentageMatrix[i][3]) + '\n')
        percOutFile.write('\n')

        print('\n---------- Max ----------')
        print('{:6s} {:10s} {:10s} {:10s} {:10s}'.format('IP','\t\t\t  npktUp','     payUp','  npktDown','   payDown'))
        for i in range(0,len(maxMatrix)):
            print('{:21s} {:10d} {:10d} {:10d} {:10d}'.format(str(ipList[i]), int(maxMatrix[i][0]), int(maxMatrix[i][1]), int(maxMatrix[i][2]), int(maxMatrix[i][3])))
            maxOutFile.write(str(maxMatrix[i][0]) + ' ' + str(maxMatrix[i][1]) + ' ' + str(maxMatrix[i][2]) + ' ' + str(maxMatrix[i][3]) + '\n')
        maxOutFile.write('\n')

        print('\n---------- Min ----------')
        print('{:6s} {:10s} {:10s} {:10s} {:10s}'.format('IP','\t\t\t  npktUp','     payUp','  npktDown','   payDown'))
        for i in range(0,len(minMatrix)):
            print('{:21s} {:10d} {:10d} {:10d} {:10d}'.format(str(ipList[i]), int(minMatrix[i][0]), int(minMatrix[i][1]), int(minMatrix[i][2]), int(minMatrix[i][3])))
            minOutFile.write(str(minMatrix[i][0]) + ' ' + str(minMatrix[i][1]) + ' ' + str(minMatrix[i][2]) + ' ' + str(minMatrix[i][3]) + '\n')
        minOutFile.write('\n')

        print('\n---------- Avg ----------')
        print('{:6s} {:10s} {:10s} {:10s} {:10s}'.format('IP','\t\t\t  npktUp','     payUp','  npktDown','   payDown'))
        for i in range(0,len(avgMatrix)):
            print('{:21s} {:10.2f} {:10.2f} {:10.2f} {:10.2f}'.format(str(ipList[i]), float(avgMatrix[i][0]), float(avgMatrix[i][1]), float(avgMatrix[i][2]), float(avgMatrix[i][3])))
            avgOutFile.write(str(avgMatrix[i][0]) + ' ' + str(avgMatrix[i][1]) + ' ' + str(avgMatrix[i][2]) + ' ' + str(avgMatrix[i][3]) + '\n')
        avgOutFile.write('\n')

        print('\n-------- Median ---------')
        print('{:6s} {:10s} {:10s} {:10s} {:10s}'.format('IP','\t\t\t  npktUp','     payUp','  npktDown','   payDown'))
        for i in range(0,len(medianMatrix)):
            print('{:21s} {:10.2f} {:10.2f} {:10.2f} {:10.2f}'.format(str(ipList[i]), float(medianMatrix[i][0]), float(medianMatrix[i][1]), float(medianMatrix[i][2]), float(medianMatrix[i][3])))
            medianOutFile.write(str(medianMatrix[i][0]) + ' ' + str(medianMatrix[i][1]) + ' ' + str(medianMatrix[i][2]) + ' ' + str(medianMatrix[i][3]) + '\n')
        medianOutFile.write('\n')

        print('\n---------- Std ----------')
        print('{:6s} {:10s} {:10s} {:10s} {:10s}'.format('IP','\t\t\t  npktUp','     payUp','  npktDown','   payDown'))
        for i in range(0,len(stdMatrix)):
            print('{:21s} {:10.2f} {:10.2f} {:10.2f} {:10.2f}'.format(str(ipList[i]), float(stdMatrix[i][0]), float(stdMatrix[i][1]), float(stdMatrix[i][2]), float(stdMatrix[i][3])))
            stdOutFile.write(str(stdMatrix[i][0]) + ' ' + str(stdMatrix[i][1]) + ' ' + str(stdMatrix[i][2]) + ' ' + str(stdMatrix[i][3]) + '\n')
        stdOutFile.write('\n')

        print('-------------------------\n\n')
        iobs += 1

    sumOutFile.close()
    totalOutFile.close()
    percOutFile.close()
    maxOutFile.close()
    minOutFile.close()
    avgOutFile.close()
    medianOutFile.close()
    stdOutFile.close()
    
    

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
    # q = 0
    for pkt in capture:
        timestamp, srcIP, dstIP, lengthIP = pkt.sniff_timestamp, pkt.ip.src, pkt.ip.dst, pkt.ip.len
        pktHandler(timestamp, srcIP, dstIP, lengthIP, sampDelta, outfile)
        # print(q)
        # q += 1
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
    
    extractFeatures(matrixSamplesFile)


if __name__ == '__main__':
    main()