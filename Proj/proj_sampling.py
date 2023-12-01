import sys
import argparse
from netaddr import IPNetwork, IPAddress, IPSet
import pyshark

# up_count      up_payload      down_count      down_payload

sampDelta = 5   # seconds

def pktHandler(timestamp, srcIP, dstIP, lengthIP, sampDelta, outfile):
    global scnets
    global ssnets
    global npkts
    global T0
    global outc
    global last_ks
    global count

    if (IPAddress(srcIP) in scnets and IPAddress(dstIP) in ssnets) or (IPAddress(srcIP) in ssnets and IPAddress(dstIP) in scnets):
        if npkts == 0:
            T0 = float(timestamp)
            last_ks = 0
            
        ks = int((float(timestamp)-T0) / sampDelta)
        
        if ks > last_ks:
            print()
            for i in outc: 
                outfile.write(i[0] + ' ' + str(i[1]) + ' ' + str(i[2]) + ' ' + str(i[3]) + ' ' + str(i[4]) + '\n')
                # outfile.write('{:20s} {:10d} {:10d} {:10d} {:10d}\n'.format(i[0], int(i[1]), int(i[2]), int(i[3]), int(i[4])))
                print('{:20s} {:10d} {:10d} {:10d} {:10d}'.format(i[0], int(i[1]), int(i[2]), int(i[3]), int(i[4])))
            outfile.write('\n')
            outc = []
            count = 0
            
        # if ks > last_ks+1:
        #     for j in range(last_ks+1,ks):
        #         outfile.write('{} {} {} {}\n'.format(*outc))
        #         print('{} {} {} {}'.format(*outc))

        if IPAddress(srcIP) in scnets: # Upload
            idx = 0
            flag = False
            for k in outc:
                if k[0] == str(dstIP):
                    flag = True
                else: idx += 1
            if not flag:
                outc.append([str(dstIP),0,0,0,0])
                count += 1
            outc[idx][1] = outc[idx][1] + 1
            outc[idx][2] = outc[idx][2] + int(lengthIP)

        if IPAddress(dstIP) in scnets: # Download
            idx = 0
            flag = False
            for k in outc:
                if k[0] == str(srcIP):
                    flag = True
                else: idx += 1
            if not flag:
                outc.append([str(srcIP),0,0,0,0])
                count += 1
            outc[idx][3] = outc[idx][3] + 1
            outc[idx][4] = outc[idx][4] + int(lengthIP)
        
        last_ks = ks
        npkts = npkts + 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', nargs='?', required= True, help='input pcap file')
    parser.add_argument('-c', '--cnet', nargs='+', required=True, help='client network(s)')
    parser.add_argument('-s', '--snet', nargs='+', required=True, help='service network(s)')
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

    fileOutput = ''.join(fileInput.split('.')[:-1])+'_samples'

    global npkts
    global T0
    global outc
    global last_ks
    global count
    

    npkts = 0
    outc = []
    count = 0

    outfile = open(fileOutput,'w')

    print('{:20s} {:10s} {:10s} {:10s} {:10s}'.format('IP','\t npktUp','   payUp','npktDown',' payDown'))

    capture = pyshark.FileCapture(fileInput, display_filter='ip')
    for pkt in capture:
        timestamp, srcIP, dstIP, lengthIP = pkt.sniff_timestamp, pkt.ip.src, pkt.ip.dst, pkt.ip.len
        pktHandler(timestamp, srcIP, dstIP, lengthIP, sampDelta, outfile)

    outfile.close()

if __name__ == '__main__':
    main()