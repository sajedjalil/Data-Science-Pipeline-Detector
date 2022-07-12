import os
import sys
import glob

import scipy.io as sio


def main():
    all_samp_rate = set()
    all_ci = set()
    all_samps = set()
    for fname in glob.glob('../input/*/*.mat'):
        try:
            mat = sio.loadmat(fname)
        except ValueError as e:
            print(fname)
            print(e)
        samp_rate = mat['dataStruct']['iEEGsamplingRate'][0][0][0][0]
        ci = mat['dataStruct']['channelIndices'][0][0][0]
        ci = '_'.join([str(int(c)) for c in ci])
        samps = mat['dataStruct']['nSamplesSegment'][0][0][0][0]
        all_samp_rate.add(samp_rate)
        all_ci.add(ci)
        all_samps.add(samps)

    print(all_samps)
    print(all_ci)
    print(all_samp_rate)



if __name__ == '__main__':
    main()
