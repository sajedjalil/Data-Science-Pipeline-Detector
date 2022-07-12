import numpy as np
import pandas as pd
import glob
import scipy.io as sio
from multiprocessing import Process

def find_subsequence(seq, subseq):
    target = np.dot(subseq, subseq)
    candidates = np.where(np.correlate(seq, subseq, mode='valid') == target)[0]
    check = candidates[:, np.newaxis] + np.arange(len(subseq))
    mask = np.all((np.take(seq, check) == subseq), axis=-1)
    return candidates[mask]

def load_data(filename):
    return sio.loadmat(filename)['dataStruct']['data'][0, 0][:, 0]

def find_similar(direction, filename, files_to_check, start, end):
    data = load_data(filename)
    needle = np.floor(data[start:end])
    files = sorted(glob.glob(files_to_check))
    for f in files:
        if filename in f:
            continue #dont find myself
        try:
            fdata = np.floor(load_data(f))
            if (len(find_subsequence(fdata, needle))):
                print(direction, ": Similar %s - %s" % (filename, f))
                find_similar(direction, f, files_to_check, start, end)
                return
        except:
            print("Failed to check %s" % f)
    print(direction, ": Nothing found")

def search(filename, files_to_check):
    rw_search = Process(target=find_similar, args=("RW", filename, files_to_check, 0, 70,))
    fw_search = Process(target=find_similar, args=("FW", filename, files_to_check, 239930, 240000,))
    rw_search.start()
    fw_search.start()
    rw_search.join()
    fw_search.join()

print("Started")
search('../input/test_1_new/new_1_3.mat', '../input/test_1_new/new_1_*.mat')
print("Done.")