import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.io import loadmat
import glob
import re
from multiprocessing import Process

def mat_to_pandas(path):
    mat = loadmat(path)
    names = mat['dataStruct'].dtype.names
    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}
    sequence = -1
    if 'sequence' in names:
        sequence = mat['dataStruct']['sequence'][0,0][0]
    return pd.DataFrame(ndata['data'], columns=ndata['channelIndices'][0]), sequence
    

def feature_process(data):
    shape = data.shape
    res = ""
    for i in range(0,shape[1]):
        res = res + str( data[i+1].var()) + ','
    return res[0:-1]

def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]
    
def train_data_process(set_id):
    print("start process patient {}".format(set_id))
    out = open("train_" + str(set_id) + ".csv", "w")
    out.write("patient_id")
    for i in range(16):
        out.write(",var_" + str(i))
    out.write(",result\n")
    
    for classfiy in range(2):
        print("process class {}".format(classfiy))
        files = sorted(glob.glob("../input/train_" + str(set_id) + "/*"+str(classfiy)+".mat"), key=natural_key)
        for file in files:
            outstr = str(set_id)
            try:
                data,sequence = mat_to_pandas(file)
            except:
                print('load data error in file {}...'.format(file))
                continue
            data = feature_process(data)
            outstr = outstr+","+data+","+str(classfiy)+"\n"
            out.write(outstr)
    out.close()
    
def test_data_process(set_id):
    print("start process test {}".format(set_id))
    out = open("test_" + str(set_id) + ".csv", "w")
    out.write("file_name")
    for i in range(16):
        out.write(",var_" + str(i))
    out.write("\n")
    files = sorted(glob.glob("../input/test_" + str(set_id) + "/*.mat"), key=natural_key)
    for file in files:
        outstr = file.split(r'/')[-1]
        try:
            data,sequence = mat_to_pandas(file)
        except:
            print('load data error in file {}...'.format(file))
            continue
        data = feature_process(data)
        outstr = outstr+","+data+"\n"
        out.write(outstr)
    out.close()
def load_train_feature():
    train1 = pd.read_csv("train_1.csv")
    train2 = pd.read_csv("train_2.csv")
    train3 = pd.read_csv("train_3.csv")
    train = pd.concat([train1, train2, train3])
    train = train[train['var_0'] > 0]
    return train
    

if __name__ == '__main__':
    p = dict()
    p[1] = Process(target=train_data_process, args=(1,))
    p[1].start()
    p[2] = Process(target=train_data_process, args=(2,))
    p[2].start()
    p[3] = Process(target=train_data_process, args=(3,))
    p[3].start()
    p[4] = Process(target=test_data_process, args=(1,))
    p[4].start()
    p[5] = Process(target=test_data_process, args=(2,))
    p[5].start()
    p[6] = Process(target=test_data_process, args=(3,))
    p[6].start()
    p[1].join()
    p[2].join()
    p[3].join()
    p[4].join()
    p[5].join()
    p[6].join()
    
    
    
    
    
    
    
    
    
    