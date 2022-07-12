import gc
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import matthews_corrcoef
from operator import itemgetter



def GrabData():
    directory = '../input/'
    trainfiles = ['train_numeric.csv']

    cols =  [['Id', 'Response']]
    traindata = None
    for i, f in enumerate(trainfiles):
        print(f)
        subset = None
        for i, chunk in enumerate(pd.read_csv(directory + f,
                                              usecols=cols[i],
                                              chunksize=50000,
                                              low_memory=False)):
            print(i)
            if subset is None:
                subset = chunk.copy()
            else:
                subset = pd.concat([subset, chunk])
            del chunk
            gc.collect()
        if traindata is None:
            traindata = subset.copy()
        else:
            traindata = pd.merge(traindata, subset.copy(), on="Id")
        del subset
        gc.collect()
    
    print("???")
    traindata.to_csv('trainResonese'+'.csv', index=False)
    print("!!!!")
    return  traindata


print('Started')
GrabData()
print('Finished')