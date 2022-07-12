import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.io import loadmat

import matplotlib.pyplot as plt

def mat_to_dataframe(path):
    mat = loadmat(path)
    names = mat['dataStruct'].dtype.names
    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}
    return pd.DataFrame(ndata['data'], columns=ndata['channelIndices'][0])

# pairs = []
# for i in range(1, 5):
#     pairs.append(['../input/train_1/1_' + str(i) + '_0.mat',
#                   '../input/train_1/1_' + str(i) + '_1.mat'])

X0 = mat_to_dataframe('../input/train_1/1_1_1.mat')
# print(X0)
plt.plot(X0.index, X0[1])
plt.show()