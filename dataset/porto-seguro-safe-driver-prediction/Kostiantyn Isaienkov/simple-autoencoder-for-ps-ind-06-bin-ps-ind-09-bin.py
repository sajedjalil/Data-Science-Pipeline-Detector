# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
SEED = 42
np.random.seed(SEED)

from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras.optimizers import SGD

def main():

    #preparing data
    trainX = pd.read_csv('../input/train.csv')
    testX = pd.read_csv('../input/test.csv')

   
    data_train = pd.concat([trainX['ps_ind_06_bin'], trainX['ps_ind_07_bin'], trainX['ps_ind_08_bin'], trainX['ps_ind_09_bin']], axis = 1)
    data_test = pd.concat([testX['ps_ind_06_bin'], testX['ps_ind_07_bin'], testX['ps_ind_08_bin'], testX['ps_ind_09_bin']], axis = 1)

    data_train = data_train.as_matrix()
    data_test = data_test.as_matrix()


  
    encoding_dim = 1
    input = Input(shape=(4,))


    encoded = Dense(4096, activation='relu')(input)
    encoded = Dense(encoding_dim, activation='relu')(encoded)
    decoded = Dense(4096, activation='relu')(encoded)
    decoded = Dense(4, activation='relu')(decoded)

    autoencoder = Model(input, decoded)
    encoder = Model(input, encoded)

    sgd = SGD(lr=1)
    autoencoder.compile(optimizer=sgd, loss='mse')

    autoencoder.fit(data_train, data_train, epochs=10, batch_size=128)

    result_train = encoder.predict(data_train)
    result_test = encoder.predict(data_test)
    
    result_train = pd.DataFrame(result_train)
    result_train.to_csv('train.csv')
    result_test = pd.DataFrame(result_test)
    result_test.to_csv('test.csv')
        



if __name__ == "__main__":
    main()