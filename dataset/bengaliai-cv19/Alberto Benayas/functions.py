import pandas as pd
import numpy as np
from io import StringIO
from keras.utils import to_categorical
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import save_img
from tqdm import tqdm
import os
import gc

'''
Read Dataframe from file, based on file extension
file is a path with an extension
'''
def toDF(file):
    extension = file.split('.')[1]
    if extension == 'csv':
        return pd.read_csv(file)
    if extension == 'feather':
        return pd.read_feather(file)
    if extension == 'parquet':
        return pd.read_parquet(file)

# Create dataframe from a list of files
'''
files is a list of paths
'''
def toDF_all(files):
    dfs = []
    for f in files:
        df = toDF(f)
        dfs.append(df)
    df_final = pd.concat(dfs)
    del dfs
    gc.collect()
    return df_final

'''
Creates a version of the dataframe by rebalancing the number of sample of each class.
The number of samples of each class will be equal to the least represented class
'''
def rebalance(df, shuffle=True):
    g = df.groupby('LABEL')
    g = g.apply(lambda x: x.sample(g.size().min())).reset_index(drop=True)
    if shuffle:
        g = g.sample(frac=1).reset_index(drop=True)
    return g


def generate_png(data, path='test_image_cnn.png'):
    save_img(path, array_to_img(data))


# Funcion generadora para entrenar en batches desde GS
def generator(filenames, batch_size):
    print('Reading new file', str(filenames[0]))
    X, y = readfile(filenames[0])
    fileindex = 1
    i = 0
    while True:
        if batch_size > len(X):
            print('Reading new file', str(fileindex))
            if fileindex > len(filenames) - 1:
                print('End of data')
                print('Set', str(i))
                i = i + 1
                yield X, y
            X_aux, y_aux = readfile(filenames[fileindex])
            X = np.concatenate((X, X_aux), axis=0)
            y = np.concatenate((y, y_aux), axis=0)
            fileindex = fileindex + 1

        batch_x = X[:batch_size]
        batch_y = y[:batch_size]

        X = X[batch_size:]
        y = y[batch_size:]
        print('Set', str(i))
        i = i + 1
        yield batch_x, batch_y