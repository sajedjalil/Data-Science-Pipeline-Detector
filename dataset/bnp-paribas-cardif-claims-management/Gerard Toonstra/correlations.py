# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import numpy as np

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

def plot_corr(df,threshold,size=14):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''

    df.drop(["ID","target"],axis=1,inplace=True)
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    locs, labels = plt.xticks(range(len(corr.columns)), corr.columns)
    plt.setp(labels, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    # plt.show()
    plt.savefig('correlations.jpg')
    print("Saved correlations graph")
    plt.clf()

    corr[corr<threshold] = np.nan
    #corr.fillna(0,inplace=True)

    pd.DataFrame(corr).to_csv('correlation.csv',index=True)

    cov = df.cov()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(cov)
    locs, labels = plt.xticks(range(len(corr.columns)), corr.columns)
    plt.setp(labels, rotation=90)
    plt.yticks(range(len(cov.columns)), cov.columns)
    # plt.show()
    plt.savefig('covariations.jpg')
    print("Saved covariations graph")

def load_data(filename):
    df = pd.read_csv(filename)
    df.drop(['v22'], axis=1, inplace=True)
    df.drop(['v91'], axis=1, inplace=True)
    return df

print("Loading and converting train set")    
train = load_data('../input/train.csv')
print("Loading and converting test set")
test  = load_data('../input/test.csv')

df_all = pd.concat((train, test), axis=0, ignore_index=True)

plot_corr(df_all,0.5)
