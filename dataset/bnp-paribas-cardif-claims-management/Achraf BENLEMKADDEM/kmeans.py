import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
from sklearn.cluster import KMeans
from sklearn import datasets

trainDF = pd.read_csv("../input/train.csv")
trainDF.set_index('ID')
print(trainDF.size)

dataDF = trainDF.drop('target',1)

print(dataDF.loc[[-4]])
'''
km = KMeans(n_clusters=2) 
km.fit(testDF)
'''


# Any results you write to the current directory are saved as output.