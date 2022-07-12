# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
#Feature weights taken from Sandro forum post.
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 03 21:00:28 2016
@author: hardy_000
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from sklearn import neighbors, datasets
from sklearn.neighbors import NearestNeighbors
from sklearn import cross_validation
from sklearn.metrics import pairwise
from scipy.spatial import distance
import scipy as sp
import sklearn as skl
from collections import Counter


fw = [500, 1000, 4, 3, 2, 10] 

def loadtraindata():
    z= pd.read_csv("../input/train.csv")
    return z
    
def loadtestdata():
    z= pd.read_csv("../input/test.csv")
def prepareData(dat):
    dat['accuracy']=normalize(train['accuracy'])
    dat=convertTime(dat)
    return dat
    return z
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""
    # By Jake VanderPlas
    # License: BSD-style

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)
def getScore(a,b):
    res=a==b
    return sum(res)/size(res)
def convertTime(df):
    df['x']=df['x']*fw[0]
    df['y']=df['y']*fw[1]
    
    initial_date = np.datetime64('2014-01-01T01:01', dtype='datetime64[m]')
    d_times = pd.DatetimeIndex(initial_date + np.timedelta64(int(mn), 'm') 
                               for mn in df.time.values)    
    df['hour'] = (d_times.hour+ d_times.minute/60) * fw[2]
    df['month'] = d_times.month * fw[4]
    df['year'] = (d_times.year - 2013) * fw[5]

    df = df.drop(['time'], axis=1) 
    return df
    
def getSquare(data,i,j,buff):
    data=data[((((j-1-buff)<=data['y']) & (data['y']<=j+buff))&(((i-1-buff)<=data['x']) & (data['x']<=i+buff)))]
    
    return data
def removeSquare(data,i,j):
    data=data[~((((j-1)<=data['y']) & (data['y']<=j))&(((i-1)<=data['x']) & (data['x']<=i)))]

    

    return data
def getPercentError(guess,label):
    correct=(guess==label)
    return (1./correct.shape[0])*sum(correct)
def plot3DScatter(data):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.xlabel('x')
    plt.ylabel('y')
    ax.set_zlabel('time')
    ax.scatter(normalize(data.x),normalize(data.y),normalize(data.time),c=data.place_id,marker='o',depthshade=False,lw = 0)
    plt.show()
#places=train.groupby('place_id')
#meanplaces=places.mean()
#meanplaces=meanplaces.reset_index(drop=False)
#plot a few locations means
#meanplaces[:100].plot(kind='scatter',x='x',y='y',c='place_id',colormap='RdYlGn')

def getSamplePlaces(data,numberOfPlaces=1):
        
    return data[data.place_id.isin((data.place_id).unique()[1:numberOfPlaces+1])]
def getSamplePlace(numberOfPlaces=1):
        
    return train[train.place_id==(train.place_id).unique()[numberOfPlaces]]

def normalize(series):
    return (series-min(series))/(max(series)-min(series))    
    
def myf(a,w):
    lookupTable, indexed_dataSet = np.unique(a, return_inverse=True)
    
    y= np.bincount(indexed_dataSet,w)
    lookupTable[y.argsort()]
    res=(lookupTable[y.argsort()][::-1][:3])
    ret=np.empty((3))
    ret.fill(res[-1])
    ret[0:res.shape[0]]=res
    return ret
def trainData(X,y):
    X_train=X
    y_train=y

    #clf = neighbors.KNeighborsClassifier(25)
    clf = neighbors.KNeighborsClassifier(n_neighbors=25, weights='distance', 
                               metric='manhattan')
    ft=clf.fit(X_train, y_train)
    return ft
    
    

def predictTest(test,clf,y_train,X_train,X_train_acc):
    distances,indices=clf.kneighbors(test)
    knearest_locs=np.array(X_train)[indices]
    knearest_labels=np.array(y_train)[indices]
    knearest_acc=np.array(X_train_acc)[indices]
    points=np.array(test)
    reppoints=np.tile(points,(25,1,1))
    reppoints=reppoints.transpose((1,0,2))

    ty=knearest_locs-reppoints
    tys=np.square(ty)
    tyssum=np.sum(tys,axis=2)
    tyssums=np.sqrt(tyssum)
    tyssumsac=knearest_acc/tyssums


    
    result = np.empty_like(knearest_labels[:,0:3])
    for i,(x,y) in enumerate(zip(knearest_labels,tyssumsac)):
        result[i] = myf(x,y)
    return result

def predictRegion(train,test):






    y_train=train.place_id
    X_train=train.drop('place_id',1)
    X_train_acc=X_train.accuracy
    X_train=X_train.drop('accuracy',1)
    X_train=X_train.drop('row_id',1)
    ft=trainData(X_train,y_train)
    


    X_test=test.drop('accuracy',1)
    row_id=X_test['row_id']
    X_test=X_test.drop('row_id',1)



    pred=predictTest(X_test,ft,y_train,X_train,X_train_acc)
    result=np.insert(pred,0,row_id,axis=1)
    return result
    
numberOfPlaces=8

train=loadtraindata()
#test=loadtestdata()
#test=prepareData(test)
train=prepareData(train)

train, test = cross_validation.train_test_split(train, test_size=0.4, random_state=0)

pred= np.empty((0,4), int)
#iterate over grid
p=7
for i in range(1,1):
    for j in range(1,1):
        temp=getSquare(test,i,j,0)
        res=predictRegion(getSquare(train,i,j,0.1),temp)
        test=removeSquare(test,i,j)
        places=temp.place_id
        pred=np.append(pred,res,0)
        p=getScore(pred,places)
        print(p)
        sys.stdout.flush()
print("blah")
#pred=pred[np.argsort(pred[:,0])]
#re=pd.DataFrame(pred)
#x=re[1].astype(str)+" "+re[2].astype(str)+" "+re[3].astype(str)
#re=pd.DataFrame(np.concatenate((pred[:,0].reshape(-1,1),x.reshape(-1,1)),1))
#re.columns=["row_id","place_id"]
#re.to_csv("C:\\Users\\hardy_000\\fbcomp\\submission.csv",index=False)
