import numpy as np
import pandas as pd
from collections import Counter
from collections import defaultdict
import matplotlib.pyplot as plt
import math
import scipy

from sklearn import svm
from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import cross_val_score as cvs
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import ExtraTreesRegressor as ETR
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDRegressor as sgdr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator

from random import random
from random import randint
from itertools import permutations

class RejectionSampling(object):
    def __init__(self, iterator, k=1):
        self.iterator = iterator
        self.k = k
        self.ans = []

    def run(self):
        n = 0.0
        for i in self.iterator:
            n += 1.0
            r = random()
            if r < (self.k/n):
                self.__accept(i)
        return self.ans

    def __accept(self, x):
        if len(self.ans) == self.k:
            idx = randint(0, self.k-1)
            del self.ans[idx]
        self.ans.append(x)

class MedianEverything(BaseEstimator):
    def __init__(self, keys):
        self.keys = keys
        self.total = 0
        self.not_found = 0

    def fit(self, X, y=None):
        tmp = pd.concat([X,y], axis=1)
        target = tmp.columns[-1]
        
        #all keys dict
        all_keys = tmp.groupby(self.keys).agg({target: np.median})
        #initalise dict list
        self.dicts = [all_keys.to_dict()[target]]
        
        #all other dicts
        for n in range(1,len(self.keys)):
            keys = self.keys[:-n]
            #print (keys, self.keys, n)
            key_grouped = tmp.groupby(keys).agg({target: np.median})
            self.dicts.append(key_grouped.to_dict()[target])
        
        #product_median2 = tmp.groupby(self.keys).agg({target: np.median})
        #self.product_median2_dict = product_median2.to_dict()[target]
        
        #product_median = tmp.groupby(self.keys[0]).agg({target: np.median})
        #self.product_median_dict = product_median.to_dict()[target]
        
        #self.global_median = np.median(y)
        #global_median = np.median(y)
        
        #global medidan for fallback dict
        global_median = np.median(y)
        self.dicts.append(defaultdict(lambda: global_median))
        
        #self.dicts = [product_median2.to_dict()[target], product_median.to_dict()[target], defaultdict(lambda: global_median)]
        return self
        
    def get_value2(self, keys, dicts):
        tkey = tuple(keys)
        #if tkey in dicts[0].keys():
        try:
            return dicts[0][tkey]
        #else:
        except KeyError:
            return self.get_value2(keys[:-1], dicts[1:])
        
    def get_value(self, key):
        self.total += 1
        key = tuple(key)
        try:
            return self.product_median2_dict[key]
        except KeyError:
            try:
                return self.product_median_dict[key[0]]
            except KeyError:
                self.not_found += 1
                return self.global_median
        
    def predict(self, X):
        #return X[self.keys].map(lambda t: self.get_value(t))
        #return X[self.keys].apply(lambda t: self.get_value(t), axis=1)
        return X[self.keys].apply(lambda t: self.get_value2(t, self.dicts), axis=1)

#A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
# vectorized error calc
def rmsle(estimator, X, y0):
    #print(estimator.best_params_)
    y = estimator.predict(X)
    if len(y[y<=-1]) != 0:
        y[y<=-1] = 0.0
    #print("here",y[y<=-1],len(y[y<=-1]))
    assert len(y) == len(y0)
    r = np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))
    if math.isnan(r):
        print("this is a nan")
        print(scipy.stats.describe(y))
        plt.hist(y, bins=10, color='blue')
        plt.savefig("nan_y.png")
        
    return r

def get_data(N, quick=False):
    chunk = 10**3 #4
    cols = ['Semana','Agencia_ID','Canal_ID','Ruta_SAK','Cliente_ID','Producto_ID','Demanda_uni_equil']
    #cols = ['Semana','Agencia_ID','Canal_ID','Ruta_SAK','Cliente_ID','Producto_ID','Venta_uni_hoy']
    
    if quick:
        df_train = pd.read_csv('../input/train.csv', usecols=cols, nrows=40000)
    else:
    
        data = pd.read_csv('../input/train.csv', usecols=cols, iterator=True, chunksize=chunk)
        chunks = RejectionSampling(data,N).run()
        df_train = pd.concat(chunks, ignore_index=True)
    
    X = df_train[['Semana','Agencia_ID','Canal_ID','Ruta_SAK','Cliente_ID','Producto_ID']]#.values
    y = df_train['Demanda_uni_equil']#.values
    #y = df_train['Venta_uni_hoy']#.values

    return [X,y]
    
def submit(estimator, cols):
    #cols = ['Agencia_ID','Canal_ID','Ruta_SAK','Cliente_ID','Producto_ID']
    df_test = pd.read_csv('../input/test.csv', usecols=cols)
    sub = pd.read_csv('../input/sample_submission.csv')
    sub['Demanda_uni_equil'] = estimator.predict(df_test)
    sub.to_csv('medians.csv', index=False)

if __name__ == "__main__":
    N = 8000 #2
    N = 4000 #3
    N = 1650
    N = 10
    N = 800
    N = 2000
    #[X,y] = get_data(N, quick=True)
    [X,y] = get_data(N)
    print(X.shape)
    
    #key = 'Producto_ID'
    #key = ['Producto_ID', 'Agencia_ID']
    
    #keys = ['Canal_ID', 'Producto_ID', 'Cliente_ID']
    #keys = ['Canal_ID', 'Producto_ID', 'Cliente_ID', 'Agencia_ID']
    
    def calc(keys):
        clf = MedianEverything(keys)
        scores = cvs(clf, X, y, cv=10, scoring=rmsle)
        #print(scores)
        print("Key: %s Accuracy: %0.4f (+/- %0.4f)" % (keys, scores.mean(), scores.std() * 2))
    
    keys = ['Producto_ID']
    calc(keys)
    
    keys = ['Producto_ID', 'Canal_ID']
    calc(keys)
    
    keys = ['Producto_ID', 'Canal_ID', 'Cliente_ID']
    calc(keys)
    
    #keys = ['Producto_ID', 'Canal_ID', 'Ruta_SAK']
    #calc(keys)

    #keys = ['Producto_ID', 'Canal_ID']
    #for k in X.columns:
    #    if k not in keys:
    #        nkeys = keys.copy()
    #        nkeys.append(k)
    #        calc(nkeys)

 
    #ans = {}
    #ans2 = {}
    #base = ['Producto_ID', 'Semana', 'Agencia_ID']
    #for keys in permutations(X.columns,3):
    ##for keys in permutations(base):
    #    #if keys[0] == 'Producto_ID' and keys[1] == 'Semana':
    #    clf = MedianEverything(list(keys))
    #    scores = cvs(clf, X, y, cv=5, scoring=rmsle)
    #    ans2[keys] = scores.mean()
    #    ans[scores.mean()] = keys
    #m = min(ans2.values())
    #mkey = list(ans2.keys())[list(ans2.values()).index(m)]
    #print (mkey, m)
    #for x in ans.keys():
    #    print (ans[x], x)
    #m = min(ans.keys())
    #print (ans[m],m)
    
    #for i,x in enumerate(ans2.keys()):
    #    print (i,x, ans2[x])
    
    #clf.fit(X,y)
    #submit(clf, keys)
    
 #   rState = 0
    #rState = None
 #   ntrees = 1000
 #   depth = 4
    #scores = cvs(RFR(random_state=rState, n_estimators=ntrees), X, y, cv=5, scoring=rmsle)
    #scores = cvs(ETR(random_state=rState, n_estimators=ntrees, max_depth=depth), X, y, cv=10, scoring=rmsle)
#    clf = LR()
#    scores = cvs(clf, X, y, cv=10, scoring=rmsle)
#    print(scores)
   # print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
    
   