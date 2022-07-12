import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from collections import Counter
from collections import defaultdict
from sklearn.cross_validation import train_test_split
import math

#A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
# vectorized error calc
def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))

def get_data():
    df_train = pd.read_csv('../input/train.csv', usecols=['Producto_ID','Demanda_uni_equil'], nrows=8000000)
    #df_train = pd.read_csv('../input/train.csv', usecols=['Producto_ID','Demanda_uni_equil'])
    X = df_train['Producto_ID']
    y = df_train['Demanda_uni_equil']

    return train_test_split(X, y, test_size=0.30, random_state=43)

def train(X,y,N):
    top_N = Counter(X).most_common(N)
    total_median = np.median(y)
    results = defaultdict(lambda: total_median)
    for item, value in top_N:
        idx = X == item
        median_value = np.median(y[idx])
        results[item] = median_value 
        
    return results
    
def predict(model, X):
    f = lambda x: model[x]
    return X.apply(f)
    
def submit(model):
    df_test = pd.read_csv('../input/test.csv', usecols=['Producto_ID'])
    sub = pd.read_csv('../input/sample_submission.csv')
    f = lambda x: model[x]
    sub['Demanda_uni_equil'] = df_test['Producto_ID'].apply(f)
    sub.to_csv('topN.csv', index=False)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_data()
    for x in range(1000,2000,100):
        model = train(X_train, y_train, N=x)
        y_pred = predict(model, X_test)

        a = rmsle(y_test, y_pred)
        print(x,a)
    
    model = train(X_train, y_train, N=1600)
    submit(model)