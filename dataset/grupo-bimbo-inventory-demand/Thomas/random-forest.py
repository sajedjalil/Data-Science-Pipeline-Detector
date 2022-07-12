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
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDRegressor as sgdr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

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

def get_data():
    N = 500000
    N = 1000000
    #N = 50000
    df_train = pd.read_csv('../input/train.csv', usecols=['Semana','Agencia_ID','Canal_ID','Ruta_SAK','Cliente_ID','Producto_ID','Demanda_uni_equil'], nrows=N)
    X = df_train[['Semana','Agencia_ID','Canal_ID','Ruta_SAK','Cliente_ID','Producto_ID']].values
    y = df_train['Demanda_uni_equil'].values

    return [X,y]
    
def submit(estimator):
    df_test = pd.read_csv('../input/test.csv', usecols=['Producto_ID'])
    sub = pd.read_csv('../input/sample_submission.csv')
    f = lambda x: model[x]
    sub['Demanda_uni_equil'] = df_test['Producto_ID'].apply(f)
    sub.to_csv('topN.csv', index=False)

if __name__ == "__main__":
    #X_train, X_test, y_train, y_test = get_data()
    [X_,y] = get_data()
    s = StandardScaler().fit(X_)
    X = s.transform(X_)
    #scores = cvs(LR(), X, y, cv=5, scoring=rmsle)
    #print(scores)
    #print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
    
    rState = 0
    #rState = None
    #ntrees = 50
    #scores = cvs(RFR(random_state=rState, n_estimators=ntrees), X, y, cv=5, scoring=rmsle)
    #print(scores)
    #print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
    
    
    #clf1 = make_pipeline(StandardScaler(), sgdr(loss="epsilon_insensitive"))
    clf1 = sgdr(loss="epsilon_insensitive", random_state=rState)
    clf1 = sgdr(loss="huber", random_state=rState)
    #clf1 = sgdr(loss="huber", random_state=rState, n_iter = np.ceil(10**6 / X.shape[0]))
    #clf1 = sgdr(loss="squared_loss", random_state=rState)
    #clf1 = sgdr(loss="squared_epsilon_insensitive", random_state=rState)
    print(clf1.get_params().keys())
    #parameters = {'sgdregressor__alpha':10.0**-np.arange(1,9)}
    parameters = {'alpha':10.0**-np.arange(1,7), 'epsilon':[0.1,0.001,0.0001]}
    clf = GridSearchCV(clf1, parameters, scoring=rmsle)
    scores = cvs(clf, X, y, cv=5, scoring=rmsle)
    print(scores)
    print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
    
    #scores = cvs(RFR(random_state=0, n_estimators=200), X, y, cv=10, scoring=rmsle)
    #print(scores)
    #print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
    
    #clf = LR().fit(X_train, y_train)
    #s = clf.score(X_test, y_test)
    #print(s)
    
    #for x in range(1000,2000,100):
    #    model = train(X_train, y_train, N=x)
    #    y_pred = predict(model, X_test)

    #    a = rmsle(y_test, y_pred)
    #    print(x,a)
    
    #model = train(X_train, y_train, N=1600)
    #submit(model)