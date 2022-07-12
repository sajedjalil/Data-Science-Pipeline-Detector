'''
This script is to illustrate a solid cross validation process for this competition.
We use 10 fold out-of-bag overall cross validation instead of averaging over folds. 
The entire process is repeated 5 times and then averaged.

You would notice that the CV value obtained by this method would be lower than the
usual procedure of averaging over folds. It also tends to have very low deviation.

Any scikit learn model can be validated using this. Models like XGBoost and 
Keras Neural Networks can also be validated using their respective scikit learn APIs.
XGBoost is illustrated here along with Ridge regression.
'''

import numpy as np 
import pandas as pd 
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
import xgboost as xgb

def R2(ypred, ytrue):
    y_avg = np.mean(ytrue)
    SS_tot = np.sum((ytrue - y_avg)**2)
    SS_res = np.sum((ytrue - ypred)**2)
    r2 = 1 - (SS_res/SS_tot)
    return r2

def cross_validate(model, x, y, folds=10, repeats=5):
    '''
    Function to do the cross validation - using stacked Out of Bag method instead of averaging across folds.
    model = algorithm to validate. Must be scikit learn or scikit-learn like API (Example xgboost XGBRegressor)
    x = training data, numpy array
    y = training labels, numpy array
    folds = K, the number of folds to divide the data into
    repeats = Number of times to repeat validation process for more confidence
    '''
    ypred = np.zeros((len(y),repeats))
    score = np.zeros(repeats)
    x = np.array(x)
    for r in range(repeats):
        i=0
        print('Cross Validating - Run', str(r + 1), 'out of', str(repeats))
        x,y = shuffle(x,y,random_state=r) #shuffle data before each repeat
        kf = KFold(n_splits=folds,random_state=i+1000) #random split, different each time
        for train_ind,test_ind in kf.split(x):
            print('Fold', i+1, 'out of',folds)
            xtrain,ytrain = x[train_ind,:],y[train_ind]
            xtest,ytest = x[test_ind,:],y[test_ind]
            model.fit(xtrain, ytrain)
            ypred[test_ind,r]=model.predict(xtest)
            i+=1
        score[r] = R2(ypred[:,r],y)
    print('\nOverall R2:',str(score))
    print('Mean:',str(np.mean(score)))
    print('Deviation:',str(np.std(score)))
    pass

def main():
    train = pd.read_csv('../input/train.csv')
    y = np.array(train['y'])
    train = train.drop(['ID','y','X0','X1','X2','X3','X4','X5','X6','X8'], axis=1)
    ridge_model = Ridge(alpha=1)
    xgb_model = xgb.XGBRegressor(max_depth=2, learning_rate=0.01, n_estimators=10, silent=True,
                                objective='reg:linear', nthread=-1, base_score=100, seed=4635,
                                missing=None)
    cross_validate(ridge_model, np.array(train), y, folds=10, repeats=5) #validate ridge regression
    #cross_validate(xgb_model, np.array(train), y, folds=10, repeats=5) #validate xgboost
    
    pass

if __name__ == '__main__':
	main()
    
