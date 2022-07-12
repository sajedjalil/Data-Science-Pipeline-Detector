## @brief Customized (soft) kappa in XGBoost
## @author Chenglong Chen
## @note You might have to spend some effort to tune the hessian (in softkappaobj function)
##  and the booster param to get it to work.

import pandas as pd
import numpy as np
import xgboost as xgb
from ml_metrics import quadratic_weighted_kappa

#####################
## Helper function ##
#####################
## softmax
def softmax(score):
    score = np.asarray(score, dtype=float)
    score = np.exp(score-np.max(score))
    score /= np.sum(score, axis=1)[:,np.newaxis]
    return score

## soft version of kappa score using the class probability
## inspired by @George Mohler in the Higgs competition
## https://www.kaggle.com/c/higgs-boson/forums/t/10286/customize-loss-function-in-xgboost/53459#post53459
## NOTE: As also discussed in the above link, it is hard to tune the hessian to get it to work.
def softkappaobj(preds, dtrain):
    ## label are in [0,1,2,3] as required by XGBoost for multi-classification
    labels = dtrain.get_label() + 1
    labels = np.asarray(labels, dtype=int)
    preds = softmax(preds)
    M = preds.shape[0]
    N = preds.shape[1]

    ## compute O (enumerator)
    O = 0.0
    for j in range(N):
        wj = (labels - (j+1.))**2
        O += np.sum(wj * preds[:,j])
    
    ## compute E (denominator)
    hist_label = np.bincount(labels)[1:]
    hist_pred = np.sum(preds, axis=0)
    E = 0.0
    for i in range(N):
        for j in range(N):
            E += pow(i - j, 2.0) * hist_label[i] * hist_pred[j]

    ## compute gradient and hessian
    grad = np.zeros((M, N))
    hess = np.zeros((M, N))
    for n in range(N):
        ## first-order derivative: dO / dy_mn
        dO = np.zeros((M))
        for j in range(N):
            indicator = float(n == j)
            dO += ((labels - (j+1.))**2) * preds[:,n] * (indicator - preds[:,j])
        ## first-order derivative: dE / dy_mn
        dE = np.zeros((M))
        for k in range(N):
            for l in range(N):
                indicator = float(n == k)
                dE += pow(k-l, 2.0) * hist_label[l] * preds[:,n] * (indicator - preds[:,k])
        ## the grad
        grad[:,n] = -M * (dO * E - O * dE) / (E**2)
        
        ## second-order derivative: d^2O / d (y_mn)^2
        d2O = np.zeros((M))
        for j in range(N):
            indicator = float(n == j)
            d2O += ((labels - (j+1.))**2) * preds[:,n] * (1 - 2.*preds[:,n]) * (indicator - preds[:,j])
       
        ## second-order derivative: d^2E / d (y_mn)^2
        d2E = np.zeros((M))
        for k in range(N):
            for l in range(N):
                indicator = float(n == k)
                d2E += pow(k-l, 2.0) * hist_label[l] * preds[:,n] * (1 - 2.*preds[:,n]) * (indicator - preds[:,k])
        ## the hess
        hess[:,n] = -M * ((d2O * E - O * d2E)*(E**2) - (dO * E - O * dE) * 2. * E * dE) / (E**4)

    grad *= -1.
    hess *= -1.
    # this pure hess doesn't work in my case, but the following works ok
    # use a const
    #hess = 0.000125 * np.ones(grad.shape, dtype=float)
    # or use the following...
    scale = 0.000125 / np.mean(abs(hess))
    hess *= scale
    hess = np.abs(hess) # It works!! no idea...
    grad.shape = (M*N)
    hess.shape = (M*N)
    return grad, hess

# evalerror is your customized evaluation function to 
# 1) decode the class probability 
# 2) compute quadratic weighted kappa
def evalerror(preds, dtrain):
    ## label are in [0,1,2,3] as required by XGBoost for multi-classification
    labels = dtrain.get_label() + 1
    ## class probability
    preds = softmax(preds)
    ## decoding (naive argmax decoding)
    pred_labels = np.argmax(preds, axis=1) + 1
    ## compute quadratic weighted kappa (using implementation from @Ben Hamner
    ## https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/quadratic_weighted_kappa.py
    kappa = quadratic_weighted_kappa(labels, pred_labels)
    return 'kappa', kappa


####################
## Model buliding ##
####################
## to use the above obj, you have to use the following task param (in addition to your general param and booster param)
## see https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
param = {
    'objective': 'reg:linear', # for linear raw predict score
    'num_class': 4 # num of classes
}

## I find it hard to tune the param, and the following seems working ok on my end
param['booster'] = 'gblinear' # gbtree has more params to tune...
param['eta'] = 1
param['lambda'] = 0.00005
param['alpha'] = 0.000001

## data
#train21 = pd.read_csv("../input/train.csv").fillna("")
#test21  = pd.read_csv("../input/test.csv").fillna("")
dtrain = xgb.DMatrix("../input/train.csv", silent=True)
dvalid = xgb.DMatrix("../input/test.csv", silent=True)

## train
num_round = 10
watchlist = [(dtrain, 'train'), (dvalid, "valid")]
bst = xgb.train(param, dtrain, num_round, watchlist, obj=softkappaobj, feval=evalerror)

## make prediction (class probability)
pred = softmax(bst.predict(dvalid))