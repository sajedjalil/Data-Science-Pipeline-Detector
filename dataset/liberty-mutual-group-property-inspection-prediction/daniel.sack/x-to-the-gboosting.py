import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder

def gini(y_true, y_pred):
    """ Simple implementation of the (normalized) gini score in numpy. 
        Fully vectorized, no python loops, zips, etc. Significantly
        (>30x) faster than previous implementions
        
        Credit: https://www.kaggle.com/jpopham91/
    """

    # check and get number of samples
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]
    
    # sort rows on prediction column 
    # (from largest to smallest)
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:,0].argsort()][::-1,0]
    pred_order = arr[arr[:,1].argsort()][::-1,0]
    
    # get Lorenz curves
    L_true = np.cumsum(true_order) / np.sum(true_order)
    L_pred = np.cumsum(pred_order) / np.sum(pred_order)
    L_ones = np.linspace(0, 1, n_samples)
    
    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)
    
    # normalize to true Gini coefficient
    return G_pred/G_true
    
    
def normalized_gini(y_true, y_pred):
    ng = gini(y_true, y_pred)/gini(y_true, y_true)
    return ng


def xgb_predict(train, target, test, max_rounds=10000):
    
    # set params
    params = dict(
        objective='reg:linear',
        eta=0.005,
        min_child_weight=6,
        subsample=0.7,
        colsample_bytree=0.7,
        scale_pos_weight=1,
        silent=1,
        max_depth=9
        )
    plist = list(params.items())
    
    # train/val split
    X_train, X_val, y_train, y_val = train_test_split(train, target, test_size=0.2)
    xgtrain = xgb.DMatrix(X_train, label=y_train)
    xgval = xgb.DMatrix(X_val, label=y_val)
    
    # set up test
    xgtest = xgb.DMatrix(test)
    
    # train using early stopping
    watchlist = [(xgtrain, 'train'), (xgval, 'val')]
    model = xgb.train(plist, xgtrain, max_rounds, watchlist, early_stopping_rounds=120)
    preds_train = model.predict(xgb.DMatrix(train), ntree_limit=model.best_iteration)
    preds = model.predict(xgtest, ntree_limit=model.best_iteration)
    
    return preds_train, preds
    
    
def prep_data1(train, target, test):
    # handle categorical data
    data = pd.get_dummies(pd.concat([train, test]))
    X_train = data.loc[train.index].values
    y_train = target.values
    X_test = data.loc[test.index].values
    
    return X_train, y_train, X_test


def prep_data2(train, target, test):
    train = train.T.to_dict().values()
    test = test.T.to_dict().values()
    
    vec = DictVectorizer(sparse=False)
    train = vec.fit_transform(train)
    test = vec.transform(test)
    
    return train, target, test


def prep_data3(train, target, test):
    train_s = np.array(train)
    test_s = np.array(test)
    
    # label encode the categorical variables
    for i in range(train_s.shape[1]):
        lbl = LabelEncoder()
        lbl.fit(list(train_s[:,i]) + list(test_s[:,i]))
        train_s[:,i] = lbl.transform(train_s[:,i])
        test_s[:,i] = lbl.transform(test_s[:,i])
    
    train_s = train_s.astype(float)
    test_s = test_s.astype(float)
    
    return train_s, target, test_s
    

if __name__ == '__main__':
    
    # load data
    train = pd.read_csv('../input/train.csv', index_col='Id')
    test = pd.read_csv('../input/test.csv', index_col='Id')
    
    # randomize data
    train = train.reindex(np.random.permutation(train.index))

    # split off target and indices
    target = train.Hazard
    train.drop('Hazard', axis=1, inplace=True)
    test_ind = test.index
    
    # prepare data
    train1, target1, test1 = prep_data1(train, target, test)
    train2, target2, test2 = prep_data2(train, target, test)
    train3, target3, test3 = prep_data3(train, target, test)
    
    # combine all data prep methods
    train = np.hstack([train1, train2, train3])
    test = np.hstack([test1, test2, test3])
    
    # fit model and predict
    preds_train, preds = xgb_predict(train, target, test, max_rounds=5000)
    
    # evaluate on training set
    print(normalized_gini(target, preds_train))
    
    # output predictions
    out = pd.DataFrame({'Hazard': preds}, index=test_ind)
    out.to_csv('submission.csv')