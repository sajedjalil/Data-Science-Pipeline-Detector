import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer


def xgb_predict(train, target, test, max_rounds=10000):
    """ Based on https://www.kaggle.com/soutik/liberty-mutual-group-property-inspection-prediction/blah-xgb/run/35158
    """
    
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
    preds = model.predict(xgtest, ntree_limit=model.best_iteration)
    
    return preds
    

if __name__ == '__main__':
    
    # load data
    train = pd.read_csv('../input/train.csv', index_col='Id')
    test = pd.read_csv('../input/test.csv', index_col='Id')

    # split off targets and indices
    targets = train.Hazard
    train.drop('Hazard', axis=1, inplace=True)
    test_ind = test.index
    
    # handle categorical data
    # data = pd.get_dummies(pd.concat([train, test]))
    # X_train = data.loc[train.index].values
    # y_train = targets.values
    # X_test = data.loc[test.index].values
    
    train = train.T.to_dict().values()
    test = test.T.to_dict().values()
    
    vec = DictVectorizer()
    train = vec.fit_transform(train)
    test = vec.transform(test)

    # randomize data
    # idx = np.random.permutation(train.shape[0])
    # train = train[idx]
    # targets = targets[idx]
    
    # run xgb
    preds = xgb_predict(train, targets, test)
    
    # output predictions
    out = pd.DataFrame({'Hazard': preds}, index=test_ind)
    out.to_csv('submission.csv')