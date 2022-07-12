import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn import pipeline, metrics, grid_search


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


def fit(train, target):
    
    # set up pipeline
    est = pipeline.Pipeline([
            ('xgb', xgb.XGBRegressor(silent=True)),
        ])
        
    # create param grid for grid search
    params = {
        'xgb__learning_rate': [0.003, 0.005, 0.01, ],
        'xgb__min_child_weight': [5, 6, 7, ],
        'xgb__subsample': [0.5, 0.7, 0.9, ],
        'xgb__colsample_bytree': [0.5, 0.7, 0.9, ],
        'xgb__max_depth': [1, 3, 5, 7, 9, 11, ],
        'xgb__n_estimators': [10, 50, 100, ],
        }

    # set up scoring mechanism
    gini_scorer = metrics.make_scorer(normalized_gini, greater_is_better=True)
    
    # initialize gridsearch
    gridsearch = grid_search.RandomizedSearchCV(
        estimator=est,
        param_distributions=params,
        scoring=gini_scorer,
        verbose=10,
        n_jobs=-1,
        cv=3,
        n_iter=3,
        )
        
    # fit gridsearch
    gridsearch.fit(train, target)
    print('Best score: %.3f' % gridsearch.best_score_)
    print('Best params:')
    for k, v in sorted(gridsearch.best_params_.items()):
        print("\t%s: %r" % (k, v))
        
    # get best estimator
    return gridsearch.best_estimator_


def predict(est, test, test_index):
    y_pred = est.predict(test)
    pred = pd.DataFrame({'Hazard': y_pred}, index=test_index)
    pred.to_csv('submission.csv')
    print('Predictions saved to submission.csv')
    

if __name__ == '__main__':
    
    # load data
    train = pd.read_csv('../input/train.csv', index_col='Id')
    test = pd.read_csv('../input/test.csv', index_col='Id')
    
    target = train.Hazard
    train.drop('Hazard', axis=1, inplace=True)

    # preprocess categorical features and convert to numpy arrays
    data = pd.get_dummies(pd.concat([train, test]))
    X_train = data.loc[train.index].values
    y_train = target.values
    X_test = data.loc[test.index].values
    
    # randomize train set
    idx = np.random.permutation(len(train))
    X_train = X_train[idx]
    y_train = y_train[idx]
    
    # fit model
    est = fit(X_train, y_train)
    
    # generate predictions
    predict(est, X_test, test.index)
    