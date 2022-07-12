#jpopham91
#Abhishek
#schurik

#and maybe few other author's i cant remmember


import pandas as pd

from sklearn import preprocessing, ensemble, feature_selection

from sklearn import pipeline, metrics, grid_search

from sklearn.linear_model import LogisticRegression


# Simple implementation of the (normalized) gini score in numpy
# Fully vectorized, no python loops, zips, etc.
# Significantly (>30x) faster than previous implementions

import numpy as np

def Gini(y_true, y_pred):
    # check and get number of samples
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]

    # sort rows on prediction column
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:,0].argsort()][:,0]
    pred_order = arr[arr[:,1].argsort()][:,0]

    # get Lorenz curves
    L_true = np.cumsum(true_order) / np.sum(true_order)
    L_pred = np.cumsum(pred_order) / np.sum(pred_order)
    L_ones = np.linspace(0, 1, n_samples)

    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)

    # normalize to true Gini coefficient
    return G_pred/G_true

def prepare_data():
    # load train data
    train    = pd.read_csv('../input/train.csv')
    test     = pd.read_csv('../input/test.csv')
    labels   = train.Hazard
    test_ind = test.ix[:,'Id']
    train.drop('Hazard', axis=1, inplace=True)
    train.drop('Id', axis=1, inplace=True)
    test.drop('Id', axis=1, inplace=True)
    train = np.array(train)
    test = np.array(test)
    for i in range(train.shape[1]):
        if type(train[1,i]) is str:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train[:,i]) + list(test[:,i]))
            train[:,i] = lbl.transform(train[:,i])
            test[:,i] = lbl.transform(test[:,i])
    return train.astype(float), labels, test.astype(float), test_ind

def test():
    train, labels, _, _ = prepare_data()
    transform = feature_selection.SelectPercentile(feature_selection.f_classif)
    # Create the pipeline
    est = pipeline.Pipeline([
                                 ('model', LogisticRegression())
                            ])
    #LogisticRegression(C=1.0, intercept_scaling=1, dual=False, fit_intercept=True, penalty='l2', tol=0.0001)
                            

    # Create a parameter grid to search for best parameters for everything in the pipeline
    param_grid = {  'model__C':[100.0],
                    'model__intercept_scaling': [1],
                    'model__dual': [False],
                    'model__fit_intercept': [True],
                    'model__penalty': ['l2'],
                    'model__tol': [0.0001]

                   }

    # Normalized Gini Scorer
    gini_scorer = metrics.make_scorer(Gini, greater_is_better = True)

    # Initialize Grid Search Model
    model = grid_search.GridSearchCV(estimator  = est,
                                     param_grid = param_grid,
                                     scoring    = gini_scorer,
                                     verbose    = 10,
                                     n_jobs     = 1,
                                     iid        = True,
                                     refit      = True,
                                     cv         = 2)
    # Fit Grid Search Model
    model.fit(train, labels)
    print("Best score: %0.3f" % model.best_score_)
    print("Best parameters set:")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    # Get best model
    best_model = model.best_estimator_

    # Fit model with best parameters optimized for normalized_gini
    best_model.fit(train,labels)
    return best_model


def score(model):
    # load test data
    _, _, test, test_ind = prepare_data()
    preds = model.predict(test)
    preds4 = pd.DataFrame({"Id": test_ind, "Hazard": preds})
    preds4 = preds4.set_index('Id')
    preds4.to_csv('RandomForestClassifier.csv')

    return

def main():

    model = test()
    print("create submission")
    score(model)

main()