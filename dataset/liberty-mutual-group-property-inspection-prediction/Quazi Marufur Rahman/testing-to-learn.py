import pandas as pd
import numpy as np 
from sklearn import preprocessing
# import xgboost as xgb
from sklearn import pipeline, metrics, grid_search
from sklearn.utils import shuffle
from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn.svm import SVC


TRAIN = '../input/train.csv'
TEST = '../input/test.csv'

def gini(solution, submission):
    df = zip(solution, submission, range(len(solution)))
    df = sorted(df, key=lambda x: (x[1],-x[2]), reverse=True)
    rand = [float(i+1)/float(len(df)) for i in range(len(df))]
    totalPos = float(sum([x[0] for x in df]))
    cumPosFound = [df[0][0]]
    for i in range(1,len(df)):
        cumPosFound.append(cumPosFound[len(cumPosFound)-1] + df[i][0])
    Lorentz = [float(x)/totalPos for x in cumPosFound]
    Gini = [Lorentz[i]-rand[i] for i in range(len(df))]
    return sum(Gini)

def normalized_gini(solution, submission):
    normalized_gini = gini(solution, submission)/gini(solution, solution)
    return normalized_gini

def prepare_data():
    # load train data  


    train    = pd.read_csv(TRAIN)
    test     = pd.read_csv(TEST)
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


    svc = SVC()
    pca = PCA()

    pipe = pipeline.Pipeline(steps = [('pca', pca), ('svc', svc)])
    n_components = [16]
    kernel = ['poly']
    gini_scorer = metrics.make_scorer(normalized_gini, greater_is_better = True)
    model = grid_search.GridSearchCV(   
                                estimator = pipe,
                                param_grid = dict
                                (
                                    pca__n_components=n_components,
                                    svc__kernel=kernel
                                ),
                                scoring    = gini_scorer,
                                verbose    = 10,
                                n_jobs     = 1,
                                iid        = True,
                                refit      = True,
                                cv         = 2
                            )
    print ('fitting')
   
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
    preds = pd.DataFrame({"Id": test_ind, "Hazard": preds})
    preds = preds.set_index('Id')
    preds.to_csv('output.csv')
    return

def main():
    model = test()
    score(model)

main()