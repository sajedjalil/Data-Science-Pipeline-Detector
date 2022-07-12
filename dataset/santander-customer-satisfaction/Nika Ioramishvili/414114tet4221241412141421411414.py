import gc
import xgboost as xgb
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

if __name__ == "__main__":
    print('Started!')
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')
    
    remove = []
    c = train.columns
    for i in range(len(c)-1):
        v = train[c[i]].values
        for j in range(i+1, len(c)):
            if np.array_equal(v, train[c[j]].values):
                remove.append(c[j])

    train.drop(remove, axis=1, inplace=True)
    test.drop(remove, axis=1, inplace=True)

    remove = []
    for col in train.columns:
        if train[col].std() == 0:
            remove.append(col)
    
    
    train.to_csv("train_clear.csv", index=False)
    
    train.drop(remove, axis=1, inplace=True)
    test.drop(remove, axis=1, inplace=True)
    features = train.columns[1:-1]
    pca = PCA(n_components=2)
    x_train_projected = pca.fit_transform(normalize(train[features], axis=0))
    x_test_projected = pca.transform(normalize(test[features], axis=0))
    tokeep = train.columns[1:-1]
    features = train.columns[1:-1]
    todrop = list(set(tokeep).difference(set(features)))
    train.drop(todrop, inplace=True, axis=1)
    test.drop(todrop, inplace=True, axis=1)
    features = train.columns[1:-1]
    split = 10
    skf = StratifiedKFold(train.TARGET.values,
                          n_folds=split,
                          shuffle=False,
                          random_state=42)

    train_preds = None
    test_preds = None
    visibletrain = blindtrain = train
    index = 0
    print('Change num_rounds to 350')
    num_rounds = 10
    params = {}
    params["objective"] = "binary:logistic"
    params["eta"] = 0.03
    params["subsample"] = 0.8
    params["colsample_bytree"] = 0.7
    params["silent"] = 1
    params["max_depth"] = 5
    params["min_child_weight"] = 1
    params["eval_metric"] = "auc"
    for train_index, test_index in skf:
        print('Fold:', index)
        visibletrain = train.iloc[train_index]
        blindtrain = train.iloc[test_index]
        dvisibletrain = \
            xgb.DMatrix(csr_matrix(visibletrain[features]),
                        visibletrain.TARGET.values,
                        silent=True)
        dblindtrain = \
            xgb.DMatrix(csr_matrix(blindtrain[features]),
                        blindtrain.TARGET.values,
                        silent=True)
        watchlist = [(dblindtrain, 'eval'), (dvisibletrain, 'train')]
        clf = xgb.train(params, dvisibletrain, num_rounds,
                        evals=watchlist, early_stopping_rounds=50,
                        verbose_eval=False)

        blind_preds = clf.predict(dblindtrain)
        print('Blind Log Loss:', log_loss(blindtrain.TARGET.values,
                                          blind_preds))
        print('Blind ROC:', roc_auc_score(blindtrain.TARGET.values,
                                          blind_preds))
        index = index+1
        del visibletrain
        del blindtrain
        del dvisibletrain
        del dblindtrain
        gc.collect()
        dfulltrain = \
            xgb.DMatrix(csr_matrix(train[features]),
                        train.TARGET.values,
                        silent=True)
        dfulltest = \
            xgb.DMatrix(csr_matrix(test[features]),
                        silent=True)
        if(train_preds is None):
            train_preds = clf.predict(dfulltrain)
            test_preds = clf.predict(dfulltest)
        else:
            train_preds *= clf.predict(dfulltrain)
            test_preds *= clf.predict(dfulltest)
        del dfulltrain
        del dfulltest
        del clf
        gc.collect()

    train_preds = np.power(train_preds, 1./index)
    test_preds = np.power(test_preds, 1./index)
    print('Average Log Loss:', log_loss(train.TARGET.values, train_preds))
    print('Average ROC:', roc_auc_score(train.TARGET.values, train_preds))
    submission = pd.DataFrame({"ID": train.ID,
                               "TARGET": train.TARGET,
                               "PREDICTION": train_preds})

    submission.to_csv("simplexgbtrain.csv", index=False)
    submission = pd.DataFrame({"ID": test.ID, "TARGET": test_preds})
    submission.to_csv("simplexgbtest.csv", index=False)
    print('Finish')
