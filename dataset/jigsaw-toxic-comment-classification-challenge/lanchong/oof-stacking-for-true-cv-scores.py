"""
this scipt is inspired by Håkon Hapnes Strand and an other person(sorry, i can't find his kernel).
I think the script can  get the true cv score compared by ths script written by  Håkon Hapnes Strand.
https://www.kaggle.com/hhstrand/oof-stacking-regime
Good luck for everyone.
I have learned a lot from this competition,but I don't have enough source to fine tune the parameters and ensembel the models.
"""




import pandas as pd
import numpy as np
import re
import lightgbm as lgb
import warnings
from sklearn.model_selection import KFold
import gc
warnings.filterwarnings(action='ignore', category=DeprecationWarning, module='sklearn')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from collections import defaultdict
import os
from sklearn.metrics import roc_auc_score


#######################
# FEATURE ENGINEERING #
#######################
"""
Main function
Input: pandas Series and a feature engineering function
Output: pandas Series
"""
def engineer_feature(series, func, normalize=True):
    feature = series.apply(func)
       
    if normalize:
        feature = pd.Series(z_normalize(feature.values.reshape(-1,1)).reshape(-1,))
    feature.name = func.__name__ 
    return feature

"""
Engineer features
Input: pandas Series and a list of feature engineering functions
Output: pandas DataFrame
"""
def engineer_features(series, funclist, normalize=True):
    features = pd.DataFrame()
    for func in funclist:
        feature = engineer_feature(series, func, normalize)
        features[feature.name] = feature
    return features

"""
Normalizer
Input: NumPy array
Output: NumPy array
"""
scaler = StandardScaler()
def z_normalize(data):
    scaler.fit(data)
    return scaler.transform(data)
    
"""
Feature functions
"""
def asterix_freq(x):
    return x.count('!')/len(x)

def uppercase_freq(x):
    return len(re.findall(r'[A-Z]',x))/len(x)
    
"""
Import submission and OOF files
"""
def get_subs(nums):
    subs = np.hstack([np.array(pd.read_csv("../input/trained-models/sub" + str(num) + ".csv")[LABELS]) for num in nums])
    oofs = np.hstack([np.array(pd.read_csv("../input/trained-models/oof" + str(num) + ".csv")[LABELS]) for num in nums])
    return subs, oofs

if __name__ == "__main__":
    
    train = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv').fillna(' ')
    test = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv').fillna(' ')
    submission = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')
    INPUT_COLUMN = "comment_text"
    LABELS = train.columns[2:]
    
    # Import submissions and OOF files
    # 29: LightGBM trained on Fasttext (CV: 0.9765, LB: 0.9620)
    # 51: Logistic regression with word and char n-grams (CV: 0.9858, LB: ?)
    # 52: LSTM trained on Fasttext (CV: ?, LB: 0.9851)
    subnums = [29,51,52]
    subs, oofs = get_subs(subnums)
    
    # Engineer features
    feature_functions = [len, asterix_freq, uppercase_freq]
    features = [f.__name__ for f in feature_functions]
    F_train = engineer_features(train[INPUT_COLUMN], feature_functions)
    F_test = engineer_features(test[INPUT_COLUMN], feature_functions)
    
    X_train = np.hstack([F_train[features].as_matrix(), oofs])
    X_test = np.hstack([F_test[features].as_matrix(), subs])
    params = {
        "objective": "binary",
        'metric': {'auc'},
        "boosting_type": "gbdt",
        "verbosity": -1,
        "num_threads": 4,
        "max_depth":3,
        "bagging_fraction": 0.8,
        "bagging_freq":5,
        #"colsample_bytree":0.45,
        "feature_fraction": 0.45,
        "learning_rate": 0.1,
        "num_leaves": 3,
        "verbose": -1,
        #"min_split_gain": .1,
        "reg_alpha": .3
    }
    # Now go through folds
    # I use K-Fold for reasons described here : 
    # https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/49964
    train.drop("comment_text",axis=1,inplace=True)
    scores = []
    scores_false = []
    folds = KFold(n_splits=10, shuffle=True, random_state=233)
    lgb_round_dict = defaultdict(int)
    trn_lgbset = lgb.Dataset(X_train, free_raw_data=False)
    del X_train
    gc.collect()
    for class_name in LABELS:
        print("Class %s scores : " % class_name)
        class_pred = np.zeros(len(train))
        train_target = train[class_name]
        trn_lgbset.set_label(train_target.values)
        submission[class_name] = np.zeros(len(X_test))
        lgb_rounds = 1000
        score_temp = 0;
        for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train, train_target)):
            watchlist = [
                trn_lgbset.subset(trn_idx),
                trn_lgbset.subset(val_idx)
            ]
            # Train lgb l1
            model = lgb.train(
                params=params,
                train_set=watchlist[0],
                num_boost_round=lgb_rounds,
                valid_sets=watchlist,
                early_stopping_rounds=50,
                verbose_eval=0
            )
            class_pred[val_idx] = model.predict(trn_lgbset.data[val_idx], num_iteration=model.best_iteration)
            submission[class_name] = submission[class_name] + model.predict(X_test, num_iteration=model.best_iteration)
            score = roc_auc_score(train_target.values[val_idx], class_pred[val_idx])
            score_temp += score
            # Compute mean rounds over folds for each class
            # So that it can be re-used for test predictions
            lgb_round_dict[class_name] += model.best_iteration
            print("\t Fold %d : %.6f in %3d rounds" % (n_fold + 1, score, model.best_iteration))
        submission[class_name] = submission[class_name] / folds.n_splits  
        print("full score : %.6f" % roc_auc_score(train_target, class_pred))
        scores.append(roc_auc_score(train_target, class_pred))
        scores_false.append(score_temp / 10)
        train[class_name + "_oof"] = class_pred


    print('Total CV score is {}'.format(np.mean(scores)))
    print("total false CV score is {}".format(np.mean(scores_false)))
    submission.to_csv("result.csv", index=False)
    