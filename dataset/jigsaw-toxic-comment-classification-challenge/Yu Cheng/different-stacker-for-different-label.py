import pandas as pd
import numpy as np
import re
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning, module='sklearn')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score


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
    sub = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')
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

   
    scores = []
    # different stacker for different label
    # A single model ( with the same hyper-parameters ) is not competent for all the labels
    # I just use xgboost with different hyper-parameter, you coulde try differnt models like lr or LightGBM
    
    stackers = {
        'toxic' : xgb.XGBClassifier(missing=np.nan, max_depth=5, n_estimators=90, learning_rate=0.03, nthread=-1, subsample=0.95, colsample_bytree=0.85, seed=4242,  scale_pos_weight=float(np.sum(train['toxic']==0)) / (np.sum(train['toxic']==1))), # 0.9844
        'severe_toxic' : xgb.XGBClassifier(missing=np.nan, max_depth=6, n_estimators=125, learning_rate=0.03, nthread=-1, subsample=0.95, colsample_bytree=0.85, seed=4242, scale_pos_weight=float(np.sum(train['severe_toxic']==0)) / (np.sum(train['severe_toxic']==1))),
        'obscene' : xgb.XGBClassifier(missing=np.nan, max_depth=6, n_estimators=25, learning_rate=0.03, nthread=-1, subsample=0.95, colsample_bytree=0.85, seed=4242, scale_pos_weight=float(np.sum(train['obscene']==0)) / (np.sum(train['obscene']==1))),
        'threat' : xgb.XGBClassifier(missing=np.nan, max_depth=5, n_estimators=100, learning_rate=0.07, nthread=-1, subsample=0.95, colsample_bytree=0.85, seed=4242, scale_pos_weight=float(np.sum(train['threat']==0)) / (np.sum(train['threat']==1))),
        'insult' : xgb.XGBClassifier(missing=np.nan, max_depth=5, n_estimators=100, learning_rate=0.05, nthread=-1, subsample=0.95, colsample_bytree=0.85, seed=4242, scale_pos_weight=float(np.sum(train['insult']==0)) / (np.sum(train['insult']==1))),
        'identity_hate' : xgb.XGBClassifier(missing=np.nan, max_depth=6, n_estimators=60, learning_rate=0.05, nthread=-1, subsample=0.95, colsample_bytree=0.85, seed=4242, scale_pos_weight=float(np.sum(train['identity_hate']==0)) / (np.sum(train['identity_hate']==1)))

    }
    
    for label in LABELS:
        print(label)
        score = cross_val_score(stackers[label], X_train, train[label], cv=5, scoring='roc_auc')
        print("AUC:", score)
        
        scores.append(np.mean(score))
        stackers[label].fit(X_train, train[label])
        sub[label] = stackers[label].predict_proba(X_test)[:,1]
        
    print("CV score:", np.mean(scores))

    sub.to_csv("submission.csv", index=False)
