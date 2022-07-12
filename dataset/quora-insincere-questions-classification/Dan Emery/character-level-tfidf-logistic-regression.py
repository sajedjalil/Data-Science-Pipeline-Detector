'''
This script took a lot of inspiration from Bojan Tunguz's Logistic Regression Baseline kernel here: https://www.kaggle.com/tunguz/lr-with-words-n-grams-baseline
'''

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from scipy.sparse import hstack
from sklearn.metrics import f1_score
import gc


train = pd.read_csv('../input/train.csv').fillna(' ')
test = pd.read_csv('../input/test.csv').fillna(' ')

train_text = train['question_text']
test_text = test['question_text']
all_text = pd.concat([train_text, test_text])

#This is completely untuned.
char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char_wb',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(2, 5),
    max_features=50000)

print('\nFitting Vectorizer')
char_vectorizer.fit(all_text)

print('\nTransforming Text')
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)

y = train['target']

# Function to run cross validation with the capability of averaging results over multiple seeds
def kfold_sklearn(train_df, test_df,y, clf, num_folds, seeds):
    print("Starting Model. Train shape: {}".format(train_df.shape))
    oofs = np.zeros(train_df.shape[0])
    subs = np.zeros(test_df.shape[0])
    # Cross validation model
    for seed in seeds:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

        # Create arrays and dataframes to store results
        oof_preds = np.zeros(train_df.shape[0])
        sub_preds = np.zeros(test_df.shape[0])

        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, y)):

            train_x, train_y = train_df[train_idx], y[train_idx]
            valid_x, valid_y = train_df[valid_idx], y[valid_idx]

            clf.fit(train_x, train_y)

            oof_preds[valid_idx] = clf.predict_proba(valid_x)[:,1]
            sub_preds += clf.predict_proba(test_df)[:,1] / folds.n_splits

            print('Fold %2d F1 Score : %.6f' % (n_fold + 1, f1_score(valid_y, (oof_preds[valid_idx]>0.27).astype(np.int))))
            del train_x, train_y, valid_x, valid_y
            gc.collect()

        oofs += oof_preds/len(seeds)
        subs += sub_preds/len(seeds)
    
    print('Full F1 score %.6f' % f1_score(y, (oofs>.27).astype(np.int)))
    return oofs, subs
    
clf = LogisticRegression(C=20, solver='sag')
oofs, preds = kfold_sklearn(train_char_features, test_char_features, y, clf, 5, [42])

score = 0
thresh = .5
for i in np.arange(0.1, 0.501, 0.01):
    temp_score = f1_score(y, (oofs > i))
    if(temp_score > score):
        score = temp_score
        thresh = i

print("CV: {}, Threshold: {}".format(score, thresh))
submission = pd.DataFrame.from_dict({'qid': test['qid']})
submission['prediction'] = (preds>thresh).astype(int)
submission.to_csv('submission.csv', index=False)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    