# reference:
# https://www.kaggle.com/featureblind/robust-lasso-patches-with-rfe-gs
# https://www.kaggle.com/aantonova/851-logistic-regression
import numpy as np
import pandas as pd
import gc
import time
from datetime import datetime
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression

train = pd.read_csv('../input/train.csv', index_col = 'id')
test = pd.read_csv('../input/test.csv', index_col = 'id')
target = train['target']

def save_submit(test_, clfs_, filename):
    subm = pd.DataFrame(np.zeros(test_.shape[0]), index = test_.index, columns = ['target'])
    for clf in clfs_:
        subm['target'] += clf.predict_proba(test_)[:, 1]
    subm['target'] /= len(clfs_)
    subm = subm.reset_index()
    subm.columns = ['id', 'target']
    subm.to_csv(filename, index = False)

scores = pd.DataFrame(columns = ['auc', 'acc', 'loss', 'tn', 'fn', 'fp', 'tp'])

# selected by RFECV with lasso
features = [ '16', '33', '43', '45', '52', '63', '65', '73', '90', '91', '117', '133', '134', '149', '189', '199', '217', '237', '258', '295']

train = train[features]
test = test[features]

def logreg_cross_validation(train_, target_, params,
                            num_folds = 5, repeats = 20, rs = 3210):
    print(params)
    clfs = []
    folds = RepeatedStratifiedKFold(n_splits = num_folds, n_repeats = repeats, random_state = rs)

    valid_pred = pd.DataFrame(index = train_.index)
    
    # Cross-validation cycle
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(target_, target_)):
        print('--- Fold {} started at {}'.format(n_fold, time.ctime()))
        
        train_x, train_y = train_.iloc[train_idx], target_.iloc[train_idx]
        valid_x, valid_y = train_.iloc[valid_idx], target_.iloc[valid_idx]
        
        clf = LogisticRegression(**params)
        clf.fit(train_x, train_y)
        clfs.append(clf)

        predict = clf.predict_proba(valid_x)[:, 1]

        tn, fp, fn, tp = confusion_matrix(valid_y, (predict >= .5) * 1).ravel()
        auc = roc_auc_score(valid_y, predict)
        acc = accuracy_score(valid_y, (predict >= .5) * 1)
        loss = log_loss(valid_y, predict)
        print('TN =', tn, 'FN =', fn, 'FP =', fp, 'TP =', tp)
        print('AUC = ', auc, 'Loss =', loss, 'Acc =', acc)

        valid_pred[n_fold] = pd.Series(predict, index = valid_x.index)

        del train_x, train_y, valid_x, valid_y, predict
        gc.collect()
    return clfs, valid_pred

params = {'random_state':300, 'n_jobs':-1, 'C':0.2, 'penalty':'l1', 'class_weight':'balanced', 'solver':'saga'}

clfs, pred = logreg_cross_validation(train, target, params)
pred_mean = pred.mean(axis = 1)
scores = scores.T
tn, fp, fn, tp = confusion_matrix(target, (pred_mean >= .5) * 1).ravel()
scores['logreg'] = [roc_auc_score(target, pred_mean), accuracy_score(target, (pred_mean >= .5) * 1), log_loss(target, pred_mean), tn, fn, fp, tp]
scores = scores.T

score_auc = scores.loc['logreg', 'auc']
score_acc = scores.loc['logreg', 'acc']
score_loss = scores.loc['logreg', 'loss']
print(score_auc, score_acc, score_loss)

save_submit(test, clfs, 'submission.csv')
