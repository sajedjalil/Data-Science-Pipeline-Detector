#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

import time
start = time.time()
import gc
import numpy as np
import pandas as pd
import xgboost as xgb

from datetime import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import KFold
from scipy.stats import skew, boxcox
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import itertools

shift = 200
COMB_FEATURE = 'cat80,cat87,cat57,cat12,cat79,cat10,cat7,cat89,cat2,cat72,' \
               'cat81,cat11,cat1,cat13,cat9,cat3,cat16,cat90,cat23,cat36,' \
               'cat73,cat103,cat40,cat28,cat111,cat6,cat76,cat50,cat5,' \
               'cat4,cat14,cat38,cat24,cat82,cat25'.split(',')

def encode(charcode):
    r = 0
    ln = len(str(charcode))
    for i in range(ln):
        r += (ord(str(charcode)[i]) - ord('A') + 1) * 26 ** (ln - i - 1)
    return r

fair_constant = 0.7
def fair_obj(preds, dtrain):
    labels = dtrain.get_label()
    x = (preds - labels)
    den = abs(x) + fair_constant
    grad = fair_constant * x / (den)
    hess = fair_constant * fair_constant / (den * den)
    return grad, hess

def xg_eval_mae(yhat, dtrain):
    y = dtrain.get_label()
    return 'xg_eval_mae', mean_absolute_error(np.exp(y)-shift,
                                      np.exp(yhat)-shift)
def mungeskewed(train, test, numeric_feats):
    ntrain = train.shape[0]
    test['loss'] = 0
    train_test = pd.concat((train, test)).reset_index(drop=True)
    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))
    skewed_feats = skewed_feats[skewed_feats > 0.25]
    skewed_feats = skewed_feats.index

    for feats in skewed_feats:
        train_test[feats] = train_test[feats] + 1
        train_test[feats], lam = boxcox(train_test[feats])
    return train_test, ntrain

if __name__ == "__main__":

    print('\nStarted')
    directory = '../input/'
    train = pd.read_csv(directory + 'train.csv')
    test = pd.read_csv(directory + 'test.csv')
    # add for oof
    id_train = train['id'].values
    # end add for oof

    numeric_feats = [x for x in train.columns[1:-1] if 'cont' in x]
    categorical_feats = [x for x in train.columns[1:-1] if 'cat' in x]
    train_test, ntrain = mungeskewed(train, test, numeric_feats)
    
    # taken from Vladimir's script (https://www.kaggle.com/iglovikov/allstate-claims-severity/xgb-1114)
    # for column in list(train.select_dtypes(include=['object']).columns):
    for column in train.dtypes[train.dtypes == 'object'].index.tolist(): # for old pandas
        if train[column].nunique() != test[column].nunique():
            set_train = set(train[column].unique())
            set_test = set(test[column].unique())
            remove_train = set_train - set_test
            remove_test = set_test - set_train

            remove = remove_train.union(remove_test)


            def filter_cat(x):
                if x in remove:
                    return np.nan
                return x


            train_test[column] = train_test[column].apply(lambda x: filter_cat(x), 1)
            
        # [ANCHOR_1] Just for testing. If uncomment this line then we can skip feature combs - code from [ANCHOR_2] to [ANCHOR_3]
        train_test[column] = pd.factorize(train_test[column].values, sort=True)[0]

    # taken from Ali's script (https://www.kaggle.com/aliajouz/allstate-claims-severity/singel-model-lb-1117)
    train_test["cont1"] = np.sqrt(preprocessing.minmax_scale(train_test["cont1"]))
    train_test["cont4"] = np.sqrt(preprocessing.minmax_scale(train_test["cont4"]))
    train_test["cont5"] = np.sqrt(preprocessing.minmax_scale(train_test["cont5"]))
    train_test["cont8"] = np.sqrt(preprocessing.minmax_scale(train_test["cont8"]))
    train_test["cont10"] = np.sqrt(preprocessing.minmax_scale(train_test["cont10"]))
    train_test["cont11"] = np.sqrt(preprocessing.minmax_scale(train_test["cont11"]))
    train_test["cont12"] = np.sqrt(preprocessing.minmax_scale(train_test["cont12"]))

    train_test["cont6"] = np.log(preprocessing.minmax_scale(train_test["cont6"]) + 0000.1)
    train_test["cont7"] = np.log(preprocessing.minmax_scale(train_test["cont7"]) + 0000.1)
    train_test["cont9"] = np.log(preprocessing.minmax_scale(train_test["cont9"]) + 0000.1)
    train_test["cont13"] = np.log(preprocessing.minmax_scale(train_test["cont13"]) + 0000.1)
    train_test["cont14"] = (np.maximum(train_test["cont14"] - 0.179722, 0) / 0.665122) ** 0.25

    # Just for testing. Feture combs can be skipped if we uncomment line [ANCHOR_1]
    # [ANCHOR_2] 
    """
    print('')
    for comb in itertools.combinations(COMB_FEATURE, 2):
        feat = comb[0] + "_" + comb[1]
        train_test[feat] = train_test[comb[0]] + train_test[comb[1]]
        train_test[feat] = train_test[feat].apply(encode)
        print('Combining Columns:', feat)

    print('')
    for col in categorical_feats:
        print('Analyzing Column:', col)
        train_test[col] = train_test[col].apply(encode)
    """
    # [ANCHOR_3]

    # I think we don't want to see this
    # print(train_test[categorical_feats])

    ss = StandardScaler()
    train_test[numeric_feats] = \
        ss.fit_transform(train_test[numeric_feats].values)
        
    # Print shape
    print('train_test shape:', train_test.shape)
    
    # Cast to np.float32 to make size a half less
    # print('Initial size of train_test: [%.2f] MB' % (train_test.memory_usage(index = True).sum() / 1000000.0))
    train_test = train_test.astype(np.float32)
    # print('Initial size of train_test: [%.2f] MB' % (train_test.memory_usage(index = True).sum() / 1000000.0))

    # We don't need copies (train = train_test.iloc[:ntrain, :].copy())
    train = train_test.iloc[:ntrain, :]
    test = train_test.iloc[ntrain:, :]
    
    # train_test isn't used any more. Del it and clean up
    del train_test
    gc.collect()

    print('\nMedian Loss:', train.loss.median())
    print('Mean Loss:', train.loss.mean())

    ids = pd.read_csv(directory + 'test.csv')['id']
    train_y = np.log(train['loss'] + shift)
    train_x = train.drop(['loss','id'], axis=1)
    test_x = test.drop(['loss','id'], axis=1)
    # calculate this here to be able to delete train
    train_shape = train.shape[0]
    
    # train and test aren't used any more. Del it and clean up
    del train
    del test
    gc.collect()

    n_folds = 10
    cv_sum = 0
    # early_stopping = 100 # !!! this isn't used in original script
    fpred = []
    # add for oof
    pred_oob = np.zeros(train_x.shape[0])
    # end add for oof
    xgb_rounds = []

    # We don't need d_train_full
    # d_train_full = xgb.DMatrix(train_x, label=train_y)
    d_test = xgb.DMatrix(test_x)

    kf = KFold(train_shape, n_folds=n_folds)
    for i, (train_index, test_index) in enumerate(kf):
        print('\n Fold %d' % (i+1))
        X_train, X_val = train_x.iloc[train_index], train_x.iloc[test_index]
        y_train, y_val = train_y.iloc[train_index], train_y.iloc[test_index]

        rand_state = 2016

        params = {
            'seed': 0,
            'colsample_bytree': 0.7,
            'silent': 1,
            'subsample': 0.7,
            'learning_rate': 0.03,
            'objective': 'reg:linear',
            'max_depth': 12,
            'min_child_weight': 100,
            'booster': 'gbtree'}

        d_train = xgb.DMatrix(X_train, label=y_train)
        d_valid = xgb.DMatrix(X_val, label=y_val)
        # To get some speed up let's calculate metric only for val set
        watchlist = [(d_valid, 'eval')]

        clf = xgb.train(params,
                        d_train,
                        10, # 100000
                        watchlist,
                        early_stopping_rounds=50,
                        obj=fair_obj,
                        feval=xg_eval_mae,
                        verbose_eval = 1)

        xgb_rounds.append(clf.best_iteration)
        scores_val = clf.predict(d_valid, ntree_limit=clf.best_ntree_limit)
        # add for oof
        pred_oob[test_index] = np.exp(clf.predict(d_valid, ntree_limit=clf.best_ntree_limit)) - shift
        # end add for oof
        cv_score = mean_absolute_error(np.exp(y_val), np.exp(scores_val))
        print('eval-MAE: %.6f' % cv_score)
        y_pred = np.exp(clf.predict(d_test, ntree_limit=clf.best_ntree_limit)) - shift

        if i > 0:
            fpred = pred + y_pred
        else:
            fpred = y_pred
        pred = fpred
        cv_sum = cv_sum + cv_score

    mpred = pred / n_folds
    score = cv_sum / n_folds
    print('Average eval-MAE: %.6f' % score)
    n_rounds = int(np.mean(xgb_rounds))

    print("Writing results")
    result = pd.DataFrame(mpred, columns=['loss'])
    result["id"] = ids
    result = result.set_index("id")
    print("%d-fold average prediction:" % n_folds)

    now = datetime.now()
    score = str(round((cv_sum / n_folds), 6))
    sub_file = 'submission_5fold-average-xgb_fairobj_' + str(score) + '_' + str(
        now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    print("Writing submission: %s" % sub_file)
    result.to_csv(sub_file, index=True, index_label='id')
    
    # add for oof
    df = pd.DataFrame({'id': id_train, 'loss': pred_oob})
    df.to_csv('preds_xgb_oob.csv', index = False)
    # end add for oof

    with open('time.txt', 'w') as f:
        f.write(str(time.time() - start))
    
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------



