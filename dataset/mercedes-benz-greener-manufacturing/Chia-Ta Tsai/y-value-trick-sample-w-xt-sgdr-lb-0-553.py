#y value trick from but using median
#https://www.kaggle.com/robertoruiz/a-magic-feature/code
#https://www.kaggle.com/c/mercedes-benz-greener-manufacturing/discussion/34180
#referancing random_projection and decomposition from
#https://www.kaggle.com/hakeem/stacked-then-averaged-models-0-5697

import time
from random import choice, sample, shuffle, uniform, seed
from math import exp, expm1, log1p, log10, log2, sqrt, ceil, floor
#
import numpy as np
import pandas as pd
#
from sklearn.decomposition import TruncatedSVD, PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection 
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


target_id, target = 'ID', 'y'
###############################################################################

def set_decompostions(tfs={}, n_comp=12, random_seed=622):
    # tSVD
    tfs['svd'] = TruncatedSVD(n_components=n_comp, random_state=random_seed)
    # PCA
    tfs['pca'] = PCA(n_components=n_comp, random_state=random_seed)
    # ICA
    tfs['ica'] = FastICA(n_components=n_comp, max_iter=1000, random_state=random_seed)
    # GRP
    tfs['grp'] = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=random_seed)
    # SRP
    tfs['srp'] = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=random_seed)
    
    return tfs

###############################################################################
def collect_predict(preds, collect, d=1., col='y_', opt_test=False):
    if opt_test:
        if col not in collect.columns:
            collect[col] = 0.
        collect[col] += preds / d
    else:        
        collect[col] = preds

###############################################################################
if __name__ == '__main__':

    options = {}
    nr_splits = 10
    fold_gen_seed = 622
    tmstmp = '{}'.format(time.strftime("%Y-%m-%d-%H-%M"))

    #load data
    input_folder = '../input/'
    #train
    df_train = pd.read_csv(input_folder + 'train.csv')
    print("original data: X_train: {}".format(df_train.shape), flush=True)
    df_train['Xid'] = df_train[target_id].apply(log1p)
    #test
    df_test = pd.read_csv(input_folder + 'test.csv')
    print("original data: X_test: {}".format(df_test.shape), flush=True)
    df_test['Xid'] = df_test[target_id].apply(log1p)

    #factorized
    f_magic = ['X0', 'X2']
    feats = list(set(df_train.columns.tolist()).difference([target, target_id]))
    print('read in {} features'.format(len(feats)), flush=True)
    for c in feats:
        if df_train[c].dtype == 'object':
            candidates = list(df_train[c].values) + list(df_test[c].values)
            lbl = LabelEncoder()
            lbl.fit(candidates)
            df_train[c] = lbl.transform(list(df_train[c].values))
            df_test[c] = lbl.transform(list(df_test[c].values))
            val_uniq = len(set(candidates))
            print('{} is object: {} uniques'.format(c, val_uniq), flush=True)
    
    #data
    train_X = df_train[feats]
    test_X = df_test[feats]
    
    #id
    train_id = df_train[target_id]
    test_id = df_test[target_id]
    
    #capping
    train_y = df_train[target]
    raw_train_y = df_train[target]

    #fold assignments
    train_sets, valid_sets = list(), list()
    fold_gen = KFold(n_splits=nr_splits, shuffle=True, random_state=fold_gen_seed)
    for train_indices, valid_indices in fold_gen.split(train_y, train_y):
        train_sets.append(train_indices)
        valid_sets.append(valid_indices)

    #regressor
    seed_val =170623
    #glm
    a = 0.0005
    l1r = 0.25
    eps = 0.03
    glm = SGDRegressor(loss='huber', penalty='elasticnet', alpha=a, l1_ratio=l1r, fit_intercept=True, max_iter=200, shuffle=True, verbose=0, epsilon=eps, random_state=seed_val, learning_rate='invscaling', eta0=0.01, power_t=0.25, warm_start=False, average=False)
    #extratree
    xt = ExtraTreesRegressor(n_estimators=1200, criterion='mse', max_depth=None, min_samples_split=4, min_samples_leaf=4, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=128, bootstrap=True, oob_score=True, n_jobs=-1, random_state=seed_val, verbose=0, warm_start=False)

    #decompositions
    nb_comp = 12
    tfs = set_decompostions(n_comp=nb_comp, random_seed=623)
        
    #feats
    f_cat = list(set(df_train.columns.tolist()).difference([target, target_id]))

    #data
    train_X = df_train[f_cat]
    test_X = df_test[f_cat]

    #preds
    train_preds = pd.DataFrame()
    test_preds = pd.DataFrame()
    test_preds[target_id] = test_id

    #start cv
    print('\n{}-fold cv'.format(nr_splits))
    for nr_fold in range(nr_splits):
        print('eval fold {:02d}'.format(nr_fold), flush=True)
        
        #data
        X_train = train_X.iloc[train_sets[nr_fold]].reset_index(drop=True)
        X_valid = train_X.iloc[valid_sets[nr_fold]].reset_index(drop=True)
        X_test = test_X.copy()
        #y
        y_train = train_y.iloc[train_sets[nr_fold]].reset_index(drop=True)
        y_valid = train_y.iloc[valid_sets[nr_fold]].reset_index(drop=True)
        raw_y_valid = raw_train_y.iloc[valid_sets[nr_fold]].reset_index(drop=True)

        #pred
        sub_train = pd.DataFrame()
        sub_train[target_id] = train_id.iloc[valid_sets[nr_fold]].tolist()
            
        #using transformer
        for k, v in tfs.items():
            trans_train = v.fit_transform(X_train)
            trans_valid = v.transform(X_valid)
            trans_test = v.transform(X_test)
            for nb in range(nb_comp):
                new_col = '{}_{:02d}'.format(k, nb+1)
                X_train[new_col] = trans_train[:, nb]
                X_valid[new_col] = trans_valid[:, nb]
                X_test[new_col] = trans_test[:, nb]

        #magic
        raw_y_train = raw_train_y.iloc[train_sets[nr_fold]].reset_index(drop=True)
        for f in f_magic:
            magic_df = pd.DataFrame()
            magic_df[target] = raw_y_train
            magic_df[f] = df_train[f].iloc[train_sets[nr_fold]].reset_index(drop=True)
            rplc = np.median(raw_y_train)
            magic_df = magic_df.groupby(f)[target].median()
            magic_dict = magic_df.to_dict()
                
            f_m = 'magic_{}'.format(f)
            X_train[f_m] = X_train[f].apply(lambda x: magic_dict.get(x, rplc))
            X_valid[f_m] = X_valid[f].apply(lambda x: magic_dict.get(x, rplc))
            X_test[f_m] = X_test[f].apply(lambda x: magic_dict.get(x, rplc))

        #clean NA
        X_train = X_train.apply(np.nan_to_num)
        X_valid = X_valid.apply(np.nan_to_num)
        X_test = X_test.apply(np.nan_to_num)
                           
        #        
        stem = 'xt'
        target_this = 'y_{}'.format(stem)
        reg = xt
        reg.fit(X_train, y_train)
        
        preds = reg.predict(X_valid)
        print('XT r2={:.6f}'.format(r2_score(y_valid, preds)), flush=True)
        collect_predict(preds, sub_train, d=nr_splits, col=target_this, opt_test=False)
        collect_predict(reg.predict(X_test), test_preds, d=nr_splits, col=target_this, opt_test=True)
        
        #
        stem = 'glm'
        target_this = 'y_{}'.format(stem)
        reg = glm
        reg.fit(X_train, y_train)

        preds = reg.predict(X_valid)
        print('SGDR r2={:.6f}'.format(r2_score(raw_y_valid, preds)))        
        collect_predict(preds, sub_train, d=nr_splits, col=target_this, opt_test=False)
        collect_predict(reg.predict(X_test), test_preds, d=nr_splits, col=target_this, opt_test=True)

        #end of one fold eval in cv
        train_preds = train_preds.append(sub_train)
        del X_train, X_valid, X_test
        print(end='\n')
    
    #merge y into dataset
    train_preds = train_preds.reset_index(drop=True)
    df_train = df_train.merge(train_preds, how='left', on=target_id)
    df_test = df_test.merge(test_preds, how='left', on=target_id)        

    #performance check
    print('summary')
    for t in train_preds.columns.tolist():
        if t.startswith('y_'):
            #score = r2_score(train_y, df_train[t])
            score = r2_score(raw_train_y, df_train[t])
            print('{} r2 = {:.6f}'.format(t[2:], score), flush=True)
                
            sub = pd.DataFrame()
            sub[target_id] = df_test[target_id]
            sub[target] = df_test[t]
            sub.to_csv("{}_{}_s{:.6f}.csv".format(tmstmp, t[2:], score), index=False)
     
    #find best weights
    print('optimizing weights')
    trials, best = 1000, 0
    best_w = []
    for i in range(trials):
        weights = []
        df_train['tmp'] = 0
        
        for t in train_preds.columns.tolist():
            if t.startswith('y_'):
                weights.append(uniform(0.25, 0.75))
                df_train['tmp'] += df_train[t] * weights[-1]

        s = sum(weights)
        score = r2_score(raw_train_y, df_train['tmp'].apply(lambda x: x/s))
        if score > best:
            best = score
            rmse = sqrt(mean_squared_error(raw_train_y, df_train['tmp']))
            best_w = [w / s for w in weights]
            print('no {:04d}: r2 = {:.5f}, rmse = {:.3f}, current best ({})'.format(i, best, rmse, best_w), flush=True)
            
    #save weighted 
    sub = pd.DataFrame()
    sub[target_id] = df_test[target_id]    
    sub[target] = 0
    i = 0
    for t in train_preds.columns.tolist():
        if t.startswith('y_'):
            sub[target] += df_test[t] * best_w[i]
            print('w {}: {:.6f}'.format(t[2:], best_w[i]), flush=True)
            i += 1
    sub.to_csv("{}_wsum_s{:.6f}.csv".format(tmstmp, best), index=False)
