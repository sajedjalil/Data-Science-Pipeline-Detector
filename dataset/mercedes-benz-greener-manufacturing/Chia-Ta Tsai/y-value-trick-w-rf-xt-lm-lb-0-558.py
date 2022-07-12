#y value trick from but using median
#https://www.kaggle.com/robertoruiz/a-magic-feature/code
#https://www.kaggle.com/c/mercedes-benz-greener-manufacturing/discussion/34180
#referancing random_projection and decomposition from
#https://www.kaggle.com/hakeem/stacked-then-averaged-models-0-5697
#kernel5686XGB
#https://www.kaggle.com/linux18/kernel-0-5686/code

import time
from random import choice, sample, shuffle, uniform, seed
from math import exp, expm1, log1p, log10, log2, sqrt, ceil, floor
#
import numpy as np
import pandas as pd
#
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD, PCA, FastICA, NMF
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection 
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDRegressor, ElasticNet
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, RandomTreesEmbedding
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

target_id, target = 'ID', 'y'
###############################################################################

def set_decompostions(tfs={}, n_comp=12, random_seed=622):
    # tSVD
    tfs['svd'] = TruncatedSVD(n_components=n_comp, random_state=random_seed)
    # PCA
    tfs['pca'] = PCA(n_components=n_comp, random_state=random_seed)
    # ICA
    tfs['ica'] = FastICA(n_components=n_comp, max_iter=100, random_state=random_seed)
    # GRP
    tfs['grp'] = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=random_seed)
    # SRP
    tfs['srp'] = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=random_seed)
    #NMF
    tfs['nmf'] = NMF(n_components=nb_comp, shuffle=True, init='random', random_state=random_seed)
    
    return tfs

#stacking
###############################################################################
def collect_predict(preds, collect, d=1., col='y_', opt_test=False):
    if opt_test:
        if col not in collect.columns:
            collect[col] = 0.
        collect[col] += preds / d
    else:        
        collect[col] = preds


def optimize_weights(trials=10000, y=None, df=None, fs=[]):
    best_w = {}
    best_r2, best_rmse = -1.0 * float('inf'), float('inf')
    
    fs = [f for f in fs if f in df.columns.tolist()]
    
    for i in range(trials):    
        weights = {}
        df['tmp'] = 0
        
        for f in fs:
            weights[f] = uniform(0.15, 0.75)        
            df['tmp'] += df[f] * weights.get(f, 0)
            
        s = sum(list(weights.values()))
        df['tmp'] = df['tmp'].apply(lambda x: x/s)
        score = r2_score(y, df['tmp'])
        rmse = sqrt(mean_squared_error(y, df['tmp']))
            
        if score > best_r2:
            best_r2, best_rmse = score, rmse
            best_w = {k: v/s for k, v in weights.items()}
            print('no {:04d}: r2 = {:.5f}, rmse = {:.3f}, current best ({})'.format(i, score, rmse, best_w), flush=True)

    return best_w, best_r2, best_rmse


#csv
###############################################################################
def write_csv(y_id, y_preds, stem=''):
    sub = pd.DataFrame()
    sub[target_id] = y_id
    sub[target] = y_preds
    sub.to_csv(stem, index=False)

###############################################################################

if __name__ == '__main__':

    options = {}
    nr_splits = 5
    fold_gen_seed = 630
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

    seed_val =17030

    #regressor    
    reg_scikit = {}
    #glm
    #a, l1r, eps = 0.0075, 0.25, 0.03
    #reg_scikit['lm_l1'] = SGDRegressor(loss='huber', penalty='elasticnet', alpha=a, l1_ratio=l1r, fit_intercept=True, n_iter=100, shuffle=True, verbose=0, epsilon=eps, random_state=seed_val, learning_rate='invscaling', eta0=0.01, power_t=0.25, warm_start=False, average=False)
    a, l1r = 2.0, 0.50
    reg_scikit['lm_l1'] = ElasticNet(alpha=a, l1_ratio=l1r, fit_intercept=True, normalize=False, precompute=False, max_iter=250, copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=seed_val, selection='cyclic')
    #a, l1r, eps = 0.0005, 0.01, 0.03
    #reg_scikit['lm_l2'] = SGDRegressor(loss='huber', penalty='elasticnet', alpha=a, l1_ratio=l1r, fit_intercept=True, n_iter=100, shuffle=True, verbose=0, epsilon=eps, random_state=seed_val, learning_rate='invscaling', eta0=0.01, power_t=0.25, warm_start=False, average=False)
    a, l1r = 0.05, 0.001
    reg_scikit['lm_l2'] = ElasticNet(alpha=a, l1_ratio=l1r, fit_intercept=True, normalize=False, precompute=False, max_iter=250, copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=seed_val, selection='cyclic')
    #extratree
    trees, leafs, depth = 560, 128, 11
    reg_scikit['xt_shallow'] = ExtraTreesRegressor(n_estimators=trees, criterion='mse', max_depth=depth, min_samples_split=4, min_samples_leaf=4, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=leafs, bootstrap=True, oob_score=True, n_jobs=-1, random_state=seed_val, verbose=0, warm_start=False)
    trees, leafs, depth = 400, 256, 14
    reg_scikit['xt_deep'] = ExtraTreesRegressor(n_estimators=trees, criterion='mse', max_depth=depth, min_samples_split=4, min_samples_leaf=4, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=leafs, bootstrap=True, oob_score=True, n_jobs=-1, random_state=seed_val, verbose=0, warm_start=False)
    #randomforest        
    trees, leafs, depth = 560, 128, 11
    reg_scikit['rf_shallow'] = RandomForestRegressor(n_estimators=trees, criterion='mse', max_depth=depth, min_samples_split=4, min_samples_leaf=4, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=leafs, bootstrap=True, oob_score=True, n_jobs=-1, random_state=seed_val, verbose=0, warm_start=False)
    trees, leafs, depth = 400, 256, 14
    reg_scikit['rf_deep'] = RandomForestRegressor(n_estimators=trees, criterion='mse', max_depth=depth, min_samples_split=4, min_samples_leaf=4, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=leafs, bootstrap=True, oob_score=True, n_jobs=-1, random_state=seed_val, verbose=0, warm_start=False)

    #
    for k, reg in reg_scikit.items():
        print('{}={}'.format(k, reg.get_params()), flush=True)

    #four known clusters
    n_clust = 4
    clust = MiniBatchKMeans(n_clusters=n_clust, max_iter=1000, init_size=n_clust*10)
    #decompositions
    nb_comp = 12
    tfs = set_decompostions(n_comp=nb_comp, random_seed=seed_val)
    #embedding
    trees, depth = 25, 8 #2 ** 8 = 256
    embed = RandomTreesEmbedding(n_estimators=trees, max_depth=depth, n_jobs=-1, random_state=seed_val)
        
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
        #
        raw_y_train = raw_train_y.iloc[train_sets[nr_fold]].reset_index(drop=True)
        raw_y_valid = raw_train_y.iloc[valid_sets[nr_fold]].reset_index(drop=True)

        #pred
        sub_train = pd.DataFrame()
        sub_train[target_id] = train_id.iloc[valid_sets[nr_fold]].tolist()

        #feats
        f_y_enc = f_magic[:]
        
        #using transformer
        for k, v in tfs.items():
            trans_train = v.fit_transform(X_train[f_cat])
            trans_valid = v.transform(X_valid[f_cat])
            trans_test = v.transform(X_test[f_cat])
            for nb in range(nb_comp):
                new_col = '{}_{:03d}'.format(k, nb+1)
                X_train[new_col] = trans_train[:, nb]
                X_valid[new_col] = trans_valid[:, nb]
                X_test[new_col] = trans_test[:, nb]

        #known cluster
        f = 'f_clu_{:03d}'.format(n_clust)
        f_y_enc.append(f)
        X_train[f] = clust.fit_predict(X_train)
        X_valid[f] = clust.predict(X_valid)
        X_test[f] = clust.predict(X_test) 

        #embed
        embed.fit(X_train)
        trans_train = embed.apply(X_train)
        trans_valid = embed.apply(X_valid)
        trans_test = embed.apply(X_test)
        
        for tree in range(trans_train.shape[1]):
            f = 'f_embed_{:04d}'.format(tree)
            f_y_enc.append(f)
            leaf_lbl = LabelEncoder()
            leaf_train = trans_train[:, tree].tolist()
            leaf_valid = trans_valid[:, tree].tolist()
            leaf_test = trans_test[:, tree].tolist()
                    
            leaf_lbl.fit(leaf_train + leaf_valid + leaf_test)
            X_train[f] = leaf_lbl.transform(leaf_train)
            X_valid[f] = leaf_lbl.transform(leaf_valid)
            X_test[f] = leaf_lbl.transform(leaf_test)

        #encode y from factorization
        for f in f_y_enc:
            magic_df = pd.DataFrame()
            magic_df[target] = raw_y_train
            magic_df[f] = X_train[f]
            rplc = np.median(raw_y_train)
            magic_df = magic_df.groupby(f)[target].median()
            magic_dict = magic_df.to_dict()
                
            f_m = 'f_y_enc_{}'.format(f)
            X_train[f_m] = X_train[f].apply(lambda x: magic_dict.get(x, rplc))
            X_valid[f_m] = X_valid[f].apply(lambda x: magic_dict.get(x, rplc))
            X_test[f_m] = X_test[f].apply(lambda x: magic_dict.get(x, rplc))

            #assume larger id get larger y
            b = 64.0
            f_m_x_id = 'f_y_enc_{}_x_id'.format(f)
            X_train[f_m_x_id] = X_train[f_m] * X_train['Xid'].apply(lambda x: exp(x/b))
            X_valid[f_m_x_id] = X_valid[f_m] * X_valid['Xid'].apply(lambda x: exp(x/b))
            X_test[f_m_x_id] = X_test[f_m] * X_test['Xid'].apply(lambda x: exp(x/b))

        #clean NA
        X_train = X_train.apply(np.nan_to_num)
        X_valid = X_valid.apply(np.nan_to_num)
        X_test = X_test.apply(np.nan_to_num)
                           
        #learning
        for k, reg in reg_scikit.items():
            stem = k
            target_this = 'y_{}'.format(stem)
            reg.fit(X_train, y_train)
    
            preds = reg.predict(X_valid)
            score = r2_score(y_valid, preds)
            rmse = sqrt(mean_squared_error(y_valid, preds))
            print('{}: r2={:.6f}, rmse={:.3f}'.format(k, score, rmse), flush=True)      
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
    collect_r2, collect_rmse = {}, {}
    print('summary', flush = True)
    f_preds = [f for f in df_train.columns.tolist() if f.startswith('y_')]
    for t in f_preds:
        collect_r2[t] = r2_score(raw_train_y, df_train[t])
        collect_rmse[t] = sqrt(mean_squared_error(raw_train_y, df_train[t]))
        score, rmse = collect_r2.get(t, 0), collect_rmse.get(t, 0)
        
        print('{}: r2={:.6f}, rmse={:.3f}'.format(t[2:], score, rmse), flush=True)
        file = '{}_{}_s{:.5f}_e{:.3f}.csv'.format(tmstmp, t[2:], score, rmse)
        write_csv(df_test[target_id].tolist(), df_test[t].tolist(), stem=file)

    #optimizing weights
    print('optimizing weights', flush=True)
    best_w, best_r2, best_rmse = optimize_weights(trials=10000, y=raw_train_y, df=df_train[f_preds + [target, target_id]].copy(), fs=f_preds)

    #save weighted 
    f_preds = [f for f in f_preds if f in df_test.columns.tolist()]
    sub = pd.DataFrame()
    sub[target_id] = df_test[target_id]    
    sub[target] = 0
    for i, t in enumerate(f_preds):
        w =  best_w.get(t, 0)
        sub[target] += df_test[t] * w
        score, rmse = collect_r2.get(t, 0), collect_rmse.get(t, 0)
        print('w {}: {:.3f}, r2 = {:.5f}, rmse = {:.3f}'.format(t[2:], w, score, rmse), flush=True)
    sub.to_csv('{}_wsum_s{:.5f}_e{:.3f}.csv'.format(tmstmp, best_r2, best_rmse), index=False)
