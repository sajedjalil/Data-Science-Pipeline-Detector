import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
np.random.seed(12324)
from sklearn.cross_validation import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from ml_metrics import auc
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

INPUT_PATH = '../input/'
OUTPUT_PATH = ''


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.cross_validation import StratifiedKFold



def process_base(train, test):
    train.loc[(train['var38']>117310.979) & (train['var38']<117310.98), 'var38'] = -999.0
    test.loc[(test['var38']>117310.979) & (test['var38']<117310.98), 'var38'] = -999.0

    train.loc[train['var3']==-999999, 'var3'] = -999.0
    test.loc[test['var3']==-999999, 'var3'] = -999.0

    for f in ['imp_op_var40_comer_ult1', 'imp_op_var40_efect_ult3', 'imp_op_var41_comer_ult3', 'imp_sal_var16_ult1']:
        train.loc[train[f]==0.0, f] = -999.0
        test.loc[test[f]==0.0, f] = -999.0

    return train, test

def drop_sparse(train, test):
    flist = [x for x in train.columns if not x in ['ID','TARGET']]
    for f in flist:
        if len(np.unique(train[f]))<2:
            train.drop(f, axis=1, inplace=True)
            test.drop(f, axis=1, inplace=True)
    return train, test

def drop_duplicated(train, test):
    # drop var6 variable (it is similar to var29)
    flist = [x for x in train.columns if not x in ['ID','TARGET']]            
    train.drop([x for x in flist if 'var6' in x], axis=1, inplace=True)
    test.drop([x for x in flist if 'var6' in x], axis=1, inplace=True)

    # remove repeated columns with _0 in the name
    flist = [x for x in train.columns if not x in ['ID','TARGET']]        
    flist_remove = []
    for i in range(len(flist)-1):
        v = train[flist[i]].values
        for j in range(i+1, len(flist)):
            if np.array_equal(v, train[flist[j]].values):
                if '_0' in flist[j]:
                    flist_remove.append(flist[j])
                elif  '_0' in flist[i]:
                    flist_remove.append(flist[i])
    train.drop(flist_remove, axis=1, inplace=True)
    test.drop(flist_remove, axis=1, inplace=True)

    flist_remove = ['saldo_medio_var13_medio_ult1', 'delta_imp_reemb_var13_1y3', 'delta_imp_reemb_var17_1y3', 
                   'delta_imp_reemb_var33_1y3', 'delta_imp_trasp_var17_in_1y3', 'delta_imp_trasp_var17_out_1y3',
                   'delta_imp_trasp_var33_in_1y3', 'delta_imp_trasp_var33_out_1y3']
    train.drop(flist_remove, axis=1, inplace=True)
    test.drop(flist_remove, axis=1, inplace=True)

    return train, test

def add_features(train, test, features):
    flist = [x for x in train.columns if not x in ['ID','TARGET']]
    if 'SumZeros' in features:
        train.insert(1, 'SumZeros', (train[flist] == 0).astype(int).sum(axis=1))
        test.insert(1, 'SumZeros', (test[flist] == 0).astype(int).sum(axis=1))
    flist = [x for x in train.columns if not x in ['ID','TARGET']]

    if 'tsne' in features:
        tsne_feats = pd.read_csv(OUTPUT_PATH + 'features/tsne_feats.csv')
        train = pd.merge(train, tsne_feats, on='ID', how='left')
        test = pd.merge(test, tsne_feats, on='ID', how='left')

    if 'pca' in features:
        pca_feats = pd.read_csv(OUTPUT_PATH + 'features/dmitry_pca_feats.csv')
        train = pd.merge(train, pca_feats, on='ID', how='left')
        test = pd.merge(test, pca_feats, on='ID', how='left')

    if 'kmeans' in features:
        kmeans_feats = pd.read_csv(OUTPUT_PATH + 'features/kmeans_feats.csv')
        train = pd.merge(train, kmeans_feats, on='ID', how='left')
        test = pd.merge(test, kmeans_feats, on='ID', how='left')

    return train, test

def normalize_features(train, test):
    flist = [x for x in train.columns if not x in ['ID','TARGET']]
    for f in flist:
        if train[f].max() == 9999999999.0:
            fmax = train.loc[train[f]<9999999999.0, f].max()
            train.loc[train[f]==9999999999.0, f] = fmax + 1

        if len(train.loc[train[f]<0, f].value_counts()) == 1:
            train.loc[train[f]<0, f] = -1.0
            test.loc[test[f]<0, f] = -1.0
            fmax = max(np.max(train[f]), np.max(test[f]))
            if fmax > 0:
                train.loc[train[f]>0, f] = 1.0*train.loc[train[f]>0, f]/fmax
                test.loc[test[f]>0, f] = 1.0*test.loc[test[f]>0, f]/fmax

        if len(train.loc[train[f]<0, f]) == 0:
            fmax = max(np.max(train[f]), np.max(test[f]))
            if fmax > 0:
                train.loc[train[f]>0, f] = 1.0*train.loc[train[f]>0, f]/fmax
                test.loc[test[f]>0, f] = 1.0*test.loc[test[f]>0, f]/fmax

        if len(train.loc[train[f]<0, f].value_counts()) > 1:
            fmax = max(np.max(train[f]), np.max(test[f]))
            if fmax > 0:
                train[f] = 1.0*train[f]/fmax
                test[f] = 1.0*test[f]/fmax

    return train, test

def add_likelihood_feature(fname, train_likeli, test_likeli, flist):
    tt_likeli = pd.DataFrame()
    np.random.seed(1232345)
    skf = StratifiedKFold(train_likeli['TARGET'].values, n_folds=5, shuffle=True, random_state=21387)
    for train_index, test_index in skf:
        ids = train_likeli['ID'].values[train_index]
        train_fold = train_likeli.loc[train_likeli['ID'].isin(ids)].copy()
        test_fold = train_likeli.loc[~train_likeli['ID'].isin(ids)].copy()
        global_avg = np.mean(train_fold['TARGET'].values)
        feats_likeli = train_fold.groupby(fname)['TARGET'].agg({'sum': np.sum, 'count': len}).reset_index()
        feats_likeli[fname + '_likeli'] = (feats_likeli['sum'] + 30.0*global_avg)/(feats_likeli['count']+30.0)
        test_fold = pd.merge(test_fold, feats_likeli[[fname, fname + '_likeli']], on=fname, how='left')
        test_fold[fname + '_likeli'] = test_fold[fname + '_likeli'].fillna(global_avg)
        tt_likeli = tt_likeli.append(test_fold[['ID', fname + '_likeli']], ignore_index=True)
    train_likeli = pd.merge(train_likeli, tt_likeli, on='ID', how='left')
    
    global_avg = np.mean(train_likeli['TARGET'].values)
    feats_likeli = train_likeli.groupby(fname)['TARGET'].agg({'sum': np.sum, 'count': len}).reset_index()
    feats_likeli[fname + '_likeli'] = (feats_likeli['sum'] + 30.0*global_avg)/(feats_likeli['count']+30.0)
    test_likeli = pd.merge(test_likeli, feats_likeli[[fname, fname + '_likeli']], on=fname, how='left')
    test_likeli[fname + '_likeli'] = test_likeli[fname + '_likeli'].fillna(global_avg)
    return train_likeli, test_likeli, flist + [fname + '_likeli']


train = pd.read_csv(INPUT_PATH + 'train.csv')
test = pd.read_csv(INPUT_PATH + 'test.csv')

train, test = process_base(train, test)
train, test = drop_sparse(train, test)
train, test = drop_duplicated(train, test)
train, test = add_features(train, test, ['SumZeros'])

flist = [x for x in train.columns if not x in ['ID','TARGET']]

### add TSNE features
X = train[flist].append(test[flist], ignore_index=True).values.astype('float64')
svd = TruncatedSVD(n_components=30)
X_svd = svd.fit_transform(X)
X_scaled = StandardScaler().fit_transform(X_svd)
model = TSNE(n_components=2, random_state=0)
feats_tsne = model.fit_transform(X_scaled)
feats_tsne = pd.DataFrame(feats_tsne, columns=['tsne1', 'tsne2'])
feats_tsne['ID'] = train[['ID']].append(test[['ID']], ignore_index=True)['ID'].values
train = pd.merge(train, feats_tsne, on='ID', how='left')
test = pd.merge(test, feats_tsne, on='ID', how='left')

feat = train[['ID', 'tsne1', 'tsne2']].append(test[['ID', 'tsne1', 'tsne2']], ignore_index=True)
feat.to_csv(OUTPUT_PATH + 'tsne_feats.csv', index=False)











