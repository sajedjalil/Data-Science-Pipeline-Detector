import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD, FastICA, FactorAnalysis
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor

print("Load data...")
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
subm = pd.read_csv('../input/sample_submission.csv')
print("Train shape: {}\nTest shape: {}".format(train.shape, test.shape))

PERC_TRESHOLD = 0.98   ### Percentage of zeros in each feature ###
N_COMP = 22            ### Number of decomposition components ###


target = np.log1p(train['target']).values
cols_to_drop = [col for col in train.columns[2:]
                    if [i[1] for i in list(train[col].value_counts().items()) 
                    if i[0] == 0][0] >= train.shape[0] * PERC_TRESHOLD]

print("Define training features...")
exclude_other = ['ID', 'target']
train_features = []
for c in train.columns:
    if c not in cols_to_drop \
    and c not in exclude_other:
        train_features.append(c)
print("Number of featuress for training: %s" % len(train_features))

train, test = train[train_features], test[train_features]
print("\nTrain shape: {}\nTest shape: {}".format(train.shape, test.shape))


print("\nStart decomposition process...")
print("PCA")
pca = PCA(n_components=N_COMP, random_state=17)
pca_results_train = pca.fit_transform(train)
pca_results_test = pca.transform(test)

print("tSVD")
tsvd = TruncatedSVD(n_components=N_COMP, random_state=17)
tsvd_results_train = tsvd.fit_transform(train)
tsvd_results_test = tsvd.transform(test)

print("ICA")
ica = FastICA(n_components=N_COMP, random_state=17)
ica_results_train = ica.fit_transform(train)
ica_results_test = ica.transform(test)

print("FA")
fa = FactorAnalysis(n_components=N_COMP, random_state=17)
fa_results_train = fa.fit_transform(train)
fa_results_test = fa.transform(test)

print("GRP")
grp = GaussianRandomProjection(n_components=N_COMP, eps=0.1, random_state=17)
grp_results_train = grp.fit_transform(train)
grp_results_test = grp.transform(test)

print("SRP")
srp = SparseRandomProjection(n_components=N_COMP, dense_output=True, random_state=17)
srp_results_train = srp.fit_transform(train)
srp_results_test = srp.transform(test)

print("Append decomposition components to datasets...")
for i in range(1, N_COMP + 1):
    train['pca_' + str(i)] = pca_results_train[:, i - 1]
    test['pca_' + str(i)] = pca_results_test[:, i - 1]
    
    train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
    test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]

    train['ica_' + str(i)] = ica_results_train[:, i - 1]
    test['ica_' + str(i)] = ica_results_test[:, i - 1]
    
    train['fa_' + str(i)] = fa_results_train[:, i - 1]
    test['fa_' + str(i)] = fa_results_test[:, i - 1]

    train['grp_' + str(i)] = grp_results_train[:, i - 1]
    test['grp_' + str(i)] = grp_results_test[:, i - 1]

    train['srp_' + str(i)] = srp_results_train[:, i - 1]
    test['srp_' + str(i)] = srp_results_test[:, i - 1]
print("\nTrain shape: {}\nTest shape: {}".format(train.shape, test.shape))


print("\nModelling...")
def rmsle(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return np.sqrt(np.mean(np.power(np.log(y_true + 1) - np.log(y_pred + 1), 2)))

folds = KFold(n_splits=5, shuffle=True, random_state=546789)
oof_preds = np.zeros(train.shape[0])
sub_preds = np.zeros(test.shape[0])

for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train)):
    trn_x, trn_y = train.ix[trn_idx], target[trn_idx]
    val_x, val_y = train.ix[val_idx], target[val_idx]
    
    cb_model = CatBoostRegressor(iterations=1000,
                                 learning_rate=0.1,
                                 depth=4,
                                 l2_leaf_reg=20,
                                 bootstrap_type='Bernoulli',
                                 subsample=0.6,
                                 eval_metric='RMSE',
                                 metric_period=50,
                                 od_type='Iter',
                                 od_wait=45,
                                 random_seed=17,
                                 allow_writing_files=False)
    
    cb_model.fit(trn_x, trn_y,
                 eval_set=(val_x, val_y),
                 cat_features=[],
                 use_best_model=True,
                 verbose=True)
    
    oof_preds[val_idx] = cb_model.predict(val_x)
    sub_preds += cb_model.predict(test) / folds.n_splits
    
    print("Fold %2d RMSLE : %.6f" % (n_fold+1, rmsle(np.exp(val_y)-1, np.exp(oof_preds[val_idx])-1)))

print("Full RMSLE score %.6f" % rmsle(np.exp(target)-1, np.exp(oof_preds)-1)) 
subm['target'] = np.exp(sub_preds)-1
subm.to_csv('submission.csv', index=False)