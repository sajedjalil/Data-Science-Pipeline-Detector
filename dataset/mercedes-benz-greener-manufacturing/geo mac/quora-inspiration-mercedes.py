import pandas as pd
from scipy.spatial.distance import cityblock, euclidean, mahalanobis, jaccard, canberra, braycurtis, correlation, cosine
from scipy.stats import skew, kurtosis
from sklearn.cross_validation import cross_val_score
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.utils import shuffle
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from tqdm import tqdm

TRAIN_FILE = '../input/train.csv'
TEST_FILE = '../input/test.csv'

MULTIPLIER_OF_NEW_DATASET = 10
SIZE_OF_TEST = 5



print ("loading...")

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

# Encode categorical data using LabelEncoder
for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))

# Creating the new train dataset by pairing randomly the existing instances
train_double = train
train_all = pd.DataFrame()

for i in range(MULTIPLIER_OF_NEW_DATASET):
    train_suffled = shuffle(train).reset_index(drop=True)
    train_suffled = train_suffled.rename(columns=lambda x: x.replace('X', 'Z'))
    train_suffled = train_suffled.rename(columns=lambda x: x.replace('y', 'y_Z'))
    train_suffled = train_suffled.rename(columns=lambda x: x.replace('ID', 'ID_Z'))

    train_double = pd.concat((train, train_suffled), axis=1)
    train_all = pd.concat((train_all, train_double), axis=0)

train_all = train_all.reset_index(drop=True)
train_all['diff'] = train_all['y'] - train_all['y_Z']
y_train = train_all['diff'].values

train_all.drop(['ID', 'ID_Z', 'y', 'y_Z', 'diff'], axis=1, inplace=True)

X_cols = [c for c in train_all.columns if c.startswith('X')]
Z_cols = [c for c in train_all.columns if c.startswith('Z')]

# Adding pair distances as features
f_cosine_similarity = []
f_euclidean_distance = []
f_cityblock_distance = []
f_correlation_distance = []
f_jaccard_distance = []
f_canberra_distance = []
f_braycurtis_distance = []
f_skew_X = []
f_skew_Z = []
f_kur_X = []
f_kur_Z = []

print('-----distances-------')
for i in tqdm(range(train_all.shape[0])):
    # if i % 1000 == 0:
    #     print('-----------------------------------------' + str(i))
    f_cosine_similarity.append(cosine(train_all[X_cols].values[i], train_all[Z_cols].values[i]))
    f_euclidean_distance.append(euclidean(train_all[X_cols].values[i], train_all[Z_cols].values[i]))
    f_cityblock_distance.append(cityblock(train_all[X_cols].values[i], train_all[Z_cols].values[i]))
    f_correlation_distance.append(correlation(train_all[X_cols].values[i], train_all[Z_cols].values[i]))
    f_jaccard_distance.append(jaccard(train_all[X_cols].values[i], train_all[Z_cols].values[i]))
    f_canberra_distance.append(canberra(train_all[X_cols].values[i], train_all[Z_cols].values[i]))
    f_braycurtis_distance.append(braycurtis(train_all[X_cols].values[i], train_all[Z_cols].values[i]))
    f_skew_X.append(skew(train_all[X_cols].values[i]))
    f_skew_Z.append(skew(train_all[Z_cols].values[i]))
    f_kur_X.append(kurtosis(train_all[X_cols].values[i]))
    f_kur_Z.append(kurtosis(train_all[Z_cols].values[i]))

train_all['f_cosine_similarity'] = f_cosine_similarity
train_all['f_euclidean_distance'] = f_euclidean_distance
train_all['f_cityblock_distance'] = f_cityblock_distance
train_all['f_correlation_distance'] = f_correlation_distance
train_all['f_jaccard_distance'] = f_jaccard_distance
train_all['f_canberra_distance'] = f_canberra_distance
train_all['f_braycurtis_distance'] = f_braycurtis_distance
train_all['f_skew_X'] = f_skew_X
train_all['f_skew_Z'] = f_skew_Z
train_all['f_kur_X'] = f_kur_X
train_all['f_kur_Z'] = f_kur_Z

# We tried and validated many algos without optimize most of them


en = make_pipeline(ElasticNet(alpha=0.091, l1_ratio=0.01))

rd = Ridge(alpha=1, fit_intercept=True, normalize=True)

rf = RandomForestRegressor(n_estimators=250, n_jobs=-1, max_depth=3, min_samples_split=11, min_samples_leaf=50)

et = ExtraTreesRegressor(n_estimators=100, n_jobs=-1, max_depth=8, min_samples_split=11, min_samples_leaf=50)

xgbm = xgb.sklearn.XGBRegressor(max_depth=4, learning_rate=0.05, subsample=0.9,
                                objective='reg:linear', n_estimators=500)

lgbm = lgb.LGBMRegressor(nthread=3, silent=True, learning_rate=0.05, max_depth=4, num_leaves=15, max_bin=255,
                         subsample_for_bin=50000,
                         min_child_weight=5, min_child_samples=10, subsample=1, subsample_freq=1,
                         colsample_bytree=1, n_estimators=90, seed=777, reg_alpha=1, reg_lambda=10)

rcv = RidgeCV()
lcv = LassoCV()
ecv = ElasticNetCV()
ompcv = OrthogonalMatchingPursuitCV()

lgbm = lgbm.fit(train_all, y_train)
pr_lst = []

for i in range(SIZE_OF_TEST):
    print('----------------------' + str(i))
    train_suffled = shuffle(train).reset_index(drop=True)
    train_suffled = train_suffled.rename(columns=lambda x: x.replace('X', 'Z'))
    train_suffled = train_suffled.rename(columns=lambda x: x.replace('y', 'y_Z'))
    train_suffled = train_suffled.rename(columns=lambda x: x.replace('ID', 'ID_Z'))

    test_double = pd.concat((test, train_suffled), axis=1)

    # Adding pair distances as features
    f_cosine_similarity = []
    f_euclidean_distance = []
    f_cityblock_distance = []
    f_correlation_distance = []
    f_jaccard_distance = []
    f_canberra_distance = []
    f_braycurtis_distance = []
    f_skew_X = []
    f_skew_Z = []
    f_kur_X = []
    f_kur_Z = []

    print('-----distances-------')
    for i in tqdm(range(test_double.shape[0])):
        # if i % 1000 == 0:
        #     print('-----------------------------------------' + str(i))
        f_cosine_similarity.append(cosine(test_double[X_cols].values[i], test_double[Z_cols].values[i]))
        f_euclidean_distance.append(euclidean(test_double[X_cols].values[i], test_double[Z_cols].values[i]))
        f_cityblock_distance.append(cityblock(test_double[X_cols].values[i], test_double[Z_cols].values[i]))
        f_correlation_distance.append(correlation(test_double[X_cols].values[i], test_double[Z_cols].values[i]))
        f_jaccard_distance.append(jaccard(test_double[X_cols].values[i], test_double[Z_cols].values[i]))
        f_canberra_distance.append(canberra(test_double[X_cols].values[i], test_double[Z_cols].values[i]))
        f_braycurtis_distance.append(braycurtis(test_double[X_cols].values[i], test_double[Z_cols].values[i]))
        f_skew_X.append(skew(test_double[X_cols].values[i]))
        f_skew_Z.append(skew(test_double[Z_cols].values[i]))
        f_kur_X.append(kurtosis(test_double[X_cols].values[i]))
        f_kur_Z.append(kurtosis(test_double[Z_cols].values[i]))

    test_double['f_cosine_similarity'] = f_cosine_similarity
    test_double['f_euclidean_distance'] = f_euclidean_distance
    test_double['f_cityblock_distance'] = f_cityblock_distance
    test_double['f_correlation_distance'] = f_correlation_distance
    test_double['f_jaccard_distance'] = f_jaccard_distance
    test_double['f_canberra_distance'] = f_canberra_distance
    test_double['f_braycurtis_distance'] = f_braycurtis_distance
    test_double['f_skew_X'] = f_skew_X
    test_double['f_skew_Z'] = f_skew_Z
    test_double['f_kur_X'] = f_kur_X
    test_double['f_kur_Z'] = f_kur_Z

    y_live = train_suffled['y_Z']
    test_double.drop(['ID', 'ID_Z', 'y_Z'], axis=1, inplace=True)

    pr_lst.append(lgbm.predict(test_double) + y_live)

preds = np.array(pr_lst).mean(0)

submission = pd.read_csv('../input/sample_submission.csv')
submission.index = submission.ID
submission.y = preds
submission.to_csv('quora_approach.csv', index=False)


# results = cross_val_score(lgbm, train_all, y_train, cv=5, scoring='r2')
# print("LGBM score: %.4f (%.4f)" % (results.mean(), results.std()))
#
# results = cross_val_score(lcv, train_all, y_train.reshape(-1,1), cv=5, scoring='r2')
# print("LassoCV score: %.4f (%.4f)" % (results.mean(), results.std()))
#
# results = cross_val_score(ecv, train_all, y_train, cv=5, scoring='r2')
# print("ElasticNetCV score: %.4f (%.4f)" % (results.mean(), results.std()))
#
# results = cross_val_score(ompcv, train_all, y_train, cv=5, scoring='r2')
# print("OrthogonalMatchingPursuitCV score: %.4f (%.4f)" % (results.mean(), results.std()))
#
# results = cross_val_score(en, train_all, y_train, cv=5, scoring='r2')
# print("ElasticNet score: %.4f (%.4f)" % (results.mean(), results.std()))
# #
# results = cross_val_score(rf, train_all, y_train, cv=5, scoring='r2')
# print("RandomForest score: %.4f (%.4f)" % (results.mean(), results.std()))
#
# results = cross_val_score(rd, train_all, y_train, cv=5, scoring='r2')
# print("Ridge score: %.4f (%.4f)" % (results.mean(), results.std()))
#
# results = cross_val_score(et, train_all, y_train, cv=5, scoring='r2')
# print("ExtraTrees score: %.4f (%.4f)" % (results.mean(), results.std()))
#
# results = cross_val_score(xgbm, train_all, y_train, cv=5, scoring='r2')
# print("XGBoost score: %.4f (%.4f)" % (results.mean(), results.std()))

