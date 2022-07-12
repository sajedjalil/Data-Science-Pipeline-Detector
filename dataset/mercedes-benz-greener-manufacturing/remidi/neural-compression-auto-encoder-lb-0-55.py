''' new compression technique for the non-probers out there, 

I am trying to do more feature-engineering, because just probing wouldn't improve my skill. 


Please provide feedback and upvote if you like it :)'''


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.

from sklearn.base import BaseEstimator,TransformerMixin, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
from sklearn.linear_model import ElasticNetCV, LassoLarsCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.utils import check_array
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import r2_score

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import NMF
from sklearn.cluster import FeatureAgglomeration

from sklearn.ensemble import RandomForestRegressor
import random

random.seed(1234)

import warnings
warnings.filterwarnings('ignore')


class StackingEstimator(BaseEstimator, TransformerMixin):
    
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None, **fit_params):
        self.estimator.fit(X, y, **fit_params)
        return self
    def transform(self, X):
        X = check_array(X)
        X_transformed = np.copy(X)
        # add class probabilities as a synthetic feature
        if issubclass(self.estimator.__class__, ClassifierMixin) and hasattr(self.estimator, 'predict_proba'):
            X_transformed = np.hstack((self.estimator.predict_proba(X), X))

        # add class prodiction as a synthetic feature
        X_transformed = np.hstack((np.reshape(self.estimator.predict(X), (-1, 1)), X_transformed))

        return X_transformed

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# removing the outlier
train = train.loc[train['y'] < 170, :]


# function for auto encoder with a compressed components n_comp = 12
def neural_compression_v2(train, test):
    dataset = pd.concat([train.drop('y', axis=1), test], axis=0)
    ids = dataset['ID']
    dataset.drop('ID', axis=1, inplace=True)
    y_train = train['y']
    
    cat_vars = [c for c in dataset.columns if dataset[c].dtype == 'object']
    for c in cat_vars:
        t_data = pd.get_dummies(dataset[c], prefix=c)
        dataset = pd.concat([dataset, t_data], axis=1)

    dataset.drop(cat_vars, axis=1, inplace=True)

    train = dataset[:train.shape[0]]
    test = dataset[train.shape[0]:]

    print("one hot encoded train shape :: {}".format(train.shape))
    print("one hot encoded test shape :: {}".format(test.shape))
    
    ''' neural network compression code '''
    
    import keras
    from keras import regularizers
    from keras.layers import Input, Dense
    from keras.models import Model

    print(keras.__version__)
    init_dim = train.shape[1]

    input_row = Input(shape=(init_dim, ))
    encoded = Dense(512, activation='relu')(input_row)
    encoded = Dense(256, activation='relu')(encoded)
    encoded = Dense(128, activation='relu')(encoded)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    
    encoded = Dense(12, activation='relu')(encoded)
    
    decoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(256, activation='relu')(decoded)
    decoded = Dense(512, activation='relu')(decoded)
    decoded = Dense(init_dim, activation='sigmoid')(decoded)

    autoencoder = Model(inputs=input_row, outputs=decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    autoencoder.fit(train.values, train.values,
                    batch_size=256,
                    shuffle=True, validation_data=(test.values, test.values), epochs=300)

    # compressing the data
    encoder = Model(inputs=input_row, outputs=encoded)
    train_compress = encoder.predict(train.values)
    test_compress = encoder.predict(test.values)

    # denoising the data
    denoised_train = autoencoder.predict(train.values)
    denoised_test = autoencoder.predict(test.values)
    
    return train_compress, test_compress, denoised_train, denoised_test

train_compress, test_compress, denoised_train, denoised_test = neural_compression_v2(train, test)


# ===================================== mean of the y for the categorical replacement ========================
mean_x0 = train[['X0', 'y']].groupby(['X0'], as_index=False).mean()
mean_x0.columns = ['X0', 'mean_x0']
train = pd.merge(train, mean_x0, on='X0', how='outer')

mean_x1 = train[['X1', 'y']].groupby(['X1'], as_index=False).mean()
mean_x1.columns = ['X1', 'mean_x1']
train = pd.merge(train, mean_x1, on='X1', how='outer')

mean_x2 = train[['X2', 'y']].groupby(['X2'], as_index=False).mean()
mean_x2.columns = ['X2', 'mean_x2']
train = pd.merge(train, mean_x2, on='X2', how='outer')

mean_x3 = train[['X3', 'y']].groupby(['X3'], as_index=False).mean()
mean_x3.columns = ['X3', 'mean_x3']
train = pd.merge(train, mean_x3, on='X3', how='outer')

mean_x4 = train[['X4', 'y']].groupby(['X4'], as_index=False).mean()
mean_x4.columns = ['X4', 'mean_x4']
train = pd.merge(train, mean_x4, on='X4', how='outer')

mean_x5 = train[['X5', 'y']].groupby(['X5'], as_index=False).mean()
mean_x5.columns = ['X5', 'mean_x5']
train = pd.merge(train, mean_x5, on='X5', how='outer')

mean_x6 = train[['X6', 'y']].groupby(['X6'], as_index=False).mean()
mean_x6.columns = ['X6', 'mean_x6']
train = pd.merge(train, mean_x6, on='X6', how='outer')

mean_x8 = train[['X8', 'y']].groupby(['X8'], as_index=False).mean()
mean_x8.columns = ['X8', 'mean_x8']
train = pd.merge(train, mean_x8, on='X8', how='outer')

test = pd.merge(test, mean_x0, on='X0', how='left')
test['mean_x0'].fillna(test['mean_x0'].dropna().mean(), inplace=True)

test = pd.merge(test, mean_x1, on='X1', how='left')
test['mean_x1'].fillna(test['mean_x1'].dropna().mean(), inplace=True)

test = pd.merge(test, mean_x2, on='X2', how='left')
test['mean_x2'].fillna(test['mean_x2'].dropna().mean(), inplace=True)

test = pd.merge(test, mean_x3, on='X3', how='left')
test['mean_x3'].fillna(test['mean_x3'].dropna().mean(), inplace=True)

test = pd.merge(test, mean_x4, on='X4', how='left')
test['mean_x4'].fillna(test['mean_x4'].dropna().mean(), inplace=True)

test = pd.merge(test, mean_x5, on='X5', how='left')
test['mean_x5'].fillna(test['mean_x5'].dropna().mean(), inplace=True)

test = pd.merge(test, mean_x6, on='X6', how='left')
test['mean_x6'].fillna(test['mean_x6'].dropna().mean(), inplace=True)

test = pd.merge(test, mean_x8, on='X8', how='left')
test['mean_x8'].fillna(test['mean_x8'].dropna().mean(), inplace=True)

# ===================================== mean of the y for the categorical replacement ========================

for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))
        
# ============================================== One hot encoding local inc, LB decrease ========================
# y_train = train['y']
# dataset = pd.concat([train.drop('y', axis=1), test], axis=0)
# print(dataset.shape)
# for c in dataset.columns:
#     if dataset[c].dtype == 'object':
#         t = pd.get_dummies(dataset[c], prefix=c)
#         dataset = pd.concat([dataset, t], axis=1)

# dataset.drop([c for c in train.columns if train[c].dtype == 'object'], axis=1, inplace=True)
# train = dataset[:train.shape[0]]
# train['y'] = y_train
# test = dataset[train.shape[0]:]

# ============================================== One hot encoding local inc, LB decrease ========================

n_comp = 12

# tSVD
tsvd = TruncatedSVD(n_components=n_comp, random_state=420)
tsvd_results_train = tsvd.fit_transform(train.drop(["y"], axis=1))
tsvd_results_test = tsvd.transform(test)

# PCA
pca = PCA(n_components=n_comp, random_state=420)
pca2_results_train = pca.fit_transform(train.drop(["y"], axis=1))
pca2_results_test = pca.transform(test)

# ICA
ica = FastICA(n_components=n_comp, random_state=420)
ica2_results_train = ica.fit_transform(train.drop(["y"], axis=1))
ica2_results_test = ica.transform(test)

# GRP
grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)
grp_results_train = grp.fit_transform(train.drop(["y"], axis=1))
grp_results_test = grp.transform(test)

# SRP
srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)
srp_results_train = srp.fit_transform(train.drop(["y"], axis=1))
srp_results_test = srp.transform(test)

# NMF
nmf = NMF(n_components=n_comp, init='nndsvdar', random_state=420)
nmf_results_train = nmf.fit_transform(train.drop(["y"], axis=1))
nmf_results_test = nmf.transform(test)

# FAG
fag = FeatureAgglomeration(n_clusters=n_comp, linkage='ward')
fag_results_train = fag.fit_transform(train.drop(["y"], axis=1))
fag_results_test = fag.transform(test)

usable_columns = list(set(train.columns) - set(['y']))

# Append decomposition components to datasets
for i in range(1, n_comp + 1):
    train['pca_' + str(i)] = pca2_results_train[:, i - 1]
    test['pca_' + str(i)] = pca2_results_test[:, i - 1]

    train['ica_' + str(i)] = ica2_results_train[:, i - 1]
    test['ica_' + str(i)] = ica2_results_test[:, i - 1]

    train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
    test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]

    train['grp_' + str(i)] = grp_results_train[:, i - 1]
    test['grp_' + str(i)] = grp_results_test[:, i - 1]

    train['srp_' + str(i)] = srp_results_train[:, i - 1]
    test['srp_' + str(i)] = srp_results_test[:, i - 1]
    
    train['nmf_' + str(i)] = nmf_results_train[:, i - 1]
    test['nmf_' + str(i)] = nmf_results_test[:, i - 1]
    
#     train['fag_' + str(i)] = fag_results_train[:, i - 1]
#     test['fag_' + str(i)] = fag_results_test[:, i - 1]

for j in range(1, train_compress.shape[1]):
    train['aen_' + str(j)] = train_compress[:, j-1]
    test['aen_' + str(j)] = test_compress[:, j-1]
    
    
# usable_columns = list(set(train.columns) - set(['y']))

y_train = train['y'].values
y_mean = np.mean(y_train)
id_test = test['ID'].values
# finaltrainset and finaltestset are data to be used only the stacked model (does not contain PCA, SVD... arrays) 
finaltrainset = train[usable_columns].values
finaltestset = test[usable_columns].values


'''Train the xgb model then predict the test data'''
print('running ....... ')

xgb_params = {
    'n_trees': 520, 
    'eta': 0.0045,
    'max_depth': 4,
    'subsample': 0.93,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean, # base prediction = mean(target)
}
# NOTE: Make sure that the class is labeled 'class' in the data file

dtrain = xgb.DMatrix(train.drop('y', axis=1), y_train)
dtest = xgb.DMatrix(test)

num_boost_rounds = 1250
# train model
model = xgb.train(xgb_params, dtrain, num_boost_round=num_boost_rounds)
y_pred = model.predict(dtest)

'''Train the stacked models then predict the test data'''

stacked_pipeline = make_pipeline(
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
    StackingEstimator(estimator=GradientBoostingRegressor(learning_rate=0.001, loss="huber", max_depth=3, max_features=0.55, min_samples_leaf=18, min_samples_split=14, subsample=0.7)),
    LassoLarsCV()
)

stacked_pipeline.fit(finaltrainset, y_train)
results = stacked_pipeline.predict(finaltestset)

'''R2 Score on the entire Train data when averaging'''

print('R2 score on train data:')
print(r2_score(y_train,stacked_pipeline.predict(finaltrainset)*0.2855 + model.predict(dtrain)*0.7145))
score = r2_score(y_train, model.predict(dtrain))
print("xgboost score : {}".format(score))


'''Save submission to csv file'''
sub = pd.DataFrame()
sub['ID'] = id_test
sub['y'] = y_pred*0.75 + results*0.25
sub.to_csv('submission.csv', index=False)