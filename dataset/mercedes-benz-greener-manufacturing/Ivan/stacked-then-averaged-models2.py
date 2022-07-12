# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
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
from sklearn.metrics import r2_score



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

for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))



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

#save columns list before adding the decomposition components

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

#usable_columns = list(set(train.columns) - set(['y']))

y_train = train['y'].values
y_mean = np.mean(y_train)
id_test = test['ID'].values
#finaltrainset and finaltestset are data to be used only the stacked model (does not contain PCA, SVD... arrays) 
finaltrainset = train[usable_columns].values
finaltestset = test[usable_columns].values


'''Train the xgb model then predict the test data'''

xgb_params = {
    'n_trees': 520, 
    'eta': 0.0045,
    'max_depth': 4,
    'subsample': 0.93,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'gamma':0.03,
    'base_score': y_mean, # base prediction = mean(target)
}
# NOTE: Make sure that the class is labeled 'class' in the data file

dtrain = xgb.DMatrix(train.drop('y', axis=1), y_train)
dtest = xgb.DMatrix(test)

num_boost_rounds = 1250
# train model
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)
y_pred = model.predict(dtest)

'''Train the stacked models then predict the test data'''

stacked_pipeline = make_pipeline(
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
    StackingEstimator(estimator=GradientBoostingRegressor(learning_rate=0.002, loss="huber", max_depth=4, max_features=0.55, min_samples_leaf=18, min_samples_split=14, subsample=0.7)),
    LassoLarsCV()

)


stacked_pipeline.fit(finaltrainset, y_train)
results = stacked_pipeline.predict(finaltestset)

'''R2 Score on the entire Train data when averaging'''

print('R2 score on train data:')
print(r2_score(y_train,stacked_pipeline.predict(finaltrainset)*0.2855 + model.predict(dtrain)*0.7145))

'''Average the preditionon test data  of both models then save it on a csv file'''
'''NN'''
# preprocessing/decomposition
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA

# Keras is a deep learning library that wraps the efficient numerical libraries Theano and TensorFlow.
# It provides a clean and simple API that allows you to define and evaluate deep learning models in just a few lines of code.from keras.models import Sequential, load_model
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.recurrent import LSTM
# define custom R2 metrics for Keras backend
from keras import backend as K
# to tune the NN
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam, RMSprop

# model evaluation
from sklearn.model_selection import cross_val_score, KFold, train_test_split, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, mean_squared_error

# feature selection
from sklearn.feature_selection import f_regression, mutual_info_regression, VarianceThreshold

# define path to save model
import os

# to make results reproducible
seed = 123

# Read datasets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# Remove the outlier
# train=train[train.y<250]

# save IDs for submission
id_test = test['ID'].copy()

y = train.y.values
###########################
# DATA PREPARATION
###########################

# glue datasets together
total = pd.concat([train, test], axis=0)
print('initial shape: {}'.format(total.shape))

# binary indexes for train/test set split
is_train = ~total.y.isnull()

#total = total.filter(items=['ID', 'X0'])
# find all categorical features
cf = total.select_dtypes(include=['object']).columns

# make one-hot-encoding convenient way - pandas.get_dummies(df) function
dummies = pd.get_dummies(
    total[cf],
    drop_first=False  # you can set it = True to ommit multicollinearity (crucial for linear models)
)

print('oh-encoded shape: {}'.format(dummies.shape))

# get rid of old columns and append them encoded
total = pd.concat(
    [
        total.drop(cf, axis=1),  # drop old
        dummies  # append them one-hot-encoded
    ],
    axis=1  # column-wise
)

print('appended-encoded shape: {}'.format(total.shape))

# recreate train/test again, now with dropped ID column
train, test = total[is_train].drop(['ID','y'], axis=1), total[~is_train].drop(['ID','y'], axis=1)

# drop redundant objects
del total

# check shape
print('\nTrain shape: {}\nTest shape: {}'.format(train.shape, test.shape))

def r2_keras(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))
def model():
    model = Sequential()
    model.add(Dense(input_dims, input_dim=input_dims, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Activation("linear"))
    model.add(Dense(input_dims, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Dropout(0.8))
    model.add(Activation("linear"))
    model.add(Dense(input_dims, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Activation("linear"))
    model.add(Dense(input_dims, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Activation("linear"))
    model.add(Dense(input_dims//2, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Dropout(0.8))
    model.add(Activation("linear"))
    # Output Layer.
    model.add(Dense(1))

    adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    rms = RMSprop(lr=0.002, rho=0.9, epsilon=1e-08, decay=0.0)

    model.compile(loss='mean_squared_error',  # one may use 'mean_absolute_error' as alternative
                  optimizer=adam,
                  metrics=[r2_keras, "mse"]  # you can add several if needed
                  )

    return model

input_dims = train.shape[1]
np.random.seed(seed)

estimator = KerasRegressor(
    build_fn=model,
    nb_epoch=30,
    batch_size=100,
    verbose=1
)

X = train.values
X_test = test.values
print('\nTrain shape No Feature Selection: {}\nTest shape No Feature Selection: {}'.format(X.shape, X_test.shape))

def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)

def sum_of_square_deviation(numbers, mean):
    return float(1 / len(numbers) * sum((x - mean) ** 2 for x in numbers))

# fit estimator
history = estimator.fit(
    X,
    y,
    epochs=30,
    verbose=2,
    shuffle=True
)

# check performance on train set
print('MSE train: {}'.format(mean_squared_error(y, estimator.predict(X)) ** 0.5))  # mse train
print('R^2 train: {}'.format(r2_score(y, estimator.predict(X))))  # R^2 train

res = estimator.predict(X_test).ravel()
#######################################
sub = pd.DataFrame()
sub['ID'] = id_test
y_pred = y_pred*0.75 + results*0.25
sub['y'] = y_pred*0.8 + res*0.2
sub.to_csv('stacked-keras.csv', index=False)
