import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing
from sklearn import metrics
from sklearn import cross_validation
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.svm import LinearSVC 
from xgboost import XGBRegressor
# Any results you write to the current directory are saved as output.

TRAINING_FILE_NAME = '../input/train.csv'
TEST_FILE_NAME = '../input/test.csv'

def load_data(test = False):
    data_frame = pd.read_csv(TRAINING_FILE_NAME if not test else TEST_FILE_NAME, header=0)

    return data_frame.iloc[160000:180000]

def get_encoders(data_frame):
    split = 116

    encoders = dict()
    for col in data_frame.columns[0:split].values:
        cats = np.unique(data_frame[col])

        label_encoder = LabelEncoder()
        label_encoder.fit(cats)

        onehot_encoder = OneHotEncoder(sparse=False, n_values=len(cats))

        encoders[col] = {'label': label_encoder, 'onehot': onehot_encoder}

    return encoders

def set_bins(data_frame):
    split = len(data_frame.columns[data_frame.columns.map(lambda column: column.startswith('cat'))])
    for i in range(split, data_frame.shape[1]):
        col = data_frame.columns[i]
        data_frame[col] = data_frame[col].map(lambda value: int(value / 0.1))

    return data_frame

def remove_low_variance(data_frame):
    sel = VarianceThreshold(threshold=(.9 * (1 - .9)))
    sel.fit(data_frame)

    return data_frame[sel.get_support(indices = True)]

def remove_univariate(X, y):
    sel = SelectKBest(f_regression, k=125)
    sel.fit(X, y)
    
    return X[sel.get_support(indices = True)]

def encode_label(data_frame, encoders):
    split = len(data_frame.columns[data_frame.columns.map(lambda column: column.startswith('cat'))])
    for i in range(0, split):
        col = data_frame.columns[i]

        label_enc = encoders[col]['label']

        data_frame[col] = label_enc.transform(data_frame[col])

def encode_onehot(X, encoders):
    features = []
    columns = X.columns[X.columns.map(lambda column: column.startswith('cat'))]
    new_split = len(columns)
    for i in range(0, new_split):
        col = columns[i]

        transformed = X[col].values
        transformedReshaped = transformed.reshape(X.shape[0], 1)

        onehot_encoder = encoders[col]['onehot']
        transformedOneHot = onehot_encoder.fit_transform(transformedReshaped)
        features.append(transformedOneHot)

        del transformed
        del transformedReshaped
        del transformedOneHot

    new_features = np.column_stack(features)
    del features

    result = np.concatenate((new_features, X.iloc[0::, new_split::].values), axis = 1)
    del new_features
    return result

def pre_process(X, y, encoders):
    #X = X.drop(['cat69', 'cat14', 'cat21', 'cat61', 'cat63', 'cat62', 'cat67', 'cat54', 'cat55', 'cat19'], axis=1)
    encode_label(X, encoders)

    #X = set_bins(X)
    
    #X = remove_low_variance(X)
    #X = remove_univariate(X, y)

    X_result = encode_onehot(X, encoders)
    
    return X_result

def fit(X, y):
    clf = XGBRegressor(n_estimators = 1000, reg_alpha = 1.6, max_depth = 12)
    clf.fit(X, y)
    return clf

print("Loading...")
train_df = load_data()
X_df = train_df.iloc[0::, 1:-1]
y_df = train_df.iloc[0::, -1]
del train_df

print("Preprocessing...")
encoders = get_encoders(X_df)

shift = 1500
X = pre_process(X_df, y_df, encoders)
y = np.log(y_df.values + shift)
del encoders
del X_df, y_df


X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.1, random_state = 0)
del X, y

print("Fitting...")
clf = fit(X_train, y_train)

print("Predicting...")
predicted = clf.predict(X_test)
output = np.exp(predicted) - shift
del predicted

print("Training set score: %f" % clf.score(X_train, y_train))
print("Test set score: %f" % clf.score(X_test, y_test))
print("Mean absolute error: %f" %  (sum(abs(output-(np.exp(y_test) - shift))) / output.shape[0]))

del X_train, X_test, y_train, y_test
del output, clf
