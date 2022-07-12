import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# import xgboost as xgb
# from keras.models import Sequential
# from keras.layers.core import Dense, Activation, Dropout
# from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils
# from hep_ml.losses import BinFlatnessLossFunction
# from hep_ml.gradientboosting import UGradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler # StandardScaler
from sklearn.cross_validation import train_test_split

# train_f = open('../input/training.csv')
# test_f = open('../input/test.csv')
print("Load the training/test data using pandas")
train = pd.read_csv('../input/training.csv')
test  = pd.read_csv('../input/test.csv')

def get_training_data():
    filter_out = ['id', 'min_ANNmuon', 'production', 'mass', 'signal']
    # filter_out = ['id', 'min_ANNmuon', 'production', 'mass', 'signal', 'SPDhits', 'IP', 'IPSig', 'isolationc']
    ids = train["id"]
    y = train["signal"]
    train.drop(filter_out, axis=1, inplace=True)
    data = train
    return ids.values, np.array(data.values), np.array(y.values)


def get_test_data():
    # filter_out = ['id', 'SPDhits', 'IP', 'IPSig', 'isolationc']
    filter_out = ['id']
    ids = test["id"]
    test.drop(filter_out, axis=1, inplace=True)
    data = test
    return ids.values, np.array(data.values)

def preprocess_data(X, scaler=None):
    # if not scaler:
    #     scaler = MinMaxScaler()
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    return X, scaler

# get training data
ids, X, y = get_training_data()
print('Data shape:', X.shape)

# shuffle the data
np.random.seed(123) # used to be 369
np.random.shuffle(X)
np.random.seed(123)
np.random.shuffle(y)

print('Signal ratio:', np.sum(y) / y.shape[0])

# preprocess the data
X, scaler = preprocess_data(X)
# y = np_utils.to_categorical(y)

print ('y', y.shape)

# split into training / evaluation data
X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size= 1 - 0.78, random_state=42)# used to be 0.97, 0.78 is better, 0.83 possible

COMPONENT_NUM = int(X.shape[1] / 2)

print('Reduction...')
pca = PCA(n_components=COMPONENT_NUM, whiten=True)
pca.fit(X)
train_data = pca.transform(X)

print('Train SVM...')
svc = SVC()
svc.fit(X, y)

print('Read testing data...')
# get training data
ids, X = get_test_data()
X, scaler = preprocess_data(X)

print('Predicting...')
predict = svc.predict(X)

print('Saving...')

submission = pd.DataFrame({"id": ids, "prediction": test_probs})
submission.to_csv("svc_preds.csv", index=False)