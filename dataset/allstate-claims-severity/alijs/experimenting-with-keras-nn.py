import numpy as np 
import pandas as pd 

from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import train_test_split
from scipy.stats import skew, boxcox
from math import exp, log

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, MaxoutDense
from keras.layers.advanced_activations import PReLU

print("Started.")
my_random = 2016
np.random.seed(my_random)

def baseline_model(sh):
    # create model
    model = Sequential()
    #model.add(Dropout(0.2))
    model.add(Dense(300, input_dim=sh, init='normal'))
    model.add(MaxoutDense(100))
    model.add(Dropout(0.3))
    model.add(Dense(200, input_dim=sh, init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(1, init='normal'))
    # Compile model
    model.compile(loss='mae', optimizer='adagrad')
    return model

DATA_TRAIN_PATH = '../input/train.csv'
DATA_TEST_PATH = '../input/test.csv'

def scale_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler

def load_data(path_train=DATA_TRAIN_PATH, path_test=DATA_TEST_PATH):
    # Tilii code from kernel https://www.kaggle.com/tilii7/allstate-claims-severity/bias-correction-xgboost/run/395245
    train_loader = pd.read_csv(path_train, dtype={'id': np.int32})
    train = train_loader.drop(['id', 'loss'], axis=1)
    test_loader = pd.read_csv(path_test, dtype={'id': np.int32})
    test = test_loader.drop(['id'], axis=1)
    ntrain = train.shape[0]
    ntest = test.shape[0]
    train_test = pd.concat((train, test)).reset_index(drop=True)
    numeric_feats = train_test.dtypes[train_test.dtypes != "object"].index

    # compute skew and do Box-Cox transformation
    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))
    print("\nSkew in numeric features:")
    print(skewed_feats)
    # transform features with skew > 0.25 (this can be varied to find optimal value)
    skewed_feats = skewed_feats[skewed_feats > 0.25]
    skewed_feats = skewed_feats.index
    for feats in skewed_feats:
        train_test[feats] = train_test[feats] + 1
        train_test[feats], lam = boxcox(train_test[feats])
    features = train.columns
    cats = [feat for feat in features if 'cat' in feat]
    # factorize categorical features
    for feat in cats:
        train_test[feat] = pd.factorize(train_test[feat], sort=True)[0]
    x_train = train_test.iloc[:ntrain, :]
    x_test = train_test.iloc[ntrain:, :]
    train_test_scaled, scaler = scale_data(train_test)
    train, _ = scale_data(x_train, scaler)
    test, _ = scale_data(x_test, scaler)

    #train_labels = np.log(np.array(train_loader['loss']))
    train_labels = np.array(train_loader['loss'])
    train_ids = train_loader['id'].values.astype(np.int32)
    test_ids = test_loader['id'].values.astype(np.int32)

    return train, train_labels, test, train_ids, test_ids

train, target, test, _, ids = load_data()


X_train, X_val, y_train, y_val = train_test_split(
    train, target, train_size=0.999, random_state=my_random)
print('Data: train shape {}, valid shape {}, test shape {}'.format(X_train.shape, X_val.shape, test.shape))

model = baseline_model(X_train.shape[1])

print("Training...")
fit = model.fit(X_train, y_train, 100,
                         nb_epoch=24,
                         validation_split=0.1, verbose=2
                         )
print("Evaluating...")
val = model.predict(X_val, 100)
#print('MAE val {}'.format(mean_absolute_error(np.exp(y_val), np.exp(val))))
print('MAE val {}'.format(mean_absolute_error(y_val, val)))

print("Predicting...")
pred = model.predict(test, 100)
#out = pd.DataFrame(np.exp(pred), index = ids, columns=['loss'])
out = pd.DataFrame(pred, index = ids, columns=['loss'])

print("Writing output...")
print(out.head(5))
out.to_csv('my_keras_%s.csv' % str(datetime.now().strftime("%Y-%m-%d_%H-%M")), index = True, index_label='id')

print("Done.")

