### Simple Keras attempt with a few dense layers - only LB 0.24
### Thankful for any comments and suggestions

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import Adam
from keras.utils import np_utils

# Read in our input data
df_train = pd.read_csv('../input/train.csv')

cat_vars = ['ps_ind_02_cat', 'ps_ind_04_cat', 'ps_ind_05_cat', 'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat', 'ps_car_04_cat', 'ps_car_05_cat', 'ps_car_06_cat', 'ps_car_07_cat', 'ps_car_08_cat', 'ps_car_09_cat', 'ps_car_10_cat', 'ps_car_11_cat']
bin_vars = ['ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin', 'ps_ind_09_bin', 'ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin', 'ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin', 'ps_calc_15_bin', 'ps_calc_16_bin', 'ps_calc_17_bin', 'ps_calc_18_bin', 'ps_calc_19_bin', 'ps_calc_20_bin']
real_vars = ['ps_ind_01', 'ps_ind_03', 'ps_ind_14', 'ps_ind_15', 'ps_reg_01', 'ps_reg_02', 'ps_reg_03', 'ps_car_11', 'ps_car_12', 'ps_car_13', 'ps_car_14', 'ps_car_15', 'ps_calc_01', 'ps_calc_02', 'ps_calc_03', 'ps_calc_04', 'ps_calc_05', 'ps_calc_06', 'ps_calc_07', 'ps_calc_08', 'ps_calc_09', 'ps_calc_10', 'ps_calc_11', 'ps_calc_12', 'ps_calc_13', 'ps_calc_14']

# Remove useless features
#useless_features = ['ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin', 'ps_ind_13_bin','ps_car_03_cat','ps_car_05_cat', 'ps_calc_01', 'ps_calc_02', 'ps_calc_03', 'ps_calc_04', 'ps_calc_05', 'ps_calc_06', 'ps_calc_07', 'ps_calc_08', 'ps_calc_09', 'ps_calc_10', 'ps_calc_11', 'ps_calc_12', 'ps_calc_13', 'ps_calc_14']
useless_features = []
df_train = df_train.drop(useless_features, axis=1)

cat_vars = [i for i in cat_vars if i not in useless_features]
bin_vars = [i for i in bin_vars if i not in useless_features]
real_vars = [i for i in real_vars if i not in useless_features]

# This prints out (rows, columns) in each dataframe
print('Train shape:', df_train.shape)
print('Columns:', df_train.columns)

# One-hot encode categorical variables
def create_dummies(data, cat_vars, cat_types):
    cat_data = data[cat_vars].values
    for i in range(len(cat_vars)):
        bins = LabelBinarizer().fit_transform(cat_data[:, 0].astype(cat_types[i]))
        cat_data = np.delete(cat_data, 0, axis=1)
        cat_data = np.column_stack((cat_data, bins))
    return cat_data

x_cat = create_dummies(df_train[cat_vars], cat_vars, [np.float32] * len(cat_vars))

# Center and rescale real variables
def standardize(data, real_vars):
    real_data = data[real_vars]
    scale = StandardScaler()
    return scale.fit_transform(real_data)

x_real = standardize(df_train[real_vars], real_vars)

# Combine preprocessed features
X = np.column_stack((df_train[bin_vars].as_matrix(), x_real, x_cat))
y = df_train['target']

# Split training and test set
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.125, random_state=2017, stratify = y)

y_train = np_utils.to_categorical(y_train.values)
y_test_pre = y_test
y_test = np_utils.to_categorical(y_test.values)

NB_EPOCH = 6
BATCH_SIZE = 32
VERBOSE = 0
OPTIMIZER = Adam()
VALIDATION_SPLIT= 0.2
FEATURES = x_train.shape[1]
DROPOUT = 0.3

print('Number of features:', FEATURES)
model = Sequential([
            Dense(64, input_shape=(FEATURES, ), activation='relu'),
            Dropout(DROPOUT),
            Dense(64, activation='relu'),
            Dropout(DROPOUT),
            Dense(64, activation='relu'),
            Dropout(DROPOUT),
            Dense(2, activation='softmax'),
        ])
model.summary()

# Define the gini metric - from https://www.kaggle.com/c/ClaimPredictionChallenge/discussion/703#5897
def gini(actual, pred, cmpcol = 0, sortcol = 1):
    assert( len(actual) == len(pred) )
    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses
    
    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)
 
def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)

model.compile(loss='binary_crossentropy', optimizer=OPTIMIZER, metrics=['acc'])

# Adjust class weights
from sklearn.utils import class_weight
class_weight = class_weight.compute_class_weight('balanced', np.unique(y), y)

# Train
history = model.fit(x_train, y_train,
                    batch_size=BATCH_SIZE, epochs=NB_EPOCH,
                    verbose=VERBOSE, validation_split=VALIDATION_SPLIT,
                    class_weight = class_weight)

score = model.evaluate(x_test, y_test, verbose=0)
print("Test score:", score[0])
print('Test accuracy:', score[1])

# Calculate gini of test
pred_gini = gini_normalized(y_test_pre, model.predict(x_test)[:,1])
print("Gini of validation:", pred_gini)

# Test data preprocessing
df_test = pd.read_csv('../input/test.csv')
print('Test shape:', df_test.shape)
df_test = df_test.drop(useless_features, axis=1)
x_cat_test = create_dummies(df_test[cat_vars], cat_vars, [np.int32] * len(cat_vars))
x_real_test = standardize(df_test[real_vars], real_vars)
X_test = np.column_stack((df_test[bin_vars].as_matrix(), x_real_test, x_cat_test))

prediction = model.predict(X_test)
id_test = df_test['id'].values

# Create a submission file
sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = prediction[:,1]
sub.to_csv('nn1.csv', index=False)

print(sub.head())