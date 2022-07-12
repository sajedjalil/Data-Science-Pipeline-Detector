
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb
from sklearn import metrics

# Read data
train = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv').drop(['id'], axis=1)
X = train.drop(['target'], axis=1)
y = train[['target']]

# Split data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20,
                                                      random_state=42)

# Impute columns
# Let's use Simple Imputer to get a baseline model
simple_imputer = SimpleImputer(strategy="most_frequent")
simple_imputer.fit(X_train)
X_train_imputed = pd.DataFrame(simple_imputer.transform(X_train))
X_train_imputed.columns = X_train.columns

# Impute validation dataset
X_valid_imputed = pd.DataFrame(simple_imputer.transform(X_valid))
X_valid_imputed.columns = X_valid.columns

# Encode Columns
# Let's use One Hot Encoder to get a baseline model
one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
one_hot_encoder.fit(X_train_imputed)
X_train_encoded = one_hot_encoder.transform(X_train_imputed)

# Encode validation dataset
X_valid_encoded = one_hot_encoder.transform(X_valid_imputed)

# Model
# Let's use LightGBM model to get a baseline model
dtrain = lgb.Dataset(X_train_encoded, label=y_train)
dvalid = lgb.Dataset(X_valid_encoded, label=y_valid)
params = {'num_leaves': 64, 'objective': 'binary', 'metric': 'auc'}
num_round = 1000

model = lgb.train(params=params, train_set=dtrain, num_boost_round=num_round,
                  valid_sets=[dvalid], early_stopping_rounds=10,
                  verbose_eval=True)
# Validation AUC score: aprox 0.751386

# Predictions on the test dataset
test = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv')
test_imputed = simple_imputer.transform(test.drop(['id'], axis=1))
test_encoded = one_hot_encoder.transform(test_imputed)
predictions_array = np.round(model.predict(test_encoded))
predictions_df = pd.DataFrame(data=predictions_array, index=test.index,
                              columns=['target'])
predictions_df = predictions_df.join(test['id'])[['id', 'target']]

predictions_df.to_csv('submission.csv', index=False)
