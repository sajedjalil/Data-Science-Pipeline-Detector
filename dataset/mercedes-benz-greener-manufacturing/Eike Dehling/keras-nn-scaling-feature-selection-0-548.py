import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, InputLayer, GaussianNoise
from keras.wrappers.scikit_learn import KerasRegressor

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


#
# Data preparation
#
y_train = train['y'].values
id_test = test['ID']

num_train = len(train)
df_all = pd.concat([train, test])
df_all.drop(['ID', 'y'], axis=1, inplace=True)

# One-hot encoding of categorical/strings
df_all = pd.get_dummies(df_all, drop_first=True)

# Sscaling features
scaler = RobustScaler()
df_all = scaler.fit_transform(df_all)

train = df_all[:num_train]
test = df_all[num_train:]

# Keep only the most contributing features
sfm = SelectFromModel(LassoCV())
sfm.fit(train, y_train)
train = sfm.transform(train)
test = sfm.transform(test)

print ('Number of features : %d' % train.shape[1])

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true - y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def build_model_fn(neurons=20, noise=0.25):
    model = Sequential()
    model.add(InputLayer(input_shape=(train.shape[1],)))
    model.add(GaussianNoise(noise))
    model.add(Dense(neurons, activation='tanh'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='nadam', metrics=[r2_keras])
    return model
    
    
#
# Tuning model parameters
#
model = KerasRegressor(build_fn=build_model_fn, epochs=75, verbose=0)

gsc = GridSearchCV(
    estimator=model,
    param_grid={
        #'neurons': range(18,31,4),
        'noise': [x/20.0 for x in range(3, 7)],
    },
    #scoring='r2',
    scoring='neg_mean_squared_error',
    cv=5
)

grid_result = gsc.fit(train, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
for test_mean, test_stdev, train_mean, train_stdev, param in zip(
        grid_result.cv_results_['mean_test_score'],
        grid_result.cv_results_['std_test_score'],
        grid_result.cv_results_['mean_train_score'],
        grid_result.cv_results_['std_train_score'],
        grid_result.cv_results_['params']):
    print("Train: %f (%f) // Test : %f (%f) with: %r" % (train_mean, train_stdev, test_mean, test_stdev, param))
    
    
#
# Train model with best params for submission
#
model = build_model_fn(**grid_result.best_params_)

model.fit(train, y_train, epochs=75, verbose=2)

y_test = model.predict(test).flatten()

df_sub = pd.DataFrame({'ID': id_test, 'y': y_test})
df_sub.to_csv('mercedes-submission.csv', index=False)