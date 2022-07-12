## Measure execution time, becaus Kaggle cloud fluctuates  
import time
start = time.time()

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

train = pd.read_csv('../input/train.csv')
y_raw = train.pop('species')
le = LabelEncoder()
y = le.fit(y_raw).transform(y_raw)
classes = le.classes_
n_classes = len(classes)
Y_train = np_utils.to_categorical(y)

train_ids = train.pop('id')
test = pd.read_csv('../input/test.csv')
test_ids = test.pop('id')
X_train = StandardScaler().fit(train).transform(train)
X_test = StandardScaler().fit(test).transform(test)
print(X_train.shape)
input_dim = X_train.shape[1]

def create_model(dropout_rate=0.0, neurons=4, second_layer=True):
    model = Sequential()
    model.add(Dense(neurons,input_dim=input_dim))
    model.add(Dropout(dropout_rate))
    model.add(Activation('relu'))
    if second_layer:
        model.add(Dense(neurons//2,input_dim=input_dim))
        model.add(Dropout(dropout_rate))
        model.add(Activation('sigmoid'))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
    return model

early_stopper=EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')
model = KerasClassifier(build_fn=create_model, nb_epoch=1, batch_size=99, verbose=0)

# # Which parameters we choose here is more a gut decision.

## best_model = grid_result.best_estimator_.model
## param_grid = dict(optimizer=['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'])

param_grid = dict(
    neurons=[input_dim * 2**k for k in range(3,6)],
    dropout_rate=[0.2, 0.3],
    second_layer=[True, False],
    )

## When running locally, which n_jobs setting is faster depends on the backend, openmp support, number of processors etc.
## scoring='neg_log_loss' not log_loss is deprecated
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, verbose=0, scoring='neg_log_loss')
grid_result = grid.fit(X_train, Y_train)

print("-----------------------------------------------------------------------------")
# Here's the result
print("Best estimator: " + str(grid.best_params_))
model=grid.best_estimator_
print("-----------------------------------------------------------------------------")
print("All grid estimators: " + str(grid.cv_results_))

best_model = grid.best_estimator_.model
print(str(best_model))

print("-----------------------------------------------------------------------------")
# Best estimator: {'second_layer': False, 'dropout_rate': 0.2, 'neurons': 3072}
# model = create_model(dropout_rate=0.2, neurons=3072, second_layer=False)

# Train on all data using early stopping to prevent overfitting
early_stopper=EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')
model.fit(X_train, Y_train, nb_epoch=200, batch_size=99, verbose=1,
    validation_split=0.1, shuffle=True, callbacks=[early_stopper])

# This returns a leaderboard rating of, which is ok I think for not adding any new features from the images
yPred = model.predict_proba(X_test)
yPred = pd.DataFrame(yPred,index=test_ids,columns=classes)
fp = open('submission.csv','w')
fp.write(yPred.to_csv())
print('finished.')

## print run time
end = time.time()
print()
print(round((end-start),2), "seconds")
