import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

train = pd.read_csv('./data/train.csv')
y_raw = train.pop('species')
le = LabelEncoder()
y = le.fit(y_raw).transform(y_raw)
classes = le.classes_
n_classes = len(classes)
Y_train = np_utils.to_categorical(y)

train_ids = train.pop('id')
test = pd.read_csv('./data/test.csv')
test_ids = test.pop('id')
X_train = StandardScaler().fit(train).transform(train)
X_test = StandardScaler().fit(test).transform(test)
print(X_train.shape)
input_dim = X_train.shape[1]

def create_model(dropout_rate=0.0, neurons=128, second_layer=True):
    model = Sequential()
    model.add(Dense(neurons,input_dim=input_dim))
    model.add(Dropout(dropout_rate))
    model.add(Activation('relu'))
    if second_layer:
        model.add(Dense(neurons//2,input_dim=input_dim))
        model.add(Dropout(dropout_rate))
        model.add(Activation('relu'))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

early_stopper=EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
model = KerasClassifier(build_fn=create_model, nb_epoch=8, batch_size=99, verbose=0)

# # Which parameters we choose here is more a gut decision.
param_grid = dict(
    neurons=[input_dim * 2**k for k in range(1,5)],
    dropout_rate=[0.0, 0.2, 0.4, 0.6, 0.8],
    second_layer=[True, False],
    )

# # When running locally, which n_jobs setting is faster depends on the backend, openmp support, number of processors etc.
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, verbose=1, scoring='neg_log_loss')
# Can't run this on kaggle, outside of running time restrictions
# grid_result = grid.fit(X_train, Y_train)

# Here's the result
# print("Best estimator: " + str(grid.best_params_))
# model=grid.best_estimator_
# Best estimator: {'second_layer': False, 'dropout_rate': 0.2, 'neurons': 3072}
model = create_model(dropout_rate=0.2, neurons=3072, second_layer=False)
# Train on all data using early stopping to prevent overfitting
early_stopper=EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
model.fit(X_train, Y_train, nb_epoch=100, batch_size=99, verbose=1,
    validation_split=0.15, shuffle=True, callbacks=[early_stopper])
# This returns a leaderboard rating of, which is ok I think for not adding any new features from the images
yPred = model.predict_proba(X_test)
yPred = pd.DataFrame(yPred,index=test_ids,columns=classes)
fp = open('submission.csv','w')
fp.write(yPred.to_csv())