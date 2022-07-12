import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.model_selection import KFold
from sklearn import *
from keras.layers import Input, Dense, Dropout, BatchNormalization, LSTM, Conv1D, Activation
from keras.models import Sequential, load_model
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
import os
import matplotlib.pyplot as plt
from pylab import savefig

import math as mt

seed = 42
np.random.seed(seed)
model_path = "keras_model.h5"
from sklearn.model_selection import KFold, train_test_split 
from sklearn.metrics import r2_score, mean_squared_error
print('DL approch!!')
print("Load data.....")
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
subm = pd.read_csv('../input/sample_submission.csv')
print("Train shape: {}\nTest shape: {}".format(train.shape, test.shape))
col = [c for c in train.columns if c not in ['ID', 'target']]

scl = preprocessing.StandardScaler()
def rmsle(y, pred):
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(pred), 2)))

x1, x2, y1, y2 = model_selection.train_test_split(train[col], train.target.values, test_size=0.20, random_state=5)
model = ensemble.RandomForestRegressor(n_jobs = -1, random_state = 7)
model.fit(scl.fit_transform(x1), y1)
print(rmsle(y2, model.predict(scl.transform(x2))))

# first 1000 features..
col = pd.DataFrame({'importance': model.feature_importances_, 'feature': col}).sort_values(by=['importance'], ascending=[False])[:1000]['feature'].values

#Added Columns from feature_selection
train = train[['ID', 'target']+list(col)]
test = test[['ID']+list(col)]
print("Train shape: {}\nTest shape: {}".format(train.shape, test.shape))

PERC_TRESHOLD = 0.98   ### Percentage of zeros in each feature ###
N_COMP = 20            ### Number of decomposition components ###

target = np.log1p(train['target']).values
cols_to_drop = [col for col in train.columns[2:]
                    if [i[1] for i in list(train[col].value_counts().items()) 
                    if i[0] == 0][0] >= train.shape[0] * PERC_TRESHOLD]

print("Define training features...")
exclude_other = ['ID', 'target']
train_features = []
for c in train.columns:
    if c not in cols_to_drop \
    and c not in exclude_other:
        train_features.append(c)
print("Number of featuress for training: %s" % len(train_features))

train, test = train[train_features], test[train_features]
print("\nTrain shape: {}\nTest shape: {}".format(train.shape, test.shape))

print("\nStart decomposition process...")
print("PCA")
pca = PCA(n_components=N_COMP, random_state=17)
pca_results_train = pca.fit_transform(train)
pca_results_test = pca.transform(test)

print("tSVD")
tsvd = TruncatedSVD(n_components=N_COMP, random_state=17)
tsvd_results_train = tsvd.fit_transform(train)
tsvd_results_test = tsvd.transform(test)

print("ICA")
ica = FastICA(n_components=N_COMP, random_state=17)
ica_results_train = ica.fit_transform(train)
ica_results_test = ica.transform(test)

print("GRP")
grp = GaussianRandomProjection(n_components=N_COMP, eps=0.1, random_state=17)
grp_results_train = grp.fit_transform(train)
grp_results_test = grp.transform(test)

print("SRP")
srp = SparseRandomProjection(n_components=N_COMP, dense_output=True, random_state=17)
srp_results_train = srp.fit_transform(train)
srp_results_test = srp.transform(test)

print("Append decomposition components to datasets...")
for i in range(1, N_COMP + 1):
    train['pca_' + str(i)] = pca_results_train[:, i - 1]
    test['pca_' + str(i)] = pca_results_test[:, i - 1]

    train['ica_' + str(i)] = ica_results_train[:, i - 1]
    test['ica_' + str(i)] = ica_results_test[:, i - 1]

    train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
    test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]

    train['grp_' + str(i)] = grp_results_train[:, i - 1]
    test['grp_' + str(i)] = grp_results_test[:, i - 1]

    train['srp_' + str(i)] = srp_results_train[:, i - 1]
    test['srp_' + str(i)] = srp_results_test[:, i - 1]
print('\nTrain shape: {}\nTest shape: {}'.format(train.shape, test.shape))

print("\nStart decomposition process...")
print("PCA")
pca = PCA(n_components=N_COMP, random_state=17)
pca_results_train = pca.fit_transform(train)
pca_results_test = pca.transform(test)

print("tSVD")
tsvd = TruncatedSVD(n_components=N_COMP, random_state=17)
tsvd_results_train = tsvd.fit_transform(train)
tsvd_results_test = tsvd.transform(test)

print("ICA")
ica = FastICA(n_components=N_COMP, random_state=17)
ica_results_train = ica.fit_transform(train)
ica_results_test = ica.transform(test)

print("GRP")
grp = GaussianRandomProjection(n_components=N_COMP, eps=0.1, random_state=17)
grp_results_train = grp.fit_transform(train)
grp_results_test = grp.transform(test)

print("SRP")
srp = SparseRandomProjection(n_components=N_COMP, dense_output=True, random_state=17)
srp_results_train = srp.fit_transform(train)
srp_results_test = srp.transform(test)

print("Append decomposition components to datasets...")
for i in range(1, N_COMP + 1):
    train['pca_' + str(i)] = pca_results_train[:, i - 1]
    test['pca_' + str(i)] = pca_results_test[:, i - 1]

    train['ica_' + str(i)] = ica_results_train[:, i - 1]
    test['ica_' + str(i)] = ica_results_test[:, i - 1]

    train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
    test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]

    train['grp_' + str(i)] = grp_results_train[:, i - 1]
    test['grp_' + str(i)] = grp_results_test[:, i - 1]

    train['srp_' + str(i)] = srp_results_train[:, i - 1]
    test['srp_' + str(i)] = srp_results_test[:, i - 1]
print('\nTrain shape: {}\nTest shape: {}'.format(train.shape, test.shape))

X = train.values
y = target
X_test = test.values

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true - y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

# modeling..
input_dims = train.shape[1] 
print("model building..")
def model():
    model = Sequential()
    
    # model.add(Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform", input_shape=train.shape))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.2))
    
    model.add(Dense(1024, input_dim=input_dims))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Dropout(0.2))
    
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Dropout(0.2))
    
    model.add(Dense(32, activation='tanh'))
    model.add(Dense(1, activation='linear'))
    
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[r2_keras])
    print(model.summary())
    return model

estimator = KerasRegressor(
    build_fn=model, 
    nb_epoch=300, 
    batch_size=500,
    verbose=1
)


def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)
   
def sum_of_square_deviation(numbers,mean):
    return float(1/len(numbers) * sum((x - mean)** 2 for x in numbers))    
  
n_splits = 5
kf = KFold(n_splits=n_splits, random_state=seed, shuffle=True)
kf.get_n_splits(X)

mse_scores = list()
r2_scores = list()
print('5 fold..')
for fold, (train_index, test_index) in enumerate(kf.split(X)):
    
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_tr, X_val = X[train_index], X[test_index]
    y_tr, y_val = y[train_index], y[test_index]

    # prepare callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_r2_keras', 
            patience=20,
            mode='max',
            verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.000001)
    ]
    # fit estimator
    history = estimator.fit(
        X_tr, 
        y_tr, 
        epochs=500,
        validation_data=(X_val, y_val),
        verbose=2,
        callbacks=callbacks,
        shuffle=True
    )
    
    pred = estimator.predict(X_val)
    
    mse = mean_squared_error(y_val, estimator.predict(X_val))**0.5
    r2 = r2_score(y_val, estimator.predict(X_val))
    mse_scores.append(mse)
    r2_scores.append(r2)
    
    print('Fold %d: Mean Squared Error %f'%(fold, mse))
    print('Fold %d: R^2 %f'%(fold, r2))

    #save results
    pred = estimator.predict(X_test)
    subm['target'] = np.expm1(pred)
    x = []
    l = subm['target'].tolist()
    for i in l:
        if  i < 0:
            x.append(0.0000001)
            continue
        x.append(i)
    subm['target'] = x
    subm.to_csv(str(r2)+'_subm_keras.csv', index=False)

mean_mse = mean(mse_scores)
mean_r2 = mean(r2_scores)

standard_deviation_mse = mt.sqrt(sum_of_square_deviation(mse_scores,mean_mse))
standard_deviation_r2 = mt.sqrt(sum_of_square_deviation(r2_scores,mean_r2))

print( 'Mean Squared Error %f'%mean_mse)
print( 'Stdev Squared Error %f'%standard_deviation_mse)
print( 'Mean R^2 %f'%mean_r2)
print( 'Stdev R^2 %f'%standard_deviation_r2)

callbacks = [
    EarlyStopping(
        monitor='val_r2_keras', 
        patience=20,
        mode='max',
        verbose=1),
    ModelCheckpoint(
        model_path, 
        monitor='val_r2_keras', 
        save_best_only=True, 
        mode='max',
        verbose=0),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.000001)
]

# train/validation split
X_tr, X_val, y_tr, y_val = train_test_split(
    X, 
    y, 
    test_size=0.05, 
    random_state=seed
)

print('fitting..')
history = estimator.fit(
    X_tr, 
    y_tr, 
    epochs=500,
    validation_data=(X_val, y_val),
    verbose=2,
    callbacks=callbacks,
    shuffle=True
)

# list all data in history
print(history.history.keys())

fig_acc = plt.figure(figsize=(10, 10))
plt.plot(history.history['r2_keras'])
plt.plot(history.history['val_r2_keras'])
plt.title('model accuracy')
plt.ylabel('R^2')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
fig_acc.savefig("model_accuracy.png")

# summarize history for loss
fig_loss = plt.figure(figsize=(10, 10))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
fig_loss.savefig("model_loss.png")

#if os.path.isfile(model_path):
#    estimator = load_model(model_path, custom_objects={'r2_keras': r2_keras})

predicted = estimator.predict(X_val)
fig_verify = plt.figure(figsize=(100, 50))
plt.plot(predicted, color="blue")
plt.plot(y_val, color="green")
plt.title('prediction')
plt.ylabel('value')
plt.xlabel('row')
plt.legend(['predicted', 'actual data'], loc='upper left')
plt.show()
fig_verify.savefig("model_verify.png")

print('MSE train: {}'.format(mean_squared_error(y_tr, estimator.predict(X_tr))**0.5)) # mse train
print('R^2 train: {}'.format(r2_score(y_tr, estimator.predict(X_tr)))) # R^2 train

# check performance on validation set
print('MSE val: {}'.format(mean_squared_error(y_val, estimator.predict(X_val))**0.5)) # mse val
print('R^2 val: {}'.format(r2_score(y_val, estimator.predict(X_val)))) # R^2 val

res = estimator.predict(X_test).ravel()
subm['target'] = np.expm1(res)
l = subm['target'].tolist()
x = []
for i in l:
    if i < 0:
        x.append(0.0000001)
        continue
    x.append(i)
subm['target'] = x
subm.to_csv('keras_baseline.csv', index=False)
# estimator.save(model_path)
