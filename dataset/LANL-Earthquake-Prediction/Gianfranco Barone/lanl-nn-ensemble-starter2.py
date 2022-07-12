# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

from sklearn.model_selection import train_test_split

# Features are based on Andrew Lukyanenko's kernel at https://www.kaggle.com/artgor/earthquakes-fe-more-features-and-samples
Xtrain = pd.read_csv('../input/lanl-competition-fe-v1/x_train')
Ytrain = pd.read_csv('../input/lanl-competition-fe-v1/y_train')
Xtest = pd.read_csv('../input/lanl-competition-fe-v1/x_test')


submission = pd.read_csv('../input/LANL-Earthquake-Prediction/sample_submission.csv', index_col='seg_id', dtype={"time_to_failure": np.float32})


def get_model(rand_state=1):
    X_train, X_val, Y_train, Y_val = train_test_split(Xtrain, Ytrain, test_size=0.2, random_state=rand_state)
    
    #Libraries for neural net
    import keras
    from keras.layers import Dense
    from keras.layers import BatchNormalization
    from keras.layers import Activation
    from keras.layers import Dropout
    from keras.models import Sequential
    from keras import optimizers
    from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
    
    #Model parameters
    
    kernel_init = 'he_normal'
    input_size = len(Xtrain.columns)
    
    
    ### Neural Network ###
    
    # Model architecture
    model = Sequential()
    #model.add(Dropout(0.4, input_shape=(input_size,)))
    model.add(Dense(16, input_dim = input_size)) 
    model.add(Activation('linear'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(32))
    model.add(Activation('tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(32))
    model.add(Activation('tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(1))    
    model.add(Activation('linear'))
    
    #compile the model
    optim = optimizers.Adam(lr = 0.0075)
    #optim = optimizers.Nadam()
    model.compile(loss = 'mean_absolute_error', optimizer = optim)
    
    #Callbacks
    csv_logger = CSVLogger('log.csv', append=True, separator=';')
    #best_model = ModelCheckpoint("model.hdf5", save_best_only=True, period=3)
    restore_best = EarlyStopping(monitor='val_loss', verbose=2, patience=99, restore_best_weights=True)
    
    model.fit(x=X_train, y=Y_train, batch_size=64, epochs=300, verbose=2, callbacks=[restore_best], validation_data=(X_val,Y_val))
    ### Neural Network End ###
    
    val_loss = model.evaluate(X_val, Y_val, batch_size=64)
    loss = model.evaluate(X_train, Y_train, batch_size=64)
    print(rand_state, loss, val_loss)
    log.append([rand_state, val_loss, loss])
    return [model, loss, val_loss]

#Ensemble
log = []
ensemble_dim = 160    
ensemble = []
for i in range(ensemble_dim):
    ensemble.append(get_model(i))
   

models = [i[0] for i in ensemble]    
losses = [i[1] for i in ensemble]
val_losses = [i[2] for i in ensemble]
print(val_losses)
av_val_loss = np.mean(val_losses)
std_val_loss = np.std(val_losses)
av_loss = np.mean(losses)
print('Average val_loss: '+ str(av_val_loss))
print(losses)
print('Average loss: '+ str(av_loss))

log = pd.DataFrame(log, columns=['state', 'validation_loss', 'loss']) 

ensemble_prediction = np.zeros(1,)
for model in models:
    ensemble_prediction = ensemble_prediction + model.predict(Xtest, verbose = 0, batch_size = 64)
nn_predictions = ensemble_prediction/ensemble_dim
#Ensemble end

print(nn_predictions)
submission['time_to_failure'] = nn_predictions
submission.to_csv('submission.csv')
flog = log.sort_values(by=['validation_loss'])
flog.loc[ensemble_dim] = ['Stats', av_val_loss, std_val_loss]
print(flog)
flog.to_csv('log.csv', index=False)

