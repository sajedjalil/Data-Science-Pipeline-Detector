####The entire data manipulation was stolen from the script from Dune_dweller
##https://www.kaggle.com/dvasyukova/talkingdata-mobile-user-demographics/a-linear-model-on-apps-and-labels

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.optimizers import SGD

from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss

import os
from scipy.sparse import csr_matrix, hstack


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

datadir = '../input'
#datadir = '/home/username/projects/talkingData/input'
ga_train = pd.read_csv(os.path.join(datadir,'gender_age_train.csv'), index_col='device_id')
ga_test = pd.read_csv(os.path.join(datadir,'gender_age_test.csv'), index_col = 'device_id')
phone = pd.read_csv(os.path.join(datadir,'phone_brand_device_model.csv'))
# Get rid of duplicate device ids in phone
phone = phone.drop_duplicates('device_id',keep='first').set_index('device_id')
events = pd.read_csv(os.path.join(datadir,'events.csv'),  parse_dates=['timestamp'], index_col='event_id')
appevents = pd.read_csv(os.path.join(datadir,'app_events.csv'), usecols=['event_id','app_id','is_active'], dtype={'is_active':bool})
applabels = pd.read_csv(os.path.join(datadir,'app_labels.csv'))


### seperate data based on events involved
dd = events.device_id.unique()
gatrain_events_index = ga_train.index.intersection(dd)
gatrain= ga_train[ga_train.index.isin(gatrain_events_index)]
gatest_events_index = ga_test.index.intersection(dd)
gatest = ga_test[ga_test.index.isin(gatest_events_index)]

gatrain['trainrow']=np.arange(gatrain.shape[0])
gatest['testrow'] = np.arange(gatest.shape[0])

# encode brand from object to number
brandencoder = LabelEncoder().fit(phone.phone_brand)
phone['brand'] = brandencoder.transform(phone['phone_brand'])
gatrain['brand'] = phone['brand'] # because phone and gatrain both use device id as index. they can align automatically
gatest['brand'] = phone['brand']
nbrands = len(brandencoder.classes_)

Xtr_brand = csr_matrix((np.ones(gatrain.shape[0]), (gatrain.trainrow, gatrain.brand)),shape=(gatrain.shape[0],nbrands))
Xte_brand = csr_matrix((np.ones(gatest.shape[0]), (gatest.testrow, gatest.brand)),shape=(gatest.shape[0],nbrands))
#Xtr_brand = csr_matrix((np.ones(gatrain.shape[0]), (gatrain.trainrow, gatrain.brand)))
#Xte_brand = csr_matrix((np.ones(gatest.shape[0]), (gatest.testrow, gatest.brand)))
print('Brand features: train shape {}, test shape {}'.format(Xtr_brand.shape, Xte_brand.shape))

# Device model
m = phone.phone_brand.str.cat(phone.device_model)
modelencoder = LabelEncoder().fit(m)
phone['model'] = modelencoder.transform(m)
gatrain['model'] =phone['model']
gatest['model'] = phone['model']
nmodels = len(modelencoder.classes_)


### model&brand overlap between training and testing 
brand_without_training = set(gatest.brand)-set(gatrain.brand)
brand_without_testing = set(gatrain.brand)-set(gatest.brand)
model_without_training = set(gatest.model)-set(gatrain.model)
model_without_testing = set(gatrain.model)-set(gatest.model) 

Xtr_model = csr_matrix((np.ones(gatrain.shape[0]), 
                       (gatrain.trainrow, gatrain.model)),shape=(gatrain.shape[0],nmodels))
Xte_model = csr_matrix((np.ones(gatest.shape[0]),
                       (gatest.testrow, gatest.model)), shape=(gatest.shape[0],nmodels))
print('Model features: train shape {}, test shape {}'.format(Xtr_model.shape, Xte_model.shape))


# App encoding
appencoder = LabelEncoder().fit(appevents.app_id)
appevents['app'] = appencoder.transform(appevents.app_id)
napps = len(appencoder.classes_)
deviceapps = (appevents.merge(events[['device_id']], how='left',left_on='event_id',right_index=True)
                       .groupby(['device_id','app'])['app'].agg(['size'])
                       .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                       .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
                       .reset_index())

appevents.merge(events[['device_id']], how='left',left_on='event_id',right_index=True)

d = deviceapps.dropna(subset=['trainrow'])
Xtr_app = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.app)), 
                      shape=(gatrain.shape[0],napps))
d = deviceapps.dropna(subset=['testrow'])
Xte_app = csr_matrix((np.ones(d.shape[0]), (d.testrow, d.app)), 
                      shape=(gatest.shape[0],napps))
print('Apps data: train shape {}, test shape {}'.format(Xtr_app.shape, Xte_app.shape))

#Label encoding
applabels = applabels[applabels.app_id.isin(appevents.app_id.unique())]
applabels['app'] = appencoder.transform(applabels.app_id)

labelencoder = LabelEncoder().fit(applabels.label_id)
applabels['label'] = labelencoder.transform(applabels.label_id)

nlabels = len(labelencoder.classes_)

devicelabels = (deviceapps[['device_id','app']]
                .merge(applabels[['app','label']])
                .groupby(['device_id','label'])['app'].agg(['size'])
                .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
                .reset_index())

d = devicelabels.dropna(subset=['trainrow'])
Xtr_label = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.label)), 
                      shape=(gatrain.shape[0],napps))
d = devicelabels.dropna(subset=['testrow'])
Xte_label = csr_matrix((np.ones(d.shape[0]), (d.testrow, d.label)), 
                      shape=(gatest.shape[0],napps))
print('Apps data: train shape {}, test shape {}'.format(Xtr_label.shape, Xte_label.shape))


# Concatenate all features

Xtrain = hstack((Xtr_brand, Xtr_model, Xtr_app, Xtr_label), format='csr')
Xtest =  hstack((Xte_brand, Xte_model, Xte_app, Xte_label), format='csr')
print('All features: train shape {}, test shape {}'.format(Xtrain.shape, Xtest.shape))

#################
# Start modeling
#################

targetencoder = LabelEncoder().fit(gatrain.group)
y = targetencoder.transform(gatrain.group)
nclasses = len(targetencoder.classes_)
dummy_y = np_utils.to_categorical(y) ## Funcion de Keras!


def batch_generator(X, y, batch_size, shuffle):
    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

def batch_generatorp(X, batch_size, shuffle):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0

# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    #model.add(Dense(10, input_dim=Xtrain.shape[1], init='normal', activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(50, input_dim=Xtrain.shape[1], init='normal', activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(12, init='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])  #logloss
    return model

model=baseline_model()

X_train, X_val, y_train, y_val = train_test_split(Xtrain, dummy_y, test_size=0.002, random_state=42)

fit= model.fit_generator(generator=batch_generator(X_train, y_train, 400, True),
                         nb_epoch=15,
                         samples_per_epoch=70496,
                         validation_data=(X_val.todense(), y_val), verbose=2
                         )

# evaluate the model
scores_val = model.predict_generator(generator=batch_generatorp(X_val, 400, False), val_samples=X_val.shape[0])
scores = model.predict_generator(generator=batch_generatorp(Xtest, 800, False), val_samples=Xtest.shape[0])

print('logloss val {}'.format(log_loss(y_val, scores_val)))

#Scaling to 1-0 probs
#for i in xrange(Xtest.shape[0]):
#    scores2[i,]=scores[i,]/sum(scores[i,])

pred = pd.DataFrame(scores, index = gatest.index, columns=targetencoder.classes_)
#pred.head()


pred.to_csv('Keras on apps -2.csv',index=True)