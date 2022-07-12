import numpy as np
seed = 7
np.random.seed(seed)

import pandas as pd
import pandas.core.algorithms as algos

from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
#from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import os
from scipy.sparse import csr_matrix, hstack

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.layers.advanced_activations import PReLU

# load dataset

datadir = '../input'
#datadir = '/home/username/projects/talkingData/input'
#datadir = 'C:\\mthayer\\competition\\talkingData\\input'
gatrain = pd.read_csv(os.path.join(datadir,'gender_age_train.csv'), index_col='device_id')
gatest = pd.read_csv(os.path.join(datadir,'gender_age_test.csv'), index_col = 'device_id')
phone = pd.read_csv(os.path.join(datadir,'phone_brand_device_model.csv'))
# Get rid of duplicate device ids in phone
phone = phone.drop_duplicates('device_id',keep='first').set_index('device_id')
events = pd.read_csv(os.path.join(datadir,'events.csv'),  parse_dates=['timestamp'], index_col='event_id')
appevents = pd.read_csv(os.path.join(datadir,'app_events.csv'), usecols=['event_id','app_id','is_active'], dtype={'is_active':bool})
applabels = pd.read_csv(os.path.join(datadir,'app_labels.csv'))


##Load the CV split.
#datadir = 'C:\\mthayer\\competition\\talkingData\\tdCode'
folds=pd.read_csv(os.path.join(datadir,"folds_10.csv"), index_col='device_id')

##Reorder train and cv so the device ids match afterwards
gatrain=gatrain.sort_index()
folds=folds.sort_index()

print("validation, must be zero!", sum(gatrain.index!=folds.index))

####Phone brand
#As preparation I create two columns that show which train or test set row a particular device_id belongs to.

gatrain['trainrow'] = np.arange(gatrain.shape[0])
gatest['testrow'] = np.arange(gatest.shape[0])

# A sparse matrix of features can be constructed in various ways. I use this constructor:
# csr_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
# where ``data``, ``row_ind`` and ``col_ind`` satisfy the
# relationship ``a[row_ind[k], col_ind[k]] = data[k]``
#
# It lets me specify which values to put into which places in a sparse matrix. For phone brand data the data array will be all ones,
# row_ind will be the row number of a device and col_ind will be the number of brand.

brandencoder = LabelEncoder().fit(phone.phone_brand)
phone['brand'] = brandencoder.transform(phone['phone_brand'])
gatrain['brand'] = phone['brand']
gatest['brand'] = phone['brand']
Xtr_brand = csr_matrix((np.ones(gatrain.shape[0]),
                       (gatrain.trainrow, gatrain.brand)))
Xte_brand = csr_matrix((np.ones(gatest.shape[0]),
                       (gatest.testrow, gatest.brand)))
print('Brand features: train shape {}, test shape {}'.format(Xtr_brand.shape, Xte_brand.shape))

# Device model
m = phone.phone_brand.str.cat(phone.device_model)

modelencoder = LabelEncoder().fit(m)
phone['model'] = modelencoder.transform(m)
gatrain['model'] = phone['model']
gatest['model'] = phone['model']

Xtr_model = csr_matrix((np.ones(gatrain.shape[0]),
                       (gatrain.trainrow, gatrain.model)))
Xte_model = csr_matrix((np.ones(gatest.shape[0]),
                       (gatest.testrow, gatest.model)))
print('Model features: train shape {}, test shape {}'.format(Xtr_model.shape, Xte_model.shape))

###############################################
###MT work: Frequency term:
# model_freq = (phone
#                 .groupby(['model'])['model'].count()
#                 .to_frame())
# model_freq.columns.values[0]='model_freq'

model_freq = phone["model"].value_counts().to_frame()
mf_encoder = LabelEncoder().fit(model_freq.model)
model_freq['model_freq']=mf_encoder.transform(model_freq['model'])
model_freq= model_freq.drop("model", 1)

gatrain=gatrain.merge(model_freq, how='left', left_on="model", right_index=True)
gatest=gatest.merge(model_freq, how='left', left_on="model", right_index=True)
gatest["model_freq"]=gatest["model_freq"].fillna(1) # fill not found frequencies with 1


Xtr_model_freq = csr_matrix((np.ones(gatrain.shape[0]),
                       (gatrain.trainrow, gatrain["model_freq"])))
Xte_model_freq = csr_matrix((np.ones(gatest.shape[0]),
                       (gatest.testrow, gatest["model_freq"])))

print('Model frequency features: train shape {}, test shape {}'.format(Xtr_model_freq.shape, Xte_model_freq.shape))

brand_freq = phone["brand"].value_counts().to_frame()
bf_encoder = LabelEncoder().fit(brand_freq.brand)
brand_freq['brand_freq']=bf_encoder.transform(brand_freq['brand'])
brand_freq= brand_freq.drop("brand", 1)

brand_freq.columns.values[0]='brand_freq'
gatrain=gatrain.merge(brand_freq, how='left', left_on="brand", right_index=True)
gatest=gatest.merge(brand_freq, how='left', left_on="brand", right_index=True)
gatest["brand_freq"]=gatest["brand_freq"].fillna(1) # fill not found frequencies with 1

Xtr_brand_freq = csr_matrix((np.ones(gatrain.shape[0]),
                       (gatrain.trainrow, gatrain.brand_freq)))

Xte_brand_freq = csr_matrix((np.ones(gatest.shape[0]),
                       (gatest.testrow, gatest.brand_freq)))

print('Brand frequency features: train shape {}, test shape {}'.format(Xtr_brand_freq.shape, Xte_brand_freq.shape))


#############################################

# Installed apps features
# For each device I want to mark which apps it has installed. So I'll have as many feature columns as there are distinct apps.
# Apps are linked to devices through events. So I do the following:
# merge device_id column from events table to app_events
# group the resulting dataframe by device_id and app and aggregate
# merge in trainrow and testrow columns to know at which row to put each device in the features matrix

appencoder = LabelEncoder().fit(appevents.app_id)
appevents['app'] = appencoder.transform(appevents.app_id)
napps = len(appencoder.classes_)
deviceapps = (appevents.merge(events[['device_id']], how='left',left_on='event_id',right_index=True)
                       .groupby(['device_id','app'])['app'].agg(['max'])
                       .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                       .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
                       .reset_index())

d = deviceapps.dropna(subset=['trainrow'])
Xtr_app = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.app)),
                      shape=(gatrain.shape[0],napps))
d = deviceapps.dropna(subset=['testrow'])
Xte_app = csr_matrix((np.ones(d.shape[0]), (d.testrow, d.app)),
                      shape=(gatest.shape[0],napps))
print('Apps data: train shape {}, test shape {}'.format(Xtr_app.shape, Xte_app.shape))

# App labels features
# These are constructed in a way similar to apps features by merging app_labels with the deviceapps dataframe we created above.
applabels = applabels.loc[applabels.app_id.isin(appevents.app_id.unique())]
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
devicelabels.head()

d = devicelabels.dropna(subset=['trainrow'])
Xtr_label = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.label)),
                      shape=(gatrain.shape[0],nlabels))
d = devicelabels.dropna(subset=['testrow'])
Xte_label = csr_matrix((np.ones(d.shape[0]), (d.testrow, d.label)),
                      shape=(gatest.shape[0],nlabels))
print('Labels data: train shape {}, test shape {}'.format(Xtr_label.shape, Xte_label.shape))

events_cout = (events.groupby('device_id')['timestamp'].agg(['size'])
                    .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                    .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
                    .reset_index())
events_cout.size = (np.log((events_cout['size'])))
events_cout.size = events_cout.size/events_cout.size.max()

d = events_cout.dropna(subset=['trainrow'])
Xtr_eventsize = csr_matrix((d.iloc[:,1], (d.trainrow, np.zeros(d.shape[0]))),
                      shape=(gatrain.shape[0],1))

d = events_cout.dropna(subset=['testrow'])
Xte_eventsize = csr_matrix((d.iloc[:,1], (d.testrow, np.zeros(d.shape[0]))),
                      shape=(gatest.shape[0],1))
print('Labels data: train shape {}, test shape {}'.format(Xtr_eventsize.shape, Xte_eventsize.shape))

events['hour'] = events.timestamp.apply(lambda x: x.hour)
events_cout_hourofday = (events.groupby(['device_id','hour'])['hour'].agg(['size'])
                    .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                    .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
                    .reset_index())
d = events_cout_hourofday.dropna(subset=['trainrow'])
Xtr_event_on_hourofday = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.hour)),
                      shape=(gatrain.shape[0],d.hour.nunique()))

d = events_cout_hourofday.dropna(subset=['testrow'])
Xte_event_on_hourofday = csr_matrix((np.ones(d.shape[0]), (d.testrow, d.hour)),
                      shape=(gatest.shape[0],d.hour.nunique()))
print('Labels data: train shape {}, test shape {}'.format(Xtr_event_on_hourofday.shape, Xte_event_on_hourofday.shape))

#####
# Spliting features
#####

### Create flag "with event":

events_grouped = (events.groupby(['device_id'], as_index=False).agg(
    {'timestamp':'count'}))
events_grouped.columns = ['device_id','with_event']
events_grouped['with_event']=1
events_grouped=events_grouped.set_index("device_id")

tr_with_event=pd.merge(gatrain[[]], events_grouped, how="left", left_index=True, right_index=True)
te_with_event=pd.merge(gatest[[]], events_grouped, how="left", left_index=True, right_index=True)
tr_with_event["with_event"]=tr_with_event["with_event"].fillna(0)
te_with_event["with_event"]=te_with_event["with_event"].fillna(0)

print("validation for tr_with_event, must be zero!", sum(gatrain.index!=tr_with_event.index))
print("validation for te_with_event, must be zero!", sum(gatest.index!=te_with_event.index))

#Add to features just in case
Xtr_with_event = csr_matrix((np.ones(gatrain.shape[0]),
                           (gatrain.trainrow, tr_with_event["with_event"])))
Xte_with_event = csr_matrix((np.ones(gatest.shape[0]),
                           (gatest.testrow, te_with_event["with_event"])))

########## End of Splitting

##################
#   App Labels
##################

print("# Read App Labels")
app_lab = pd.read_csv("../input/app_labels.csv")
app_lab = app_lab.groupby("app_id")["label_id"].apply(
    lambda x: " ".join(str(s) for s in x))

##################
#   App Events
##################
print("# Read App Events")
app_ev = pd.read_csv("../input/app_events.csv")
app_ev["app_lab"] = app_ev["app_id"].map(app_lab)
app_ev = app_ev.groupby("event_id")["app_lab"].apply(
    lambda x: " ".join(str(s) for s in x))

del app_lab

##################
#     Events
##################
print("# Read Events")
events = pd.read_csv("../input/events.csv")
events["app_lab"] = events["event_id"].map(app_ev)
events = events.groupby("device_id")["app_lab"].apply(
    lambda x: " ".join(str(s) for s in x))

del app_ev

##################
#   Phone Brand
##################
print("# Read Phone Brand")
pbd = pd.read_csv("../input/phone_brand_device_model.csv")
pbd.drop_duplicates('device_id', keep='first', inplace=True)


##################
#  Train and Test
##################
print("# Generate Train and Test")
#train = pd.read_csv("../input/gender_age_train.csv")
train = pd.read_csv(os.path.join(datadir,'gender_age_train.csv'), index_col='device_id')

train["dev_id"]=train.index
train["app_lab"] = train["dev_id"].map(events)
# train = pd.merge(train, pbd, how='left',
#                  on='device_id', left_index=True)
train=pd.merge(train, pbd, how='left', left_index=True, right_on="device_id")
train.index=train["dev_id"]

train=train.sort_index()

print("Before hash: must be zero: ", sum(train.index != gatrain.index))
#print("Before hash: must be zero: ", sum(train["dev_id"] != gatrain.index))


# test = pd.read_csv("../input/gender_age_test.csv",
#                    dtype={'device_id': np.str})
test = pd.read_csv(os.path.join(datadir,'gender_age_test.csv'), index_col='device_id')
test["dev_id"]=test.index

test["app_lab"] = test["dev_id"].map(events)
# test = pd.merge(test, pbd, how='left',
#                 on='device_id', left_index=True)
test=pd.merge(test, pbd, how='left', left_index=True, right_on="device_id")

del pbd
del events


####Phone brand
#
def get_hash_data(train, test):
    df = pd.concat((train, test), axis=0, ignore_index=True)
    split_len = len(train)

    # TF-IDF Feature
    tfv = TfidfVectorizer(min_df=1)
    df = df[["phone_brand", "device_model", "app_lab"]].astype(np.str).apply(
        lambda x: " ".join(s for s in x), axis=1).fillna("Missing")
    df_tfv = tfv.fit_transform(df)

    train = df_tfv[:split_len, :]
    test = df_tfv[split_len:, :]
    return train, test

def get_hash_data2(train, test):
    df = pd.concat((train, test), axis=0, ignore_index=True)
    split_len = len(train)

    # TF-IDF Feature
    tfv = TfidfVectorizer(min_df=1)
    df = df[["phone_brand", "device_model"]].astype(np.str).apply(
        lambda x: " ".join(s for s in x), axis=1).fillna("Missing")
    df_tfv = tfv.fit_transform(df)

    train = df_tfv[:split_len, :]
    test = df_tfv[split_len:, :]
    return train, test


trainrow = np.arange(train.shape[0])
testrow = np.arange(test.shape[0])
superrow= np.arange(train.shape[0]+ test.shape[0])

train_device_id = train["device_id"].values
test_device_id = test["device_id"].values

train_bag, test_bag = get_hash_data(train,test)

#bags only brand and model:
train_bag2, test_bag2 = get_hash_data2(train,test)


del train
del test

print("After hash: must be zero: ", sum(train_device_id != gatrain.index))


Xtrain = hstack((Xtr_brand, Xtr_model, Xtr_brand_freq, Xtr_model_freq, Xtr_app, Xtr_label,Xtr_eventsize,Xtr_event_on_hourofday,
                 train_bag, Xtr_with_event), format='csr')
Xtest =  hstack((Xte_brand, Xte_model, Xte_brand_freq, Xte_model_freq, Xte_app, Xte_label,Xte_eventsize,Xte_event_on_hourofday,
                 test_bag, Xte_with_event), format='csr')


Xtrain_ne = hstack((Xtr_brand, Xtr_model, Xtr_brand_freq, Xtr_model_freq, train_bag2, Xtr_with_event), format='csr')
Xtest_ne =  hstack((Xte_brand, Xte_model, Xte_brand_freq, Xte_model_freq, test_bag2, Xte_with_event), format='csr')

print('All features: train shape {}, test shape {}'.format(Xtrain.shape, Xtest.shape))

# Reduce dimensionality
indices = np.nonzero(Xtrain)
columns_non_unique = indices[1]
unique_columns = sorted(set(columns_non_unique))
Xtrain=Xtrain.tocsc()[:,unique_columns]
Xtest=Xtest.tocsc()[:,unique_columns]

print('All features after dimensionality reduction: train shape {}, test shape {}'.format(Xtrain.shape, Xtest.shape))

#################
# Start modeling
#################

np.random.seed(seed)

targetencoder = LabelEncoder().fit(gatrain.group)
y = targetencoder.transform(gatrain.group)
nclasses = len(targetencoder.classes_)

##Keras stuff
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

def baseline_model(num_columns):
    # create model
    model = Sequential()
    model.add(Dropout(0.4, input_shape=(num_columns,)))
    model.add(Dense(75))
    model.add(PReLU())
    model.add(Dropout(0.30))
    model.add(Dense(50, init='normal', activation='tanh'))
    model.add(PReLU())
    model.add(Dropout(0.20))

    model.add(Dense(12, init='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model

#End of keras stuff


#Create predictions repository:
pred = np.zeros((y.shape[0],nclasses*2))
pred_test = np.zeros((gatest.shape[0],nclasses*2))
n_folds=len(folds["fold"].unique())


for fold_id in xrange(1, n_folds + 1):
    #fold_id=1
    train_id = folds["fold"].values != fold_id
    valid_id = folds["fold"].values == fold_id

    # With no events
    train_id_ne = np.logical_and(train_id, tr_with_event["with_event"].values == 0)
    valid_id_ne = np.logical_and(valid_id, tr_with_event["with_event"].values == 0)
    test_id_ne = te_with_event["with_event"].values == 0

    # With events: Training using only common features
    train_id_we = np.logical_and(train_id, tr_with_event["with_event"].values == 1)
    valid_id_we = np.logical_and(valid_id, tr_with_event["with_event"].values == 1)
    test_id_we = te_with_event["with_event"].values == 1

    # First, train on all data, but only no-events feature. Validate with no events:
    Xtr, Ytr = Xtrain_ne[train_id, :], y[train_id]
    Xva, Yva = Xtrain_ne[valid_id_ne, :], y[valid_id_ne]

    # Logistic regression >
    clf1 = LogisticRegression(C=0.06, multi_class='multinomial', solver='lbfgs')  # 2.38715733092
    # Fitting logistic regression 1
    clf1.fit(Xtr, Ytr)

    # Predicting only in those with no events!
    pred[valid_id_ne, 0:12] = clf1.predict_proba(Xva)
    pred_test[test_id_ne, 0:12] = pred_test[test_id_ne, 0:12] + clf1.predict_proba(Xtest_ne[test_id_ne, :])

    score_val = log_loss(Yva, pred[valid_id_ne, 0:12])
    print("No-events: Logistic logloss for fold {} is {}".format(fold_id, score_val))

    # 2.- After, train only rows with events
    Xtr, Ytr = Xtrain[train_id_we, :], y[train_id_we]
    Xva, Yva = Xtrain[valid_id_we, :], y[valid_id_we]

    clf2 = LogisticRegression(C=0.016, multi_class='multinomial', solver='lbfgs')  # 1.99914889909
    clf2.fit(Xtr, Ytr)

    # Predicting only in those with events!
    pred[valid_id_we, 0:12] = clf2.predict_proba(Xva)
    pred_test[test_id_we, 0:12] = pred_test[test_id_we, 0:12] + clf2.predict_proba(Xtest[test_id_we, :])

    score_val = log_loss(Yva, pred[valid_id_we, 0:12])
    print("With-events: Logistic logloss for fold {} is {}".format(fold_id, score_val))

    Xva, Yva = Xtrain[valid_id, :], y[valid_id]
    score_val = log_loss(Yva, pred[valid_id, 0:12])
    print("Total: Logistic logloss for fold {} is {}".format(fold_id, score_val))

    ## Fitting Keras! ------------------------------------------------------------------>
    # First, train on all data, but only no-events feature. Validate with no events:
    Xtr, Ytr_dum = Xtrain_ne[train_id, :], dummy_y[train_id]
    Xva, Yva_dum = Xtrain_ne[valid_id_ne, :], dummy_y[valid_id_ne]

    model = baseline_model(Xtr.shape[1])
    fit = model.fit_generator(generator=batch_generator(Xtr, Ytr_dum, 381, True),
                              nb_epoch=20,
                              samples_per_epoch=Xtr.shape[0], verbose=2,
                              validation_data=(Xva.todense(), Yva_dum)
                              )

    # evaluate the model
    pred[valid_id_ne, 12:25] = model.predict_generator(generator=batch_generatorp(Xva, 400, False),
                                                       val_samples=Xva.shape[0])
    pred_test[test_id_ne, 12:25] = pred_test[test_id_ne, 12:25] + \
                                   model.predict_generator(
                                       generator=batch_generatorp(Xtest_ne[test_id_ne, :], 400, False),
                                       val_samples=Xtest_ne[test_id_ne, :].shape[0])

    # 2.- After, train all data (keras)
    Xtr, Ytr_dum = Xtrain[train_id, :], dummy_y[train_id]
    Xva, Yva_dum = Xtrain[valid_id_we, :], dummy_y[valid_id_we]

    model = baseline_model(Xtr.shape[1])
    fit = model.fit_generator(generator=batch_generator(Xtr, Ytr_dum, 381, True),
                              nb_epoch=20,
                              samples_per_epoch=Xtr.shape[0], verbose=2,
                              validation_data=(Xva.todense(), Yva_dum)
                              )

    # evaluate the model, and predict only with events:
    pred[valid_id_we, 12:25] = model.predict_generator(generator=batch_generatorp(Xva, 400, False),
                                                       val_samples=Xva.shape[0])
    pred_test[test_id_we, 12:25] = pred_test[test_id_we, 12:25] + \
                                   model.predict_generator(generator=batch_generatorp(Xtest[test_id_we, :], 400, False),
                                                           val_samples=Xtest[test_id_we, :].shape[0])

    # pred_test[test_id_ne,0:12] = pred_test[test_id_ne,0:12] + clf1.predict_proba(Xtest_ne[test_id_ne, :])

    Xva, Yva = Xtrain[valid_id, :], y[valid_id]
    score_val = log_loss(Yva, pred[valid_id, 12:25])
    print("Total: Keras logloss for fold {} is {}".format(fold_id, score_val))

print("## Enf of folds work --------")

col_names=np.concatenate((targetencoder.classes_, targetencoder.classes_), axis=0)

##Averaging predictions for all folds in the test set
pred_test /= float(n_folds)

score_val=log_loss(y, pred[:,0:12])
print("Logistic: logloss for {} folds is {}". format(n_folds, score_val))

sum(pred[6,12:25])
sum(pred_test[1,12:25])


score_val=log_loss(y, pred[:,12:25])
print("Keras: logloss for {} folds is {}". format(n_folds, score_val))


sum(pred_test[1,12:25])

pred_train_df = pd.DataFrame(pred, index = gatrain.index, columns=col_names)

pred_test_df = pd.DataFrame(pred_test, index = gatest.index, columns=col_names)

pred_train_df.to_csv('/home/username/projects/talkingData/keras_pred_train_bags5_wEvents_allData_20160824.csv', index=True, index_label='device_id')
pred_test_df.to_csv('/home/username/projects/talkingData/keras_pred_test_bags5_wEvents_allData_20160824.csv', index=True, index_label='device_id')

print(pred_test_df.head(1))

#generate prediction:
# submission = pd.DataFrame(pred_test[:,0:12], index = gatest.index, columns=targetencoder.classes_)
# submission.to_csv('/home/username/projects/talkingData/keras_cv10_plus_regression_80_reg.csv',index=True)

submission = pd.DataFrame(pred_test[:,12:25], index = gatest.index, columns=targetencoder.classes_)
submission.to_csv('/home/username/projects/talkingData/keras_cv10_with_bags5_wEvents_AllData.csv',index=True)


#generate mixed prediction (no events=logistic; events=keras)

pred_mix=pred[:,12:25]
pred_mix[train_id_ne]=pred[train_id_ne,0:12]

score_val=log_loss(y, pred_mix[:,0:12])
print("Mixed ne:Logisttic: logloss for {} folds is {}". format(n_folds, score_val))

pred_test_mix=pred_test[:,12:25]
pred_test_mix[test_id_ne,0:12]=pred_test[test_id_ne,0:12]

submission = pd.DataFrame(pred_test_mix[:,0:25], index = gatest.index, columns=targetencoder.classes_)
submission.to_csv('/home/username/projects/talkingData/keras_cv10_with_bags5_wEvents_AllData_Mix.csv',index=True)

