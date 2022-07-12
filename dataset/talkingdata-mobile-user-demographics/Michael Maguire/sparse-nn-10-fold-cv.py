import pandas as pd
import numpy as np
import os
from scipy.sparse import csr_matrix
from scipy.sparse import hstack, vstack
from scipy import io
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.advanced_activations import PReLU
from sklearn.feature_selection import SelectPercentile, f_classif, chi2, SelectKBest
from sklearn.preprocessing import LabelEncoder
lbl = LabelEncoder()
datapath = '../input'
#===============================================================================================================================
###  Model Parameters
#===============================================================================================================================

np.random.seed(7)
k = 10                          # number of stratified folds
feature_select_method = 'none'  # 'percentile' or Kbest' to use the feature selection tools
perc   = 100                    # if feature_selection_method = 'percentile', set this
K_best = 11115                  # if feature_selection_method = 'Kbest', set this
data_subset = 2                 # train/predict for 0=devices with no events, 1=with events, 2=all devices
epochs = 15
samp_per_epoch = 69984

#===============================================================================================================================
###  Load Data Files
#===============================================================================================================================
groups = ['F23-','F24-26','F27-28','F29-32','F33-42','F43+','M22-','M23-26','M27-28','M29-31','M32-38','M39+']

print ('# Loading data files...')
app_events  = pd.read_csv(os.path.join(datapath,'app_events.csv'), sep=',', dtype={'event_id':str,'app_id':str,'is_installed':float,'is_active':float})
app_labels  = pd.read_csv(os.path.join(datapath,'app_labels.csv'), sep=',', dtype={'app_id':str,'label_id':str})
label_cat   = pd.read_csv(os.path.join(datapath,'label_categories.csv'), sep=',', dtype={'label_id':str,'category':str})
events      = pd.read_csv(os.path.join(datapath,'events.csv'), sep=',', usecols=['event_id','device_id','timestamp'], dtype={'event_id':str,'device_id':str,'timestamp':object})
phone       = pd.read_csv(os.path.join(datapath,'phone_brand_device_model.csv'), sep=',', dtype={'device_id':str,'phone_brand':str,'device_model':str})
gatrain     = pd.read_csv(os.path.join(datapath,'gender_age_train.csv'), sep=',', dtype={'device_id':str,'gender':str,'age':float,'group':str})
gatest      = pd.read_csv(os.path.join(datapath,'gender_age_test.csv'), sep=',', dtype={'device_id':str})

#===========================================================
#  Train & Test data
gatrain['trainrow'] = np.arange(gatrain.shape[0])
gatrain.drop(['age','gender'], axis=1, inplace=True)
gatrain.group = lbl.fit_transform(gatrain.group)
target = np.array(gatrain.group)

gatest['testrow'] = np.arange(gatest.shape[0])
gatest['group'] = np.nan
split_len = gatrain.shape[0]

devices = gatrain.append(gatest, ignore_index=True)
devices['row_num'] = np.arange(devices.shape[0])
devices = devices[['device_id','row_num']]
rows = devices.shape[0]

#===========================================================
def build_sparse(data, feature):
    data[feature] = lbl.fit_transform(data[feature])
    cols = len(lbl.classes_)
    sparse_matrix = csr_matrix((np.ones(data.shape[0]), (data.row_num, data[feature])), shape=(rows, cols))
    sparse_matrix = sparse_matrix[:, sparse_matrix.getnnz(0) > 0] #remove columns that are all zeroes

    Xtr = sparse_matrix[ :split_len, : ]
    Xte = sparse_matrix[split_len: , : ]
    return Xtr, Xte

#===============================================================================================================================
###  Phone brand and model data
print ('# Processing phone data')
#===============================================================================================================================
phone.drop_duplicates('device_id', keep='last', inplace=True)

#===========================================================
devices = pd.merge(devices, phone, on=['device_id'], how='left')
devices['brand_model'] = map(lambda x,y: str(x)+'-'+str(y), devices.phone_brand, devices.device_model)
tr1, te1 = build_sparse(devices[['row_num','phone_brand']], 'phone_brand')
tr2, te2 = build_sparse(devices[['row_num','device_model']], 'device_model')
#===========================================================
del phone
devices = devices[['device_id','row_num']]

#===============================================================================================================================
###  App data
print ('# Processing app data')
#===========================================================
app_data    = pd.merge(app_events, events[['event_id', 'device_id']], on=['event_id'], how='left')
app_data.dropna(inplace=True)
app_install = app_data.drop_duplicates(['device_id','app_id'], keep='first')
app_install = app_install[['device_id','app_id']]

app_dev = pd.merge(devices, app_install, on=['device_id'], how='left')
app_dev = app_dev[['row_num','app_id']]
tr3, te3 = build_sparse(app_dev, 'app_id')
#===========================================================
del app_dev
devices = devices[['device_id','row_num']]

#===============================================================================================================================
###  Category data
print ('# Processing category data')

app_labels  = pd.merge(app_labels, label_cat, on=['label_id'], how='left')
lbl_install = pd.merge(app_install, app_labels, on=['app_id'], how='left')
cat_dev_ins = lbl_install[['device_id','category']].drop_duplicates(['device_id','category'], keep='first')
cat_dev_ins = pd.merge(cat_dev_ins, devices[['device_id','row_num']], on=['device_id'], how='left')
cat_dev_ins.dropna(inplace=True)
tr4, te4 = build_sparse(cat_dev_ins[['row_num','category']], 'category')

#===============================================================================================================================
###  Assemble the data

sparse_tr = hstack((tr1, tr2, tr3, tr4), format='csr')
sparse_te = hstack((te1, te2, te3, te4), format='csr')
print('Data: train shape {}, test shape {}'.format(sparse_tr.shape, sparse_te.shape))

#  Remove all zero columns from the data
trn_len = sparse_tr.shape[0]
all_data = vstack((sparse_tr, sparse_te), format='csr')
nnz_data = all_data[:, all_data.getnnz(0)>0]
train_mtx = nnz_data[ :trn_len, :]
test_mtx  = nnz_data[trn_len: , :]
print('NNZ Data: train shape {}, test shape {}'.format(train_mtx.shape, test_mtx.shape))

#===============================================================================================================================
###  Model functions
#===============================================================================================================================
# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(150, input_dim=num_feats, init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.4))
    model.add(Dense(50, input_dim=num_feats, init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(12, init='normal', activation='softmax'))
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy',
        optimizer='adadelta',
        metrics=['accuracy'])
    return model

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

def k_folds(k):
    kf  = pd.read_csv(os.path.join(datapath,'gender_age_train.csv'), sep=',', usecols=['device_id','group'], dtype={'device_id':str})
    events = pd.read_csv(os.path.join(datapath,'events.csv'), sep=',', usecols=['device_id'], dtype={'device_id':str})
    events.drop_duplicates(['device_id'], keep='first', inplace=True)
    events['event'] = ['1']*events.shape[0]
    kf = pd.merge(kf, events, on=['device_id'], how='left')
    kf.fillna('0', inplace=True)
    kf['folds'] = kf.group+kf.event
    kf.folds = lbl.fit_transform(kf.folds)
    return list(StratifiedKFold(target, n_folds=k, shuffle=True, random_state=11))

def feature_sel_P(trn, tst, y, p):
    selector = SelectPercentile(f_classif, percentile=p)
    selector.fit(trn, y)
    out_tr = selector.transform(trn)
    out_ts = selector.transform(tst)
    return out_tr, out_ts
    
def feature_sel_K(trn, tst, y, kbest):
    selector = SelectKBest(chi2, k=kbest).fit(trn, y)
    out_tr = selector.transform(trn)
    out_ts = selector.transform(tst)
    return out_tr, out_ts

def data_sub(s):
    tr_split = pd.read_csv(os.path.join(datapath,'gender_age_train.csv'), sep=',', usecols=['device_id'], dtype={'device_id':str})
    te_split = pd.read_csv(os.path.join(datapath,'gender_age_test.csv'), sep=',', dtype={'device_id':str})
    
    events = pd.read_csv(os.path.join(datapath,'events.csv'), sep=',', usecols=['device_id'], dtype={'device_id':str})
    events.drop_duplicates(['device_id'], keep='first', inplace=True)
    events['event'] = [1]*events.shape[0]
    
    tr_split = pd.merge(tr_split, events, on=['device_id'], how='left')
    tr_split.fillna(0, inplace=True)
    te_split = pd.merge(te_split, events, on=['device_id'], how='left')
    te_split.fillna(0, inplace=True)

    tr_idx = np.array( tr_split[tr_split.event==s].index.tolist() )
    te_idx = np.array( te_split[te_split.event==s].index.tolist() )

    return train_mtx[tr_idx, :], test_mtx[te_idx, :], target[tr_idx], te_split.device_id[te_split.event==s]

#===============================================================================================================================
###  CV Model
#===============================================================================================================================

if data_subset<2:
    train_mtx, test_mtx, target, ids = data_sub(data_subset)
else:  ids = gatest.device_id

print('Subsetted Data: train shape {}, test shape {}'.format(train_mtx.shape, test_mtx.shape))

if feature_select_method == 'percentile':
    Xtrain, Xtest = feature_sel_P(train_mtx, test_mtx, target, perc)
elif feature_select_method == 'Kbest':
    Xtrain, Xtest = feature_sel_K(train_mtx, test_mtx, target, K_best)
else:
    Xtrain = train_mtx.copy()
    Xtest  = test_mtx.copy()
num_feats = Xtrain.shape[1]

print('Feature-selected Data: train shape {}, test shape {}'.format(Xtrain.shape, Xtest.shape))

fold_def = k_folds(k)
CV = []
oof_preds = pd.DataFrame()
for i in range(10):
    Xtr, oof   = Xtrain[fold_def[i][0], :], Xtrain[fold_def[i][1], :]
    ytr, y_oof = target[fold_def[i][0]],    target[fold_def[i][1]]
    oof_id        = gatrain.device_id[fold_def[i][1]]
    
    model = baseline_model()
    fold_tr, val_tr, y_fold, y_val = train_test_split(Xtr, ytr, train_size=.98, random_state=10)
    
    fit = model.fit_generator(generator=batch_generator(fold_tr, y_fold, 400, True),
                         nb_epoch=epochs,
                         samples_per_epoch=samp_per_epoch,
                         validation_data=(val_tr.todense(), y_val),
                         verbose=2)
    
    preds = model.predict_generator(generator=batch_generatorp(oof, 400, False), val_samples=oof.shape[0] )
    score = log_loss(y_oof, preds)
    CV = CV+[score]
    print ('# Fold '+str(i)+': ', score)
    oof_preds= oof_preds.append(pd.DataFrame(preds, index=oof_id, columns=groups))
    
print (str(k)+'-Fold CV: ', np.mean(CV))
oof_preds.to_csv('# sparseNN_XXXTR.csv', index=True)

#####  Train and predict on full training set
model = baseline_model()
fold_tr, val_tr, y_fold, y_val = train_test_split(Xtr, ytr, train_size=.98, random_state=10)

fit = model.fit_generator(generator=batch_generator(fold_tr, y_fold, 800, True),
                         nb_epoch=epochs,
                         samples_per_epoch=samp_per_epoch,
                         validation_data=(val_tr.todense(), y_val),
                         verbose=2)
preds = model.predict_generator(generator=batch_generatorp(Xtest, 800, False), val_samples=Xtest.shape[0] )
submission = pd.DataFrame(preds, index=ids, columns=groups)
submission.to_csv('sparseNN_XXX.csv', index=True)

print (str(k)+'-Fold CV mean: ', np.mean(CV))
print (str(k)+'-Fold CV  std: ', np.std(CV))