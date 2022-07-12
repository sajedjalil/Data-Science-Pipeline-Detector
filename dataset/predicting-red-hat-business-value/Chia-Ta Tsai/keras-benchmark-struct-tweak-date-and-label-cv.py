import time
import datetime as dt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.cross_validation import StratifiedKFold, LabelKFold
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint, Callback
from keras import backend as K
from keras.layers import Input, Embedding, LSTM, Dense,Flatten, Dropout, merge, Convolution1D,MaxPooling1D,Lambda,AveragePooling1D
from keras.layers.advanced_activations import PReLU, LeakyReLU, ELU, SReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2, activity_l2
from keras.optimizers import SGD
from keras.models import Model

seed = 10147426
np.random.seed(seed)

#keras
dim = int(2 ** 4)
hidden1 = 7
hidden2 = 5

default_batch_size = 4 * 2 ** 10
L2_reg=10 ** -6

path = '../input/'
digit = 6


class AucCallback(Callback):  #inherits from Callback
    
    def __init__(self, validation_data=(), patience=25,is_regression=True,best_model_name='best_keras.mdl',feval='roc_auc_score',batch_size=1024*8):
        super(Callback, self).__init__()
        
        self.patience = patience
        self.X_val, self.y_val = validation_data  #tuple of validation X and y
        self.best = -np.inf
        self.wait = 0  #counter for patience
        self.best_model=None
        self.best_model_name = best_model_name
        self.is_regression = is_regression
        self.y_val = self.y_val#.astype(np.int)
        self.feval = feval
        self.batch_size = batch_size
        
    def on_epoch_end(self, epoch, logs={}):
        p = self.model.predict(self.X_val,batch_size=self.batch_size, verbose=0)#.ravel()
        if self.feval=='roc_auc_score':
            current = roc_auc_score(self.y_val,p)

        if current > self.best:
            self.best = current
            self.wait = 0
            self.model.save_weights(self.best_model_name,overwrite=True)
            

        else:
            if self.wait >= self.patience:
                self.model.stop_training = True
                print('\nEpoch %05d: early stopping' % (epoch))
                
                
            self.wait += 1 #incremental the number of times without improvement
        print(', Epoch %d Auc: %f | Best Auc: %f \n' % (epoch,current,self.best))


############feature
def date_process(d, f):
    print('split date info')

    #d['mm'] = d[f].dt.month
    d['dd'] = d[f].dt.day    
    d['weekday'] = d[f].dt.weekday
    
    #hash_digit = 10 ** 3
    d['yyyy-mm'] = d[f].apply(lambda x: 100 * x.year + x.month)
    
    d['day_no'] = d[f].apply(lambda x: x.day).astype(int)
    d['week_no'] = d[f].apply(lambda x: x.week).astype(int)
    
    d.drop(f, axis=1, inplace=True)
    return d


def act_process(d):
    print('process act')
    #d['null_count'] = d.isnull().sum(axis=1)
    
    hash_digit = 10 ** 6
    
    d['act_id'] = d['activity_id'].str[3:4]
    #d['activity_category'] = d['activity_category'].str.lstrip('type ').fillna('null')
    #d['char_10'] = d['char_10'].str.lstrip('type ').fillna('null')
    
    d['act_id_cate'] = d['act_id'] + '_' + d['activity_category']
    #d['act_id_cate'] = d['act_id_cate'].apply(lambda x: hash(x) % hash_digit)

    d['act_id_c10'] = d['act_id'] + ': ' + d['char_10']
    #d['act_id_c10'] = d['act_id_c10'].apply(lambda x: hash(x) % hash_digit)
    
    return d
 
    
    
##########################################################
def mask_columns(columns = {}, mask = {}):
    for str1 in mask:
        if columns.count(str1) > 0:
            columns.remove(str1)
    return columns


def label_encode(data, mask = {}):
    columns = mask_columns(data.columns.tolist(), mask)
    #encode
    for c in columns:
        data[c] = LabelEncoder().fit_transform(data[c].values)
    return data
    

def split_train_valid(X, y, n_folds=5, shuffle=True, random_state=seed):
    skf = StratifiedKFold(y, n_folds=n_folds, shuffle=shuffle, random_state=random_state)
    for ind_tr, ind_va in skf:
        X_train = X[ind_tr]
        X_valid = X[ind_va]

        y_train = y[ind_tr]
        y_valid = y[ind_va]
        break
    
    #split array into samples
    X_train = [X_train[:,i] for i in range(X.shape[1])]
    X_valid = [X_valid[:,i] for i in range(X.shape[1])]
    return X_train, y_train, X_valid, y_valid


def split_train_valid_by_label(X, y, label, n_folds=3):
    skf = LabelKFold(label, n_folds=n_folds)
    for ind_tr, ind_va in skf:
        X_train = X[ind_tr]
        X_valid = X[ind_va]

        y_train = y[ind_tr]
        y_valid = y[ind_va]
        break
    
    #split array into samples
    X_train = [X_train[:,i] for i in range(X.shape[1])]
    X_valid = [X_valid[:,i] for i in range(X.shape[1])]
    return X_train, y_train, X_valid, y_valid


def create_submission(activity_id, outcome, prefix, score, digit):
    now = dt.datetime.now()
    filename = 'submission_residual_' + prefix + '_d' + str(dim) + '_a' + str(round(score, digit)) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv.gz'
    print('Make submission:{}'.format(filename))    
    
    submission = pd.DataFrame()
    submission['activity_id'] = activity_id
    submission['outcome'] = np.round(outcome, decimals=digit)
    #submission['outcome'] = submission['outcome'].apply(lambda x: round(x, digit))
    #submission.to_csv(filename, index=False, compression='gzip')
    submission.to_csv(filename, index=False)


def main():
    #
    start_time = time.time()
    
    print('read in data')
    train = pd.read_csv(path+'act_train.csv', parse_dates=['date'])
    train = date_process(train, 'date')
    train = act_process(train)

    test = pd.read_csv(path+'act_test.csv', parse_dates=['date'])
    test = test.assign(outcome=np.nan)
    test = date_process(test, 'date')
    test = act_process(test)


    print('incorp w/ ppl')
    people = pd.read_csv(path+'people.csv', parse_dates=['date'])
    people = date_process(people, 'date')
    #people = people.rename(columns={'date':'ppl_date'}, inplace=True)
    people = label_encode(people, mask = {'people_id', 'char_38'})
    #people = people[['people_id', 'char_38']]
    #merge
    train = pd.merge(train, people, how='left', on='people_id').fillna('null')
    X_ppl = train['people_id'].values
    train.drop('people_id', axis=1, inplace=True)
    
    test = pd.merge(test, people, how='left', on='people_id').fillna('null')
    test.drop('people_id', axis=1, inplace=True)
    
    del people
    
    #concate and encode
    print('encode data')
    mask = {'activity_id', 'outcome', 'char_38', 'people_id'}
    data = pd.concat([train, test])
    data = label_encode(data, mask)
    train = data[:train.shape[0]]
    test = data[train.shape[0]:]

    
    print('Load data: {} minutes'.format(round((time.time() - start_time)/60, 2)))
    print('design model structure')
    
    #remove target and id by remove its column name
    mask = {'activity_id', 'outcome'}   
    columns = mask_columns(train.columns.tolist(), mask)    
    print('Features [{}]: {}'.format(len(columns), sorted(columns)))    

    flatten_layers = []
    flatten_layers_2 = []
    inputs = []

    for c in columns:
        #numerical features
        if c == 'char_38':
            inputs_c = Input(shape=(1,), dtype='float32')
            inputs.append(inputs_c)
            
            ds_c = Dense(dim * 4, activation='relu')(inputs_c)
            
            #flatten_layers.append(ds_c)
            flatten_layers_2.append(ds_c)

        #factoerized features            
        else:
            inputs_c = Input(shape=(1,), dtype='int32')
            inputs.append(inputs_c)
    
            num_c = len(np.unique(data[c].values))
            
            #fold = int(round(math.log10(num_c) + 0.5))
    
            embed_c = Embedding(
                            num_c,
                            dim,
                            dropout=0.10,
                            input_length=1
                            )(inputs_c)
            flatten_c= Flatten()(embed_c)

            flatten_layers.append(flatten_c)
    
    #end embedding add
    del data
    flatten = merge(flatten_layers, mode='concat')
    
    #stacking a dense on the flatten layer
    #W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)
    fc1 = Dense(hidden1 ** 2, activation='relu')(flatten)
    flatten_layers_2.append(fc1)
    #
    deep = merge(flatten_layers_2, mode='concat')
    #deep = Dropout(0.25)(fc1)
    
    #deep layers
    #deep = Dense(hidden2 * hidden2, activation='relu')(deep)
    deep = Dropout(0.10)(deep)
    deep = Dense(hidden2 ** 2, activation='relu', 
                  W_regularizer=l2(L2_reg), activity_regularizer=activity_l2(L2_reg)
                  )(deep)

    #set output
    outputs = Dense(1, activation='sigmoid')(deep)

    model = Model(input=inputs, output=outputs)
    model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
              )
              
    #del data
    print('model structure complied')

    y = train['outcome'].values
    X = train[columns].values
    del train
    
    X_t = test[columns].values
    test_activity_id = test['activity_id']
    del test


        #model config
    load_model = False
    model_name = 'mlp_residual_%s_%s_%s.hdf5'%(dim, hidden1,hidden2)
    
    if load_model:
        print('Load Model')
        model.load_weights(path+model_name)
        # model.load_weights(path+'best_keras.mdl')    
    
    #model_checkpoint = ModelCheckpoint(model_name, monitor='val_loss', save_best_only=True)


    #outer fold
    X_train, y_train, X_valid, y_valid = split_train_valid_by_label(X, y, X_ppl, n_folds=4)
    #X_train, y_train, X_valid, y_valid = split_train_valid(X, y, n_folds=4, shuffle=True, random_state=seed)                                                 

    
    #outer fold training
    #auc_callback = AucCallback(validation_data=(X_valid, y_valid.tolist()), patience=10, 
    #                           is_regression=True, best_model_name=path+'best_keras.mdl', feval='roc_auc_score')
                               
    #training param
    nb_epoch = 6         
    
    model.fit(
        X_train, y_train,
        batch_size=default_batch_size, 
        nb_epoch=nb_epoch, 
        verbose=1, shuffle=True,
        validation_data=[X_valid, y_valid],
        #callbacks = [model_checkpoint, auc_callback,],
        )

    # model.load_weights(model_name)
    # model.load_weights(path+'best_keras.mdl')
    
    #show
    #y_preds = model.predict(X_valid, batch_size=default_batch_size)
    #auc = roc_auc_score(y_valid.tolist(), y_preds)    
    #print('\nValidation auc = {}(test on outer fold)\n\n'.format(round(auc, digit)))
    
    
    print('\nPredict test')
    X_t = [X_t[:,i] for i in range(X_t.shape[1])]
    outcome = model.predict(X_t, batch_size=default_batch_size)
    create_submission(test_activity_id, outcome, 'test', 0, digit)

    
main()

