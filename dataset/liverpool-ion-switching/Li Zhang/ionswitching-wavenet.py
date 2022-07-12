'''
This is a script for the Liverpool-Ion-Switching Competition. (top 4%)

Reference notebooks:
https://www.kaggle.com/nxrprime/wavenet-with-shifted-rfc-proba-and-cbr
https://www.kaggle.com/siavrez/wavenet-keras
https://www.kaggle.com/siavrez/wavenet-keras

Main modifications:
* add global variables to test different features and models
* working environment in both kaggle and colab
* add data augmentation and gaussian noice of batch 5

'''

# Import Packages
!pip install tensorflow-addons==0.9.1
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import gc

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Dense, Dropout, Lambda, BatchNormalization, concatenate, Add, LSTM, Multiply, Activation, RepeatVector, Dot, Concatenate, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping, LearningRateScheduler
import tensorflow_addons as tfa

from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.utils.class_weight import compute_class_weight

import zipfile
import pprint

### Set Enviroments, kaggle or colab
ENV = 'kaggle' # colab

if ENV == 'kaggle':
    train_data = pd.read_csv('/kaggle/input/data-without-drift/train_clean.csv', dtype={'time': np.float32, 'signal': np.float32, 'open_channels':np.int32})
    test_data = pd.read_csv('/kaggle/input/data-without-drift/test_clean.csv', dtype={'time': np.float32, 'signal': np.float32})
    
elif ENV == 'colab':
    from google.colab import drive
    drive.mount('/content/drive')
    !kaggle competitions download liverpool-ion-switching -f sample_submission.csv
    !kaggle datasets download -d cdeotte/data-without-drift
    !kaggle datasets download -d sggpls/ion-shifted-rfc-proba    
    
    zf = zipfile.ZipFile('./data-without-drift.zip')
    train_data = pd.read_csv(zf.open('train_clean.csv'), dtype={'time': np.float32, 'signal': np.float32, 'open_channels':np.int32})
    test_data = pd.read_csv(zf.open('test_clean.csv'), dtype={'time': np.float32, 'signal': np.float32})

# # If you want to submit to kaggle, please add kaggle.json file using
# !mkdir -p ~/.kaggle
# !cp [PATH_TO_KAGGLE_JSON] ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json
    
### Helper Functions
def seed_everything(seed):
    '''
    set random seeds
    '''
    os.environ['PYTHONHASHSEED'] = str(seed)    
    np.random.seed(seed)
    tf.random.set_seed(seed)


def generate_train_sample(X, y, sample_length):
    '''
    generate training time-series samples
    '''
    if len(X.shape) == 1:
        X = X[:,np.newaxis]

    n_batch = X.shape[0]//SAMPLE_BATCH_LENGTH
    X_sample = []
    y_sample = []
    sample_batch_id = []
    
    for id_batch in range(1,n_batch+1):
        X_batch = X[SAMPLE_BATCH_LENGTH*(id_batch-1):SAMPLE_BATCH_LENGTH*id_batch]
        y_batch = y[SAMPLE_BATCH_LENGTH*(id_batch-1):SAMPLE_BATCH_LENGTH*id_batch]
        X_tmp = X_batch.reshape((-1, sample_length, X.shape[-1]))
        y_tmp = y_batch.reshape((-1, sample_length, 1))
        X_sample.append(X_tmp)
        y_sample.append(y_tmp)
        sample_batch_id.append(id_batch*np.ones((X_tmp.shape[0])))
    
    X_train = np.vstack(X_sample)
    y_train = np.vstack(y_sample)
    sample_batch_id = np.concatenate(sample_batch_id)
    
    return X_train, y_train, sample_batch_id


def generate_test_sample(X, sample_length):
    '''
    generate test time-series samples
    '''
    if len(X.shape) == 1:
        X = X[:,np.newaxis]
    X_test = X.reshape((-1, sample_length, X.shape[-1]))
    return X_test


def reduce_mem_usage(X):
    '''
    reduce memory usage
    '''
    X_min = np.min(X)
    X_max = np.max(X)
    if X.dtype == 'int':
        if (X_min > np.iinfo(np.int32).min and X_max < np.iinfo(np.int32).max):
            X = X.astype(np.int32)
        else:
            X = X.astype(np.int64)
    elif X.dtype == 'float':
        if (X_min > np.finfo(np.float32).min and X_max < np.finfo(np.float32).max):
            X = X.astype(np.float32)
        else:
            X = X.astype(np.float64)
    return X


def model_score(y, y_pred):
    """
    calculate precision, recall, f1 score of model prediction
    """
    report = classification_report(y, y_pred, digits=3)
    print(report)
    
    print('Micro precision training score for all = {}'.format(precision_score(y,y_pred,average='micro')))
    print('Micro recall training score for all = {}'.format(recall_score(y,y_pred,average='micro'))) 
    print('Micro f1 training score for all = {}'.format(f1_score(y,y_pred,average='micro')))
    
    print('Macro precision training score for all = {}'.format(precision_score(y,y_pred,average='macro')))
    print('Macro recall training score for all = {}'.format(recall_score(y,y_pred,average='macro'))) 
    print('Macro f1 training score for all = {}'.format(f1_score(y,y_pred,average='macro')))

    macro_f1 = f1_score(y,y_pred,average='macro')

    return report, macro_f1


def save_submit(preds, sample_submit):
    '''
    save test prediction and submit to kaggle
    '''
    filename = SUBMIT_NAME
    message = SUBMIT_MSG

    if ENV == 'kaggle':
        path_save = os.path.join('.', filename)
        path_submit = os.path.join('.', filename)

    elif ENV == 'colab':
        path_save = os.path.join('./drive/My Drive/submission', filename)
        path_submit = os.path.join('./drive/My\ Drive/submission', filename)

    df = pd.DataFrame(data={'time':sample_submit.time, 'open_channels':preds})
    df.to_csv(path_save, index=False, float_format='%.4f')

    if SUBMIT_TO_KAGGLE == True:
        !kaggle competitions submit liverpool-ion-switching -f {path_submit} -m {message}
        
    return


class Val_MacroF1(Callback):
    '''
    model callback for computing validation F1 score
    '''
    def __init__(self, model, X_val, y_val):
        self.model = model
        self.X_val = X_val
        self.y_val = y_val.reshape(-1)        
        self.score = []

    def on_epoch_end(self, epoch, logs):
        preds_val = np.argmax(self.model.predict(self.X_val), axis=-1).reshape(-1)
        score_val = f1_score(self.y_val, preds_val, average="macro")
        print(f' val_f1: {score_val:.5f}')
        self.score.append(score_val)


def lr_schedule(epoch):
    '''
    learning rate decay schedule
    '''
    period = 25
    div = epoch//period
    decay = 1/2**(div)
    lr = decay*LEARNING_RATE
    if epoch%period == 0 and epoch > 0:
        print('== LEARNING RATE REDUCED TO {} \n'.format(lr))
    return lr


def plot_history(history):
    '''
    plot training history
    '''
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    ax[0].plot(history['accuracy'], label='train')
    ax[0].plot(history['val_accuracy'], label='val')
    ax[0].set_ylabel('accuracy')
    ax[0].set_ylim([0.9, 0.98])
    ax[0].legend()
    ax[1].plot(history['loss'], label='train')
    ax[1].plot(history['val_loss'], label='val')
    ax[1].set_ylabel('loss')
    ax[1].set_ylim([0.075, 0.1])
    ax[1].legend()
    ax[2].plot(history['val_F1'], label='val_F1')
    ax[2].set_ylabel('macro F1')
    ax[2].set_ylim([0.930, 0.945])
    ax[2].legend()
    plt.title(MODEL_NAME)
    plt.tight_layout()
    plt.show()
    return


def export_results_report(parameters, eval_report, history):
    '''
    export classification report
    '''
    with open('results_report_'+MODEL_NAME+'.txt', 'a') as f:
        f.write(MODEL_NAME)
        print(parameters, file=f)
        print('\n',file=f)
        print(eval_report, file=f)
        print('\n',file=f)
        print(history, file=f)
    return


# Data augmentation by flipping and adding Gaussian noise
def add_augment_data(X_tr, y_tr, train_id=None, reverse=False, gaussian_noise=False):
    '''
    data augmentation
    '''
    if train_id is None:
        X_tr = np.concatenate((X_tr, np.flip(X_tr, axis=1)), axis=0)
        y_tr = np.concatenate((y_tr, np.flip(y_tr, axis=1)), axis=0)
        
    else:
        # only augment batch 5
        batch = [5]
        n_sample_per_batch = SAMPLE_BATCH_LENGTH//SEQUENCE_LENGTH
        id_aug = [i for i, id_ in enumerate(train_id) if id_//n_sample_per_batch+1 in batch]
        
        X_aug, y_aug = X_tr[id_aug], y_tr[id_aug]
        if reverse == True: X_aug, y_aug = np.flip(X_aug, axis=1), np.flip(y_aug, axis=1)
        if gaussian_noise == True: X_aug += np.random.normal(0, 0.01, X_aug.shape)

        X_tr = np.concatenate((X_tr, X_aug), axis=0)
        y_tr = np.concatenate((y_tr, y_aug), axis=0)
        
    print('Training Sample Dimension after Augment: {}'.format(X_tr.shape))
    
    return X_tr, y_tr


# Add sample weights for dealing with imbalanced data
def add_sample_weights(y_tr, y_val, weighted_sample):
    '''
    add sample weights
    '''
    y = np.concatenate((y_tr, y_val), axis=0)
    class_weight = compute_class_weight('balanced', [i for i in range(0,11)], y.flatten())
    if weighted_sample == 'sqrt':
        class_weight = np.sqrt(class_weight)

    dict_class_weight = {key: value for key, value in zip(range(0,11), class_weight)}
    train_sample_weight = np.vectorize(dict_class_weight.get)(y_tr[:,:,0])
    val_sample_weight = np.vectorize(dict_class_weight.get)(y_val[:,:,0])
    sample_weights = [train_sample_weight, val_sample_weight]

    ### Test
    print('Training Sample Weights Shape: {}'.format(sample_weights[0].shape))
    print('Training Sample Weights[2:5,0] : {}'.format(sample_weights[0][2:5,0]))
    ###
    
    return sample_weights


### Prepare Data
def data_feature_prep(train_data, test_data, sample_length=5000, shifted_feature=False, grouped_feature=False, squared_feature=False, rfprob_feature=False, knnprob_feature=False):
    '''
    data preparation pipeline
    '''
    X_train_orig = train_data['signal'].values
    y_train_orig = train_data['open_channels'].values
    X_test_orig = test_data['signal'].values

    # reduce memory usage
    X_train_orig = reduce_mem_usage(X_train_orig)
    y_train_orig = reduce_mem_usage(y_train_orig)

    # remove outlier
    def remove_outlier(X):
        batches = [1,2]
        x_max = -0.5
        for batch in batches:
            X_batch = X[(batch-1)*SAMPLE_BATCH_LENGTH:batch*SAMPLE_BATCH_LENGTH].copy()
            np.clip(X_batch, None, x_max, out=X_batch)
            X[(batch-1)*SAMPLE_BATCH_LENGTH:batch*SAMPLE_BATCH_LENGTH] = X_batch

        batch = 8
        X_batch = X[(batch-1)*SAMPLE_BATCH_LENGTH:batch*SAMPLE_BATCH_LENGTH].copy()
        batch_ref = 4
        X_batch_ref = X[(batch_ref-1)*SAMPLE_BATCH_LENGTH:batch_ref*SAMPLE_BATCH_LENGTH].copy()
        x_max = np.max(X_batch_ref)
        x_min = np.min(X_batch_ref)
        min_mask = X_batch<x_min
        max_mask = X_batch>x_max
        np.clip(X_batch, x_min, x_max, out=X_batch)
        # add some noice
        X_batch[min_mask] = X_batch[min_mask] + 0.5*np.random.rand(len(X_batch[min_mask]))
        X_batch[max_mask] = X_batch[max_mask] - 0.5*np.random.rand(len(X_batch[max_mask]))

        X[(batch-1)*SAMPLE_BATCH_LENGTH:batch*SAMPLE_BATCH_LENGTH] = X_batch
        return X
    
    X_train_orig = remove_outlier(X_train_orig)   
    
    # normalize X_train
    train_mean = X_train_orig.mean()
    train_sigma = X_train_orig.std()
    # pd.mean is not accurate
    # train_mean = train_data['signal'].mean()
    # train_sigma = train_data['signal'].std()    
    X_train_norm = (X_train_orig-train_mean)/train_sigma
    X_test_norm = (X_test_orig-train_mean)/train_sigma
    
    X_train_final = X_train_norm[:,np.newaxis,np.newaxis]
    X_test_final = X_test_norm[:,np.newaxis, np.newaxis]

    ### Test
    print('X dimension after normalization: {}'.format(X_train_final.shape))
    ###

    # stats feature
    # Add statistical features of longer sample lengthes to give more receptive field of the sample, 
    # but this features seems not working well as all the features are a constant number for the sequence
    def stat_feature(X):
        X_out = X
        if len(STAT_RECEPT_LENGTHES) > 0:
            lists = []

            for stat_recept_length in STAT_RECEPT_LENGTHES:
                mean = np.mean(X.reshape((stat_recept_length, -1), order='F'), axis=0)
                std = np.std(X.reshape((stat_recept_length, -1), order='F'), axis=0)
                mean_feature = np.tile(mean, (stat_recept_length, 1)).reshape(-1, order='F')
                std_feature = np.tile(std, (stat_recept_length, 1)).reshape(-1, order='F')   
                lists.extend([mean_feature, std_feature])

            X_stat = np.concatenate(lists, axis=-1)
            X_out = np.concatenate((X_out, X_stat), axis=-1)
        return X_out

    X_train_final = stat_feature(X_train_final)
    X_test_final = stat_feature(X_test_final)

    ### Test    
    print('X dimension after stat feature: {}'.format(X_train_final.shape))
    ###

    if squared_feature == True:
        X_sqrt = X_train_final[:,:,0][:,:,np.newaxis]**2
        X_train_final = np.concatenate((X_train_final, X_sqrt), axis=-1)

        X_sqrt = X_test_final[:,:,0][:,:,np.newaxis]**2
        X_test_final = np.concatenate((X_test_final, X_sqrt), axis=-1)        
    
    ### Test
    print('X dimension after squared feature: {}'.format(X_train_final.shape))
    ###

    # the signal seems being generated by 5 different types of models, by explorative analysis, we can give each signal sample a group type feature
    # the grorup feature is a one-hot feature
    if grouped_feature == True:
        train_group = [[(0,500000),(500000, 1000000)],
                       [(1000000,1500000),(3000000, 3500000)],
                       [(1500000,2000000),(3500000, 4000000)],
                       [(2500000,3000000),(4000000, 4500000)],
                       [(2000000,2500000),(4500000, 5000000)]]
        test_group = [[(0,100000),(300000,400000),(800000,900000),(1000000,2000000)],
                      [(400000,500000)], 
                      [(100000,200000),(900000,1000000)],
                      [(200000,300000),(600000,700000)],
                      [(500000,600000),(700000,800000)]]
        
        X_train_group = np.zeros((X_train_final.shape[0], 5))
        X_test_group = np.zeros((X_test_final.shape[0], 5))
        
        for k in range(5):
            tr_group = train_group[k]
            te_group = test_group[k]
            
            for g in tr_group: X_train_group[g[0]:g[1], k] = 1
            for g in te_group: X_test_group[g[0]:g[1], k] = 1
    
        X_train_final = np.hstack((X_train_final, X_train_group))
        X_test_final = np.hstack((X_test_final, X_test_group))
    
    ### Test        
    print('X dimension after group feature: {}'.format(X_train_final.shape))
    ### 

    # rf proba feature
    if rfprob_feature == True:
        if ENV == 'kaggle':
             X_train_rfprob = np.load("/kaggle/input/ion-shifted-rfc-proba/Y_train_proba.npy")
             X_test_rfprob = np.load("/kaggle/input/ion-shifted-rfc-proba/Y_test_proba.npy")
            
        elif ENV == 'colab':
            # zf = zipfile.ZipFile("./ion-shifted-rfc-proba.zip")
            X_train_rfprob = np.load("./ion-shifted-rfc-proba.zip")['Y_train_proba.npy']
            X_test_rfprob = np.load("./ion-shifted-rfc-proba.zip")['Y_test_proba.npy']
        
        X_train_final = np.concatenate((X_train_final, X_train_rfprob[:,0:11][:,np.newaxis,:]), axis=-1)
        X_test_final = np.concatenate((X_test_final, X_test_rfprob[:,0:11][:,np.newaxis,:]), axis=-1)

    ### Test            
    print('X dimension after RFprob feature: {}'.format(X_train_final.shape))
    ### 

    if knnprob_feature == True:
        if ENV == 'kaggle':
            X_train_knnprob = np.load('/kaggle/input/knn-preds-prob-2fold/KNN_val_prob.npy', allow_pickle=True)              
#             X_test_knnprob = 
            
        elif ENV == 'colab':
            X_train_knnprob = np.load('drive/My Drive/model/KNN_preds_prob.npy', allow_pickle=True)            
            # X_test_knnprob
        
        X_train_final = np.concatenate((X_train_final, X_train_knnprob[:,0:11][:,np.newaxis,:]), axis=-1)
        # TEST DATA TO BE ADDED
    
    ### Test    
    print('X dimension after KNNprob feature: {}'.format(X_train_final.shape))
    ###
    
    # generate data samples
    X_train, y_train, sample_batch_id = generate_train_sample(X_train_final, y_train_orig, sample_length)
    X_test = generate_test_sample(X_test_final, sample_length)
    
    # add shift
    # Adding shift helps to give more context to the end data points of the sample. Padding with zero is to avoid any data leakage or cyclic features 
    def add_shift_feature(X, shifts):
        X_signal = X[:,:,0][:,:,np.newaxis]  # keep dimension
        X_shift = []

        for shift in shifts:
            X_pos_shift = np.roll(X_signal, shift, axis=1)
            X_pos_shift[:,0:shift,:] = 0.0
            X_neg_shift = np.roll(X_signal, -shift, axis=1)
            X_neg_shift[:,-1:-(shift+1),:] = 0.0
            X_shift.append(X_pos_shift)
            X_shift.append(X_neg_shift)

        X_shift = np.concatenate(X_shift, axis=-1)
        X_out = np.concatenate((X, X_shift), axis=-1)
        return X_out
    
    if shifted_feature == True:
        shifts = [1,2,3]
        X_train = add_shift_feature(X_train, shifts)
        X_test = add_shift_feature(X_test, shifts)

    ### Test
    print('X dimension after shifted feature: {}'.format(X_train.shape))        
    ###
    
    X_train = reduce_mem_usage(X_train)
    ### Test
    print('Data type of X: {}'.format(X_train.dtype))
    print('Data type of y: {}'.format(y_train.dtype))
    ###

    gc.collect()
    
    return X_train, y_train, X_test, sample_batch_id


def data_split(X_train, y_train, sample_batch_id, n_fold=None, stratified_split=True, partial_oversampling=False):
    '''
    data split
    '''
    if n_fold is not None:
        print('== DIVIDE TRAINING SET INTO {} FOLDS =='.format(n_fold))
        
        # generator for training set split
        if stratified_split == True:
            # (Stratified sampling make the training and validation distribution similar to the final test)
            fold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=SEED)
            splits = fold.split(X_train, sample_batch_id)

        else:
            fold = KFold(n_splits=n_fold, shuffle=True, random_state=SEED)
            splits = fold.split(X_train, y_train)
    
    else:
        print('== ONE-TIME TRAIN TEST SPLIT ==')
        
        # split training and validation sets 
        indices = np.arange(X_train.shape[0])
        # (Stratified sampling make the training and validation distribution similar to the final test)
        if stratified_split == True:
            X_tr, X_val, y_tr, y_val, id_tr, id_val = train_test_split(X_train, y_train, indices, test_size=0.2, shuffle=True, random_state=SEED, stratify=sample_batch_id)

        else:
            X_tr, X_val, y_tr, y_val, id_tr, id_val = train_test_split(X_train, y_train, indices, test_size=0.2, shuffle=True, random_state=SEED)

        splits = [(id_tr, id_val)]

    return splits


def build_WaveNet(n_seq=100, n_feature=1, n_stack_layer=[12,8,4,1], n_first_filter=16, batchnorm=False, preprocess=None, postprocess=None, post_dropout=None):
    '''
    build WaveNet model
    '''
    # Input
    x = Input(shape=(None, n_feature))
    encode = x
    
    # preprocessing
    if preprocess is not None:
        if preprocess['name'] == 'LSTM':
            if 'units' in preprocess:
                units = preprocess['units']
            else:
                units = n_first_filter

            encode = LSTM(units=units, return_sequences=True)(inputs=encode)

            if batchnorm == True:
                encode = BatchNormalization()(encode)

            encode = Activation('relu')(encode)

        elif preprocess['name'] == 'Conv':
            if 'units' in preprocess:
                units = preprocess['units']
            else:
                units = n_first_filter

            if 'kernel_size' in preprocess:
                kernel_size = preprocess['kernel_size']
            else:
                kernel_size = 1

            encode = Conv1D(units, kernel_size=kernel_size, padding='same')(encode)

#             if batchnorm == True:
#                 encode = BatchNormalization()(encode) 

            encode = Activation('relu')(encode)
            
            if batchnorm == True:
                encode = BatchNormalization()(encode)    


    # WaveNet encoder
    for id_stack, n_layer in enumerate(n_stack_layer):
        n_filter = n_first_filter*2**id_stack
        encode = wavenet_block(encode, n_layer, n_filter, kernel_size=3)  # [n_batch, n_seq, n_filter]
#         if batchnorm == True and id_stack+1 < len(n_stack_layer):
#             encode = BatchNormalization()(encode)
        if batchnorm == True:
            encode = BatchNormalization()(encode)            
    
    # decoder
    decode = encode

    if postprocess is not None:
        if postprocess['name'] == 'LSTM':
            if 'units' in postprocess:
                units = postprocess['units']
            else:
                units = n_filter
            decode = LSTM(units=units, return_sequences=True)(inputs=decode)

            if batchnorm == True:
                decode = BatchNormalization()(decode)

            decode = Activation('relu')(decode)

        elif postprocess['name'] == 'Conv':
            if 'units' in postprocess:
                units = postprocess['units']
            else:
                units = n_filter

            if 'kernel_size' in postprocess:
                kernel_size = postprocess['kernel_size']
            else:
                kernel_size = 1

            decode = Conv1D(units, kernel_size=kernel_size, padding='same')(decode)

#             if batchnorm == True:
#                 decode = BatchNormalization()(decode)

            decode = Activation('relu')(decode)
    
            if batchnorm == True:
                decode = BatchNormalization()(decode)

    if post_dropout is not None:
        if 'rate' in post_dropout:
            dropout_rate = post_dropout['rate']
        else:
            dropout_rate = 0.2
        decode = Dropout(dropout_rate)(decode)

    outputs = Dense(11, activation='softmax')(decode)

    model = Model(inputs=x, outputs=outputs)
    return model

def wavenet_block(x, n_layer, n_filter, kernel_size):
    '''
    build wavenet block
    '''
    dilation_rates = [2**i for i in range(n_layer)]
    x = Conv1D(n_filter, kernel_size=1, padding='same')(x)

    res_x = x
    for dilation_rate in dilation_rates:
        # filter convolution
        x_f = Conv1D(filters = n_filter,
                    kernel_size = kernel_size, 
                    padding = 'same',
                    activation = 'tanh',
                    dilation_rate = dilation_rate)(x)

        # gating convolution
        x_g = Conv1D(filters = n_filter,
                     kernel_size = kernel_size, 
                     padding = 'same',
                     activation = 'sigmoid',
                     dilation_rate = dilation_rate)(x)

        # multiply filter and gating branches
        x = Multiply()([x_f,x_g])

        # postprocessing - equivalent to time-distributed dense
        x = Conv1D(n_filter, kernel_size=1, padding = 'same')(x)

        # residual connection
        res_x = Add()([res_x, x])

    return res_x


def run_model(model, X_tr, y_tr, X_val, y_val, X_test, callbacks, verbose=1, sample_weights=None):
    '''
    run model training
    '''
    print('START RUNNING {} \n'.format(MODEL_NAME))

    val_data = (X_val, y_val)
    if sample_weights is not None:
        train_sample_weight = sample_weights[0]
        val_sample_weight = sample_weights[1]
        val_data = (*val_data, val_sample_weight)
    else:
        train_sample_weight = None
        
    history = model.fit(X_tr, 
                        y_tr, 
                        batch_size = BATCH_SIZE, 
                        epochs = EPOCHS, 
                        shuffle = True, 
                        validation_data = val_data, 
                        callbacks = callbacks,
                        verbose = verbose,
                        sample_weight=train_sample_weight)
    
    # evaluation of validation set
    y_val_prob = model.predict(X_val)
    eval_report, macro_f1 = model_score(y_val.reshape(-1), np.argmax(y_val_prob, axis=-1).reshape(-1))
    
    # prediction
    y_test_prob = model.predict(X_test)
    
    # add validation F1 history         
    history.history['val_F1'] = callbacks[0].score
    
    gc.collect()
    
    return history, model, eval_report, y_val_prob, macro_f1, y_test_prob


def full_pipeline(train_data, test_data, model=None, sample_weights=None):
    """
    Input:
        parameters = dict of all the relevant parameters to assign model and dataset
        train_data = Dataframe of train data
        test_data = Dataframe of test data
    """
    
    seed_everything(SEED)
    
    # global parameters
    parameters = {'SEQUENCE LENGTH':SEQUENCE_LENGTH,
                  'Number of Folds:':N_FOLD,
                  'IF TRAIN SAMPLE AUGMENT:':DATA_AUGMENT,
                  'IF AUGMENT REVERSED:':AUGMENT_REVERSE,
                  'IF AUGMENT GAUSSIAN:':AUGMENT_GAUSSIAN,
                  'IF ADD SHIFTED FEATURE:':SHIFTED_FEATURE,
                  'IF USE SAMPLE WEIGHTS:':WEIGHTED_SAMPLE,
                  'IF ADD GROUPED FEATURE:':GROUPED_FEATURE,
                  'IF SQUARED FEATURE:':SQUARED_FEATURE,
                  'IF STRATIFIED SPLIT:':STRATIFIED_SPLIT,
                  'IF RFPROB FEATURE:':RFPROB_FEATURE,
                  'IF KNNPROB FEATURE:':KNNPROB_FEATURE,
                  'BATCH SIZE':BATCH_SIZE, 
                  'EPOCHS':EPOCHS,
                  'LEARNING RATE':LEARNING_RATE,              
                  'STACK OF LAYERS:':STACK_LAYERS,
                  'FIRST FILTER:':FIRST_FILTER,
                  'PREPROCESS:':PREPROCESS,
                  'POSTPROCESS:':POSTPROCESS,
                  'IF BATCH NORMALIZATION:':BATCHNORM,
                  'IF POST DROPOUT:':POST_DROPOUT,
                 }

    pprint.pprint(parameters)    
    
    # prepare data feature
    X_train, y_train, X_test, sample_batch_id = data_feature_prep(train_data,
                                                                  test_data,
                                                                  sample_length=SEQUENCE_LENGTH,
                                                                  shifted_feature=SHIFTED_FEATURE,
                                                                  grouped_feature=GROUPED_FEATURE,
                                                                  squared_feature=SQUARED_FEATURE,
                                                                  rfprob_feature=RFPROB_FEATURE,
                                                                  knnprob_feature=KNNPROB_FEATURE
                                                                 )

    # split training set
    splits = data_split(X_train, y_train, sample_batch_id, n_fold=N_FOLD, stratified_split=STRATIFIED_SPLIT)

#     # USE TPU
#     # detect and init the TPU
#     tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
#     tf.config.experimental_connect_to_cluster(tpu)
#     tf.tpu.experimental.initialize_tpu_system(tpu)

#     # instantiate a distribution strategy
#     tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

#     # instantiating the model in the strategy scope creates the model on the TPU
#     with tpu_strategy.scope():
#         model = build_Encoder_WaveNet(n_seq=SEQUENCE_LENGTH, 
#                                       n_feature=X_tr.shape[-1], 
#                                       n_stack_layer=STACK_LAYERS, 
#                                       n_first_filter=FIRST_FILTER, 
#                                       batchnorm=BATCHNORM, 
#                                       preprocess=PREPROCESS, 
#                                       postprocess=POSTPROCESS, 
#                                       post_dropout=POST_DROPOUT)

    oof_f1 = []
    oof_val_prob = np.zeros((y_train.shape[0], y_train.shape[1], 11))
    oof_test_prob = np.zeros((X_test.shape[0]*X_test.shape[1], 11, N_FOLD))
    
    for fold, (train_id, val_id) in enumerate(splits):
        print('== START FOLD {} =='.format(fold))

        X_tr, y_tr = X_train[train_id], y_train[train_id]
        X_val, y_val = X_train[val_id], y_train[val_id]

        if DATA_AUGMENT: X_tr, y_tr = add_augment_data(X_tr, y_tr, train_id, reverse=AUGMENT_REVERSE, gaussian_noise=AUGMENT_GAUSSIAN)  # augment data by flipping time dimension
        if WEIGHTED_SAMPLE: sample_weights = add_sample_weights(y_tr, y_val, WEIGHTED_SAMPLE)  # weighted sample

        # build model
        model = build_WaveNet(n_seq=SEQUENCE_LENGTH,
                              n_feature=X_train.shape[-1],
                              n_stack_layer=STACK_LAYERS,
                              n_first_filter=FIRST_FILTER,
                              batchnorm=BATCHNORM,
                              preprocess=PREPROCESS,
                              postprocess=POSTPROCESS,
                              post_dropout=POST_DROPOUT)

        opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        opt = tfa.optimizers.SWA(opt)
        model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'], sample_weight_mode='temporal')

        # define model callbacks
        lr_scheduler = LearningRateScheduler(lr_schedule)
        val_F1 = Val_MacroF1(model, X_val, y_val)
        early_stop = EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE, verbose=1, mode='min', restore_best_weights=True)
        callbacks = [val_F1, lr_scheduler, early_stop]
        
        # run model        
        history, model, eval_report, y_val_prob, macro_f1, y_test_prob = run_model(model, X_tr, y_tr, X_val, y_val, X_test, callbacks, verbose=VERBOSE, sample_weights=sample_weights)

        # save model
        model.save('{}_fold_{}.h5'.format(MODEL_NAME, fold))
        
        # plot
        plot_history(history.history)

        # export results
        export_results_report(parameters, eval_report, history.history)  
    
        # fold prediction
        oof_f1.append(macro_f1)
        if fold == 0:
            best_f1 = macro_f1            
        elif macro_f1 > best_f1:
            best_f1 = macro_f1

        oof_val_prob[val_id,:,:] = y_val_prob.astype(np.float32)
        oof_test_prob[:,:,fold] = y_test_prob.reshape(-1,11).astype(np.float32)
    
        gc.collect()
    
    print('FINISH TRAINING')
    print('BEST OOF F1 SCORE = {} IN FOLD {}'.format(np.max(oof_f1), np.argmax(oof_f1)))

    # save oof prediction
    np.save('oof_val_prob.npy', oof_val_prob)
    np.save('oof_test_prob.npy', oof_test_prob)
    
    # predict on the test set
    if SAVE_SUBMIT == True:
        preds = np.argmax(np.mean(oof_test_prob, axis=-1), axis=-1)
        save_submit(preds, test_data)
    
    return history, model, splits


### BASELINE Model
# baseline model following https://www.kaggle.com/siavrez/wavenet-keras

SAMPLE_BATCH_LENGTH = 500000
STAT_RECEPT_LENGTHES = []
SEQUENCE_LENGTH = 4000
BATCH_SIZE = 20
EPOCHS = 120

N_FOLD = None
DATA_AUGMENT = False
AUGMENT_REVERSE = False
AUGMENT_GAUSSIAN = False
GROUPED_FEATURE = False
SHIFTED_FEATURE = False
WEIGHTED_SAMPLE = False
SQUARED_FEATURE = False
STRATIFIED_SPLIT = True
RFPROB_FEATURE = False
KNNPROB_FEATURE = False

MODEL_NAME = 'BASELINE_4000'
SAVE_SUBMIT = False
SUBMIT_TO_KAGGLE = False
SUBMIT_NAME = 'submission.csv'
SUBMIT_MSG = 'wavenet'

LEARNING_RATE = 0.001
VERBOSE = 1
STACK_LAYERS = [12,8,4,1]
EARLY_STOP_PATIENCE = 40
FIRST_FILTER = 16
PREPROCESS = None # 'LSTM', 'Conv'
POSTPROCESS = None # 'LSTM', 'Conv'
POST_DROPOUT = None
BATCHNORM = False


### BEST LB SETTINGS
SEED = 24
SEQUENCE_LENGTH = 5000
MODEL_NAME = 'BEST_LB'
EPOCHS = 180
EARLY_STOP_PATIENCE = 100

STACK_LAYERS = [13,10,7,4,1]
PREPROCESS = {'name':'Conv', 'units':64, 'kernel_size':7}
POSTPROCESS = {'name':'Conv', 'units':32, 'kernel_size':7}
POST_DROPOUT = {'rate':0.2}
BATCHNORM = True
LEARNING_RATE = 0.0015
VERBOSE = 0

N_FOLD = 5
DATA_AUGMENT = True
AUGMENT_REVERSE = True
AUGMENT_GAUSSIAN = True
BATCH_SIZE = 16
SHIFTED_FEATURE = True
GROUPED_FEATURE = False
WEIGHTED_SAMPLE = False
SQUARED_FEATURE = True
STRATIFIED_SPLIT = True
RFPROB_FEATURE = True

SAVE_SUBMIT = True
SUBMIT_TO_KAGGLE = False

### Start Trraining
history, model, splits = full_pipeline(train_data, test_data)