import pandas as pd
import numpy as np
from scipy import sparse as ssp
import pylab as plt
from sklearn.preprocessing import LabelEncoder,LabelBinarizer,MinMaxScaler,OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.decomposition import TruncatedSVD,NMF,PCA,FactorAnalysis
from sklearn.feature_selection import SelectFromModel,SelectPercentile,f_classif
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import log_loss,roc_auc_score
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.cross_validation import StratifiedKFold,KFold
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint,Callback
from keras import backend as K
from keras.layers import Input, Embedding, LSTM, Dense,Flatten, Dropout, merge,Convolution1D,MaxPooling1D,Lambda,AveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.layers.advanced_activations import PReLU,LeakyReLU,ELU,SReLU
from keras.models import Model

seed = 1
np.random.seed(seed)
dim = 32
hidden=64

path = "../input/"

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
                print('Epoch %05d: early stopping' % (epoch))
                
                
            self.wait += 1 #incremental the number of times without improvement
        print('Epoch %d Auc: %f | Best Auc: %f \n' % (epoch,current,self.best))


def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size/float(batch_size)))
    return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch)]



def main():
    train = pd.read_csv(path+'act_train.csv')
    test = pd.read_csv(path+'act_test.csv')
    people = pd.read_csv(path+'people.csv')
    columns = people.columns
    test['outcome'] = np.nan
    data = pd.concat([train,test])
    
    data = pd.merge(data,people,how='left',on='people_id').fillna('missing')
    train = data[:train.shape[0]]
    test = data[train.shape[0]:]



    columns = train.columns.tolist()
    columns.remove('activity_id')
    columns.remove('outcome')
    data = pd.concat([train,test])
    for c in columns:
        data[c] = LabelEncoder().fit_transform(data[c].values)

    train = data[:train.shape[0]]
    test = data[train.shape[0]:]
    
    data = pd.concat([train,test])
    columns = train.columns.tolist()
    columns.remove('activity_id')
    columns.remove('outcome')
    flatten_layers = []
    inputs = []
    for c in columns:
        
        inputs_c = Input(shape=(1,), dtype='int32')

        num_c = len(np.unique(data[c].values))

        embed_c = Embedding(
                        num_c,
                        dim,
                        dropout=0.2,
                        input_length=1
                        )(inputs_c)
        flatten_c= Flatten()(embed_c)

        inputs.append(inputs_c)
        flatten_layers.append(flatten_c)

    flatten = merge(flatten_layers,mode='concat')
    
    fc1 = Dense(hidden,activation='relu')(flatten)
    dp1 = Dropout(0.5)(fc1)

    outputs = Dense(1,activation='sigmoid')(dp1)

    model = Model(input=inputs, output=outputs)
    model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
              )

    del data

    X = train[columns].values
    X_t = test[columns].values
    y = train["outcome"].values
    people_id = train["people_id"].values
    activity_id = test['activity_id']
    del train
    del test

    skf = StratifiedKFold(y, n_folds=4, shuffle=True, random_state=seed)
    for ind_tr, ind_te in skf:
        X_train = X[ind_tr]
        X_test = X[ind_te]

        y_train = y[ind_tr]
        y_test = y[ind_te]
        break
    
    X_train = [X_train[:,i] for i in range(X.shape[1])]
    X_test = [X_test[:,i] for i in range(X.shape[1])]
    
    del X

    model_name = 'mlp_residual_%s_%s.hdf5'%(dim,hidden)
    model_checkpoint = ModelCheckpoint(model_name, monitor='val_loss', save_best_only=True)
    auc_callback = AucCallback(validation_data=(X_test,y_test), patience=5,is_regression=True,best_model_name=path+'best_keras.mdl',feval='roc_auc_score')
    
    nb_epoch = 10

    batch_size = 1024*8
    load_model = False
    
    if load_model:
        print('Load Model')
        model.load_weights(path+model_name)
        # model.load_weights(path+'best_keras.mdl')

    model.fit(
        X_train, 
        y_train,
        batch_size=batch_size, 
        nb_epoch=nb_epoch, 
        verbose=1, 
        shuffle=True,
        validation_data=[X_test,y_test],
        # callbacks = [
            # model_checkpoint,
            # auc_callback,
            # ],
        )
    
    # model.load_weights(model_name)
    # model.load_weights(path+'best_keras.mdl')
    
    y_preds = model.predict(X_test,batch_size=1024*8)
    # print('auc',roc_auc_score(y_test,y_preds))
    
    # print('Make submission')
    X_t = [X_t[:,i] for i in range(X_t.shape[1])]
    outcome = model.predict(X_t,batch_size=1024*8)
    submission = pd.DataFrame()
    submission['activity_id'] = activity_id
    submission['outcome'] = outcome
    submission.to_csv('submission_residual_%s_%s.csv'%(dim,hidden),index=False)

main()


