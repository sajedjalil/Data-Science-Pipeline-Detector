'''
    This script provides code for training a neural network with entity embeddings
    of the 'cat' variables. For more details on entity embedding, see:
    https://github.com/entron/entity-embedding-rossmann
    
    8-Fold training with 3 averaged runs per fold. Results may improve with more folds & runs.
'''

import numpy as np
import pandas as pd

#random seeds for stochastic parts of neural network 
np.random.seed(10)
from tensorflow import set_random_seed
set_random_seed(15)

from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Reshape, Dropout
from keras.layers.embeddings import Embedding

from sklearn.model_selection import StratifiedKFold

#Data loading & preprocessing
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

X_train, y_train = df_train.iloc[:,2:], df_train.target
X_test = df_test.iloc[:,1:]

cols_use = [c for c in X_train.columns if (not c.startswith('ps_calc_'))]

X_train = X_train[cols_use]
X_test = X_test[cols_use]

col_vals_dict = {c: list(X_train[c].unique()) for c in X_train.columns if c.endswith('_cat')}

embed_cols = []
for c in col_vals_dict:
    if len(col_vals_dict[c])>2:
        embed_cols.append(c)
        print(c + ': %d values' % len(col_vals_dict[c])) #look at value counts to know the embedding dimensions

print('\n')

def build_embedding_network():
    
    inputs = []
    embeddings = []
    
    input_ps_ind_02_cat = Input(shape=(1,))
    embedding = Embedding(5, 3, input_length=1)(input_ps_ind_02_cat)
    embedding = Reshape(target_shape=(3,))(embedding)
    inputs.append(input_ps_ind_02_cat)
    embeddings.append(embedding)
    
    input_ps_ind_04_cat = Input(shape=(1,))
    embedding = Embedding(3, 2, input_length=1)(input_ps_ind_04_cat)
    embedding = Reshape(target_shape=(2,))(embedding)
    inputs.append(input_ps_ind_04_cat)
    embeddings.append(embedding)
    
    input_ps_ind_05_cat = Input(shape=(1,))
    embedding = Embedding(8, 5, input_length=1)(input_ps_ind_05_cat)
    embedding = Reshape(target_shape=(5,))(embedding)
    inputs.append(input_ps_ind_05_cat)
    embeddings.append(embedding)
    
    input_ps_car_01_cat = Input(shape=(1,))
    embedding = Embedding(13, 7, input_length=1)(input_ps_car_01_cat)
    embedding = Reshape(target_shape=(7,))(embedding)
    inputs.append(input_ps_car_01_cat)
    embeddings.append(embedding)
    
    input_ps_car_02_cat = Input(shape=(1,))
    embedding = Embedding(3, 2, input_length=1)(input_ps_car_02_cat)
    embedding = Reshape(target_shape=(2,))(embedding)
    inputs.append(input_ps_car_02_cat)
    embeddings.append(embedding)
    
    input_ps_car_03_cat = Input(shape=(1,))
    embedding = Embedding(3, 2, input_length=1)(input_ps_car_03_cat)
    embedding = Reshape(target_shape=(2,))(embedding)
    inputs.append(input_ps_car_03_cat)
    embeddings.append(embedding)
    
    input_ps_car_04_cat = Input(shape=(1,))
    embedding = Embedding(10, 5, input_length=1)(input_ps_car_04_cat)
    embedding = Reshape(target_shape=(5,))(embedding)
    inputs.append(input_ps_car_04_cat)
    embeddings.append(embedding)
    
    input_ps_car_05_cat = Input(shape=(1,))
    embedding = Embedding(3, 2, input_length=1)(input_ps_car_05_cat)
    embedding = Reshape(target_shape=(2,))(embedding)
    inputs.append(input_ps_car_05_cat)
    embeddings.append(embedding)
    
    input_ps_car_06_cat = Input(shape=(1,))
    embedding = Embedding(18, 8, input_length=1)(input_ps_car_06_cat)
    embedding = Reshape(target_shape=(8,))(embedding)
    inputs.append(input_ps_car_06_cat)
    embeddings.append(embedding)
    
    input_ps_car_07_cat = Input(shape=(1,))
    embedding = Embedding(3, 2, input_length=1)(input_ps_car_07_cat)
    embedding = Reshape(target_shape=(2,))(embedding)
    inputs.append(input_ps_car_07_cat)
    embeddings.append(embedding)
    
    input_ps_car_09_cat = Input(shape=(1,))
    embedding = Embedding(6, 3, input_length=1)(input_ps_car_09_cat)
    embedding = Reshape(target_shape=(3,))(embedding)
    inputs.append(input_ps_car_09_cat)
    embeddings.append(embedding)
    
    input_ps_car_10_cat = Input(shape=(1,))
    embedding = Embedding(3, 2, input_length=1)(input_ps_car_10_cat)
    embedding = Reshape(target_shape=(2,))(embedding)
    inputs.append(input_ps_car_10_cat)
    embeddings.append(embedding)
    
    input_ps_car_11_cat = Input(shape=(1,))
    embedding = Embedding(104, 10, input_length=1)(input_ps_car_11_cat)
    embedding = Reshape(target_shape=(10,))(embedding)
    inputs.append(input_ps_car_11_cat)
    embeddings.append(embedding)
    
    input_numeric = Input(shape=(24,))
    embedding_numeric = Dense(16)(input_numeric) 
    inputs.append(input_numeric)
    embeddings.append(embedding_numeric)

    x = Concatenate()(embeddings)
    x = Dense(80, activation='relu')(x)
    x = Dropout(.35)(x)
    x = Dense(20, activation='relu')(x)
    x = Dropout(.15)(x)
    x = Dense(10, activation='relu')(x)
    x = Dropout(.15)(x)
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, output)

    model.compile(loss='binary_crossentropy', optimizer='adam')
    
    return model

#converting data to list format to match the network structure
def preproc(X_train, X_val, X_test):

    input_list_train = []
    input_list_val = []
    input_list_test = []
    
    #the cols to be embedded: rescaling to range [0, # values)
    for c in embed_cols:
        raw_vals = np.unique(X_train[c])
        val_map = {}
        for i in range(len(raw_vals)):
            val_map[raw_vals[i]] = i       
        input_list_train.append(X_train[c].map(val_map).values)
        input_list_val.append(X_val[c].map(val_map).fillna(0).values)
        input_list_test.append(X_test[c].map(val_map).fillna(0).values)
     
    #the rest of the columns
    other_cols = [c for c in X_train.columns if (not c in embed_cols)]
    input_list_train.append(X_train[other_cols].values)
    input_list_val.append(X_val[other_cols].values)
    input_list_test.append(X_test[other_cols].values)
    
    return input_list_train, input_list_val, input_list_test    

#gini scoring function from kernel at: 
#https://www.kaggle.com/tezdhar/faster-gini-calculation
def ginic(actual, pred):
    n = len(actual)
    a_s = actual[np.argsort(pred)]
    a_c = a_s.cumsum()
    giniSum = a_c.sum() / a_c[-1] - (n + 1) / 2.0
    return giniSum / n
 
def gini_normalizedc(a, p):
    return ginic(a, p) / ginic(a, a)

#network training
K = 8
runs_per_fold = 3
n_epochs = 15

cv_ginis = []
full_val_preds = np.zeros(np.shape(X_train)[0])
y_preds = np.zeros((np.shape(X_test)[0],K))

kfold = StratifiedKFold(n_splits = K, 
                            random_state = 231, 
                            shuffle = True)    

for i, (f_ind, outf_ind) in enumerate(kfold.split(X_train, y_train)):

    X_train_f, X_val_f = X_train.loc[f_ind].copy(), X_train.loc[outf_ind].copy()
    y_train_f, y_val_f = y_train[f_ind], y_train[outf_ind]
    
    X_test_f = X_test.copy()
    
    #upsampling adapted from kernel: 
    #https://www.kaggle.com/ogrellier/xgb-classifier-upsampling-lb-0-283
    pos = (pd.Series(y_train_f == 1))
    
    # Add positive examples
    X_train_f = pd.concat([X_train_f, X_train_f.loc[pos]], axis=0)
    y_train_f = pd.concat([y_train_f, y_train_f.loc[pos]], axis=0)
    
    # Shuffle data
    idx = np.arange(len(X_train_f))
    np.random.shuffle(idx)
    X_train_f = X_train_f.iloc[idx]
    y_train_f = y_train_f.iloc[idx]
    
    #preprocessing
    proc_X_train_f, proc_X_val_f, proc_X_test_f = preproc(X_train_f, X_val_f, X_test_f)
    
    #track oof prediction for cv scores
    val_preds = 0
    
    for j in range(runs_per_fold):
    
        NN = build_embedding_network()
        NN.fit(proc_X_train_f, y_train_f.values, epochs=n_epochs, batch_size=4096, verbose=0)
   
        val_preds += NN.predict(proc_X_val_f)[:,0] / runs_per_fold
        y_preds[:,i] += NN.predict(proc_X_test_f)[:,0] / runs_per_fold
        
    full_val_preds[outf_ind] += val_preds
        
    cv_gini = gini_normalizedc(y_val_f.values, val_preds)
    cv_ginis.append(cv_gini)
    print ('\nFold %i prediction cv gini: %.5f\n' %(i,cv_gini))
    
print('Mean out of fold gini: %.5f' % np.mean(cv_ginis))
print('Full validation gini: %.5f' % gini_normalizedc(y_train.values, full_val_preds))

y_pred_final = np.mean(y_preds, axis=1)

df_sub = pd.DataFrame({'id' : df_test.id, 
                       'target' : y_pred_final},
                       columns = ['id','target'])
df_sub.to_csv('NN_EntityEmbed_10fold-sub.csv', index=False)

pd.DataFrame(full_val_preds).to_csv('NN_EntityEmbed_10fold-val_preds.csv',index=False)