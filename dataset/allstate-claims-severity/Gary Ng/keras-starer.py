## https://www.kaggle.com/mtinti/allstate-claims-severity/keras-starter-with-bagging-1111-84364/code
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Activation,Dense,Dropout
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack,csr_matrix
from scipy.stats import skew,boxcox
import itertools
from keras.layers.normalization import BatchNormalization
import datetime

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
shift = 200

target = np.log(train.loss.values + shift)
train_ids = train.id.values
test_ids = test.id.values
ntrain = train.shape[0]
train.drop(['id','loss'],axis=1,inplace=True)
test.drop(['id'],axis=1,inplace=True)
all_df = pd.concat((train,test),axis=0)

## category
cat_list = [col for col in all_df.columns if 'cat' in col]
## numeric
cont_list = [col for col in all_df.columns if 'cont' in col]

sparsed_data = []
for col in cat_list:
    dummies = pd.get_dummies(all_df[col].astype('category'))
    sparse = csr_matrix(dummies)
    sparsed_data.append(sparse)


scaler = StandardScaler()
sparsed_data.append(csr_matrix(scaler.fit_transform(all_df[cont_list])))
data = hstack(sparsed_data,format='csr')

x_train = data[:ntrain,:]
x_test = data[ntrain:,:]
del (data,scaler,sparsed_data,cat_list,cont_list)

### Batch generator ###

def batch_generator(X,y,batch_size,shuffle):
    number_of_batch = np.ceil(X.shape[0] / batch_size)
    count = 0 
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size * count : batch_size *(count +1)]
        X_batch = X[batch_index,:].toarray()
        y_batch = y[batch_index]
        count +=1
        yield X_batch,y_batch
        
        if count == number_of_batch:
            if shuffle:
                np.random.shuffle(sample_index)
            count =0

def batch_generatorP(X,batch_size,shuffle):
    number_of_batch = X.shape[0] / np.ceil(X.shape[0] / batch_size)
    count = 0 
    sample_index = np.arange(X.shape[0])
    
    while True:
        batch_index = sample_index[batch_size * count : batch_size *(count +1)]
        X_batch = X[batch_index,:].toarray()
        count +=1
        yield X_batch
        if count == number_of_batch:
            count = 0
##########    

### Build model ####

def create_model():
    
    model = Sequential()
    model.add(Dense(400,input_dim=x_train.shape[1],activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.499))
    
    
    model.add(Dense(200,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.499))
    
    model.add(Dense(50,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Dense(1))
    model.compile(loss='mae',optimizer='adadelta')
    
    return model
###########


nfold = 2
kf = KFold(len(target),n_folds=nfold,shuffle=True,random_state=42)
num_fold = 0
nb_epoch = 3
nb_bag = 3
pred_train = np.zeros(x_train.shape[0])
pred_test = np.zeros(x_test.shape[0])

for i,(train_idx,test_idx) in enumerate(kf):
    
    X_train = x_train[train_idx]
    X_valid = x_train[test_idx]
    y_train = target[train_idx]
    y_valid = target[test_idx]
    pred = np.zeros(X_valid.shape[0])

    for j in range(nb_bag):
        model = create_model()
        clf = model.fit_generator(generator=batch_generator(X_train,y_train,128,True),
                                  nb_epoch = nb_epoch,
                                  samples_per_epoch = X_train.shape[0],
                                  verbose=0)
        pred += np.exp(model.predict_generator(generator=batch_generatorP(X_valid,800,False),
                                val_samples = X_valid.shape[0])[:,0]) - shift
        
        pred_test += np.exp(model.predict_generator(generator=batch_generatorP(x_test,800,False),
                            val_samples=x_test.shape[0])[:,0]) - shift
    pred /= nb_bag
    pred_train[test_idx]=pred
    score = mean_absolute_error(np.exp(y_valid) - shift,pred)
    print('Fold {} , MAE : {}'.format(i,score))

print('Total MAE : {}'.format(mean_absolute_error(np.exp(target) - shift,pred_train)))

print('Writing.....')
### train predictions
train_pred_df = pd.DataFrame({'id':train_ids,'loss':pred_train})
train_pred_df.to_csv('train_prediction.csv',index=False)


## test predictions
pred_test /= (nfold * nb_bag) ## 
test_pred_df = pd.DataFrame({'id':test_ids,'loss':pred_test})
test_pred_df.to_csv('test_prediction.csv',index=False)
print('Finished.....')





