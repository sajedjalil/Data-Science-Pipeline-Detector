
#dataloading etc adopted from @jeffd23 script



import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D,ZeroPadding2D
from keras.optimizers import Adam , RMSprop, Adadelta, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.layers.core import Dense, Dropout, Activation, Flatten, MaxoutDense
from keras.layers.advanced_activations import PReLU,ELU


act_train = pd.read_csv('../input/act_train.csv')
act_test = pd.read_csv('../input/act_test.csv')
people = pd.read_csv('../input/people.csv')

# Save the test IDs for Kaggle submission
test_ids = act_test['activity_id']

def preprocess_acts(data, train_set=True):
    
    # Getting rid of data feature for now
    data = data.drop(['date', 'activity_id'], axis=1)
    if(train_set):
        data = data.drop(['outcome'], axis=1)
    
    ## Split off _ from people_id
    data['people_id'] = data['people_id'].apply(lambda x: x.split('_')[1])
    data['people_id'] = pd.to_numeric(data['people_id']).astype(int)
    
    columns = list(data.columns)
    
    # Convert strings to ints
    for col in columns[1:]:
        data[col] = data[col].fillna('type 0')
        data[col] = data[col].apply(lambda x: x.split(' ')[1])
        data[col] = pd.to_numeric(data[col]).astype(int)
    return data

def preprocess_people(data):
    
    # TODO refactor this duplication
    data = data.drop(['date'], axis=1)
    data['people_id'] = data['people_id'].apply(lambda x: x.split('_')[1])
    data['people_id'] = pd.to_numeric(data['people_id']).astype(int)
    
    #  Values in the people df is Booleans and Strings    
    columns = list(data.columns)
    bools = columns[11:]
    strings = columns[1:11]
    
    for col in bools:
        data[col] = pd.to_numeric(data[col]).astype(int)        
    for col in strings:
        data[col] = data[col].fillna('type 0')
        data[col] = data[col].apply(lambda x: x.split(' ')[1])
        data[col] = pd.to_numeric(data[col]).astype(int)
    return data
    
    # Preprocess each df
peeps = preprocess_people(people)
actions_train = preprocess_acts(act_train)
actions_test = preprocess_acts(act_test, train_set=False)

# Merege into a unified table

# Training 
features = actions_train.merge(peeps, how='left', on='people_id')
labels = act_train['outcome']

# Testing
test = actions_test.merge(peeps, how='left', on='people_id')

# Check it out...
features.sample(10)



from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping

def create_model_v1( input_dim):
    nb_classes = 1
    # number of convolutional filters to use
 
    model = Sequential()

  
    model.add(Dense(100,input_dim=input_dim,activation='relu'))
   
  
    model.add(Dense(100,activation='relu'))
  
   
    
    model.add(Dense(nb_classes))
    model.add(Activation('sigmoid'))

    sgd = SGD(lr=0.05, decay=0, momentum=0.95, nesterov=True)
    #sgd = SGD(lr=1e-2, decay=1e-6)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

from sklearn import preprocessing


from sklearn.cross_validation import train_test_split
features = features.as_matrix()
scaler = preprocessing.StandardScaler().fit(features)
features = scaler.transform(features)   
num_test = 0.20
X_train, X_test, y_train, y_test = train_test_split(features, labels.as_matrix(), test_size=num_test, random_state=1337)

model_checkpoint = ModelCheckpoint('redhat1.hdf5', monitor='val_loss', save_best_only=True)

input_dim = X_train.shape[1]

model= create_model_v1(input_dim)

print("Start fitting the model")

model.fit(X_train , y_train, batch_size=100, nb_epoch=1, validation_data =(X_test,y_test) ,
          verbose=1, shuffle=True,callbacks=[model_checkpoint])

test= scaler.transform(test.as_matrix())

model.load_weights('redhat1.hdf5') 
proba= model.predict(X_test, verbose=1)
test_proba = model.predict(test, verbose=1)

print(np.shape(proba))

## Out of box random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.grid_search import GridSearchCV
print("start predicting")
#clf = RandomForestClassifier()
#clf.fit(X_train, y_train)

#proba = clf.predict_proba(X_test)
preds = proba
score = roc_auc_score(y_test, preds)
print("Area under ROC {0}".format(score))


#test_proba = clf.predict_proba(test)
test_preds = test_proba.flatten()

print(np.shape(test_preds))
# Format for submission
output = pd.DataFrame({ 'activity_id' : test_ids, 'outcome': test_preds })

output.to_csv('redhat.csv', index = False)
    
    
    