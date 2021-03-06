from keras.regularizers import l2, activity_l2
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss, auc, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from keras.optimizers import Adagrad,SGD,Adadelta
from keras.callbacks import EarlyStopping
from keras.layers import containers
from keras.layers.core import Dense, AutoEncoder

np.random.seed(1778)  # for reproducibility
need_normalise=True
need_validataion=True
nb_epoch=2#8
golden_feature=[("CoverageField1B","PropertyField21B"),
                ("GeographicField6A","GeographicField8A"),
                ("GeographicField6A","GeographicField13A"),
                ("GeographicField8A","GeographicField13A"),
                ("GeographicField11A","GeographicField13A"),
                ("GeographicField8A","GeographicField11A")]

def save2model(submission,file_name,y_pre):
    assert len(y_pre)==len(submission)
    submission['QuoteConversion_Flag']=y_pre
    submission.to_csv(file_name,index=False)
    print ("saved files %s" % file_name)

def generateFileName(model,params):
     file_name="_".join([(key+"_"+ str(val))for key,val in params.items()])
     return model+"_"+file_name+".csv"

def load_data():
    train=pd.read_csv('../input/train.csv')
    test=pd.read_csv('../input/test.csv')
    train = train.drop(['QuoteNumber','PersonalField7','PersonalField8','PersonalField13','PersonalField34','PersonalField35','PersonalField36','PersonalField37','PersonalField38','PersonalField39','PersonalField40','PersonalField41','PersonalField42','PersonalField43','PersonalField49','PersonalField50','PersonalField51','PersonalField52','PersonalField53','PersonalField54','PersonalField55','PersonalField56','PersonalField57','PersonalField58','PersonalField59','PersonalField60','PersonalField61','PersonalField62','PersonalField63','PersonalField64','PersonalField65','PersonalField66','PersonalField67','PersonalField68','PersonalField69','PersonalField70','PersonalField71','PersonalField72','PersonalField73','PropertyField5','PropertyField9','PropertyField20','PropertyField22','PropertyField10','PropertyField6', 'GeographicField10A','GeographicField10B'], axis=1)
    
    submission=pd.DataFrame()
    submission["QuoteNumber"]= test["QuoteNumber"]
    
    test = test.drop(['QuoteNumber','PersonalField7','PersonalField8','PersonalField13','PersonalField34','PersonalField35','PersonalField36','PersonalField37','PersonalField38','PersonalField39','PersonalField40','PersonalField41','PersonalField42','PersonalField43','PersonalField49','PersonalField50','PersonalField51','PersonalField52','PersonalField53','PersonalField54','PersonalField55','PersonalField56','PersonalField57','PersonalField58','PersonalField59','PersonalField60','PersonalField61','PersonalField62','PersonalField63','PersonalField64','PersonalField65','PersonalField66','PersonalField67','PersonalField68','PersonalField69','PersonalField70','PersonalField71','PersonalField72','PersonalField73','PropertyField5','PropertyField9','PropertyField20','PropertyField22','PropertyField10','PropertyField6', 'GeographicField10A','GeographicField10B'],axis=1)
    train['Date'] = pd.to_datetime(pd.Series(train['Original_Quote_Date']))
    train = train.drop('Original_Quote_Date', axis=1)
    train['Year'] = train['Date'].apply(lambda x: int(str(x)[:4]))
    train['Month'] = train['Date'].apply(lambda x: int(str(x)[5:7]))
    train['weekday'] = [train['Date'][i].dayofweek for i in range(len(train['Date']))]
    
    test['Date'] = pd.to_datetime(pd.Series(test['Original_Quote_Date']))
    test = test.drop('Original_Quote_Date', axis=1)
    test['Year'] = test['Date'].apply(lambda x: int(str(x)[:4]))
    test['Month'] = test['Date'].apply(lambda x: int(str(x)[5:7]))
    test['weekday'] = [test['Date'][i].dayofweek for i in range(len(test['Date']))]
    
    train = train.drop('Date', axis=1)
    test = test.drop('Date', axis=1)
    
    #fill na
    train = train.fillna(-1)
    test = test.fillna(-1)
    
    for f in test.columns:# train has QuoteConversion_Flag
        if train[f].dtype=='object':
            lbl = LabelEncoder()
            lbl.fit(list(train[f])+list(test[f]))
            train[f] = lbl.transform(list(train[f].values))
            test[f] = lbl.transform(list(test[f].values))
    
    train_y=train['QuoteConversion_Flag'].values
    encoder = LabelEncoder()
    train_y = encoder.fit_transform(train_y).astype(np.int32)
    train_y = np_utils.to_categorical(train_y)
    train=train.drop('QuoteConversion_Flag',axis=1)
    
    #add golden feature:
    for featureA,featureB in golden_feature:
        train["_".join([featureA,featureB,"diff"])]=train[featureA]-train[featureB]
        test["_".join([featureA,featureB,"diff"])]=test[featureA]-train[featureB]        

    print ("processsing finished")
    valid=None
    valid_y=None
    train = np.array(train)
    train = train.astype(np.float32)
    test=np.array(test)
    test=test.astype(np.float32)
    if need_normalise:
        scaler = StandardScaler().fit(train)
        train = scaler.transform(train)
        test = scaler.transform(test)
    
    if need_validataion:
        train,valid,train_y,valid_y=train_test_split(train,train_y,test_size=20000,random_state=218)
    return [(train,train_y),(test,submission),(valid,valid_y)]

print('Loading data...')

datasets=load_data()

X_train, y_train = datasets[0]
X_test, submission = datasets[1]
X_valid, y_valid = datasets[2]

nb_classes = y_train.shape[1]
print(nb_classes, 'classes')

dims = X_train.shape[1]
print(dims, 'dims')

model = Sequential()

model.add(Dense(128, input_shape=(dims,),W_regularizer=l2(0.0001)))
model.add(PReLU())
model.add(BatchNormalization())
model.add(Dropout(0.5))

#model.add(Dense(300,W_regularizer=l2(0.0001)))
#model.add(PReLU())
#model.add(BatchNormalization())
#model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))
opt=Adadelta(lr=1,decay=0.995,epsilon=1e-5)
model.compile(loss='binary_crossentropy', optimizer=opt)
auc_scores=[]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
best_score=-1
best_model=None
print('Training model...')
if need_validataion:
    for i in range(nb_epoch):
    #early_stopping=EarlyStopping(monitor='val_loss', patience=0, verbose=1)
    #model.fit(X_train, y_train, nb_epoch=nb_epoch,batch_size=256,validation_split=0.01,callbacks=[early_stopping])
        model.fit(X_train, y_train, nb_epoch=1,batch_size=256,validation_split=0.01)
        y_pre = model.predict_proba(X_valid)
        scores = roc_auc_score(y_valid,y_pre)
        auc_scores.append(scores)
        print (i,scores)
        if scores>best_score:
            best_score=scores
            best_model=model
    plt.plot(auc_scores)
    plt.show()
else:
    model.fit(X_train, y_train, nb_epoch=nb_epoch, batch_size=256)

#print('Generating submission...')
y_pre = model.predict_proba(X_test)[:,1]
#print roc_auc_score(y_test,y_pre)
save2model(submission, 'keras_nn_test.csv',y_pre)