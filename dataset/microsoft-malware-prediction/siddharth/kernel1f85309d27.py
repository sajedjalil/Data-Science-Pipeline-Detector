# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os






# Any results you write to the current directory are saved as output.
#../input/train.csv
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adam
import pandas as pd
import math


flag=1

FE = ['EngineVersion','AppVersion','AvSigVersion','Census_OSVersion']
OHE = [ 'RtpStateBitfield','IsSxsPassiveMode','DefaultBrowsersIdentifier',
        'AVProductStatesIdentifier','AVProductsInstalled', 'AVProductsEnabled',
        'CountryIdentifier', 'CityIdentifier', 
        'GeoNameIdentifier', 'LocaleEnglishNameIdentifier',
        'Processor', 'OsBuild', 'OsSuite',
        'SmartScreen','Census_MDC2FormFactor',
        'Census_OEMNameIdentifier', 
        'Census_ProcessorCoreCount',
        'Census_ProcessorModelIdentifier', 
        'Census_PrimaryDiskTotalCapacity', 'Census_PrimaryDiskTypeName',
        'Census_HasOpticalDiskDrive',
        'Census_TotalPhysicalRAM', 'Census_ChassisTypeName',
        'Census_InternalPrimaryDiagonalDisplaySizeInInches',
        'Census_InternalPrimaryDisplayResolutionHorizontal',
        'Census_InternalPrimaryDisplayResolutionVertical',
        'Census_PowerPlatformRoleName', 'Census_InternalBatteryType',
        'Census_InternalBatteryNumberOfCharges',
        'Census_OSEdition', 'Census_OSInstallLanguageIdentifier',
        'Census_GenuineStateName','Census_ActivationChannel',
        'Census_FirmwareManufacturerIdentifier',
        'Census_IsTouchEnabled', 'Census_IsPenCapable',
        'Census_IsAlwaysOnAlwaysConnectedCapable', 'Wdft_IsGamer',
        'Wdft_RegionIdentifier']

dtypes = {}
for x in FE+OHE: dtypes[x] = 'category'
dtypes['MachineIdentifier'] = 'str'
dtypes['HasDetections'] = 'int8'
num=-1500000
print("importing data")
df_train1 = pd.read_csv('../input/train.csv',nrows=1500000, usecols=dtypes.keys(), dtype=dtypes)
print ('Loaded',len(df_train1),'rows of TRAIN.CSV!')
for i in range(1):      
        num=num+1500000
        num1=num+1500000
        
        df_train = df_train1.loc[num:num1]
        
        #print(df_train)
        cols = []; dd = []
        
        
        
        def nan_check(x):
            if isinstance(x,float):
                if math.isnan(x):
                    return True
            return False
        
        # FREQUENCY ENCODING
        def encode_FE(df,col,verbose=1):
            d = df[col].value_counts(dropna=False)
            n = col+"_FE"
            df[n] = df[col].map(d)/d.max()
            #if verbose==1:
             #   print('FE encoded',col)
            return [n]
        
        # ONE-HOT-ENCODE ALL CATEGORY VALUES THAT COMPRISE MORE THAN
        # "FILTER" PERCENT OF TOTAL DATA AND HAS SIGNIFICANCE GREATER THAN "ZVALUE"
        def encode_OHE(df, col, filter, zvalue, tar='HasDetections', m=0.5, verbose=1):
            cv = df[col].value_counts(dropna=False)
            cvd = cv.to_dict()
            vals = len(cv)
            th = filter * len(df)
            sd = zvalue * 0.5/ math.sqrt(th)
            #print(sd)
            n = []; ct = 0; d = {}
            for x in cv.index:
                try:
                    if cv[x]<th: break
                    sd = zvalue * 0.5/ math.sqrt(cv[x])
                except:
                    if cvd[x]<th: break
                    sd = zvalue * 0.5/ math.sqrt(cvd[x])
                if nan_check(x): r = df[df[col].isna()][tar].mean()
                else: r = df[df[col]==x][tar].mean()
                if abs(r-m)>sd:
                    nm = col+'_BE_'+str(x)
                    if nan_check(x): df[nm] = (df[col].isna()).astype('int8')
                    else: df[nm] = (df[col]==x).astype('int8')
                    n.append(nm)
                    d[x] = 1
                ct += 1
                if (ct+1)>=vals: break
            if verbose==1:
                print('OHE encoded',col,'- Created',len(d),'booleans')
            return [n,d]
        
        # ONE-HOT-ENCODING from dictionary
        def encode_OHE_test(df,col,dt):
            n = []
            for x in dt: 
                n += encode_BE(df,col,x)
            return n
        
        # BOOLEAN ENCODING
        def encode_BE(df,col,val):
            n = col+"_BE_"+str(val)
            if nan_check(val):
                df[n] = df[col].isna()
            else:
                df[n] = df[col]==val
            df[n] = df[n].astype('int8')
            return [n]
        
        #print(df_train)
        for x in FE:
            cols += encode_FE(df_train,x)
        for x in OHE:
            tmp = encode_OHE(df_train,x,0.005,5)
            cols += tmp[0]; dd.append(tmp[1])
        from keras import callbacks
        from sklearn.metrics import roc_auc_score
        
        class printAUC(callbacks.Callback):
            def __init__(self, X_train, y_train):
                super(printAUC, self).__init__()
                self.bestAUC = 0
                self.X_train = X_train
                self.y_train = y_train
                
            def on_epoch_end(self, epoch, logs={}):
                pred = self.model.predict(np.array(self.X_train))
                auc = roc_auc_score(self.y_train, pred)
                print("Train AUC: " + str(auc))
                pred = self.model.predict(self.validation_data[0])
                auc = roc_auc_score(self.validation_data[1], pred)
                print ("Validation AUC: " + str(auc))
                if (self.bestAUC < auc) :
                    self.bestAUC = auc
                    self.model.save("bestNet.h5", overwrite=True)
                return
        
        
        for x in FE+OHE:
            del df_train[x]
        
        X_train, X_val, Y_train, Y_val = train_test_split(
            df_train[cols], df_train['HasDetections'], test_size = 0.3)
        print("pre process done")
        
        
        if flag==1:
            print("new model")
            model = Sequential()
            model.add(Dense(400,input_dim=len(cols)))
            model.add(Dropout(0.4))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dense(900))
            model.add(Dropout(0.4))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dense(900))
            model.add(Dropout(0.4))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dense(900))
            model.add(Dropout(0.4))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dense(900))
            model.add(Dropout(0.4))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            
            model.add(Dense(900))
            model.add(Dropout(0.4))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dense(900))
            model.add(Dropout(0.4))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dense(900))
            model.add(Dropout(0.4))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer=Adam(lr=0.01), loss="binary_crossentropy", metrics=["accuracy"])
        annealer = LearningRateScheduler(lambda x: 1e-2 * 0.95 ** x)
        flag=0
            
        print("training")
        model.fit(X_train,Y_train, batch_size=32, epochs = 2, callbacks=[annealer,
                  printAUC(X_train, Y_train)], validation_data = (X_val,Y_val), verbose=2)
                  
                  
          
del df_train
del X_train, X_val, Y_train, Y_val

# LOAD BEST SAVED NET
from keras.models import load_model
model = load_model('bestNet.h5')

pred = np.zeros((7853253,1))
id = 1
chunksize = 2000000
for df_test in pd.read_csv('../input/test.csv', 
            chunksize = chunksize, usecols=list(dtypes.keys())[0:-1], dtype=dtypes):
    print ('Loaded',len(df_test),'rows of TEST.CSV!')
    # ENCODE TEST
    cols = []
    for x in FE:
        cols += encode_FE(df_test,x,verbose=0)
    for x in range(len(OHE)):
        cols += encode_OHE_test(df_test,OHE[x],dd[x])
    # PREDICT TEST
    end = (id)*chunksize
    if end>7853253: end = 7853253
    pred[(id-1)*chunksize:end] = model.predict(df_test[cols])
    print('  encoded and predicted part',id)
    id += 1          
df_test = pd.read_csv('../input/test.csv', usecols=['MachineIdentifier'])
df_test['HasDetections'] = pred
df_test.to_csv('submission1.csv', index=False)          