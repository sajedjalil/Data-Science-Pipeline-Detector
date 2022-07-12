from pandas import Series,DataFrame
names=[
'SalesField9'
'SalesField8',
'SalesField4',
'SalesField3',
'SalesField2B',
'SalesField1B',
'SalesField1A',
'SalesField15',
'SalesField12',
'PropertyField9',
'PropertyField8',
'PropertyField7',
'PropertyField4',
'PropertyField38',
'PropertyField37',
'PropertyField36',
'PropertyField34',
'PropertyField33',
'PropertyField32',
'PropertyField31',
'PropertyField30',
'PropertyField2B',
'PropertyField29',
'PropertyField28',
'PropertyField24B',
'PropertyField22',
'PropertyField21B',
'PropertyField20',
'PropertyField1B',
'PropertyField1A',
'PropertyField18',
'PropertyField16B',
'PropertyField15',
'PropertyField14',
'PropertyField13',
'PersonalField9',
'PersonalField83',
'PersonalField82',
'PersonalField8',
'PersonalField77',
'PersonalField7',
'PersonalField66',
'PersonalField62',
'PersonalField6',
'PersonalField5',
'PersonalField4B',
'PersonalField42',
'PersonalField40',
'PersonalField33',
'PersonalField27',
'PersonalField2',
'PersonalField19',
'PersonalField18',
'PersonalField16',
'PersonalField15',
'PersonalField12',
'PersonalField11',
'PersonalField10B',
'PersonalField10A',
'PersonalField1',
'Month',
'GeographicField63',
'GeographicField44A',
'GeographicField42B',
'GeographicField41B',
'GeographicField38A',
'GeographicField36B',
'GeographicField32A',
'GeographicField29B',
'GeographicField24A',
'GeographicField23B',
'GeographicField23A',
'GeographicField22B',
'GeographicField21B',
'GeographicField17B',
'GeographicField17A',
'GeographicField13B',
'Field8',
'Field7',
'Field6',
'Field10',
'CoverageField8',
'CoverageField6B',
'CoverageField5B',
'CoverageField5A',
'CoverageField4B',
'CoverageField3A',
'CoverageField2B',
'CoverageField2A',
'CoverageField11B',
'CoverageField11A',
'SalesField6',
'SalesField13',
'PropertyField39A',
'PropertyField35',
'PropertyField27',
'PropertyField25',
'PersonalField81',
'GeographicField62A',
'GeographicField39B',
'GeographicField22A',
'GeographicField1A',
'GeographicField15A',
'CoverageField6A',
'SalesField10',
'PropertyField23',
'PersonalField52',
'PersonalField36',
'GeographicField18A',
'GeographicField8A',
'GeographicField60B',
'GeographicField46B',
'PersonalField69',
'PersonalField59',
'GeographicField5A',
'GeographicField26A'
]

import random
from datetime import datetime
import pandas as pd
from pandas import DataFrame as df
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import train_test_split,cross_val_score
from sklearn import preprocessing
from sklearn.preprocessing import maxabs_scale

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train_sample = np.random.choice(train.index.values,100000)   
train = train.ix[train_sample]

y = train.QuoteConversion_Flag.values

# Converting date into datetime format
train['Date'] = pd.to_datetime(pd.Series(train['Original_Quote_Date']))
# Dropping original date column
train = train.drop('Original_Quote_Date', axis=1)   

test['Date'] = pd.to_datetime(pd.Series(test['Original_Quote_Date']))
test = test.drop('Original_Quote_Date', axis=1)

## Seperating date into 3 columns
train['Year'] = train['Date'].apply(lambda x: int(str(x)[:4]))
train['Month'] = train['Date'].apply(lambda x: int(str(x)[5:7]))
train['weekday'] = train['Date'].dt.dayofweek

test['Year'] = test['Date'].apply(lambda x: int(str(x)[:4]))
test['Month'] = test['Date'].apply(lambda x: int(str(x)[5:7]))
test['weekday'] = test['Date'].dt.dayofweek 

train = train.drop('Date', axis=1)
test = test.drop('Date', axis=1)    


## Filing NA values with -1

train.fillna(-1, inplace=True)
test.fillna(-1,inplace=True)

#columns choice--gmm
train=DataFrame(train,columns=names)
test=DataFrame(test,columns=names)

print(train.shape)

for f in train.columns:
    if train[f].dtype=='object':
        lbl=preprocessing.LabelEncoder()
        lbl.fit(list(train[f].values)+list(test[f].values))
        train[f]=lbl.transform(list(train[f].values))
        test[f]=lbl.transform(list(test[f].values))

#train = train.drop(['QuoteNumber', 'QuoteConversion_Flag'], axis=1)
#test = test.drop('QuoteNumber', axis=1)  

print(train.shape)


from sklearn.preprocessing import Imputer

X = train.ix[:,0:113]
#X=train

X = Imputer().fit_transform(X)

print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)       
      
extc = ExtraTreesClassifier(n_estimators=580,max_features= 112,criterion= 'entropy',min_samples_split= 3,
                          max_depth= 30, min_samples_leaf= 8)   
extc.fit(X,y)    

## Creating submission file
preds = extc.predict_proba(test)[:,1]
sample = pd.read_csv('../input/sample_submission.csv')
sample.QuoteConversion_Flag = preds
sample.to_csv('extc_p.csv', index=False)        
   


