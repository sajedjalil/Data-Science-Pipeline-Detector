
#read data
import pandas as pd
import sys

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

#explore data
#print(train.shape)
#print(test.shape)
#print(train.dtypes)
#print(train.head(n=5))
#print(test.head(n=5))

#print(train_columns)
#print(train.apply(lambda x:x.nunique()))
#print(test.apply(lambda x:x.nunique()))

#print(test['QuoteNumber'].values)
#sys.exit()

train = train.drop('Original_Quote_Date',1)
test  = test.drop('Original_Quote_Date', 1)

train_columns = list(train.columns)
test_columns  = list(test.columns)

QuoteNumber = pd.DataFrame(test['QuoteNumber'],columns=['QuoteNumber'])

#feature engineering
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for col in train_columns:
    if(train[col].dtype=='object'):
        print(col)
        train[col] = le.fit_transform(list(train[col].values))
        #train[col] = train[col].astype(float)
    if(col=='QuoteConversion_Flag'): continue
    else:
        test[col] = le.fit_transform(list(test[col].values))
        #test[col] = test[col].astype(float)
        
x_train = train.drop(['QuoteNumber','QuoteConversion_Flag'],1)
y_train = train['QuoteConversion_Flag']
x_train = x_train.fillna(-1)

x_test  = test.drop('QuoteNumber',1)
x_test  = x_test.fillna(-1)
#modeling
from sklearn.ensemble import RandomForestClassifier
import numpy as np

rfc  = RandomForestClassifier(n_estimators=150)
rfc.fit(x_train, y_train)
importances = rfc.feature_importances_
std = np.std([tree.feature_importances_ for tree in rfc.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
# Print the feature ranking
print("Feature ranking:")
for f in range(x_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
#print(x_test)

#from sklearn.metrics import roc_auc_score
#print(roc_auc_score(y_train, rfc.predict_proba(x_train)))
    
#from sklearn.cross_validation import cross_val_score
#scores = cross_val_score(rfc,x_train,y_train,cv=1,scoring="roc_auc")
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

predictions = rfc.predict_proba(x_test)[:,1]
predictions = pd.DataFrame(predictions)
print(predictions)

#submission

submission = pd.concat([QuoteNumber,predictions],axis=1)
submission.columns = ['QuoteNumber','QuoteConversion_Flag']

submission.to_csv("homesite_submission.csv",index=False)
