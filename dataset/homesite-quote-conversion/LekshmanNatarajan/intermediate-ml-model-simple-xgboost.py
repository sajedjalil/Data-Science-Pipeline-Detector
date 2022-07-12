# %% [code]
# Intermediate ML model
# Involves simple data cleaning & XGboost at default settings 


#library import
import pandas as pd
import numpy as np

pd.options.display.max_columns = 250
pd.options.display.max_rows = 300

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

import xgboost as xgb

# %% [code]
#data import
train_df = pd.read_csv('/kaggle/input/homesite-quote-conversion/train.csv.zip')
test_df = pd.read_csv('/kaggle/input/homesite-quote-conversion/test.csv.zip')

# %% [code]
for i in ['year','month','day']:
     train_df[i] = np.nan
     test_df[i] = np.nan

train_df[['year','month','day']] = list(train_df.Original_Quote_Date.str.split("-"))
test_df[['year','month','day']] = list(test_df.Original_Quote_Date.str.split("-"))

# %% [code]
train_df['weekday'] = pd.to_datetime(train_df['Original_Quote_Date']).dt.dayofweek
test_df['weekday'] = pd.to_datetime(test_df['Original_Quote_Date']).dt.dayofweek

quote_numbers = test_df.QuoteNumber

# %% [code]
train_df.drop(['Original_Quote_Date','QuoteNumber'],axis=1,inplace=True)
test_df.drop(['Original_Quote_Date','QuoteNumber'],axis=1,inplace=True)

# %% [code]
# xgboost can handle missing values by default
# so no fillna required

# %% [code]
#label encoding
for f in train_df.columns:
    if train_df[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_df[f].values) + list(test_df[f].values))
        train_df[f] = lbl.transform(list(train_df[f].values))
        test_df[f] = lbl.transform(list(test_df[f].values))

# %% [code]
X = train_df.drop('QuoteConversion_Flag',axis=1)
y = train_df.QuoteConversion_Flag


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify = y, random_state=42)

clf = xgb.XGBClassifier(tree_method='gpu_hist')
                        
xgb_model = clf.fit(X_train,y_train)

print("train accuracy score = ", accuracy_score(y_train,xgb_model.predict(X_train)))
print("test accuracy score = ", accuracy_score(y_test,xgb_model.predict(X_test)))

# %% [code]
#submitting output
output_submission = pd.DataFrame(zip(quote_numbers,xgb_model.predict_proba(test_df)[:,1]), columns = ['QuoteNumber','QuoteConversion_Flag'])
output_submission.to_csv('/kaggle/working/output_submission.csv',index=False)