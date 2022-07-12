#library import
import pandas as pd
import numpy as np

pd.options.display.max_columns = 250

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import plot_confusion_matrix

import matplotlib.pyplot as plt


#data import
train_df = pd.read_csv('/kaggle/input/homesite-quote-conversion/train.csv.zip')
test_df = pd.read_csv('/kaggle/input/homesite-quote-conversion/test.csv.zip')


# ## Dropping Original_quote_date
def drop_cols(df):
    df.drop(['Original_Quote_Date'],axis=1,inplace=True)
    return df

train_df = drop_cols(train_df)
test_df = drop_cols(test_df)


# ## Removing columns with na values
cols_to_delete = train_df.isna().sum()[train_df.isna().sum() > 0].index

def drop_cols_from_list(df,cols_to_delete):
    df.drop(cols_to_delete,axis=1,inplace=True)
    return df

train_df = drop_cols_from_list(train_df,cols_to_delete)
test_df = drop_cols_from_list(test_df,cols_to_delete)


#dropping categorical columns with nunique > 2
cols_to_drop = []

for i in set(train_df.columns) - set(train_df._get_numeric_data().columns):
    if (train_df.loc[:,i].nunique() >= 3):
        cols_to_drop.append(i)
        
train_df = drop_cols_from_list(train_df,cols_to_drop)
test_df = drop_cols_from_list(test_df,cols_to_drop)


#one hot encoding:
cls_to_encode = set(train_df.columns) - set(train_df._get_numeric_data().columns)

def ohe(df,cls_to_encode):
    df = pd.get_dummies(df,columns=cls_to_encode,drop_first=True)
    return df

train_df = ohe(train_df,cls_to_encode)
test_df = ohe(test_df,cls_to_encode)


#dropping any extra columns in test set that are not present in train set
test_df.drop(list(set(test_df.columns) - set(train_df.columns)),axis=1,inplace=True)


#training Randomforest
X = train_df.drop('QuoteConversion_Flag',axis=1)
y = train_df.QuoteConversion_Flag

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify = y, random_state=42)

rfc = RandomForestClassifier(n_jobs=-1)
rfc.fit(X_train,y_train)

print("train accuracy score = ", accuracy_score(y_train,rfc.predict(X_train)))
print("test accuracy score = ", accuracy_score(y_test,rfc.predict(X_test)))

plot_roc_curve(rfc, X_test, y_test)
plt.show()

plot_confusion_matrix(rfc, X_test, y_test,values_format='d')
plt.show()



#submitting output
output_submission = pd.DataFrame(zip(test_df.QuoteNumber,rfc.predict_proba(test_df)[:,1]), columns = ['QuoteNumber','QuoteConversion_Flag'])
output_submission.to_csv('/kaggle/working/output_submission.csv',index=False)