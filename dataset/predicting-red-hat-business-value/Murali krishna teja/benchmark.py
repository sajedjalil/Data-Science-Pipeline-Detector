# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages
# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats
# For example, here's several helpful packages to load in 

#library(ggplot2) # Data visualization
#library(readr) # CSV file I/O, e.g. the read_csv function

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#system("ls ../input")

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression 
df_train=pd.read_csv('../input/act_train.csv')
df_test=pd.read_csv('../input/act_test.csv')

activity_train=df_train["activity_id"]
activity_test=df_test["activity_id"]

y_train=df_train["outcome"]
del df_train["outcome"]
cols_tbd=["people_id","activity_id","date"]
for i in cols_tbd:
    del df_train[i]
    del df_test[i]

finding_max_value_cat_variable=lambda column_as_series: column_as_series.value_counts().sort_values(ascending=False).keys()[0]
replace_convert_object_category= lambda column_as_series, value_to_be_replaced_with: column_as_series.replace(np.NaN,value_to_be_replaced_with).astype('category')

#df_train=df_train.dropna()
print(df_train.shape)
for i in df_train.columns:
    #print(df_train[i].value_counts())
    df_train[i]=replace_convert_object_category(df_train[i],finding_max_value_cat_variable(df_train[i]))
    df_test[i]=replace_convert_object_category(df_test[i],finding_max_value_cat_variable(df_test[i]))
    df_train[i]=df_train[i].str.replace('type ','').astype("category")
    df_test[i]=df_test[i].str.replace('type ','').astype("category")

       
df_train=np.array(df_train)       
model=LogisticRegression()
model.fit(df_train,y_train)
probs=model.predict_proba(np.array(df_test))[:,1]
op_df = pd.DataFrame()
op_df['activity_id'] = activity_test
op_df['outcome'] = probs
op_df.to_csv('beat_the_benchmark.csv', index=False)


    
