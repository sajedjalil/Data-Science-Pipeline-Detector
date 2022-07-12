# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
print("Load Packages")
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from fastai.tabular import *  #fast.ai tabular models

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
print("Print Directories")
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
# Any results you write to the current directory are saved as output.
#read in training data, outcomes and testing  
print("load train and train and labels")
train_without_dep = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')#train 
train_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')#train labels

#join training data with outcomes
print("merge train with labels")
df=train_without_dep.merge(train_labels,on=['game_session','installation_id','title'])
#delete tables we don't need 
del(train_without_dep)
del(train_labels)

#read in test data 
print("read in test and save a copy")
test_df = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')#test
#test_copy=test_df #save a copy for final submission 

#drop variables that might be making our AI run slowly 
print("delete columns that might not be useful")
df=df.drop(['event_id','game_session','event_data','installation_id'],axis=1)
test_df=test_df.drop(['event_id','game_session','event_data','installation_id'],axis=1)

#create date variables in train and test 
print("create time variables in both train and test")
make_date(df, 'timestamp')
make_date(test_df, 'timestamp')
df = add_datepart(df, 'timestamp')
test_df = add_datepart(test_df, 'timestamp')

#examine data 
#df.head(1)
#test_df.head(1)
print("print out the names and types of both data sets, train and test")
df.dtypes
print("#############################")
test_df.dtypes

#procedures for cleaning data 
print("set the procedures for cleaning")
procs = [FillMissing, Categorify, Normalize]

#validation data
#valid_idx = range(len(df)-2000, len(df))

#set up the model 
print("set the dependent variable,categories and continuous variables")
dep_var = 'accuracy_group'
cat_names = ['title', 'type','world']
cont_names = ['event_count', 'event_code', 
              'game_time', 'timestampYear', 
              'timestampMonth', 'timestampWeek', 
              'timestampDay', 'timestampDayofweek', 
              'timestampDayofyear', 'timestampElapsed']

#build data bunch 
print("create the test databunch and main databunch, data")
test = TabularList.from_df(test_df, cat_names=cat_names, cont_names=cont_names, procs=procs)
data = (TabularList.from_df(df, path='.', cat_names=cat_names, cont_names=cont_names, procs=procs)
                           .split_by_idx(list(range(0,200)))
                           .label_from_df(cols=dep_var)
                           .add_test(test, label=0)
                           .databunch())

#delete df and test_df
del(df)
del(test_df)

#build model 
#learn = tabular_learner(data, layers=[200,100],metrics=accuracy)
print("run and create learner")
np.random.seed(101)
learn = tabular_learner(data, layers=[60, 20], metrics=accuracy)
learn.fit(5)

#make predictions and submission file 
print("make predictions and create submission file ")
predictions, *_ = learn.get_preds(DatasetType.Test)
labels = np.argmax(predictions, 1)
test_copy = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')#test
sub_df = pd.DataFrame({'installation_id': test_copy['installation_id'], 'accuracy_group': labels})
sub_df.to_csv('submission.csv', index=False)
sub_df.head(50)
