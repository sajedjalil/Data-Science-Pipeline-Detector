
# coding: utf-8

# # Introduction

# This notebook intends to analyse and implement a basic classification on a Kaggle competition dataset (https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection). The dataset is from https://www.talkingdata.com, a Chinese big data service platform. The interesting - and challenging the same time - about this competition was the nature of the data. It greatly resembles online web analytics/clickstream data both in size but also in the way that most people are used to intepret attribution; particularly the Last Touch Channel/Value model.

# # Setup, data loading and formatting

# Let's import all the libraries needed first

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import gc
import sys
main_dir = "../input/"
# main_dir = ""
# Import all necessary libraries
# datetime for time/date-related manipulations
from datetime import datetime
from datetime import timedelta

# Model training and evaluation tools
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, auc,f1_score, recall_score, precision_score,roc_auc_score
from sklearn.pipeline import Pipeline

from sklearn.externals import joblib

import matplotlib as plt
# get_ipython().magic(u'matplotlib inline')


# Load the training data set. It is important to set the data types manually so that the size remains manageable even in a small 8GB Ram laptop.

# In[3]:


data_types = {'click_id':'uint32','ip':'uint32', 'app':'uint16','device':'uint16',
                      'os':'uint16','channel':'uint16', 'is_attributed': 'uint8'}

# Select only appropriate lines to keep data as small as possible.
starting_line = 9308568 + 1
number_of_lines = 122578385

train_sample = pd.read_csv(
    main_dir + 'train.csv', 
    dtype = data_types,
    skiprows= starting_line, 
    nrows= number_of_lines, 
    header= None, 
    names = ['ip','app','device','os','channel','click_time','attributed_time','is_attributed']
)

train_sample.info()


# #### Columns descriptions (from Kaggle):
# 
# Each row of the training data contains a click record, with the following features.
# 
# - ip: ip address of click.<br>
# - app: app id for marketing.<br>
# - device: device type id of user mobile phone (e.g., iphone 6 plus, iphone 7, huawei mate 7, etc.)<br>
# - os: os version id of user mobile phone<br>
# - channel: channel id of mobile ad publisher<br>
# - click_time: timestamp of click (UTC)<br>
# - attributed_time: if user download the app for after clicking an ad, this is the time of the app download<br>
# - is_attributed: the target that is to be predicted, indicating the app was downloaded<br>
# Note that ip, app, device, os, and channel are encoded.<br>
# 
# The test data is similar, with the following differences:<br>
# 
# - click_id: reference for making predictions<br>
# - is_attributed: not included

# The attribution time column is not needed any more, let's save some memory.

# In[4]:


main_data = train_sample.drop(['attributed_time'], axis = 1)
del train_sample
gc.collect()


# The feature extraction functions are defined below. The basic operations executed are:<br>
# 1. Format the column click_time into appropriate day, hour, minute columns<br>
# 2. Perform per day and hour groupings of the four key dimensions; app, os, device & channel

# In[5]:


def grp_dim_by_hour(df, dimension):
    return df[['ip','day','hour',dimension]].groupby(by = ['day','hour',dimension]).count().reset_index().rename(index=str, columns={"ip": str(dimension + "_instances_by_hour")})

def format_data(df):
    df['hour'] = pd.to_datetime(df.click_time).dt.hour.astype('uint8')
    df['day'] = pd.to_datetime(df.click_time).dt.day.astype('uint8')
    df['minute'] = pd.to_datetime(df.click_time).dt.minute.astype('uint8')
    df = df.drop(['click_time'], axis = 1)
    gc.collect()
    
    fixed_dimensions = ['app','device','channel','os']
    for dim in fixed_dimensions:
        grouping_by_dim = grp_dim_by_hour(df, dim)
        df = pd.merge(df, grouping_by_dim, how = 'left', on = [dim,'day','hour'])
    
    del grouping_by_dim
    gc.collect()
    
    return(df)


# Plotting helper functions

# In[131]:


def print_conversion_by_dim(df, dimension, sort_by = 'ip', asc = False):
    df = df.groupby(by = dimension)
    conversions = df[['is_attributed']].sum().reset_index()
    totals = df[['ip']].count().reset_index()
    grouping = pd.merge(totals, conversions, on = dimension) 
    grouping['conv'] = grouping.apply(lambda r: r['is_attributed']/r['ip'], axis = 1)
    grouping = grouping.sort_values(by = sort_by, ascending= asc)
    grouping[[dimension,'conv','ip']].plot(kind = 'bar', x = dimension, y = ['conv','ip'], figsize=(20, 8), secondary_y = 'conv')
    del conversions, totals
    gc.collect()
    return grouping

def print_conversion_by_dim_for_converting_values(df, dimension, conversion_threshold = 0):
    rows_with_conv = df[df['conv'] > conversion_threshold][dimension]
    final_df = main_data[main_data[dimension].isin(rows_with_conv)][[dimension,'ip','is_attributed']]
    res = print_conversion_by_dim(final_df, dimension, 'ip')


# Let's format the main data.

# In[6]:


main_data = format_data(main_data)


# We can see below that now extra columns have been added along with their types and same sample values.

# In[7]:


print(main_data.info())
print(main_data.head())


# In[8]:


desc = main_data.describe()


# In[9]:


desc.applymap("{0:.0f}".format)


# Basic data coverage (date and hour range). This helps us confirm that we only have data for day 7 and 8. Day 7 will act as our training set and 8 as our validation set.

# In[10]:


print(pd.crosstab(main_data['hour'],main_data['day']))


# As I am working on online analytics, conversion rate is never left out of the discussion. So let's check the typical conversion rate over hour for both days.

# In[11]:


tmp = pd.crosstab(main_data['hour'], main_data['is_attributed']).apply(lambda r: r/r.sum(), axis=1)


# In[13]:


tmp.reset_index()[1].plot()


# In[14]:


tmp.reset_index()[1].plot(kind = 'box')


# Conversion rate in general fluctuates from 0.1% to 0.3%. Very low rate and highly unbalanced classification problem. The positive label is extremely rare.

# # Examination of core dimensions

# It is worth investigating how our core dimensions are distributed (volume of "visits" spread over different values) in comparison to their respective conversion rate. In other words, is there a particular segment of those dimensions that converts particularly well?

# Our core dimensions are:<br>
# 1) Operating System<br>
# 2) Channel (Similar to marketing channels in online analytics? Maybe)<br>
# 3) Device<br>
# 4) Application<br>

# ### 1) Operating System

# The OS dimension has decent amount of unique values but not the highest.

# In[15]:


print('Unique OS values: ' + str(len(main_data.os.unique())))


# Let's examing the conversion in relation to its volume across the different values.

# In[16]:


os_conv = print_conversion_by_dim(main_data[['os','ip','is_attributed']], dimension = 'os')


# It looks like a small portion of values contain the majority of the volume. Regarding Conversion, There are some spikes here and there but nothing stands out. Let's see what happens when we inspect only OS values that display a conversion value greater than 0.

# In[17]:


print_conversion_by_dim_for_converting_values(os_conv, 'os')


# Now the total volume of OS values has decreased dramatically but for the high volume Operating Systems, conversion remains at small levels.

# ### 2) Channel

# Applying the same logic on Channel, we notice the same behaviour. The only difference is that there are more high volume values that have decent conversion while the total number of unique values is relatively small. That probably indicates that this dimension has bigger impact on predicting the positive outcome.

# In[18]:


print('Unique Channel values: ' + str(len(main_data.channel.unique())))


# In[19]:


res = print_conversion_by_dim(main_data[['channel','ip','is_attributed']], 'channel')


# In[20]:


print_conversion_by_dim_for_converting_values(res, 'channel')


# ### 3) Device

# Device does not look as promising as the previous two dimensions. It has way more unique values and by applying the same logic as before, one can infeer that it is not a stable and reliable feature (at least for conversion).

# In[21]:


print('Unique Device values: ' + str(len(main_data.device.unique())))


# In[22]:


res = print_conversion_by_dim(main_data[['device','ip','is_attributed']], 'device')


# In[23]:


print_conversion_by_dim_for_converting_values(res, 'device')


# ### 4) App

# App is similar to the Operating System in terms of volume and conversion to volume alignment. So we would expect that to be affecting on similar levels the final outcome.

# In[24]:


print('Unique App values: ' + str(len(main_data.app.unique())))


# In[25]:


res = print_conversion_by_dim(main_data[['app','ip','is_attributed']], 'app')


# In[26]:


print_conversion_by_dim_for_converting_values(res, 'app')


# ## Model Training

# Building the test and training sets for training the model.

# In[112]:


features = ['app','device','os','channel','hour','day','minute', 'app_instances_by_hour', 'os_instances_by_hour', 'device_instances_by_hour',  'channel_instances_by_hour']

# Selecting these timeslot to make the data set smaller and minimize the training period. 
# The reason for that is the final test set that will be uploaded on Kaggle contains only those timeslots.
# So this simplifies the work.
is_test_set_hour_slots = (
    (main_data['hour'] >= 4) & (main_data['hour'] <= 5) |
    (main_data['hour'] >= 9) & (main_data['hour'] <= 10) |
    (main_data['hour'] >= 13) & (main_data['hour'] <= 14)
)

is_train_set = (main_data['day'] == 7) & is_test_set_hour_slots

is_test_set = (main_data['day'] == 8) & is_test_set_hour_slots


# In[113]:


X_train = main_data[is_train_set][features]
y_train = main_data[is_train_set]['is_attributed']

X_test = main_data[is_test_set][features]
y_test = main_data[is_test_set]['is_attributed']

del is_train_set
del is_test_set
# del main_data
gc.collect()


# For the actual model fitting, I had to experiment for different parameters' values. There is more room for improvement here given adequate time.

# In[119]:


print('Start model building')
rf_cls = RandomForestClassifier(class_weight = 'balanced', min_samples_leaf = 5, min_samples_split= 4, max_depth= 10)
gc.collect()
rf_cls.fit(X_train , y_train)


# Saving a model is always a good idea!

# In[130]:


print('Start model saving')
joblib.dump(rf_cls, 'model_rf_cls-965-auc.pkl') 
print('End model saving')


# ### Model evaluation on test set.

# We need to calculate different key metrics but we will focus on the ROC curce (https://en.wikipedia.org/wiki/Receiver_operating_characteristic) as this is used by Kaggle to rank the results.

# In[134]:


predictions = rf_cls.predict(X_test)
print('Accuracy ' + str(accuracy_score(y_test,predictions)))
print('Precision '+ str(precision_score(y_test,predictions)))
print('Recall ' + str(recall_score(y_test,predictions)))
print('F1-score ' + str(f1_score(y_test,predictions)))
predictions_proba = rf_cls.predict_proba(X_test)
print('ROC AUC : ' + str(roc_auc_score(y_test, predictions_proba[:,1])))


# By ispecting the confusion matrix, we notice that we fail to incorectly identify nearly 6.5k instances of possitive labels. There are also 500k instances of false negatives. This initially might sound a lot. However out of total 20M negative labels that's around 2.5% of incorrectly classified negative labels. 
# 
# Also our recal value is quite decent 0.874; true positives / (true positives + false negatives).

# In[143]:


print(pd.crosstab(y_test,predictions, rownames=['Actual Value'],colnames=['Predicted Value']))


# Based on the ROC value (>0.96) along with the fact we are dealing with highly unbalanced labels, we can safely say the model is performing well on our test set.

# But which features had a big impact on the results?

# In[121]:


print('Feature Importance')
print('------------------')
for index in np.flip(np.argsort(rf_cls.feature_importances_, ), axis = 0):
    print(X_test.columns[index] + ' : ' + str(rf_cls.feature_importances_[index]))
print('------------------')
print('End model building')


# In[ ]:


false_negatives = (y_test == 1) & (predictions == 0) # This is the data-set that we need to improve, incorrectly classified positive labels
false_positives = (y_test == 0) & (predictions == 1)
true_positives = (y_test == 1) & (predictions == 1)
true_negatives = (y_test == 0) & (predictions == 0)


# In[103]:


false_negatives = main_data.loc[false_negatives.values.tolist()]
false_positives = main_data.loc[false_positives.values.tolist()]
true_positives = main_data.loc[true_positives.values.tolist()]
true_negatives = main_data.loc[true_negatives.values.tolist()]


# ## Final predictions

# In[123]:


## Load Training Set
test_sample = pd.read_csv(main_dir + 'test.csv', dtype = data_types) 


# In[124]:


test_sample.info()


# In[126]:


test_sample = format_data(test_sample)


# In[127]:


test_sample.head()


# In[128]:


print(pd.crosstab(test_sample['hour'],test_sample['day']))


# In[129]:


predictions = rf_cls.predict_proba(test_sample[features])
d = {'click_id': test_sample['click_id'], 'is_attributed': predictions[:,1]}
df = pd.DataFrame(data=d)
df.to_csv('output.csv', index= False)


# The specific model achieves <b>0.9231350</b> on the public score. Given the small drop in AUC compared to our validation set, I would say this might have to do with the day of the week as well. For example, it might be the case that the data we use to train and validate the model are from weekdays and the test set from weekend (or vise versa).  But overall it is a good and promising result.
