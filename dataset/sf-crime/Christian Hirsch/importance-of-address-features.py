
# coding: utf-8

## San Francisco Crime Data -- Feature Engineering

# First, we set certain initial parameters. To improve fault-tolerance, we analyze the training set in batches of a specified size.

# In[1]:

import numpy as np
import math
import pandas as pd
import pickle

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler,FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.utils import shuffle

#path to data
DATA_PATH = "../input/"

#path for storing classes that are actually used for training
FEATURE_PATH = "../features/"

#batch number of the large training and data files
BATCH_SIZE = 100000


# Next, we load the data and drop columns from the training set that cannot be used in the test set. 

# In[2]:

train_data = pd.read_csv(DATA_PATH + "train.csv")
#test = pd.read_csv(DATA_PATH + "test.csv")
train_data = train_data.drop(['Resolution', 'Descript'], axis=1)
train_data['Dates'] = pd.to_datetime(train_data['Dates'])


# An exploratory analysis shows that certain categories are exceptionally rare. To make our classifier more robust, we exclude these categories from training.

# In[3]:

rare_cats = set(['FAMILY OFFENSES', 'BAD CHECKS', 'BRIBERY', 'EXTORTION',
       'SEX OFFENSES NON FORCIBLE', 'GAMBLING', 'PORNOGRAPHY/OBSCENE MAT',
       'TREA'])
all_cats = set(train_data['Category'].unique())
common_cats = all_cats-rare_cats

#pickle the categories
#pickle.dump(sorted(list(all_cats)), open(FEATURE_PATH + 'all_cats', 'wb'))
#pickle.dump(sorted(list(common_cats)), open(FEATURE_PATH + 'common_cats', 'wb'))

train_data = train_data[train_data['Category'].isin(common_cats)]
train = train_data.reset_index(drop = True)


# Next, we split the training set into two folds,  to prevent overfitting of classifiers to be constructed later. We proceed parallel to the construction of the test set given in the problem statement and split according to specific weeks.

# In[4]:

fold0 = train_data[train_data['Dates'].dt.week%(4)/2==0]
fold1 = train_data[train_data['Dates'].dt.week%(4)/2!=0]
fold0 = shuffle(fold0, random_state = 123).iloc[0:300000,:]
fold1 = shuffle(fold1, random_state = 123).iloc[0:300000,:]


### Auxiliary Transformers

# It will be convenient to use the ItemSelector from http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html and a LabelEncoder that can act on one-dimensional data frames.

# In[5]:

class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[data_dict.columns.intersection(self.key)]

class DFLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        return

    def fit(self, x, y = None):
        return self

    def transform(self, data):
        return LabelEncoder().fit_transform(data)


### Encode Category

# As preliminary operation, we encoder the category labels.

# In[6]:

cat_enc = LabelEncoder()
train.loc[:,'Category'] = cat_enc.fit_transform(train.loc[:,'Category'])
fold0.loc[:,'Category'] = cat_enc.fit_transform(fold0.loc[:,'Category'])
fold1.loc[:,'Category'] = cat_enc.fit_transform(fold1.loc[:,'Category'])


### Latitude and Longitude (2 Features)

# The first feature that we engineer is obtained by standardizing and pca-ing latitude and longitude.

# In[7]:

xy_is=ItemSelector(['X', 'Y'])
xy_scaler = StandardScaler()
pca = PCA()
xy_pipe = Pipeline([('xy_is', xy_is), ('xy_scaler', xy_scaler), ('pca', pca)])
xy_names = ['x', 'y']


### Date (11 Features)

# Second, we extract several date features. We need to take into account that minute, hour, week, month are of cyclic nature  https://www.kaggle.com/sergeylebedev/sf-crime/initial-benchmark-need-tuning/code.

# In[8]:

date_names = ['minute', 'minute30',
               'hour', 'hour12',
               'day',
               'week', 'week26',
               'month', 'month6',
               'year', 
               'weekday'
               ]
def recenter_cyclic(data, period):
    return (data - period/2) % period
def DayWeekMonthYear(data):
        dateTuple = data.apply(lambda x: pd.Series([x.minute, recenter_cyclic(x.minute, 60), 
                                                    x.hour, recenter_cyclic(x.hour, 24),
                                                    x.day, 
                                                    x.week, recenter_cyclic(x.week, 52),
                                                    x.month, recenter_cyclic(x.month, 12),
                                                    x.year, 
                                                    x.dayofweek]))
        dateTuple.columns = date_names
        return dateTuple
    
date_is = ItemSelector(['Dates'])
dateFun = FunctionTransformer(lambda x: DayWeekMonthYear(pd.to_datetime(x.iloc[:, 0])), validate=False)
date_pipe = Pipeline([('date_is', date_is),
                      ('date_fun', dateFun)])


### Address (32 Features)

# Regarding the addresses, we implement an idea from https://www.kaggle.com/papadopc/sf-crime/neural-nets-and-address-featurization, which suggests count featurization of addresses.

# This step requires some preliminary data munging. First, we count the categories for each address and then determine the log of the relative frequency. Then, the log-ratios are obtained using a pivot table. We replace the log of 0 by a large negative number. In order to ensure that most entries are 0, we subtract that number from the pivot table afterwards. Finally, we merge these log_ratios back to our original data frame.

# In[9]:

#minimum number of crimes that are required for log-ratios to be computed
MIN_COUNT = 5

#default log ratio
DEFAULT_RAT = math.log(1.0 / len(common_cats))

class CountFeatureEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.log_ratios = None

    def fit(self, data, y = None):
        
        #determine total number of crimes
        address_counts = pd.DataFrame(data.groupby(['Address']).size(), columns = ['total_count'])
        
        #determine number of crimes by category and address
        address_category_counts = pd.DataFrame(data.groupby(['Address', 'Category']).size(), columns = ['count']).reset_index()
        
        address_category_counts['total_count'] = address_category_counts.groupby('Address')['count'].transform(lambda x: max(MIN_COUNT, x.sum()))
        address_category_counts = address_category_counts[address_category_counts['count'] >= MIN_COUNT]
        address_category_counts['log'] = (address_category_counts['count']/address_category_counts['total_count']).apply(math.log)
        
        #form pivot table
        self.log_ratios = pd.pivot_table(address_category_counts, index = 'Address', 
                          columns = 'Category', values = 'log', fill_value = DEFAULT_RAT) - DEFAULT_RAT  
        self.log_ratios.columns = ['LogRatio_' + str(x) for x in range(len(self.log_ratios.columns))]
        
        #join total counts
        self.log_ratios = self.log_ratios.merge(address_counts, 'left', left_index = True, right_index = True)
        return self

    def transform(self, data):
        
        #merge with log_ratios
        merged_data = data.loc[:, 'Address'].reset_index().merge(self.log_ratios, 'left', left_on = 'Address', right_index = True).iloc[:,2:]        
        
        #replace NAs with default values
        default_df = pd.DataFrame(np.zeros(merged_data.shape), columns = merged_data.columns) 
        default_df.iloc[:, -1] = MIN_COUNT

        return merged_data.combine_first(default_df)


# Finall, we set up the pipeline for the count featurizer.

# In[10]:

address_cat_is = ItemSelector(['Address', 'Category'])
count_feat_fun = CountFeatureEncoder()

count_feat_pipe = Pipeline([('address_cat_is', address_cat_is), 
                            ('count_feat_fun', count_feat_fun)])
                              
count_feat_names = ['count_feature_' + i for i in common_cats] + ['total_count']


### Collocated crimes (1 Feature)

# Finally, we add a column indicating  how many other crimes nearby at the same time and scene. This is proposed in https://www.kaggle.com/luventu/sf-crime/title/discussion.

# In[11]:

reshaper = FunctionTransformer(lambda X: X.reshape(-1,1),validate=False)

def cocrime(df):
 #group by date and time and check whether id corresponds to multiple incidents
 groupedFrame = df.reset_index().groupby(['Dates', 'X', 'Y'])['index'] 
 return groupedFrame.transform(lambda x: len(list(x)))
    
cocrimeFun = FunctionTransformer(lambda df : cocrime(df), validate = False)
cocrime_pipe = Pipeline([('cocrimeFun', cocrimeFun), ('reshaper', reshaper)])
cocrime_names = ['CollocatedCrime']


### Concluding feature union (47 Features)

# In order to mitigate failures, we set up a batch-processing procedure.

# In[12]:

def batch_process(data, pipeline, column_names, extra_col_idx, file_name):
    pipelined_data = pipeline.transform(data)
    return pd.DataFrame(pipelined_data, columns = featureNames).reset_index(drop = True)
    #iterations = int(data.shape[0] / BATCH_SIZE) + 1
    
    #process initial batch
    #for i in range(0, iterations):
    #    print("Round:{}".format(i))
        
        #pipelined_data = pipeline.transform(data.iloc[i * BATCH_SIZE: (i + 1) * BATCH_SIZE, :])
        #pipelined_df = pd.DataFrame(pipelined_data, columns = featureNames).reset_index(drop = True)
        #pipelined_df = pipelined_df.join(data.iloc[i * BATCH_SIZE: (i + 1) * BATCH_SIZE, extra_col_idx].reset_index(drop = True))
        
        #write_mode = 'w' if i==0 else 'a'
        #pipelined_df.to_csv(DATA_PATH + file_name, mode = write_mode, header = (i == 0), index = False)


# Now, we reap the benefits of setting up the pipelines and congregate everything into a concise FeatureUnion. In the forthcoming notebook, we will use feature importances to check which of our elaborately constructed features are worth their salt.

# In[13]:

features = FeatureUnion([('xy',xy_pipe),
              ('date', date_pipe),
              ('count_feat', count_feat_pipe),
              ('cocrime', cocrime_pipe)
              ])
featureNames = xy_names + date_names  + count_feat_names + cocrime_names

#features.fit(train)
cat_idx = train.columns.tolist().index('Category')
#id_idx = test.columns.tolist().index('Id')

#print("Transforming train data")
#batch_process(train, features, featureNames, cat_idx, 'trainTransformed.csv');

#print("Transforming test data")
#batch_process(test, features, featureNames, id_idx, 'testTransformed.csv');


#create test and cv set for cross validation

features.fit(fold1)
print("Transforming fold 0 of cv set")
Xfold0 = batch_process(fold0, features, featureNames, cat_idx, 'fold0Transformed.csv');

features.fit(fold0)
print("Transforming fold 1 of cv set")
Xfold1 = batch_process(fold1, features, featureNames, cat_idx, 'fold1Transformed.csv');


# Finally, we shuffle the data to prevent overfitting. And merge the two folds into a single file.

# In[14]:

#train = pd.read_csv(DATA_PATH + "trainTransformed.csv")
#train = shuffle(train)
#train.to_csv(DATA_PATH + 'trainTransformed.csv', index = False)

#fold0 = pd.read_csv(DATA_PATH + "fold0Transformed.csv")
#fold1 = pd.read_csv(DATA_PATH + "fold1Transformed.csv")
#fold0 = shuffle(fold0)
#fold1 = shuffle(fold1)
#cv_train = pd.concat([fold0, fold1])
#cv_train = shuffle(cv_train)
#cv_train.to_csv(DATA_PATH + 'cvTrainTransformed.csv', index = False)


# In[ ]:




## San Francisco Crime Data -- Feature Importance

# In this notebook, we explore the importance of the features via xgb. First, we fix initial parameters.

# In[1]:

import numpy as np
import operator
import pandas as pd

from xgboost.sklearn import XGBClassifier



#score function of xgb classifier
SCORING = 'mlogloss'

#params for xgb
HYPER_PARAMS = {
 'learning_rate': 0.2,
 'n_estimators': 20,
 'max_depth': 6,
 'subsample': 0.8,
 'colsample_bytree': 0.8,
 'max_delta_step': 1,
 'objective': 'multi:softmax',
 'nthread': 8,
 'seed': 1747
}


# Now, we load training data and separate it into validation and training data.

# In[2]:


X_train = Xfold0
y_train = fold0.loc[:, 'Category']

X_val = Xfold1
y_val = fold1.loc[:, 'Category']


# Now comes the time-consuming step of training xgb.

# In[3]:

xgb = XGBClassifier(**HYPER_PARAMS)
xgb.fit(X_train, y_train, eval_set = [(X_val, y_val)], eval_metric = SCORING, verbose = 10)


# Now, we can gaze at the important features.

# In[4]:

gbdt = xgb.booster()
importance = gbdt.get_fscore()
importance = sorted(importance.items(), key = operator.itemgetter(1), reverse = True)
df=pd.DataFrame(importance, columns = ['feature', 'fscore'])
print(df)


# This provides us with a good idea as to which features are particularly relevant. 
# 
# - clearly, the timing in terms of minute, hour and year are critical
# - the collocated-crime feature scores surprisingly high
# - the spatial coordinates are useful
# - the total number of crimes in a steet is an important indicator, as well as some of the log-ratios
# - the month is not particularly essential, presumably as seasonal information can be recovered from the week
