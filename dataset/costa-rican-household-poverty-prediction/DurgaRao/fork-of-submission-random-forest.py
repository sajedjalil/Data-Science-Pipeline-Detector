# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import Counter
from sklearn import model_selection
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
# Any results you write to the current directory are saved as output.
os.chdir("../input")
train=pd.read_csv('train.csv')
test=pd.read_csv("test.csv")

#### impute missing values
mean_imputer = preprocessing.Imputer() #By defalut parameter is mean and let it use default one.
mean_imputer.fit(train[['v2a1','meaneduc','SQBmeaned']]) 
train[['v2a1','meaneduc','SQBmeaned']] = mean_imputer.transform(train[['v2a1','meaneduc','SQBmeaned']])

#### droping columns with object data type
x_train=train.drop(columns=['dependency','edjefa','edjefe','idhogar','Id','rez_esc','v18q1'])

x_train=x_train.drop_duplicates()

mean_imputer = preprocessing.Imputer() #By defalut parameter is mean and let it use default one.
mean_imputer.fit(test[['v2a1','meaneduc','SQBmeaned']]) 
test[['v2a1','meaneduc','SQBmeaned']] = mean_imputer.transform(test[['v2a1','meaneduc','SQBmeaned']])


x_test=test.drop(columns=['dependency','edjefa','edjefe','idhogar','Id','rez_esc','v18q1'])


############################# DETECT OUTLIERS ############################


def detect_outliers(df,n,features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers   

# detect outliers from all fields
Outliers_to_drop = detect_outliers(x_train,12,list(x_train))

x_train = x_train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)
y_train=x_train['Target']
x_train=x_train.drop(columns=['Target'])




kfold=model_selection.StratifiedKFold(n_splits=10,random_state=105641)


############ RANDOM FOREST###################3
rdt=RandomForestClassifier()
dt_grid = {'max_depth':list(range(3,8)), 'min_samples_split':[2,3,6,7,8], 'criterion':['gini','entropy']}
grid_tree_estimator = model_selection.GridSearchCV(rdt,dt_grid,cv=kfold)
grid_tree_estimator.fit(x_train, y_train)


x_test['Target']=grid_tree_estimator.predict(x_test)
