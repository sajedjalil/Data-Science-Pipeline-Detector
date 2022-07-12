# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

#Importing the dataset
train_csv = pd.read_csv('../input/train.csv')
test_csv = pd.read_csv('../input/test.csv')

#drop nan rows

train_csv.columns

#Choose my predictors - pay attention to test for nan values before selecting the columns
cat_features = ["title","description"]
#cat_features = ["category_name"]
num_features = []
target = ["deal_probability"]

X_train = train_csv.loc[:, cat_features+num_features]
X_test = test_csv.loc[:,cat_features+num_features]

#Define target variable
y_train = train_csv.loc[:, target]

#Firstly I will build the entire features vectors so that when I fit my encoder I make sure I fit it through all the possible values in the cateorical data
frames = [X_train,X_test]
X = pd.concat(frames,keys=['training','test'])

X = X.replace(np.nan, '', regex=True)

#Check if number of words in title and description has any significance

#noofwords = pd.DataFrame(columns = ['title','description'], index=range(0,len(X["title"]))) - I did not know how to create an additional data frame that woul contain the number of words but at the same time preserve the 'train' and 'test' keys so I would later on use for split

for i in range(len(X["title"])):
    X['title'][i] = len(X['title'][i])
    X['description'][i] = len(X['description'][i])

X_train = X.loc['training']
X_test = X.loc['test']

#example of pearson correlation
test=[1,2,3,4]
train=[-1,-2,-3,-4]
np.corrcoef(test,train)

#Check person correlation
#np.corrcoef(X_train['title'].values,y_train.values) - this does not work

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set result
y_pred = regressor.predict(X_test)
y_pred = np.concatenate( y_pred, axis=0 )

my_submission = pd.DataFrame({'item_id': test_csv.item_id, 'deal_probability': y_pred})

my_submission.to_csv('submission_wc.csv', index=False)