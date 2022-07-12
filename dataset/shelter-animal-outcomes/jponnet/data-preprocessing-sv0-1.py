# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# Data load and initial manipulation
data_file_train = '../input/train.csv'
data_file_test = '../input/test.csv'
train_data = pd.read_csv(data_file_train, delimiter=',',index_col='AnimalID')
print (train_data.dtypes)
train_data_x = train_data[['AnimalType']].values
train_data_y = train_data[['OutcomeType']].values

# Convert Outcome data to 2 categories:
outcome_dict = {'Adoption':1, 'Died':0, 'Euthanasia':0, 'Return_to_owner':1, 'Transfer':0}
for idx, label in enumerate(train_data_y):
    train_data_y[idx] = outcome_dict.get(str(label[0]))
print (train_data_y)


'''Converting the category variables'''

# Put labels on AnimalType and Split in 2 columns
le = LabelEncoder()
train_data_x[:,0] = le.fit_transform(train_data_x[:,0])
ohe = OneHotEncoder(categorical_features=[0])
lr_train_data_x = ohe.fit_transform(train_data_x).toarray()