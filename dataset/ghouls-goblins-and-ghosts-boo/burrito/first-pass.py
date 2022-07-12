# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import model_selection 
from sklearn import linear_model

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

train_data = pd.read_csv(r'../input/train.csv')
test_data = pd.read_csv(r'../input/test.csv')
sample_sub = pd.read_csv(r'../input/sample_submission.csv')
print("train_data_size : \n")
print(train_data.shape)
print("\n")
print("test_data_size : \n")
print(test_data.shape)

type_dict = {'Ghost':1,'Ghoul':2,'Goblin':3}
type_dict2 = {1:'Ghost',2:'Ghoul',3:'Goblin'}

train_data['int_type'] = train_data['type'].apply(lambda x: type_dict[x])
x_variable = ['bone_length','rotting_flesh','hair_length','has_soul']
y_variable = 'int_type'
color_dummy = list(train_data.color.drop_duplicates())
for c in color_dummy:
    col_name = c + '_dummy'
    x_variable.append(col_name)
    test_data[col_name] = 0
    train_data[col_name] = 0
    train_data.loc[train_data.color == c, col_name] = 1
    test_data.loc[test_data.color == c, col_name] = 1
    
X_train = train_data[x_variable].as_matrix()
Y_train = train_data[y_variable].as_matrix()
X_test = test_data[x_variable].as_matrix()

logreg = linear_model.LogisticRegression(C = 10000 )
logreg.fit(X_train,Y_train)


Y_test = logreg.predict(X_test)
id_col = list(test_data['id'])
data_dict = {'id':id_col,'type':Y_test}
df_output = pd.DataFrame(data_dict)
df_output['type'] = df_output['type'].apply(lambda x: type_dict2[x])

print(df_output)
df_output.to_csv(r'submission.csv',index = False)