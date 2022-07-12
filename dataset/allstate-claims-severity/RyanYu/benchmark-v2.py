# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import xgboost as xgb
df = pd.read_csv("../input/train.csv")
loss = np.log(df['loss'])

# drop 
drop_var = ['cat9','cat16','cat23','cat66','cat90','cat103',\
  'cat111','cat114','cont6','cont7','cont8','cont9','cont11',\
  'cont12','cont13','cont14']
df.drop([],axis=1,inplace=True)

colnames = df.columns.values.tolist()

cat_var = [x for x in colnames if x[0:3] == 'cat' ]
dummy_df = pd.get_dummies(df[cat_var], prefix = cat_var) 
df_dev = pd.concat([df,dummy_df],axis = 1)
df_dev.drop(cat_var,axis=1,inplace=True)
df_dev.drop(['id','loss'],axis=1,inplace=True)
# df_dev.head(n=4).to_csv('output.csv')
dev_colnames = df_dev.columns.values.tolist()
#print(dev_colnames)


dtrain = xgb.DMatrix(df_dev, label=loss,missing=0)

#  param #1 : mae = 1225
#param = {'max_depth':10, 'eta':0.3, 'silent':1, 'objective':'reg:linear'}
#num_round = 10

#print ('running cross validation')

#xgb.cv(param, dtrain, num_round, nfold=10,
#       metrics={'mae'}, seed = 0,
#       callbacks=[xgb.callback.print_evaluation(show_stdv=True)])


param = {'max_depth':5, 'eta':0.3, 'silent':1, 'objective':'reg:linear'}
num_round = 10

bst = xgb.train( param, dtrain, num_round)

print("----train ends----")
del dtrain
del df_dev
del dummy_df

df_test = pd.read_csv("../input/test.csv")
df_id = pd.DataFrame(df_test['id'])
colnames_test = df_test.columns.values.tolist()

cat_var_test = [x for x in colnames if x[0:3] == 'cat' ]
dummy_df_test = pd.get_dummies(df_test[cat_var_test], prefix = cat_var_test) 
df_val = pd.concat([df_test,dummy_df_test],axis = 1)
del df_test
del dummy_df_test
df_val.drop(cat_var_test,axis=1,inplace=True)
df_val.drop('id',axis=1,inplace=True)

print("----prediction----")
val_colnames = df_val.columns.values.tolist()
for x in dev_colnames:
    if x not in val_colnames:
        df_val[x] = 0
        
val_colnames = df_val.columns.values.tolist()


dtest = xgb.DMatrix(df_val[dev_colnames])        
ypred = pd.DataFrame(bst.predict(dtest),columns=['loss'])
ypred = np.exp(ypred)
output = pd.concat([df_id,ypred],axis=1)
output.to_csv('benchmark_v2.csv',index=False)






