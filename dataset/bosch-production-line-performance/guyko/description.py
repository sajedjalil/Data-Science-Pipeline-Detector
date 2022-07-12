# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.tree import DecisionTreeClassifier as DecTreeClass
from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


train = pd.read_csv ("../input/train_numeric.csv", nrows = 1000)
cols = list(train.columns)
#pd.options.display.max_rows = 1000
#dsc = train.describe().T
#dsc[:,1]
for t in range(len(cols)):
    print(cols[t])
    print(np.reshape(train.loc[pd.notnull(train[cols[t]]),cols[t]].describe(),(1,8)))
    #dsc[:,t] = np.reshape(train.loc[pd.notnull(train[cols[t]]),cols[t]].describe(),(1,8))
#dsc

train['_Intercept_']=1
train_w = train.copy()
num_features = [x for x in df.columns if x not in ['Id','Response']]
target=['Response']

for f in range(len(num_features)):
#for f in range (1,2):
    features2 = [num_features[f],'_Intercept_']
        
    dt = DecTreeClass(max_leaf_nodes=6,min_samples_split=split, random_state=99)

    dt.fit(train[pd.notnull(train[num_features[f]])][features2],train[pd.notnull(train[num_features[f]])][target])

    pred=np.minimum(0.9999,np.maximum(0.0001,dt.predict_proba(train[pd.notnull(train[num_features[f]])][features2])[:,1]))    

    train_w.loc[pd.notnull(train[num_features[f]]),num_features[f]] = np.log(pred/(1-pred)) - np.log(mean_target/(1-mean_target))


    if train_w.loc[pd.isnull(train[num_features[f]]),num_features[f]].shape[0] > 0:

        pred = pysqldf("""
            select distinct
                a.""" + num_features[f] + """_target as pred
            from 
            (
                select 
                    """ + num_features[f] + """, 
                    sum(1) as volume, 
                    avg(""" + target + """) as """ + num_features[f] + """_target 
                from train 
                group by """ + num_features[f] + """ 
                order by """ + num_features[f] + """_target
            ) a
            inner join 
            train b on 1=1
            where b.""" + num_features[f] + """ is null and a.""" + num_features[f] + """ is null
            """).as_matrix()[0,0]
        
        train_w.loc[pd.isnull(train[num_features[f]]),num_features[f]] = np.log(pred/(1-pred)) - np.log(mean_target/(1-mean_target))
      
    var = train.groupby(by=train_w[num_features[f]])[num_features[f]].describe().unstack()   
    var.rename(columns={'min': 'var_min', 'max': 'var_max'}, inplace=True)
    var['WoE'] = var.index
    tar = train.groupby(by=train_w[num_features[f]])[target].describe().unstack()
    summ = pd.concat([var['var_min'], var['var_max'], tar['count'], tar['mean'], var['WoE']], axis=1)
    summ.sort(['var_min'], ascending=[1], inplace=True)

summ