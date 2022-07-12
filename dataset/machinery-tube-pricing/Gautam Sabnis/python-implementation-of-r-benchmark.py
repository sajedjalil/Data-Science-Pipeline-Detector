import os
os.system("ls ../input")
os.system("echo \n\n")
os.system("head ../input/*")

"""
Author:Tanay Chowdhury
Following the R script from arnaud demytt 
"""

import pandas as pd
import numpy as np
import glob
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
### Load train and test
test = pd.read_csv("../input/test_set.csv")
train = pd.read_csv("../input/train_set.csv")

train['id']=pd.Series(np.arange(-1,-train.shape[0],-1))
test['cost']=0

train=train.append(test)

continueLoop = True

while(continueLoop):
   continueLoop = False
   for f in glob.glob(os.path.join("../input/", '*.csv')):
        d=pd.read_csv(f)
        commonVariables=train.columns.difference(train.columns.difference(d.columns))
        if len(commonVariables)==1:
            train=pd.merge(train, d[d.columns.difference(train.columns)], left_index=True, right_index=True, how='outer')
            continueLoop=True
            print (train.shape)
        

test = train[train['id']>0]
train = train[train['id']<0]


y=np.log(train['cost']+1)
train1= train[train.columns.difference(train.columns[[2, 3]])]
test1=test[test.columns.difference(test.columns[[2, 3]])]


#### Separating numeric and character columns #####
numcols=[];charcols=[]
for i in range(len(train1.columns)):
    if train1.ix[:,i].dtypes != 'O':
        numcols.append(train1.columns[i])
    else:
        charcols.append(train1.columns[i])

for i in range(len(train1.columns)):
    if train1.ix[:,i].dtypes  != 'O':
      train1.ix[:,i]=train1.ix[:,i].fillna(-1) 
    else:
       train1.ix[:,i]=train1.ix[:,i].fillna("NAvalue") 
       
for i in range(len(test1.columns)):
    if test1.ix[:,i].dtypes  != 'O':
      test1.ix[:,i]=test1.ix[:,i].fillna(-1) 
    else:
       test1.ix[:,i]=test1.ix[:,i].fillna("NAvalue") 



### Clean variables with too many categories
#for i in range(len(charcols)):
#    print charcols[i],len(set(train1[charcols[i]]))
#    tmp=train1[charcols[i]].value_counts()<30
#    for j in range(len(train1[charcols[i]])):
#        val=train1[charcols[i]][[j]].values[0]
#        if val in tmp.index and tmp[val]:
#            train1[charcols[i]][[j]]='rarevalue'
#            
#for i in range(len(charcols)):
#    print charcols[i],len(set(test1[charcols[i]]))
#    tmp=test1[charcols[i]].value_counts()<30
#    for j in range(len(test1[charcols[i]])):
#        val=test1[charcols[i]][[j]].values[0]
#        if val in tmp.index and tmp[val]:
#            test1[charcols[i]][[j]]='rarevalue'

################################################


rf = RandomForestRegressor(random_state=0, n_estimators=40,max_depth=10)
rf.fit(train1[numcols],y)

yhat=rf.predict(train1[numcols])
print (mean_squared_error(yhat,y))

pred=rf.predict(test1[numcols])
pred=np.exp(pred)-1

tmp=test['id'].reshape(30235,1).astype('int')
tmp1=pred.reshape(30235,1)
tmp2=np.hstack([tmp,tmp1])
df=pd.DataFrame.from_records(tmp2,columns=['id','cost'])
df['id']=df['id'].astype(int)

df.to_csv("submission.csv",index=False)
