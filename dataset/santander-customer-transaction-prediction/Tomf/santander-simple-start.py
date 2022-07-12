import pandas as pd
from sklearn.linear_model import LogisticRegression as lr

data = pd.read_csv('../input/train.csv')
data2 = pd.read_csv('../input/test.csv')

x=data.iloc[:,2:]
y=data.iloc[:,1].values.ravel()
x2=data2.iloc[:,1:]

mod=lr(random_state=0, solver='lbfgs', max_iter=10000, multi_class='ovr').fit(x,y)

preds=mod.predict(x2)

labels=[]
for i in range(0,len(preds)):
 labels.append('test_'+str(i))

results=pd.DataFrame({'ID_code':labels,'target':preds})
results.to_csv('sub.csv',index=False)
