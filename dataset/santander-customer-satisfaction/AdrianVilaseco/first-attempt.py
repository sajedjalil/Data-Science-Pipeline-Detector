import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt

train=pd.read_csv('../input/train.csv',header=0)
testpd=pd.read_csv('../input/test.csv',header=0)
test = testpd.values


rf = RandomForestClassifier(n_estimators=100)

target= train.TARGET
train = train.drop(['TARGET'],axis=1)

train = train.values
target = target.values

rf.fit(train,target)
pred = rf.predict_proba(test)
submission2 = pd.DataFrame({"ID":testpd.ID, "TARGET":pred[:,1]})
submission2.to_csv("submission.csv", index=False)


