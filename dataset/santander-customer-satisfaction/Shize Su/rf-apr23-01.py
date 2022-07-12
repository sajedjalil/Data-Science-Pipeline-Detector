import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt

np.random.seed(8)


train=pd.read_csv('../input/train.csv',header=0)
testpd=pd.read_csv('../input/test.csv',header=0)

print(train.shape, testpd.shape)
test = testpd.values
print("done reading data.")



rf = RandomForestClassifier(n_estimators=400)



target= train.TARGET
train = train.drop(['TARGET'],axis=1)

train = train.values
target = target.values


train=train[:,0:160]
test=test[:,0:160]


print("Start modeling:")
rf.fit(train,target)

print("Start prediction:")
pred = rf.predict_proba(test)
submission2 = pd.DataFrame({"ID":testpd.ID, "TARGET":pred[:,1]})
submission2.to_csv("sub_ssz_rf_apr23_03.csv", index=False)

print("Done!")
