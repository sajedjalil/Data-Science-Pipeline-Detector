#Training set has 61878 rows and 95 columns

import csv
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.cross_validation import KFold

x=[]
y=[]
with open("../input/train.csv",'r') as f:
    r=csv.reader(f)
    header=next(r)
    #print(header)
    for row in r:
        x.append([float(e) for e in row[1:-1]])
        y.append(float(row[-1][-1]))
s=StandardScaler()
x=s.fit_transform(x)
y=np.array(y)

nrow=len(y)
nlab=9

def docv(clf,x,y,kf):
    pred=np.zeros((nrow,nlab))
    for trainidx,testidx in kf:
        clf.fit(x[trainidx],y[trainidx])
        pred[testidx]=clf.predict_proba(x[testidx])
    return pred

kf=KFold(nrow,5,shuffle=True,random_state=0)
for clf,clfname in (
#        [LogisticRegression(C=10),'logreg10'],
#        [LogisticRegression(C=50),'logreg50'],
#        [LogisticRegression(C=100),'logreg100'],
#        [RandomForestClassifier(n_estimators=10),'rf10'],
#        [RandomForestClassifier(n_estimators=20),'rf20'],
#        [RandomForestClassifier(n_estimators=50),'rf50'],
#        [RandomForestClassifier(n_estimators=100),'rf100'],
#        [RandomForestClassifier(n_estimators=200),'rf200'],
        ):
    print('%s %.3f'%(clfname,log_loss(y,docv(clf,x,y,kf))))
"""
logreg10/50/100 0.672 without StandardScaler
logreg10 0.672 with StandardScaler
rf10 1.545
rf20 0.938
rf50 0.669
rf100 0.600
rf200 0.590
"""
    


