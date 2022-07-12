# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
from sklearn.preprocessing import OneHotEncoder,LabelBinarizer,LabelEncoder
print(check_output(["ls", "../input"]).decode("utf8"))

train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")

lb=LabelBinarizer()

colors=pd.DataFrame(lb.fit_transform(train['color']),columns=[ 'white','black','clear','blue','green','blood'])
types=pd.DataFrame(lb.fit_transform(train['type']),columns=['Ghost', 'Goblin','Ghoul'])
y_train=train['type']
le = LabelEncoder().fit(y_train)
y_train = le.transform(y_train)
data=train.drop(['id','color','type'],axis=1)
data=pd.concat([data,colors,types],axis=1)

print(data.describe())

traindata=data.iloc[:,:-3]

targetdata=data[['Ghost', 'Goblin','Ghoul']]
print(targetdata)
print(traindata.shape,targetdata.shape)

testcolors=pd.DataFrame(lb.fit_transform(test['color']),columns=[ 'white','black','clear','blue','green','blood'])
testdata=pd.concat([test[:-1],testcolors],axis=1)
#testdata=testdata.dropna()
ids=testdata['id'].fillna(899)
testdata=testdata.drop(['id','color'],axis=1)

testdata["bone_length"]=testdata["bone_length"].fillna(testdata["bone_length"].median())
testdata["rotting_flesh"]=testdata["rotting_flesh"].fillna(testdata["rotting_flesh"].median())
testdata["hair_length"]=testdata["hair_length"].fillna(testdata["hair_length"].median())
testdata["has_soul"]=testdata["has_soul"].fillna(testdata["has_soul"].median())


# Any results you write to the current directory are saved as output.

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.svc import SVM

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2))
#clf=RandomForestClassifier(n_estimators=100)
clf.fit(traindata,y_train)
print(testdata.describe())
predictions=clf.predict_proba(testdata)
print(predictions)
def maxval(row):
    m=row[0]
    p=0
    for i in range(3):
        if row[i]>m:
            p=i
            m=row[i]
    return p
results=np.apply_along_axis(maxval,1,predictions)
print(results)
q=[]
for i in ids:
    print(i)
    q.append(int(i))

a=pd.DataFrame({'type':[['Ghost', 'Goblin','Ghoul'][p] for p in results],'id':q})
b=a[['id','type']]
b.to_csv("submission.csv",index=False)
#pd.DataFrame(a['id'],a['Type'])
#pd.DataFrame(a.iloc[:,1],a.iloc[:,0])


    
    
    
    
    