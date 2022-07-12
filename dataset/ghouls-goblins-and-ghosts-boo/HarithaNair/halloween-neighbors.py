# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelBinarizer
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
ids=train.values[:,0]
testid=test.values[:,0]
train=train.drop('id',axis=1)
lb=LabelBinarizer()
coloring=lb.fit_transform(train["color"])
#print(train.shape)
for i in range(371):
    colored=pd.DataFrame({"white":coloring[:,0],"black":coloring[:,1],"clear":coloring[:,2],"blue":coloring[:,3],"green":coloring[:,4],"blood":coloring[:,5]})
train['white'] = pd.Series(colored['white'])
train['black'] = pd.Series(colored['black'])
train['clear'] = pd.Series(colored['clear'])
train['blue'] = pd.Series(colored['blue'])
train['green'] = pd.Series(colored['green'])
train['blood'] = pd.Series(colored['blood'])
train=train.drop('color',axis=1)

coloring1=lb.fit_transform(test["color"])
#print(train.shape)
for i in range(371):
    colored1=pd.DataFrame({"white":coloring1[:,0],"black":coloring1[:,1],"clear":coloring1[:,2],"blue":coloring1[:,3],"green":coloring1[:,4],"blood":coloring1[:,5]})
test['white'] = pd.Series(colored1['white'])
test['black'] = pd.Series(colored1['black'])
test['clear'] = pd.Series(colored1['clear'])
test['blue'] = pd.Series(colored1['blue'])
test['green'] = pd.Series(colored1['green'])
test['blood'] = pd.Series(colored1['blood'])
test=test.drop(['color','id'],axis=1)
"""
ghastly=lb.fit_transform(train["type"])
for i in range(371):
    ghoost=pd.DataFrame({"ghost":ghastly[:,0],"goblin":ghastly[:,1],"ghoul":ghastly[:,2]})"""
    
new=train["type"]
train=train.drop('type',axis=1)
X=pd.DataFrame.as_matrix(train)
y=pd.DataFrame.as_matrix(new)

# Any results you write to the current directory are saved as output.
params={"n_neighbors":[5,8,11,14],'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],'leaf_size':[20,30,40],'p':[1,2]}
alg=neighbors.KNeighborsClassifier()
alg.fit(X,y)
prediction=alg.predict(test)
submission=pd.DataFrame({'id':testid, 'type': prediction})
submission.to_csv("submissionghost2.csv",index=False)