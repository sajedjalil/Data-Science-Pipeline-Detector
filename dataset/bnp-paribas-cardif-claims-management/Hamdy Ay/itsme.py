# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import pandas as pd
import csv
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import Imputer
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import scipy, pylab
from sklearn.utils import shuffle



def converter(wd):
    if wd==wd:  #catch NaN
        sum = 0
        for i in range(len(wd)):
            sum += (ord(wd[i].lower())-ord('a')+1)
        return sum
    else:
        return wd

train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
train=train.drop(['ID'],axis=1)
testID=['ID']
testID= test.ID
test=test.drop(['ID'],axis=1)
target= train.target
train=train.drop(['target'],axis=1)
#testTarget=test.target
#testTarget1=test.target
#test=test.drop(['target'],axis=1)

print(testID)
train.v3=train.v3.apply(converter)
train.v22=train.v22.apply(converter)
train.v24=train.v24.apply(converter)
train.v30=train.v30.apply(converter)
train.v31=train.v31.apply(converter)
train.v47=train.v47.apply(converter)
train.v52=train.v52.apply(converter)
train.v56=train.v56.apply(converter)
train.v66=train.v66.apply(converter)
train.v71=train.v71.apply(converter)
train.v74=train.v74.apply(converter)
train.v75=train.v75.apply(converter)
train.v79=train.v79.apply(converter)
train.v91=train.v91.apply(converter)
train.v107=train.v107.apply(converter)
train.v110=train.v110.apply(converter)
train.v112=train.v112.apply(converter)
train.v113=train.v113.apply(converter)
train.v125=train.v125.apply(converter)

test.v3=test.v3.apply(converter)
test.v22=test.v22.apply(converter)
test.v24=test.v24.apply(converter)
test.v30=test.v30.apply(converter)
test.v31=test.v31.apply(converter)
test.v47=test.v47.apply(converter)
test.v52=test.v52.apply(converter)
test.v56=test.v56.apply(converter)
test.v66=test.v66.apply(converter)
test.v71=test.v71.apply(converter)
test.v74=test.v74.apply(converter)
test.v75=test.v75.apply(converter)
test.v79=test.v79.apply(converter)
test.v91=test.v91.apply(converter)
test.v107=test.v107.apply(converter)
test.v110=test.v110.apply(converter)
test.v112=test.v112.apply(converter)
test.v113=test.v113.apply(converter)
test.v125=test.v125.apply(converter)

imp=Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(train)
imp.fit(test)
train=imp.transform(train)
test=imp.transform(test)

columns =['v1','v2','v3','v4','v5','v6','v7','v8','v9','v10','v11','v12','v13','v14','v15','v16','v17','v18','v19','v20','v21','v22','v23','v24','v25','v26','v27','v28','v29','v30','v31','v32','v33','v34','v35','v36','v37','v38','v39','v40','v41','v42','v43','v44','v45','v46','v47','v48','v49','v50','v51','v52','v53','v54','v55','v56','v57','v58','v59','v60','v61','v62','v63','v64','v65','v66','v67','v68','v69','v70','v71','v72','v73','v74','v75','v76','v77','v78','v79','v80','v81','v82','v83','v84','v85','v86','v87','v88','v89','v90','v91','v92','v93','v94','v95','v96','v97','v98','v99','v100','v101','v102','v103','v104','v105','v106','v107','v108','v109','v110','v111','v112','v113','v114','v115','v116','v117','v118','v119','v120','v121','v122','v123','v124','v125','v126','v127','v128','v129','v130','v131']

scaler=StandardScaler()
scaler.fit(train)
train=scaler.transform(train)
test=scaler.transform(test)

print('\n')
train.shape
print('\n')


lsvc = LinearSVC(C=0.25,loss='hinge', penalty="l2", dual=True).fit(train, target)
model = SelectFromModel(lsvc, prefit=True)
train  = model.transform(train)
test = model.transform(test)



#data = pd.concat((train,test),axis=0,ignore_index=True)
#print(train_updated.shape)
#print(test.shape)
#clf=svm.SVC(probability=True)
#train = shuffle(train,random_state=0)

#clf = SGDClassifier(loss="log",penalty="l2")
#clf.fit(train,target)
#results=['Results']
#results=clf.predict_proba(test)
#score = log_loss(testTarget,results)

clf = RandomForestClassifier(n_estimators=200,max_features=None,min_samples_leaf=2,max_depth=30)
clf.fit(train,target)
resultsRandom=['Results']
resultsRandom=clf.predict_proba(test)
#scoreRandom = log_loss(testTarget1,resultsRandom)

final=pd.concat([pd.DataFrame(testID),pd.DataFrame(resultsRandom)],axis=1)
final.columns = ['ID','0','PredictedProb']
results = final.drop(final.columns[0],axis=1)

final.drop(final.columns[1], axis=1, inplace=True) # Note: zero indexed
final.to_csv("submission.csv",index=False)
#print(score)
#print(scoreRandom)




# Any results you write to the current directory are saved as output.