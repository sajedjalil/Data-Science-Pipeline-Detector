# Note: Kaggle only runs Python 3, not Python 2

# We'll use the pandas library to read CSV files into dataframes
import pandas as pd
import numpy as np

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")
sample=pd.read_csv("../input/sample_submission.csv") 

#print(sample)

# Write summaries of the train and test sets to the log
#print('\nSummary of train dataset:\n')
#print(train.describe())
#print('\nSummary of test dataset:\n')
#print(test.describe())

z=[]
for p in range(1,3):
    for j in range(1,18):
        m="T"+str(p)+"_"+"V"+str(j)
        z.append(m)
        
z.remove("T2_V16")
z.remove("T2_V17")
#print(z)

#replace N and Y with 0,1
True_False=["T1_V6","T1_V17","T2_V3","T2_V11","T2_V12"]
for tf in True_False:
    train[tf]=train[tf].replace('N',0)
    train[tf]=train[tf].replace('Y',1)
    test[tf]=test[tf].replace('N',0)
    test[tf]=test[tf].replace('Y',1)

#the structures of dataset
#for tag in z:
#    print(tag)
#    print(train[tag].value_counts())

#print(train.head(1))

alp_list=[]
for i in range(26):
    s=i+65
    a=chr(s)
    alp_list.append(a)
    s=0

#get header of the dataset
#header=train.columns
#print(header[1:3])

#replace alphabet with number
str_list=["T1_V4","T1_V5","T1_V7","T1_V8","T1_V9","T1_V11","T1_V12","T1_V15","T1_V16","T2_V5","T2_V13"]
for i in str_list:
    train[i]=train[i].replace(alp_list,range(1,27))
    test[i]=test[i].replace(alp_list,range(1,27))

score=train['Hazard']

#return the type of the tag
def tag_type(a):
    if a in True_False:
        return "judge"
    if a in str_list:
        return "choose"
    else:
        return "number"

#calculate the correlation between each tag and score
"""
score_max=0
for tag in z:
    corr_score=score.corr(train[tag])
    type_=tag_type(tag)
    print('%s correlation is %.3f type is %s' %(tag,corr_score,type_))
    if corr_score>score_max:
        score_max=corr_score
        max_tag=tag
        max_type=type_
print("max correlation is %.2f %s type is %s" %(score_max,max_tag,max_type))
"""

x=train[z]
y=score
test_x=test[z]

"""
from sklearn import linear_model

clf = linear_model.Lasso(alpha = 0.01)
clf.fit(x,y)
pre_=clf.predict(test_x)
sample['Hazard']=pre_
sample.to_csv("submit.csv",index=False)
print(clf.score(x,y))
"""

"""
try_max=0
for try_x in range(32):
    for try_y in range(32):
        zx=train[z[try_x:try_y]]
        if len(zx.columns)>0:
            clf2 = linear_model.BayesianRidge()
            clf2.fit(zx,y)
            sc=clf2.score(zx,y)
            if sc>try_max:
                try_max=sc
                tb=z[try_x:try_y]
                print(tb,try_max)                
"""
"""
las_max=0
for k in np.linspace(0.001,1,20):
    clf = linear_model.Lasso(alpha = k)
    clf.fit(x,y)
    pre_=clf.predict(test_x)
    sample['Hazard']=pre_
    #sample.to_csv("submit.csv",index=False)
    if clf.score(x,y)>las_max:
        las_max=clf.score(x,y)
        print(clf.score(x,y),k)
"""
"""
try_max=0
for try_x in range(32):
    for try_y in range(32):
        zx=train[z[try_x:try_y]]
        if len(zx.columns)>0:
           for k in np.linspace(0.01,1,20):
                clf = linear_model.Lasso(alpha = k)
                clf.fit(zx,y)
                #sample.to_csv("submit.csv",index=False)
                if clf.score(zx,y)>try_max:
                    try_max=clf.score(zx,y)
                    print(try_max,k,z[try_x:try_y])
"""

from sklearn import preprocessing
x=preprocessing.scale(x)

from sklearn.linear_model import SGDRegressor
clf =  SGDRegressor(loss="squared_loss", penalty="l2")
clf.fit(x,y)
pre_=clf.predict(test_x)
sample['Hazard']=pre_
sample.to_csv("submit.csv",index=False)
print(clf.score(x,y))
