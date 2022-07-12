import pandas as pd

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 11:21:42 2017

@author: Shresth
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.learning_curve import validation_curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation 
#train=pd.read_csv("train.csv")
#test=pd.read_csv("test.csv")

ids=test["Id"]
test=test.drop("Id",axis=1)
train=train.drop("Id",axis=1)

y=train["Cover_Type"]
xtrain=train.drop("Cover_Type",axis=1)

predictors=list(xtrain.columns)
print (predictors)

alg=RandomForestClassifier(max_features='sqrt')

def feat_eng(df):
    
    #absolute distance to water
    df['Distance_To_Hydrology']=(df['Vertical_Distance_To_Hydrology']**2.0+
                                 df['Horizontal_Distance_To_Hydrology']**2.0)**0.5
    
feat_eng(xtrain)
feat_eng(test)
print(xtrain.describe())
print("now cols")
cols=list(xtrain.columns[:10])
cols.append('Distance_To_Hydrology')
cats=list(xtrain.columns[10:-1])
print(cols)
print(cats)
#X_Norm=xtrain.copy()

#normalization
from sklearn.preprocessing import Normalizer
def norm(xtrain):
    X_Norm = Normalizer().fit_transform(xtrain[cols])
    X_Norm2=pd.DataFrame(X_Norm,columns=cols)
    print(X_Norm2.describe(),X_Norm2.shape)
    #print (X_Norm.describe())
    print(X_Norm2[cols[:-1]].shape,xtrain[cats].shape,X_Norm2[cols[-1]].shape)
    new=pd.concat([X_Norm2[cols[:-1]],xtrain[cats],X_Norm2[cols[-1]]],axis=1)
    return new
new=norm(xtrain)
newtest=norm(test)
#X_con=np.concatenate((X_Norm[cols[:-1]],xtrain[cats],X_Norm[cols[-1]]),axis=1)
#xconpd=pd.DataFrame(X_con,columns=list(xtrain.columns))
#print(xconpd.head())

#PCA

from sklearn import decomposition,datasets
"""print("hi")
pca = decomposition.PCA(n_components=54) #Finds first 200 PCs
pca.fit(new)
plt.plot(pca.explained_variance_ratio_)
plt.ylabel('% of variance explained')
plt.show()
plt.savefig("test.png")
print("oo")"""
"""
prange=np.arange(50,650,50)
train_scores, cv_scores = validation_curve(alg, xtrain, y,param_name='n_estimators',
                                              param_range=prange)


plt.xlabel('n_estimators')
plt.ylabel('Mean CV score')
plt.plot(prange, np.mean(cv_scores, axis=1), label="Cross-validation score",color="g")
"""

alg.set_params(n_estimators=350)

alg.fit(new,y)
ranks= list(pd.DataFrame(alg.feature_importances_,index=new.columns).sort([0], ascending=False).axes[0])
alg.fit(new[ranks[:38]],y)
print ("hi")
"""prange=np.arange(200,600,50)
train_scores, cv_scores = validation_curve(alg, xtrain[ranks[:38]], y,param_name='n_estimators',
                                              param_range=prange)


plt.xlabel('n_estimators')
plt.ylabel('Mean CV score')
plt.plot(prange, np.mean(cv_scores, axis=1), label="Cross-validation score",color="g")
"""
predictions=alg.predict(newtest[ranks[:38]])
sub=pd.DataFrame({ "Cover_Type": predictions,
                      "Id": ids
                     
                     })
newhh=sub[['Id','Cover_Type']]
print (newhh)
newhh.to_csv("submissiongradboost.csv", index=False)

