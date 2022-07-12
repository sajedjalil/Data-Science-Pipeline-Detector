# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#import libraries
import time
from functools import wraps

#import data analytics libraries installed
import pandas as pd 
import numpy as np

from sklearn import cross_validation
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel

import xgboost as xgb

#import data visualization libraries
import matplotlib.pyplot as plt 
import seaborn as sns



#show run time of a function
'''
def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result=function(*args, **kwargs)
        t1 = time.time()
        print("Total running time %s: %s secconds" % (function.func_name, str(t1-t0)))
        return result
    return function_timer
'''

#----------------------------------------load data-----------------------------------
start=time.clock()


#@fn_timer
def readFile(fileName):
    return pd.read_csv(fileName)
    
df_train = readFile("../input/train.csv")#load data into a dataframe
df_test = readFile("../input/test.csv")



#loading time
end=time.clock()
print ("loading time: %f seconds" %(end -start ))
#------------------------------------------------------------------------------------



#------------------------------Data Preparation-----------------------------------------------------------
#Clean data


#@fn_timer 
def removeDuplicatedRowsAndColumns():
    #remove duplicated rows
    df_train.drop_duplicates()
    df_test.drop_duplicates()

    #remove duplicated columns
    remove = []
    cols = df_train.columns
    for i in range(len(cols)-1):
        v = df_train[cols[i]].values
        for j in range(i+1,len(cols)):
            if np.array_equal(v,df_train[cols[j]].values):
                remove.append(cols[j])

    df_train.drop(remove, axis=1, inplace=True)
    df_test.drop(remove, axis=1, inplace=True)
    
   
removeDuplicatedRowsAndColumns()


#@fn_timer
def removeConstantColumns():
    remove = []
    for col in df_train.columns:
        if df_train[col].std() == 0:
            remove.append(col)

    df_train.drop(remove, axis=1, inplace=True)
    df_test.drop(remove, axis=1, inplace=True)

    
removeConstantColumns()

#df_train.describe()
#var3 contains -999999, outlier?
df_train.var3.replace(-999999,2)


#-----------------------------------------Explore Data------------------------------------------
#line number
Nb_clients=df_train.TARGET.count()
print (Nb_clients)


#In TARGET column: 0 means happy, 1 means unhappy
#Distribution of Customer Satisfaction
df = df_train.TARGET.value_counts(1)
#df

#show distribution in Pie chart (just for fun)
rate=[df[0],df[1]]
labels = ['happy', 'unhappy']
colors = ['blue','orange']

plt.pie(rate, labels=labels, autopct='%1.2f%%', colors=colors)
plt.show()
#unbalanced positive and negative samples



'''
df_train.var15.describe()
#var15 is suspected to be the age of the customer
#show distribution in histogram chart
df_train.var15.hist(bins=100) 
plt.show()

'''

#remove ID and TARGET
Y = df_train.TARGET.values
X = df_train.drop(["ID","TARGET"], axis=1) 

test_id = df_test.ID
df_test = df_test.drop(["ID"], axis=1)



# Add PCA components as features
#@fn_timer 
def addPCAfeatures():
    X_normalized = normalize(X, axis=0)
    test_normalized= normalize(df_test, axis=0)
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_normalized)
    test_pca=pca.fit_transform(test_normalized)
    
    X['PCA1'] = X_pca[:,0]
    X['PCA2'] = X_pca[:,1]
    X['PCA3'] = X_pca[:,2]

    df_test['PCA1'] = test_pca[:,0]
    df_test['PCA2'] = test_pca[:,1]
    df_test['PCA3'] = test_pca[:,2]

    
addPCAfeatures()    
#X.ix[0:5, 306:309]


#split data
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.30, random_state=1000)
print(X_train.shape, X_test.shape, df_test.shape)

#Feature selection
clf = ExtraTreesClassifier(random_state=1000)
selector = clf.fit(X_train, y_train)




'''
#FeatureName = X_train.columns.values
impScore = pd.Series(clf.feature_importances_)
c = pd.DataFrame({'impScore':impScore})
c.index=X_train.columns.values

# plot most important features
c.sort_index(axis=0, by='impScore', ascending=False)[0:40].plot(kind='bar', title='Feature Importances according to ExtraTreesClassifier', figsize=(12, 8))
plt.ylabel('Feature Importance Score')
plt.subplots_adjust(bottom=0.3)
#plt.savefig('Figure1_FeatureImps.png')
plt.show()
'''


fs = SelectFromModel(selector, prefit=True)

X_train = fs.transform(X_train)
X_test = fs.transform(X_test)
df_test = fs.transform(df_test)

print(X_train.shape, X_test.shape, df_test.shape)






#--------------------Train Model-------------------------------------------
#xgboost
#@fn_timer 
def trainModel(i):
    m2_xgb = xgb.XGBClassifier(n_estimators=110, nthread=-1, max_depth = i, \
    seed=1000)
    m2_xgb.fit(X_train, y_train, eval_metric="auc", verbose = False,
               eval_set=[(X_test, y_test)])
    return m2_xgb

    
'''    
# calculate the auc score
for i in range(8):
    print("max_depth =%i" ,i+1 )
    print("Roc AUC: ", roc_auc_score(y_test, trainModel(i).predict_proba(X_test)[:,1]))
'''

#max_depth=6 is the best 
print("Roc AUC: ", roc_auc_score(y_test, trainModel(6).predict_proba(X_test)[:,1]))

#Submission
probs = trainModel(6).predict_proba(df_test)

submission = pd.DataFrame({"ID":test_id, "TARGET": probs[:,1]})
submission.to_csv("submission.csv", index=False)
