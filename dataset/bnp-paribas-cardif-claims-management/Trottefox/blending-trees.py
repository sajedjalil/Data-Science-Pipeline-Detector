# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import random
rnd=57
maxCategories=20

train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
random.seed(rnd)
train.index=train.ID
test.index=test.ID
del train['ID'], test['ID']
target=train.target
del train['target']


#prepare data
traindummies=pd.DataFrame()
testdummies=pd.DataFrame()

for elt in train.columns:
    vector=pd.concat([train[elt],test[elt]], axis=0)

    #count as categorial if number of unique values is less than maxCategories
    if len(vector.unique())<maxCategories:
        traindummies=pd.concat([traindummies, pd.get_dummies(train[elt],prefix=elt,dummy_na=True)], axis=1).astype('int8')
        testdummies=pd.concat([testdummies, pd.get_dummies(test[elt],prefix=elt,dummy_na=True)], axis=1).astype('int8')
        del train[elt], test[elt]
    else:
        typ=str(train[elt].dtype)[:3]
        if (typ=='flo') or (typ=='int'):
            minimum=vector.min()
            maximum=vector.max()
            train[elt]=train[elt].fillna(int(minimum)-2)
            test[elt]=test[elt].fillna(int(minimum)-2)
            minimum=int(minimum)-2
            traindummies[elt+'_na']=train[elt].apply(lambda x: 1 if x==minimum else 0)
            testdummies[elt+'_na']=test[elt].apply(lambda x: 1 if x==minimum else 0)
            

            #resize between 0 and 1 linearly ax+b
            a=1/(maximum-minimum)
            b=-a*minimum
            train[elt]=a*train[elt]+b
            test[elt]=a*test[elt]+b
        else:
            if (typ=='obj'):
                list2keep=vector.value_counts()[:maxCategories].index
                train[elt]=train[elt].apply(lambda x: x if x in list2keep else np.nan)
                test[elt]=test[elt].apply(lambda x: x if x in list2keep else np.nan)                
                traindummies=pd.concat([traindummies, pd.get_dummies(train[elt],prefix=elt,dummy_na=True)], axis=1).astype('int8')
                testdummies=pd.concat([testdummies, pd.get_dummies(test[elt],prefix=elt,dummy_na=True)], axis=1).astype('int8')
                
                #Replace categories by their weights
                tempTable=pd.concat([train[elt], target], axis=1)
                tempTable=tempTable.groupby(by=elt, axis=0).agg(['sum','count']).target
                tempTable['weight']=tempTable.apply(lambda x: .5+.5*x['sum']/x['count'] if (x['sum']>x['count']-x['sum']) else .5+.5*(x['sum']-x['count'])/x['count'], axis=1)
                tempTable.reset_index(inplace=True)
                train[elt+'weight']=pd.merge(train, tempTable, how='left', on=elt)['weight']
                test[elt+'weight']=pd.merge(test, tempTable, how='left', on=elt)['weight']
                train[elt+'weight']=train[elt+'weight'].fillna(.5)
                test[elt+'weight']=test[elt+'weight'].fillna(.5)
                del train[elt], test[elt]
            else:
                print('error', typ)

#remove na values too similar to v2_na
from sklearn import metrics
for elt in train.columns:
    if (elt[-2:]=='na') & (elt!='v2_na'):
        dist=metrics.pairwise_distances(train.v2_na.reshape(1, -1),train[elt].reshape(1, -1))
        if dist<8:
            del train[elt],test[elt]
        else:
            print(elt, dist)
            
            
train=pd.concat([train,traindummies, target], axis=1)
test=pd.concat([test,testdummies], axis=1)
del traindummies,testdummies


#remove features only present in train or test
for elt in list(set(train.columns)-set(test.columns)):
    del train[elt]
for elt in list(set(test.columns)-set(train.columns)):
    del test[elt]
    
#run cross validation
from sklearn import cross_validation
X, Y, Xtarget, Ytarget=cross_validation.train_test_split(train, target, test_size=0.2)
del train

from sklearn import ensemble

clfs=[
    ensemble.RandomForestClassifier(bootstrap=False, class_weight='auto', criterion='entropy',
            max_depth=None, max_features='sqrt', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=4,
            min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=-1,
            oob_score=False, random_state=rnd, verbose=0,
            warm_start=False),
    ensemble.ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='entropy',
           max_depth=None, max_features='sqrt', max_leaf_nodes=None,
           min_samples_leaf=1, min_samples_split=3,
           min_weight_fraction_leaf=1e-5, n_estimators=500, n_jobs=-1,
           oob_score=False, random_state=rnd, verbose=0, warm_start=False),
    ensemble.GradientBoostingClassifier(init=None, learning_rate=0.1, loss='deviance',
              max_depth=2, max_features=None, max_leaf_nodes=None,
              min_samples_leaf=1, min_samples_split=3,
              min_weight_fraction_leaf=0.0, n_estimators=50,
              presort='auto', random_state=rnd, subsample=1.0, verbose=0,
              warm_start=False)
]

indice=0
preds=[]
predstest=[]

#run models
for model in clfs:
    
    model.fit(X, Xtarget)

    preds.append(model.predict_proba(Y)[:,1])
    print('model ',indice,': loss=',metrics.log_loss(Ytarget,preds[indice]))

    noms=pd.DataFrame(test.columns[abs(model.feature_importances_)>1e-10][:30])
    noms.columns=['noms']
    coefs=pd.DataFrame(model.feature_importances_[abs(model.feature_importances_)>1e-10][:30])
    coefs.columns=['coefs']
    df=pd.concat([noms, coefs], axis=1).sort_values(by=['coefs'])

    plt.figure(indice)
    df.plot(kind='barh', x='noms', y='coefs', legend=True, figsize=(6, 10))
    plt.savefig('clf'+str(indice)+'_ft_importances.jpg')

    predstest.append(model.predict_proba(test)[:,1])
    indice+=1

#find best weights
step=0.1 * (1./len(preds))
print("step:", step)
poidsref=np.zeros(len(preds))
poids=np.zeros(len(preds))
poidsreftemp=np.zeros(len(preds))
poidsref=poidsref+1./len(preds)

bestpoids=poidsref.copy()
blend_cv=np.zeros(len(preds[0]))

for k in range(0,len(preds),1):
    blend_cv=blend_cv+bestpoids[k]*preds[k]
bestscore=metrics.log_loss(Ytarget.values,blend_cv)

getting_better_score=True
while getting_better_score:
    getting_better_score=False
    for i in range(0,len(preds),1):
        poids=poidsref
        if poids[i]-step>-step:
            #decrease weight in position i
            poids[i]-=step
            for j in range(0,len(preds),1):
                if j!=i:
                    if poids[j]+step<=1:
                        #try an increase in position j
                        poids[j]+=step
                        #score new weights
                        blend_cv=np.zeros(len(preds[0]))
                        for k in range(0,len(preds),1):
                            blend_cv=blend_cv+poids[k]*preds[k]
                        actualscore=metrics.log_loss(Ytarget.values,blend_cv)
                        #if better, keep it
                        if actualscore<bestscore:
                            bestscore=actualscore
                            bestpoids=poids.copy()
                            getting_better_score=True
                        poids[j]-=step
            poids[i]+=step
    poidsref=bestpoids.copy()

print("weights: ", bestpoids)
print("optimal blend loss: ", bestscore)


blend_to_submit=np.zeros(len(predstest[0]))

for i in range(0,len(preds),1):
    blend_to_submit=blend_to_submit+bestpoids[i]*predstest[i]

#submit
submission=pd.read_csv('../input/sample_submission.csv')
submission.PredictedProb=blend_to_submit
submission.to_csv('simpleblend.csv', index=False)

plt.figure(indice)
submission.PredictedProb.hist(bins=30)
plt.savefig('distribution.jpg')