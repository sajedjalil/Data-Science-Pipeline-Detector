import kagglegym
import numpy as np
import pandas as pd
import random
from sklearn import ensemble, linear_model, metrics

env = kagglegym.make()
o = env.reset()
train = o.train
print(train.shape)
d_mean= train.median(axis=0)
train["nbnulls"]=train.isnull().sum(axis=1)
col=[x for x in train.columns if x not in ['id', 'timestamp', 'y']]

rnd=17

#keeping na information on some columns (best selected by the tree algorithms)
add_nas_ft=True
nas_cols=['technical_9', 'technical_0', 'technical_32', 'technical_16', 'technical_38', 
'technical_44', 'technical_20', 'technical_30', 'technical_13']
#columns kept for evolution from one month to another (best selected by the tree algorithms)
add_diff_ft=True
diff_cols=['technical_22','technical_20', 'technical_30', 'technical_13', 'technical_34']

#homemade class used to infer randomly on the way the model learns
class createLinearFeatures:
    
    def __init__(self, n_neighbours=1, max_elts=None, verbose=True, random_state=None):
        self.rnd=random_state
        self.n=n_neighbours
        self.max_elts=max_elts
        self.verbose=verbose
        self.neighbours=[]
        self.clfs=[]
        
    def fit(self,train,y):
        if self.rnd!=None:
            random.seed(self.rnd)
        if self.max_elts==None:
            self.max_elts=len(train.columns)
        list_vars=list(train.columns)
        random.shuffle(list_vars)
        
        lastscores=np.zeros(self.n)+1e15

        for elt in list_vars[:self.n]:
            self.neighbours.append([elt])
        list_vars=list_vars[self.n:]
        
        for elt in list_vars:
            indice=0
            scores=[]
            for elt2 in self.neighbours:
                if len(elt2)<self.max_elts:
                    clf=linear_model.LinearRegression(fit_intercept=False, normalize=True, copy_X=True, n_jobs=-1) 
                    clf.fit(train[elt2+[elt]], y)
                    scores.append(metrics.mean_squared_error(y,clf.predict(train[elt2 + [elt]])))
                    indice=indice+1
                else:
                    scores.append(lastscores[indice])
                    indice=indice+1
            gains=lastscores-scores
            if gains.max()>0:
                temp=gains.argmax()
                lastscores[temp]=scores[temp]
                self.neighbours[temp].append(elt)

        indice=0
        for elt in self.neighbours:
            clf=linear_model.LinearRegression(fit_intercept=False, normalize=True, copy_X=True, n_jobs=-1) 
            clf.fit(train[elt], y)
            self.clfs.append(clf)
            if self.verbose:
                print(indice, lastscores[indice], elt)
            indice=indice+1
                    
    def transform(self, train):
        indice=0
        for elt in self.neighbours:
            #this line generates a warning. Could be avoided by working and returning
            #with a copy of train.
            #kept this way for memory management
            train['neighbour'+str(indice)]=self.clfs[indice].predict(train[elt])
            indice=indice+1
        return train
    
    def fit_transform(self, train, y):
        self.fit(train, y)
        return self.transform(train)

#a home-made class attempt to remove outliers by successive quantization on residuals
class recurrent_linear_approx():
    def __init__(self, quant=.999, limit_size_train=.9):
        self.quant=quant
        self.limit_size_train=limit_size_train
        self.bestmodel=[]
       
    def fit(self, train, y):
        internal_model=linear_model.Ridge(fit_intercept=False)
        bestscore=1e15
        better=True
        indextrain=train.dropna().index
        limitlen=len(train)*self.limit_size_train
        while better:
            internal_model.fit(train.ix[indextrain], y.ix[indextrain])
            score=metrics.mean_squared_error(internal_model.predict(train.ix[indextrain]), y.ix[indextrain])
            if score < bestscore:
                bestscore=score
                self.bestmodel=internal_model
                residual=y.ix[indextrain]-internal_model.predict(train.ix[indextrain])
                indextrain=residual[abs(residual)<=abs(residual).quantile(self.quant)].index
                if len(indextrain)<limitlen:
                    better=False
            else:
                better=False
                self.bestmodel=internal_model

    def predict(self, test):
        return self.bestmodel.predict(test)

if add_nas_ft:
    for elt in nas_cols:
        train[elt + '_na'] = pd.isnull(train[elt]).apply(lambda x: 1 if x else 0)
        #no need to keep columns with no information
        if len(train[elt + '_na'].unique())==1:
            print("removed:", elt, '_na')
            del train[elt + '_na']
            nas_cols.remove(elt)


if add_diff_ft:
    train=train.sort_values(by=['id','timestamp'])
    for elt in diff_cols:
        #a quick way to obtain deltas from one month to another but it is false on the first
        #month of each id
        train[elt+"_d"]= train[elt].rolling(2).apply(lambda x:x[1]-x[0]).fillna(0)
    #removing month 0 to reduce the impact of erroneous deltas
    train=train[train.timestamp!=0]

print(train.shape)
cols=[x for x in train.columns if x not in ['id', 'timestamp', 'y']]


#generation of linear models
cols2fit=['technical_22','technical_20', 'technical_30_d', 'technical_20_d', 'technical_30', 
'technical_13', 'technical_34']
models=[]
columns=[]
residuals=[]
for elt in cols2fit:
    print("fitting linear model on ", elt)
    model=recurrent_linear_approx(quant=.99, limit_size_train=.9)
    model.fit(train.loc[:,[elt]],train.loc[:, 'y'])
    models.append(model)
    columns.append([elt])
    residuals.append(abs(model.predict(train[[elt]].fillna(d_mean))-train.y))

train=train.fillna(d_mean)
    
#adding all trees generated by a tree regressor
print("adding new features")
featureexpander=createLinearFeatures(n_neighbours=30, max_elts=2, verbose=True, random_state=rnd)
index2use=train[abs(train.y)<0.086].index
featureexpander.fit(train.ix[index2use,cols],train.ix[index2use,'y'])
trainer=featureexpander.transform(train[cols])
treecols=trainer.columns

print("training trees")
model = ensemble.ExtraTreesRegressor(n_estimators=100, max_depth=4, n_jobs=-1, random_state=rnd, verbose=0)
model.fit(trainer,train.y)
print(pd.DataFrame(model.feature_importances_,index=treecols).sort_values(by=[0]).tail(30))
for elt in model.estimators_:
    models.append(elt)
    columns.append(treecols)
    residuals.append(abs(elt.predict(trainer)-train.y))


#model selection : create a new target selecting models with lowest asolute residual for each line
#the objective at this step is to keep only the few best elements which should
#lead to a better generalization
num_to_keep=10
targetselector=np.array(residuals).T
targetselector=np.argmin(targetselector, axis=1)
print("selecting best models:")
print(pd.Series(targetselector).value_counts().head(num_to_keep))

tokeep=pd.Series(targetselector).value_counts().head(num_to_keep).index
tokeepmodels=[]
tokeepcolumns=[]
tokeepresiduals=[]
for elt in tokeep:
    tokeepmodels.append(models[elt])
    tokeepcolumns.append(columns[elt])
    tokeepresiduals.append(residuals[elt])

#creating a new target for a model in charge of predicting which model is best for the current line
targetselector=np.array(tokeepresiduals).T
targetselector=np.argmin(targetselector, axis=1)

print("training selection model")
modelselector = ensemble.ExtraTreesClassifier(n_estimators=100, max_depth=4, n_jobs=-1, random_state=rnd, verbose=0)
modelselector.fit(trainer, targetselector)
print(pd.DataFrame(modelselector.feature_importances_,index=treecols).sort_values(by=[0]).tail(30))

lastvalues=train[train.timestamp==905][['id']+diff_cols].copy()

print("end of trainind, now predicting")
indice=0
countplus=0
rewards=[]
while True:
    indice+=1
    test = o.features
    test["nbnulls"]=test.isnull().sum(axis=1)
    if add_nas_ft:
        for elt in nas_cols:
            test[elt + '_na'] = pd.isnull(test[elt]).apply(lambda x: 1 if x else 0)
    test=test.fillna(d_mean)

    pred = o.target
    if add_diff_ft:
        #creating deltas from lastvalues
        indexcommun=list(set(lastvalues.id) & set(test.id))
        lastvalues=pd.concat([test[test.id.isin(indexcommun)]['id'],
            pd.DataFrame(test[diff_cols][test.id.isin(indexcommun)].values-lastvalues[diff_cols][lastvalues.id.isin(indexcommun)].values,
            columns=diff_cols, index=test[test.id.isin(indexcommun)].index)],
            axis=1)
        #adding them to test data    
        test=test.merge(right=lastvalues, how='left', on='id', suffixes=('','_d')).fillna(0)
        #storing new lastvalues
        lastvalues=test[['id']+diff_cols].copy()
    
    testid=test.id
    test=featureexpander.transform(test[cols])
    #prediction using modelselector and models list
    selected_prediction = modelselector.predict_proba(test.loc[: ,treecols])
    for ind,elt in enumerate(tokeepmodels):
        pred['y']+=selected_prediction[:,ind]*elt.predict(test[tokeepcolumns[ind]])

    indexbase=pred.index
    pred.index=testid    
    oldpred=pred['y']
    pred.index=indexbase
    
    o, reward, done, info = env.step(pred)
    rewards.append(reward)
    if reward>0:
        countplus+=1
    
    if indice%100==0:
        print(indice, countplus, reward, np.mean(rewards))
        
    if done:
        print(info["public_score"])
        break