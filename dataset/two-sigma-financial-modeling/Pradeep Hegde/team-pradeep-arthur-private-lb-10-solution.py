import kagglegym
import numpy as np
import pandas as pd
import random
from sklearn import ensemble, linear_model, metrics
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.linear_model import HuberRegressor
from itertools import combinations
import gc
from threading import Thread
import multiprocessing
from multiprocessing import Manager
from sklearn import preprocessing as pp
from numpy.fft import fft
from sklearn.naive_bayes import GaussianNB


env = kagglegym.make()
o = env.reset()
train = o.train
print(train.shape)
d_mean= train.median(axis=0)


low_y_cut = -0.075
high_y_cut = 0.075
y_is_above_cut = (train.y > high_y_cut)
y_is_below_cut = (train.y < low_y_cut)
y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)  

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

class huber_linear_model():
    def __init__(self):

        self.bestmodel=None
        self.scaler = pp.MinMaxScaler()
       
    def fit(self, train, y):

        indextrain=train.dropna().index
        tr = self.scaler.fit_transform(train.ix[indextrain])
        self.bestmodel = HuberRegressor().fit(tr, y.ix[indextrain])
        

    def predict(self, test):
        te = self.scaler.transform(test)
        return self.bestmodel.predict(te)

class LGB_model():
    def __init__(self, num_leaves=25, feature_fraction=0.6, bagging_fraction=0.6):
        self.lgb_params = {
                'task': 'train',
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': {'l2'},
                'learning_rate': 0.05,
                'bagging_freq': 5,
                'num_thread':4,
                'verbose': 0
            }
        
        self.lgb_params['feature_fraction'] = feature_fraction
        self.lgb_params['bagging_fraction'] = bagging_fraction
        self.lgb_params['num_leaves'] = num_leaves
        

        self.bestmodel=None
       
    def fit(self, train, y):
        
        X_train, X_val, y_train, y_val = train_test_split(train, y, test_size=0.2, random_state=343)
        
        lgtrain = lgb.Dataset(X_train, y_train)
        lgval = lgb.Dataset(X_val, y_val, reference=lgtrain)
                
        self.bestmodel = lgb.train(self.lgb_params,
                                    lgtrain,
                                    num_boost_round=100,
                                    valid_sets=lgval,
                                    verbose_eval=False,
                                    early_stopping_rounds=5)


    def predict(self, test):
        return self.bestmodel.predict(test, num_iteration=self.bestmodel.best_iteration)

def calcHuberParallel(df_train, train_cols, result):
    model=huber_linear_model()
    model.fit(df_train.loc[:,train_cols], df_train.loc[:, 'y'])
    residual = abs(model.predict(df_train[train_cols].fillna(d_mean))-df_train.y)
    
    result.append([model, train_cols, residual])
    
    return 0


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


cols2fit=['technical_22','technical_20', 'technical_30_d', 'technical_20_d', 'technical_30', 
          'technical_13', 'technical_34']

models=[]
columns=[]
residuals=[]

num_threads = 4
result = Manager().list()
threads = []

for (col1, col2) in combinations(cols2fit, 2):
    print("fitting Huber model on ", [col1, col2])
    threads.append(multiprocessing.Process(target=calcHuberParallel, args=(train, [col1, col2], result)))

    if (len(threads) == num_threads):
        for thread in threads:
            thread.start()
    
        for thread in threads:
            thread.join()
        
        print(len(result))
        
        threads = []
        
        
''' Last bit '''
print("running last threads ..")
if (len(threads)>0):
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
    print(len(result))
    
    threads = []


for data in result:
    model, train_cols, residual = data
    models.append(model)
    columns.append(train_cols)
    residuals.append(residual) 

del result, threads
gc.collect()



train=train.fillna(d_mean)

print("adding new features")
featureexpander=createLinearFeatures(n_neighbours=20, max_elts=2, verbose=True, random_state=rnd)
index2use=train[abs(train.y)<0.086].index
featureexpander.fit(train.ix[index2use,cols],train.ix[index2use,'y'])
trainer=featureexpander.transform(train[cols])

treecols=trainer.columns

print("training LGB model ")
num_leaves = [70]
feature_fractions = [0.2, 0.6]
bagging_fractions = [0.7]

#with Timer("running LGB models "):
for num_leaf in num_leaves:
    for feature_fraction in feature_fractions:
        for bagging_fraction in bagging_fractions:
            print("fitting LGB tree model with ", num_leaf, feature_fraction, bagging_fraction)
            model = LGB_model(num_leaves=num_leaf, feature_fraction=feature_fraction, bagging_fraction=bagging_fraction)
            model.fit(trainer,train.y)
            models.append(model)
            columns.append(treecols)
            residuals.append(abs(model.predict(trainer)-train.y))

print("training trees")
model = ensemble.ExtraTreesRegressor(n_estimators=100, max_depth=4, n_jobs=-1, random_state=rnd, verbose=0)
model.fit(trainer,train.y)
print(pd.DataFrame(model.feature_importances_,index=treecols).sort_values(by=[0]).tail(30))
for elt in model.estimators_:
    models.append(elt)
    columns.append(treecols)
    residuals.append(abs(elt.predict(trainer)-train.y))

num_to_keep=7
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


for modelp in tokeepmodels:
    print("")
    print(modelp)


#creating a new target for a model in charge of predicting which model is best for the current line
targetselector=np.array(tokeepresiduals).T
targetselector=np.argmin(targetselector, axis=1)

#with Timer("Training ET selection model "):
print("training selection model")
modelselector = ensemble.ExtraTreesClassifier(n_estimators= 120, max_depth=4, n_jobs=-1, random_state=rnd, verbose=0)
modelselector.fit(trainer, targetselector)

model2 = GaussianNB()
model2.fit(trainer,targetselector)

print(pd.DataFrame(modelselector.feature_importances_,index=treecols).sort_values(by=[0]).tail(30))


lastvalues=train[train.timestamp==max(train.timestamp)][['id']+diff_cols].copy()

print("end of training, now predicting")
indice=0
countplus=0
rewards=[]


del models
del columns
del residuals
del tokeepresiduals
gc.collect()

while True:
    indice+=1
    test = o.features
    test["nbnulls"]=test.isnull().sum(axis=1)
    if add_nas_ft:
        for elt in nas_cols:
            test[elt + '_na'] = pd.isnull(test[elt]).apply(lambda x: 1 if x else 0)
    test=test.fillna(d_mean)
    
    timestamp = o.features.timestamp[0]

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
    selected_prediction2 = model2.predict_proba(test.loc[: ,treecols])
    for ind,elt in enumerate(tokeepmodels):
        pred['y']+= (selected_prediction[:,ind]*elt.predict(test[tokeepcolumns[ind]])*1.00) +  (selected_prediction2[:,ind]*elt.predict(test[tokeepcolumns[ind]])*0.05)
    
    
    pred['y'] = pred['y'].clip(low_y_cut, high_y_cut)

    o, reward, done, info = env.step(pred)

    rewards.append(reward)
    if reward>0:
        countplus+=1
    
    if indice%100==0:
        print(indice, countplus, reward, np.mean(rewards), info)
        
    if done:
        print(info["public_score"])
        break