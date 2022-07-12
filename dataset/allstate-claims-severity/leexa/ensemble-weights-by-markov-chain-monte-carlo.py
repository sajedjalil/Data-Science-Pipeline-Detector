# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
np.random.seed(123)
from subprocess import check_output
import xgboost as xgb
import gc

# Any results you write to the current directory are saved as output.

#### ANYTHING ABOVE IS KAGGLE JUNK #####

#''' This code gets a xgboost ensemble of 5 models, and tries to find the optimum weights through MCMC magic''''



num_folds = 2 #should be larger, but kaggle scirpts has run time 

def MAE(y,dtrain):
    answer = dtrain.get_label()
    answer = np.array(answer)
    prediction = np.array(y)
    error = np.exp(prediction) -np.exp(answer)
    error = np.mean((error**2)**.5)
    return 'mcc error',error
    
def MAE2(y,dtrain):
    answer = dtrain.loss2
    answer = np.array(answer)
    prediction = np.array(y)
    error = prediction - answer
    error = np.mean((error**2)**.5)
    return 'mcc error',error


## smaller dataset for faster training ###
train=pd.read_csv('../input/train.csv',nrows=10000)
test=pd.read_csv('../input/test.csv',nrows=10000)
train['loss']=np.log(train['loss']+200)
train['loss2']=np.exp(train['loss'])-200

## encode cat variables as discrete integers 
for i in list(train.keys()):
	if 'cat' in i:
		dictt = {}
		var = sorted(list(train[i].unique()))
		for ii in range(0,len(var)):
			dictt[var[ii]]=ii
		train[i] = train[i].map(dictt)
		test[i] = test[i].map(dictt)
        
parameters =[]
for i in (6,12):
    for j in (60,):
            for l in (1,2):
                depth = i
                min_child_weight = j
                gamma=l
                parameters += [[depth,min_child_weight,gamma],]
predictors = ['cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat8', 'cat9', 'cat10', 'cat11', 'cat12', 'cat13', 'cat14', 'cat15', 'cat16', 'cat17', 'cat18', 'cat19', 'cat20', 'cat21', 'cat22', 'cat23', 'cat24', 'cat25', 'cat26', 'cat27', 'cat28', 'cat29', 'cat30', 'cat31', 'cat32', 'cat33', 'cat34', 'cat35', 'cat36', 'cat37', 'cat38', 'cat39', 'cat40', 'cat41', 'cat42', 'cat43', 'cat44', 'cat45', 'cat46', 'cat47', 'cat48', 'cat49', 'cat50', 'cat51', 'cat52', 'cat53', 'cat54', 'cat55', 'cat56', 'cat57', 'cat58', 'cat59', 'cat60', 'cat61', 'cat62', 'cat63', 'cat64', 'cat65', 'cat66', 'cat67', 'cat68', 'cat69', 'cat70', 'cat71', 'cat72', 'cat73', 'cat74', 'cat75', 'cat76', 'cat77', 'cat78', 'cat79', 'cat80', 'cat81', 'cat82', 'cat83', 'cat84', 'cat85', 'cat86', 'cat87', 'cat88', 'cat89', 'cat90', 'cat91', 'cat92', 'cat93', 'cat94', 'cat95', 'cat96', 'cat97', 'cat98', 'cat99', 'cat100', 'cat101', 'cat102', 'cat103', 'cat104', 'cat105', 'cat106', 'cat107', 'cat108', 'cat109', 'cat110', 'cat111', 'cat112', 'cat113', 'cat114', 'cat115', 'cat116', 'cont1', 'cont2', 'cont3', 'cont4', 'cont5', 'cont6', 'cont7', 'cont8', 'cont9', 'cont10', 'cont11', 'cont12', 'cont13', 'cont14']
target='loss'
result={}

## train 4 models with different paremeters ###
for i,j,l in parameters:
    xgtest=xgb.DMatrix(test[predictors].values,missing=np.NAN,feature_names=predictors)
    depth,min_child_weight,gamma=i,j,l
    result[(depth,min_child_weight,gamma)]=[]
    ### name of prediction ###
    name = 'feature_L2_%s_%s_%s_%s' %(str(depth), str(min_child_weight), str(gamma),str(num_folds))
    train  [name]=0
    test[name]=0
    for fold in range(0,num_folds):
        print ('\ntraining  parameters', i,j,l,',fold',fold)
        gc.collect() #to clear ram of garbage
        train_i = [x for x in train.index if x%num_folds != fold]
        cv_i = [x for x in train.index if x%num_folds == fold]
        dtrain= train.iloc[train_i]
        dcv = train.iloc[cv_i]
        xgcv    = xgb.DMatrix(dcv[predictors].values, label=dcv[target].values,missing=np.NAN,feature_names=predictors)
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values,missing=np.NAN,feature_names=predictors)

        #watchlist  = [ (xgtrain,'train'),(xgcv,'eval')] #i got valueerror in this
        params = {}
        params["objective"] =  "reg:linear"
        params["eta"] = 0.1
        params["min_child_weight"] = min_child_weight
        params["subsample"] = 0.5
        params["colsample_bytree"] = 0.5
        params["scale_pos_weight"] = 1.0
        params["silent"] = 1
        params["max_depth"] = depth
        params['seed']=1
        params['lambda']=1
        params[ 'gamma']= gamma
        plst = list(params.items())
        early_stopping_rounds=5
        result_d=xgb.train(plst,xgtrain,50,maximize=0,feval = MAE)
        #print (result_d.predict(xgcv))
        print ('train_result',MAE(result_d.predict(xgcv),xgcv))
        ### write predictions onto train and test set ###
        train.set_value(cv_i,name,np.exp(result_d.predict(xgcv))-200)
        test.set_value(test.index,name,test[name]+(np.exp(result_d.predict(xgtest)-200)/num_folds))
        gc.collect()


#### NOW THE MCMC PART to find individal weights for ensemble####

features = [x for x in  train.keys() if 'feature' in x]
print ('features are these:', features)
num=len(features)
#intialize weights
weight = np.array([1.0/num,]*num)

# This is to define variables to be used later
train['pred_new']=0
train['pred_old']=0
counter = 0
n=1000 ###MCMC steps
result={}

for i in range(0,len(features)):
    train['pred_new'] += train[features[i]]*weight[i]
    print ('feature:',features[i],',MAE=',MAE2(train[features[i]],train))
print ('combined all features',',MAE=', MAE2(train.pred_new,train))
train['pred_old']=train['pred_new']
#### MCMC  #### 
### MCMC algo for dummies 
### 1. Get initialize ensemble weights
### 2. Generate new weights 
### 3. if MAE is lower, accept new weights immediately , or else accept new weights with probability of np.exp(-diff/.3)
### 4. repeat 2-3
for i in range(0,n):
     new_weights = weight+ np.array([0.005,]*num)*np.random.normal(loc=0.0, scale=1.0, size=num)
     new_weights[new_weights < 0.01]=0.01
     train['pred_new']=0
     for ii in range(0,len(features)):
         train['pred_new'] += train[features[ii]]*new_weights[ii]
     diff = MAE2(train.pred_new,train)[1] - MAE2(train.pred_old,train)[1]
     prob = min(1,np.exp(-diff/.3))
     random_prob = np.random.rand()
     if random_prob < prob:
         weight= new_weights
         train['pred_old']=train['pred_new']
         result[i] = (MAE2(train.pred_new,train)[1] ,MAE2(train.pred_old,train)[1],prob,random_prob ,weight)
         #print (MAE2(train.pred_new,train)[1] ,MAE2(train.pred_old,train)[1],prob,random_prob),
         counter +=1
print (counter *1.0 / n, 'Acceptance Ratio') #keep this [0.4,0.6] for best results
print ('best result MAE', sorted([result[i] for i in result])[0:1][0])

weight=sorted([result[i] for i in result])[0:1][-1]
train['pred_new']=0
for i in range(0,len(features)):
    train['pred_new'] += train[features[i]]*weight[i]
print ('combined all features plus MCMC weights:',',MAE=', MAE2(train.pred_new,train))

print ('weights:', weight[-1])
### notice the weights do not necessarily sum to 1 ###
