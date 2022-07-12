
#forked from Gilberto Titericz Junior


import pandas as pd
import numpy as np
from sklearn import ensemble, preprocessing
import xgboost as xgb

import random
import time

# reproductibility
random.seed(1)
np.random.seed(2)
start_time = time.time()

# load training and test datasets
train = pd.read_csv('../input/train_set.csv', parse_dates=[2,])
test = pd.read_csv('../input/test_set.csv', parse_dates=[3,])
tube_data = pd.read_csv('../input/tube.csv')
bill_of_materials_data = pd.read_csv('../input/bill_of_materials.csv')
specs_data = pd.read_csv('../input/specs.csv')

print("train columns")
#print(train.columns)
print("test columns")
#print(test.columns)
print("tube.csv df columns")
#print(tube_data.columns)
print("bill_of_materials.csv df columns")
#print(bill_of_materials_data.columns)
print("specs.csv df columns")
#print(specs_data.columns)

#print(specs_data[2:3])

train = pd.merge(train, tube_data, on ='tube_assembly_id')
train = pd.merge(train, bill_of_materials_data, on ='tube_assembly_id')
test = pd.merge(test, tube_data, on ='tube_assembly_id')
test = pd.merge(test, bill_of_materials_data, on ='tube_assembly_id')

print("new train columns")
#print(train.columns)
#print(train[1:10])
#print(train.columns.to_series().groupby(train.dtypes).groups)

# create some new features
train['year'] = train.quote_date.dt.year
train['month'] = train.quote_date.dt.month
#train['dayofyear'] = train.quote_date.dt.dayofyear
#train['dayofweek'] = train.quote_date.dt.dayofweek
#train['day'] = train.quote_date.dt.day

test['year'] = test.quote_date.dt.year
test['month'] = test.quote_date.dt.month
#test['dayofyear'] = test.quote_date.dt.dayofyear
#test['dayofweek'] = test.quote_date.dt.dayofweek
#test['day'] = test.quote_date.dt.day

# drop useless columns and create labels
idx = test.id.values.astype(int)
test = test.drop(['id', 'tube_assembly_id', 'quote_date'], axis = 1)
labels = train.cost.values
#'tube_assembly_id', 'supplier', 'bracket_pricing', 'material_id', 'end_a_1x', 'end_a_2x', 'end_x_1x', 'end_x_2x', 'end_a', 'end_x'
#for some reason material_id cannot be converted to categorical variable
train = train.drop(['quote_date', 'cost', 'tube_assembly_id'], axis = 1)

train['material_id'].replace(np.nan,' ', regex=True, inplace= True)
test['material_id'].replace(np.nan,' ', regex=True, inplace= True)
for i in range(1,9):
    column_label = 'component_id_'+str(i)
    #print(column_label)
    train[column_label].replace(np.nan,' ', regex=True, inplace= True)
    test[column_label].replace(np.nan,' ', regex=True, inplace= True)

train.fillna(0, inplace = True)
test.fillna(0, inplace = True)

print("train columns")
#print(train.columns)

# convert data to numpy array
train = np.array(train)
test = np.array(test)


# label encode the categorical variables
for i in range(train.shape[1]):
    if i in [0,3,5,11,12,13,14,15,16,20,22,24,26,28,30,32,34]:
        #print(i,list(train[1:5,i]) + list(test[1:5,i]))
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[:,i]) + list(test[:,i]))
        train[:,i] = lbl.transform(train[:,i])
        test[:,i] = lbl.transform(test[:,i])


# object array to float
train = train.astype(float)
test = test.astype(float)

# i like to train on log(1+x) for RMSLE ;) 
# The choice is yours :)
label_log = np.log1p(labels)

# fit a random forest model

params = {}
params["objective"] = "reg:linear"
params["eta"] = 0.02
params["min_child_weight"] = 6
params["subsample"] = 0.7
params["colsample_bytree"] = 0.6
params["scale_pos_weight"] = 0.8
params["silent"] = 1
params["max_depth"] = 8
params["max_delta_step"]=2

plst = list(params.items())

print("End preparing data. Cross-validation set-up...")

## Cross validation
CV=False
trainrowsNB=train.shape[0]
print(trainrowsNB)
if CV==False:
    prob=1
else:
    prob=0.8

crossrowsNB=int((1-prob)*trainrowsNB)
rowcrossval=np.random.choice(trainrowsNB,crossrowsNB , replace=False)
crossval=train[rowcrossval,:]
crossvallog=label_log[rowcrossval]

train=np.delete(train, rowcrossval, axis=0)
trainrowsNB=train.shape[0]
label_log=np.delete(label_log, rowcrossval, axis=0)

print(crossval.shape)
print(train.shape)



xgtrain = xgb.DMatrix(train, label=label_log)
xgtest = xgb.DMatrix(test)
xgcross = xgb.DMatrix(crossval)

print("Preprocessing done")
print(str(np.round(time.time() - start_time, 1))+' seconds')


#############################      Training  & Prediction   ###############################################


############################################################################
num_rounds = 2  
print(str(num_rounds)+' ...')
start_time = time.time()

model1 = xgb.train(plst, xgtrain, num_rounds)

# Prediction
preds1 = model1.predict(xgtest)
    
## Cross-validation & test performance
if CV==True :
    preds1CV =  model1.predict(xgcross)
    preds1TR =  model1.predict(xgtrain)
    
    CVScore=np.sqrt(sum(np.power(preds1CV-crossvallog,2))/crossrowsNB)
    TrainScore=np.sqrt(sum(np.power(preds1TR-label_log,2))/trainrowsNB)
    print('CV Score : '+str(np.round(CVScore,4))+ ' Train score : '+str(np.round(TrainScore,4)) + ' in '+str(np.round(time.time() - start_time, 1))+' seconds')

############################################################################
num_rounds = 3  
print(str(num_rounds)+' ...')
start_time = time.time()

model2 = xgb.train(plst, xgtrain, num_rounds)

# Prediction
preds2 = model2.predict(xgtest)

## Cross-validation & test performance
if CV==True :
    preds2CV =  model2.predict(xgcross)
    preds2TR =  model2.predict(xgtrain)
    
    CVScore=np.sqrt(sum(np.power(preds2CV-crossvallog,2))/crossrowsNB)
    TrainScore=np.sqrt(sum(np.power(preds2TR-label_log,2))/trainrowsNB)
    print('CV Score : '+str(np.round(CVScore,4))+ ' Train score : '+str(np.round(TrainScore,4)) + ' in '+str(np.round(time.time() - start_time, 1))+' seconds')

############################################################################
num_rounds = 8000 
print(str(num_rounds)+' ...')
start_time = time.time()

model3 = xgb.train(plst, xgtrain, num_rounds)

# Prediction
preds3 = model3.predict(xgtest)

## Cross-validation & test performance
if CV==True :
    preds3CV =  model3.predict(xgcross)
    preds3TR =  model3.predict(xgtrain)
    
    CVScore=np.sqrt(sum(np.power(preds3CV-crossvallog,2))/crossrowsNB)
    TrainScore=np.sqrt(sum(np.power(preds3TR-label_log,2))/trainrowsNB)
    print('CV Score : '+str(np.round(CVScore,4))+ ' Train score : '+str(np.round(TrainScore,4)) + ' in '+str(np.round(time.time() - start_time, 1))+' seconds')


## Global CV
if CV==True :
    predsAvgCV = (preds1CV+preds2CV+preds3CV)/3
    predsAvgTR = (preds1TR+preds2TR+preds3TR)/3
    CVScore=np.sqrt(sum(np.power(predsAvgCV-crossvallog,2))/crossrowsNB)
    TrainScore=np.sqrt(sum(np.power(predsAvgTR-label_log,2))/trainrowsNB)
    print('CV Score : '+str(np.round(CVScore,4))+ ' Train score : '+str(np.round(TrainScore,4)) + ' in '+str(np.round(time.time() - start_time, 1))+' seconds')




#print('power 1/16 4000')  ############################################################################
#start_time = time.time()
#num_rounds = 4000

#label_16=np.power(np.delete(labels, rowcrossval, axis=0),1/16)
#label_log = np.power(labels,1/16)

#xgtrain = xgb.DMatrix(train, label=label_16)

#model = xgb.train(plst, xgtrain, num_rounds)
#preds3 = model.predict(xgtest)

#predsCV =  model.predict(xgcross)
#predsTR =  model.predict(xgtrain)

## Cross-validation & test performance
#CVScore=np.sqrt(sum(np.power(np.power(predsCV,16)-np.expm1(crossvallog),2))/crossrowsNB)
#TrainScore=np.sqrt(sum(np.power(np.power(predsTR,16)-np.power(label_16,16),2))/trainrowsNB)
#print('CV Score : '+str(np.round(CVScore,4))+ ' Train score : '+str(np.round(TrainScore,4)) + ' in '+str(np.round(time.time() - start_time, 1))+' seconds')


#preds = (0.58*np.expm1( (preds1+preds2+preds4)/3))+(0.42*np.power(preds3,16))

preds = np.expm1( (preds1+preds2+preds3)/3)
preds = pd.DataFrame({"id": idx, "cost": preds})
preds.to_csv('PY-XGB-2k-3k-4k.csv', index=False)


preds = np.expm1( preds3)
preds = pd.DataFrame({"id": idx, "cost": preds})
preds.to_csv('PY-XGB-8k.csv', index=False)