
#forked from Gilberto Titericz Junior


import pandas as pd
import numpy as np
from sklearn import ensemble, preprocessing
import xgboost as xgb

import random
import time

# reproductibility
random.seed(1)
np.random.seed(1)
start_time = time.time()

# load training and test datasets
train = pd.read_csv('../input/train_set.csv', parse_dates=[2,])
test = pd.read_csv('../input/test_set.csv', parse_dates=[3,])
tube_data = pd.read_csv('../input/tube.csv')
bill_of_materials_data = pd.read_csv('../input/bill_of_materials.csv')
specs_data = pd.read_csv('../input/specs.csv')


train = pd.merge(train, tube_data, on ='tube_assembly_id')
train = pd.merge(train, bill_of_materials_data, on ='tube_assembly_id')
train = pd.merge(train, specs_data, on ='tube_assembly_id')

test = pd.merge(test, tube_data, on ='tube_assembly_id')
test = pd.merge(test, bill_of_materials_data, on ='tube_assembly_id')
test = pd.merge(test, specs_data, on ='tube_assembly_id')


#compFiles = dir(base)[grep("comp_", dir(base))]
filelist=['comp_adaptor.csv','comp_boss.csv','comp_elbow.csv','comp_float.csv','comp_hfl.csv','comp_nut.csv','comp_other.csv','comp_sleeve.csv','comp_straight.csv','comp_tee.csv','comp_threaded.csv']

idComp = 1
keyMerge = 0


for idComp in range(1,9):
    for f in filelist:
        #d = pd.read_csv('../input/comp_'+str(f)+'.csv')
        d = pd.read_csv('../input/'+str(f))
        d.columns = d.columns+'_M_'+str(keyMerge)
        train = pd.merge(train, d, how='left', left_on = 'component_id_'+str(idComp), right_on = 'component_id_M_'+str(keyMerge))
        test = pd.merge(test, d, how='left', left_on = 'component_id_'+str(idComp), right_on = 'component_id_M_'+str(keyMerge))
        keyMerge = keyMerge + 1
    print(train.shape)
    print(test.shape)
    print ('idComp '+str(idComp)+' done, ' + str(train.shape[1]) +' columns')

### Feature engineering ###################################################
# create some new features
train['year'] = train.quote_date.dt.year
train['month'] = train.quote_date.dt.month
#train['dayofyear'] = train.quote_date.dt.dayofyear
#train['dayofweek'] = train.quote_date.dt.dayofweek
#train['day'] = train.quote_date.dt.day

## Mat volume
train['vol_full']=np.power(train['diameter'],2)*3.1415957/4*train['length']
train['vol_wall']=(np.power(train['diameter'],2)-np.power(train['diameter']-train['wall'],2))*3.1415957/4*train['length']
train['cummonth']= train.quote_date.dt.year * 12 + train.quote_date.dt.month

## Tube assemblies
#dFull$tube_assembly_id<-as.numeric(substr(dFull$tube_assembly_id,4,8))

test['year'] = test.quote_date.dt.year
test['month'] = test.quote_date.dt.month
#test['dayofyear'] = test.quote_date.dt.dayofyear
#test['dayofweek'] = test.quote_date.dt.dayofweek
#test['day'] = test.quote_date.dt.day

test['vol_full']=np.power(test['diameter'],2)*3.1415957/4*test['length']
test['vol_wall']=(np.power(test['diameter'],2)-np.power(test['diameter']-test['wall'],2))*3.1415957/4*test['length']
test['cummonth']= test.quote_date.dt.year * 12 + test.quote_date.dt.month



# drop useless columns and create labels
idx = test.id.values.astype(int)
test = test.drop(['id', 'tube_assembly_id', 'quote_date'], axis = 1)
labels = train.cost.values
#'tube_assembly_id', 'supplier', 'bracket_pricing', 'material_id', 'end_a_1x', 'end_a_2x', 'end_x_1x', 'end_x_2x', 'end_a', 'end_x'
#for some reason material_id cannot be converted to categorical variable
train = train.drop(['quote_date', 'cost', 'tube_assembly_id'], axis = 1)



### Clean NA values
for i in range(train.shape[1]):
    # first try convert to numeric
    NumConverted=train.iloc[:,i].convert_objects(convert_numeric=True)
    if (sum(train.iloc[:,i]==NumConverted)>20000) and (NumConverted.dtype!='object'):
        train.iloc[:,i]=NumConverted
        test.iloc[:,i]=test.iloc[:,i].convert_objects(convert_numeric=True)
        #print (str(i)+' converted')
        
    if (train.iloc[:,i].dtypes=='int64') or (train.iloc[:,i].dtypes=='float64'): # numeric 
        #print (str(i)+' '+train.columns[i]+'    numeric')
        train.iloc[:,i].replace(-1,' ', regex=True, inplace= True)
        test.iloc[:,i].replace(-1,' ', regex=True, inplace= True)
        train.iloc[:,i].replace(-1,'', regex=True, inplace= True)
        test.iloc[:,i].replace(-1,'', regex=True, inplace= True)
    else:
        #print (str(i)+' '+train.columns[i]+'    data')
        train.iloc[:,i].replace(np.nan,' ', regex=True, inplace= True)
        test.iloc[:,i].replace(np.nan,' ', regex=True, inplace= True)
        train.iloc[:,i].replace(np.nan,'', regex=True, inplace= True)
        test.iloc[:,i].replace(np.nan,'', regex=True, inplace= True)




train.fillna(0, inplace = True)
test.fillna(0, inplace = True)


# label encode the categorical variables
for i in range(train.shape[1]):
    #if i in [0,3,5,11,12,13,14,15,16,20,22,24,26,28,30,32,34,36,37,38,39,40,41,42,43,44,45]: ## initially up to 34 only
    if not((train.iloc[:,i].dtype=='int64') or (train.iloc[:,i].dtype=='float64')):
        #print (str(i)+ ' label encoded')
        #print(i,list(train[1:5,i]) + list(test[1:5,i]))
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train.iloc[:,i]) + list(test.iloc[:,i]))
        train.iloc[:,i] = lbl.transform(train.iloc[:,i])
        test.iloc[:,i] = lbl.transform(test.iloc[:,i])

# convert data to numpy array
train = np.array(train)
test = np.array(test)

# object array to float
train = train.astype(float)
test = test.astype(float)

# i like to train on log(1+x) for RMSLE ;) 
# The choice is yours :)
label_log = np.log1p(labels)

params = {}
params["objective"] = "reg:linear"
params["eta"] = 0.02
params["min_child_weight"] = 6
params["subsample"] = 0.8
params["colsample_bytree"] = 0.8
params["scale_pos_weight"] = 1
params["silent"] = 1
params["max_depth"] = 8

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
if CV==True :
    xgcross = xgb.DMatrix(crossval)

print("Preprocessing done in " + str(np.round(time.time() - start_time, 1))+' seconds')



#############################      Training  & Prediction   ###############################################


############################################################################
num_rounds = 4000
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

#preds = np.expm1( (preds1+preds2)/2)
#preds = pd.DataFrame({"id": idx, "cost": preds})
#preds.to_csv('Ens_xgb1408Full.csv', index=False)


preds = np.expm1( preds1)
preds = pd.DataFrame({"id": idx, "cost": preds})
preds.to_csv('xgb_1408Full.csv', index=False)