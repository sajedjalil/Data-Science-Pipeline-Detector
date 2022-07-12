# -*- coding: utf-8 -*-

## loading packages 
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cross_validation import KFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
import gc



### setup
ID = 'id'
TARGET = 'loss'
NFOLDS = 4
#NFOLDS = 10
SEED = 1
NROWS = None
DATA_DIR = "../input"
shift=200


### reading data 
TRAIN_FILE = "{0}/train.csv".format(DATA_DIR)
TEST_FILE = "{0}/test.csv".format(DATA_DIR)


train = pd.read_csv(TRAIN_FILE, nrows=NROWS)
test = pd.read_csv(TEST_FILE, nrows=NROWS)
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = np.log(train[TARGET]+shift).ravel()
id_train= train[ID]
id_test= test[ID]
train_test = pd.concat((train, test)).reset_index(drop=True)


### remeber the order
train_test[ID]=pd.Categorical(train_test[ID], train_test[ID].values.tolist())

####  preprocessing
train_test["cont1"] = (preprocessing.minmax_scale(train_test["cont1"]))**(1/4)
train_test["cont2"] = (preprocessing.minmax_scale(train_test["cont2"]))**(1/4)    
train_test["cont3"] = (preprocessing.minmax_scale(train_test["cont3"]))**(4)
train_test["cont4"] = np.sqrt(preprocessing.minmax_scale(train_test["cont4"]))
train_test["cont5"] = (preprocessing.minmax_scale(train_test["cont5"]))**2
train_test["cont6"] = np.exp(preprocessing.minmax_scale(train_test["cont6"]))
train_test["cont7"] = (preprocessing.minmax_scale(train_test["cont7"]))**4
train_test["cont8"] = (preprocessing.minmax_scale(train_test["cont8"]))**(1/4)
train_test["cont9"] = (preprocessing.minmax_scale(train_test["cont9"]))**4
train_test["cont10"] = np.log1p(preprocessing.minmax_scale(train_test["cont10"]))
train_test["cont11"] = (preprocessing.minmax_scale(train_test["cont11"]))**4
train_test["cont12"] = (preprocessing.minmax_scale(train_test["cont12"]))**4
train_test["cont13"] = np.log(preprocessing.minmax_scale(train_test["cont13"]))
train_test["cont14"] = (preprocessing.minmax_scale(train_test["cont14"]))**4

### custom sorting 
def custom_sorting(mylist):
    mylist_len=[]
    for i in mylist:
        mylist_len.append(len(str(i)))

    all_list=[]
    for i in np.unique(sorted(mylist_len)):
        i_list=[]
        for j in mylist:
            if len(j)==i:
                i_list.append(j)
        all_list=all_list + i_list
    return(all_list)



### factorize
cats = [feat for feat in train.columns if 'cat' in feat]
for cat in cats:
    mylist=(np.unique(train_test[cat])).tolist()
    sorting_list=custom_sorting(mylist)
    train_test[cat]=pd.Categorical(train_test[cat], sorting_list)
    train_test=train_test.sort_values(cat)
    train_test[cat] = pd.factorize(train_test[cat], sort=True)[0]

### reorder 
train_test=train_test.sort_values(ID)
gc.collect()




### define x_train, x_test
train_test.drop([ID, TARGET], axis=1, inplace=True)
x_train = np.array(train_test.iloc[:ntrain,:])
x_test = np.array(train_test.iloc[ntrain:,:])
print("{},{}".format(x_train.shape, x_test.shape))


### setup training functions

def xg_eval_mae(yhat, dtrain):
    y = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(y)-shift,
                                      np.exp(yhat)-shift)

class XgbWrapper(object):
    def __init__(self, seed=0, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 250)

    def train(self, x_train, y_train):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds,verbose_eval=300,feval=xg_eval_mae, maximize=False)

    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x))



### xgb2     
xgb2_params = {
    'seed': 1,
    'colsample_bytree': 0.3085,
    'subsample': 0.9930,
    'eta': 0.1,
    'gamma': 0.5290,
    'booster' :  'gbtree',    
    'max_depth': 7,
    'min_child_weight': 4.2922,
    'eval_metric': 'mae'
}


    
xgb2_params['nrounds']=398
print(xgb2_params)

xgb2 = XgbWrapper(seed=SEED, params=xgb2_params)
xgb2.train(x_train,y_train)
xgb2_test= xgb2.predict(x_test)
xgb2_test = pd.DataFrame(np.exp(xgb2_test)-shift, columns=[TARGET])
xgb2_test[ID] = id_test
xgb2_test.to_csv('xgb2_test.csv', index=0)