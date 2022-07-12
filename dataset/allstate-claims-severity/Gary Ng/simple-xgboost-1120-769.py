 # http://stackoverflow.com/questions/15723628/pandas-make-a-column-dtype-object-or-factor
 # https://www.kaggle.com/tilii7/allstate-claims-severity/bias-correction-xgboost/code
 # https://www.kaggle.com/mmueller/allstate-claims-severity/stacking-starter/run/390867
 # https://www.kaggle.com/aliajouz/allstate-claims-severity/singel-model-lb-1117/code
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.stats import skew
from scipy.sparse import csr_matrix,hstack
from scipy.stats import boxcox
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import minmax_scale
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor

import datetime
import xgboost as xgb
import time
import itertools

#def fmean_squared_error(ground_truth,prediction):
    
#    fmean_squared_error_ = mean_sauared_error(ground_truth,prediction)**0.5
#    return fmean_squared_error_


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
shift = 200
test_ids = test.id
target = np.log1p(train.loss.values + shift)
num_trains = train.shape[0]
num_tests = test.shape[0]
COMB_FEATURE = 'cat80,cat87,cat57,cat12,cat79,cat10,cat7,cat89,cat2,cat72'.split(
    ',')

df_train = train.drop(['id','loss'],axis=1)
df_test = test.drop(['id'],axis=1)
all_data = pd.concat((df_train,df_test),axis=0).reset_index(drop=True)
features = all_data.columns
cat_list = [feat for feat in features if 'cat' in feat]
cont_list =[feat for feat in features if 'cont' in feat]

fair_constant = 2
def fair_obj(y_pred,dtrain):
    y = dtrain.get_label()
    x = (y_pred - y)
    den = abs(x) + fair_constant
    grad = x * fair_constant / den
    hess = fair_constant * fair_constant / (den * den)
    return grad,hess

def xgb_eval_mae(y_pred,dtrain):
    y = dtrain.get_label()
    return 'mae' , mean_absolute_error(np.exp(y)-shift,
                                    np.exp(y_pred)-shift) 



numeric_feat = all_data.dtypes[all_data.dtypes != 'object'].index
skewed = all_data[numeric_feat].apply(lambda x:skew(x.dropna()))
skewed_less = skewed[skewed <= 0.25].index
skewed = skewed[skewed > 0.25].index


boxcox_column =  ['cont4','cont5','cont6','cont7',
                  'cont8','cont9','cont10','cont11',
                  'cont12']
                  

for column in skewed:
    all_data[column] = all_data[column] +1
    all_data[column],lam = boxcox(all_data[column])
    #if column == 'cont1':
    #    all_data[column],lam = boxcox(all_data[column])
    #elif column == 'cont13' or column == 'cont14':
    #    all_data[column] = np.abs(all_data[column] - np.mean(all_data[column]))
    #elif column in boxcox_column:
        ## boxcox data must be positive
    #    all_data[column] = all_data[column] +1
    #    all_data[column],lam = boxcox(all_data[column])
'''        
all_data['cont2'] = np.tan(all_data['cont2'])
all_data["cont1"] = np.sqrt(minmax_scale(all_data["cont1"]))
all_data["cont4"] = np.sqrt(minmax_scale(all_data["cont4"]))
all_data["cont5"] = np.sqrt(minmax_scale(all_data["cont5"]))
all_data["cont8"] = np.sqrt(minmax_scale(all_data["cont8"]))
all_data["cont10"] = np.sqrt(minmax_scale(all_data["cont10"]))
all_data["cont11"] = np.sqrt(minmax_scale(all_data["cont11"]))
all_data["cont12"] = np.sqrt(minmax_scale(all_data["cont12"]))
all_data["cont6"] = np.log(minmax_scale(all_data["cont6"]) + 0000.1)
all_data["cont7"] = np.log(minmax_scale(all_data["cont7"]) + 0000.1)
all_data["cont9"] = np.log(minmax_scale(all_data["cont9"]) + 0000.1)
all_data["cont13"] = np.log(minmax_scale(all_data["cont13"]) + 0000.1)
all_data["cont14"] = (np.maximum(all_data["cont14"] - 0.179722, 0) / 0.665122) ** 0.25
'''

for col in list(train.select_dtypes(include=['object']).columns):
    if train[col].nunique() != test[col].nunique():
        set_train = set(train[col].unique())
        set_test = set(test[col].unique())
        remove_train = set_train - set_test
        remove_test = set_test - set_train
        remove = remove_train.union(set_test)
        
        def filter_cat(x):
            if x in remove:
                return np.nan
            else:
                return x
        all_data[col].apply(lambda x:filter_cat(x),1)

for comb in itertools.combinations(COMB_FEATURE,2):
    feat = comb[0] + '_' + comb[1]
    all_data[comb] = train[comb[0]] + test[comb[1]]
    print(feat)

cat_list = all_data.dtypes[all_data.dtypes == 'object'].index

for cat in cat_list:
    all_data[cat] = pd.factorize(all_data[cat],sort=True)[0]

scaler = StandardScaler()
all_data[numeric_feat] = scaler.fit_transform(all_data[numeric_feat].values)
x_train = all_data.iloc[:num_trains,:].copy()
x_test = all_data.iloc[num_trains:,:].copy()


ind_params = {'eta':0.1,
              'gamma':0.5290,
             # 'n_estimators':300,
              'max_depth':7,
              'min_child_weight':4.2922,
              'seed':42,
              'subsample':0.9930,
              'colsample_bytree':0.3085,
              'objective':'reg:linear',
              'silent':1}



kf = KFold(x_train.shape[0],n_folds = 5)


x_test = xgb.DMatrix(x_test)
for i,(train_idx,valid_idx) in enumerate(kf):
    print('Fold %d \n' %(i+1))
    
    X_train,X_val = x_train.iloc[train_idx],x_train.iloc[valid_idx]
    y_train,y_val = target[train_idx],target[valid_idx]
    
    dtrain = xgb.DMatrix(X_train,label=y_train)
    dvalid = xgb.DMatrix(X_val,label=y_val)
    watchlist = [(dtrain,'train'),(dvalid,'eval')]
    
    clf = xgb.train(ind_params,dtrain,100,
                    watchlist,early_stopping_rounds = 25,
                    obj=fair_obj,feval=xgb_eval_mae)
    
    score = clf.predict(dvalid,ntree_limit=clf.best_ntree_limit )
    cv_score = mean_absolute_error(np.exp(y_val),np.exp(score))
    print('MAE %0.4f' %(cv_score))
    y_pred = np.exp(clf.predict(x_test,
                    ntree_limit = clf.best_ntree_limit)) - shift
    
    if i>0:
        fpred = pred+y_pred
    else:
        fpred = y_pred
    
    pred = fpred
    
print('KFold Finished ....')
y_pred = fpred / 5


now = datetime.datetime.now()
sub_file = 'Claims_Severity_submission_'  + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'

pd.DataFrame({'id':test_ids,'loss':y_pred}).to_csv(sub_file,index=False)


    