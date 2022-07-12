
import numpy as np 
import pandas as pd 
import xgboost as xgb

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import LabelKFold
from sklearn.metrics import roc_auc_score

from scipy.sparse import hstack

def reduce_dimen(dataset,column,toreplace):
    for index,i in dataset[column].duplicated(keep=False).iteritems():
        if i==False:
            dataset.set_value(index,column,toreplace)
    return dataset

def act_data_treatment(dsname):
    dataset = dsname
    
    for col in list(dataset.columns):
        if col not in ['people_id', 'activity_id', 'date', 'char_38', 'outcome']:
            if dataset[col].dtype == 'object':
                dataset[col].fillna('type 0', inplace=True)
                dataset[col] = dataset[col].apply(lambda x: x.split(' ')[1]).astype(np.int32)
            elif dataset[col].dtype == 'bool':
                dataset[col] = dataset[col].astype(np.int8)
    
    dataset['year'] = dataset['date'].dt.year
    dataset['month'] = dataset['date'].dt.month
    dataset['day'] = dataset['date'].dt.day
    dataset['isweekend'] = (dataset['date'].dt.weekday >= 5).astype(int)
    #dataset = dataset.drop('date', axis = 1)
    
    return dataset
    
if __name__ == '__main__':
    
    act_train_data = pd.read_csv("../input/act_train.csv",dtype={'people_id': np.str, 
    'activity_id': np.str, 'outcome': np.int8}, parse_dates=['date'])
    act_test_data  = pd.read_csv("../input/act_test.csv", dtype={'people_id': np.str, 
    'activity_id': np.str}, parse_dates=['date'])
    people_data    = pd.read_csv("../input/people.csv", dtype={'people_id': np.str, 
    'activity_id': np.str, 'char_38': np.int32}, parse_dates=['date'])

    act_train_data=act_train_data.drop('char_10',axis=1)
    act_test_data=act_test_data.drop('char_10',axis=1)

    print("Train data shape: " + format(act_train_data.shape))
    print("Test data shape: " + format(act_test_data.shape))
    print("People data shape: " + format(people_data.shape))

    act_train_data  = act_data_treatment(act_train_data)
    act_test_data   = act_data_treatment(act_test_data)
    people_data = act_data_treatment(people_data)

    train = act_train_data.merge(people_data, on='people_id', how='left', left_index=True)
    test  = act_test_data.merge(people_data, on='people_id', how='left', left_index=True)

    ppl_ids = train.people_id

    del act_train_data
    del act_test_data
    del people_data

    #train=train.sort_values(['people_id'], ascending=[1])
    #test=test.sort_values(['people_id'], ascending=[1])

    train_columns = train.columns.values
    test_columns = test.columns.values
    features = list(set(train_columns) & set(test_columns))

    train.fillna('NA', inplace=True)
    test.fillna('NA', inplace=True)

    y = train.outcome.values
    train=train.drop('outcome',axis=1)

    whole=pd.concat([train,test],ignore_index=True)
    
    whole['bus_days'] = np.busday_count(whole['date_x'].values.astype('<M8[D]'), 
    whole['date_y'].values.astype('<M8[D]') )

    categorical=['activity_category','char_1_x','char_2_x','char_3_x','char_4_x',
    'char_5_x','char_6_x','char_7_x','char_8_x','char_9_x','char_2_y','char_3_y','char_4_y',
    'char_5_y','char_6_y','char_7_y','char_8_y','char_9_y']

    for category in categorical:
        whole=reduce_dimen(whole,category,9999999)
    
    X=whole[:len(train)]
    X_test=whole[len(train):]

    del train
    del whole
    
    #X=X.sort_values(['people_id'], ascending=[1])

    X = X[features].drop(['people_id', 'activity_id','date_x','date_y'], axis = 1)
    X_test = X_test[features].drop(['people_id', 'activity_id','date_x','date_y'], axis = 1)

    #Label Encoder
    not_categorical=[]
    for category in X.columns:
        if category not in categorical:
            not_categorical.append(category)
        else:
            temp = pd.concat([X[category], X_test[category]])
            le = LabelEncoder()
            le.fit(temp.values)
            X[category] = le.transform(X[category].values)
            X_test[category] = le.transform(X_test[category].values)
    
    #OneHot Encoding
    enc = OneHotEncoder(handle_unknown='ignore')
    enc=enc.fit(pd.concat([X[categorical],X_test[categorical]]))

    X_cat_sparse=enc.transform(X[categorical])
    X_test_cat_sparse=enc.transform(X_test[categorical])

    X_sparse=hstack((X[not_categorical], X_cat_sparse)).tocsr()
    X_test_sparse=hstack((X_test[not_categorical], X_test_cat_sparse)).tocsr()

    print("Training data: " + format(X_sparse.shape))
    print("Test data: " + format(X_test_sparse.shape))
    print("###########")
    print("One Hot enconded Test Dataset Script")
    
    '''
    XGBoost params
    '''
    param = {'max_depth':11, 'eta':0.049, 'silent':1, 'objective':'binary:logistic'}
    param['nthread'] = 4
    param['seed'] = 42
    param['eval_metric'] = 'auc'
    param['subsample'] = 0.85
    param['colsample_bytree']= 0.92
    param['colsample_bylevel']= 0.9
    param['gamma']= 0.005
    param['min_child_weight'] = 0
    param['booster'] = "gbtree"
    param['lambda ']= 0.1
    param['alpha']= 0.00000001
    num_round = 200
    early_stopping_rounds=10
    verbose_eval = 20
    num_of_folds = 3

    '''
    Validation with 'num_of_folds' folds on people_ids
    '''
    lkf = LabelKFold(ppl_ids.values, n_folds=num_of_folds)
    test_preds = np.zeros((X_test.shape[0],))
    scores=[]

    for train_ind, val_ind in lkf:
        print ('/////////// New fold processing')
        X_train, X_val = X_sparse[train_ind,:], X_sparse[val_ind,:]
        y_train, y_val = y[train_ind], y[val_ind]
        print('///// Fold made')
        dtrain = xgb.DMatrix(X_train, y_train)
        dval = xgb.DMatrix(X_val, y_val)
        dtest = xgb.DMatrix(X_test_sparse)
        print('///// DMatrixes made')
    
        watchlist  = [(dtrain,'train'),(dval,'val')]
        xgb_model = xgb.train(param, dtrain, num_round, watchlist,
        early_stopping_rounds=early_stopping_rounds, verbose_eval = verbose_eval)
    
        y_val_preds = xgb_model.predict(dval, ntree_limit = xgb_model.best_iteration)
        score = roc_auc_score(y_val, y_val_preds)
        print('Validation AUC ROC:', score) 
        scores.append(score)
        
        y_test_preds = xgb_model.predict(dtest, ntree_limit = xgb_model.best_iteration)
    
        test_preds = test_preds + y_test_preds
    #averaging test preds from different folds
    test_preds = test_preds/num_of_folds
    print('///// Mean score: ', np.mean(scores))

    output = pd.DataFrame({ 'activity_id' : test['activity_id'], 'outcome': test_preds })
    
    output.to_csv('sub20.csv', index = False)