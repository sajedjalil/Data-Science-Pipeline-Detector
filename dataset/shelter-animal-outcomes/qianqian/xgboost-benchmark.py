import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.metrics import log_loss
import xgboost as xgb
seed = 1024
np.random.seed(seed)

train = pd.read_csv('../input/train.csv',parse_dates=['DateTime']).fillna('')
test = pd.read_csv('../input/test.csv',parse_dates=['DateTime']).fillna('')


y = train['OutcomeType'].values
le_y = LabelEncoder()

y = le_y.fit_transform(y)
print(le_y.classes_)

train = train.drop('OutcomeType',axis=1)
train = train.drop('OutcomeSubtype',axis=1)

train['ColorCount'] = train['Color'].apply(lambda x: len(x.split('/')))
test['ColorCount'] = test['Color'].apply(lambda x: len(x.split('/')))
train['MixOrNot'] = train['Breed'].apply(lambda x: 'Mix' in x)
test['MixOrNot'] = test['Breed'].apply(lambda x: 'Mix' in x)
train['BreedCount'] = train['Breed'].apply(lambda x: len(x.split('/')))
test['BreedCount'] = test['Breed'].apply(lambda x: len(x.split('/')))
train['SexPrefix'] = train['SexuponOutcome'].apply(lambda x: x.split(' ')[0])
test['SexPrefix'] = test['SexuponOutcome'].apply(lambda x: x.split(' ')[0])


train['DateTime_dayofweek'] = train['DateTime'].dt.dayofweek
train['DateTime_dayofyear'] = train['DateTime'].dt.dayofyear
train['DateTime_days_in_month'] = train['DateTime'].dt.days_in_month


test['DateTime_dayofweek'] = test['DateTime'].dt.dayofweek
test['DateTime_dayofyear'] = test['DateTime'].dt.dayofyear
test['DateTime_days_in_month'] = test['DateTime'].dt.days_in_month


# train = train.drop('DateTime',axis=1)
# test = test.drop('DateTime',axis=1)

data_all = pd.concat([train,test])

data_all = data_all.drop('AnimalID',axis=1)
# data_all = data_all.drop('Name',axis=1)
data_all = data_all.drop('ID',axis=1)



X = []
X_t = []
for c in data_all.columns:
    le = LabelEncoder()
    le.fit(data_all[c].values)
    X.append(le.transform(train[c].values))
    X_t.append(le.transform(test[c].values))

X = np.vstack(X).T
X_t = np.vstack(X_t).T


def make_mf_classifier(X ,y, clf, X_test,n_folds=2, n_round=5):
    n = X.shape[0]
    '''
    Fit metafeature by @clf and get prediction for test. Assumed that @clf -- classifier
    '''
    len_y = len(np.unique(y))
    mf_tr = np.zeros((X.shape[0],len_y))

    mf_te = np.zeros((X_test.shape[0],len_y))

    for i in range(n_round):
        skf = StratifiedKFold(y, n_folds=n_folds, shuffle=True, random_state=42+i*1000)
        for ind_tr, ind_te in skf:
            X_tr = X[ind_tr]
            X_te = X[ind_te]
            
            y_tr = y[ind_tr]
            y_te = y[ind_te]
            clf.fit(X_tr, y_tr)

            mf_tr[ind_te] += clf.predict_proba(X_te)
            mf_te += clf.predict_proba(X_test)*0.5
            y_pred = clf.predict_proba(X_te)
            score = log_loss(y_te, y_pred)
            print('pred[{}],score[{}]'.format(i,score))
    return (mf_tr / n_round, mf_te / n_round)


'''
metafeature
'''

# xgboost = xgb.XGBClassifier(
#         n_estimators=400, 
#         learning_rate = 0.05, 
#         max_depth=6, 
#         subsample=0.7, 
#         colsample_bytree = 0.7, 
#         # gamma = 0.7, 
#         # max_delta_step=0.1, 
#         reg_lambda = 4, 
#         # min_child_weight=50, 
#         seed = seed, 
#         ) 

# X_xgboost_mf,X_t_xgboost_mf = make_mf_classifier(X,y,xgboost, X_t,n_folds=2, n_round=10)

# X = np.hstack([X,X_xgboost_mf])
# X_t = np.hstack([X_t,X_t_xgboost_mf])

'''
train and valid
'''
skf = StratifiedKFold(y, n_folds=5, shuffle=True, random_state=seed)
for ind_tr, ind_te in skf:
    X_train = X[ind_tr]
    X_test = X[ind_te]
    
    y_train = y[ind_tr]
    y_test = y[ind_te]
    break

print(X_train.shape,X_test.shape)


xgboost = xgb.XGBClassifier(
        n_estimators=1000, 
        learning_rate = 0.03, 
        max_depth=6, 
        subsample=0.7, 
        colsample_bytree = 0.7, 
        # gamma = 0.7, 
        # max_delta_step=0.1, 
        reg_lambda = 4, 
        # min_child_weight=50, 
        seed = seed, 
        ) 

xgboost.fit(
    X_train,
    y_train,
    eval_metric='mlogloss',
    eval_set=[(X_train,y_train),(X_test,y_test)],
    early_stopping_rounds=100,
    )
y_preds = xgboost.predict_proba(X_test)


res = xgboost.predict_proba(X_t)

submission = pd.DataFrame()
submission["ID"] = np.arange(res.shape[0])+1
submission["Adoption"]= res[:,0]
submission["Died"]= res[:,1]
submission["Euthanasia"]= res[:,2]
submission["Return_to_owner"]= res[:,3]
submission["Transfer"]= res[:,4]

submission.to_csv("sub.csv",index=False)