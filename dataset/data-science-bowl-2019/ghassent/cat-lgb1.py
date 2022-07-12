import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import StratifiedKFold
from catboost import Pool, CatBoostClassifier, cv,CatBoostRegressor
from lightgbm import LGBMClassifier

from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import gc
train = pd.read_csv('../input/data-science-bowl-2019/train.csv')
test = pd.read_csv('../input/data-science-bowl-2019/test.csv')
train_labels = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')
#specs= pd.read_csv('../input/data-science-bowl-2019/specs.csv')
submission_sample= pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')


test = pd.read_csv('../input/data-science-bowl-2019/test.csv')
test_df=test[['installation_id', 'title','world','type','timestamp','event_count','event_code','game_time']]
test_df=test_df[test_df['type'] == 'Assessment']
test_df=test_df.sort_values('timestamp', ascending=False)
test_dfa=test_df.drop_duplicates(subset=['installation_id'])
test_df=test_dfa.drop(['installation_id','type','timestamp'],axis=1)
del test
gc.collect()


result = pd.merge(left=train_labels,right=train , how='inner', left_on=['installation_id','game_session', 'title'], right_on=['installation_id','game_session','title'])
result=result.drop(['installation_id', 'num_correct','num_incorrect','accuracy','event_id','event_data','timestamp','type','game_session'],axis=1)
train_=result[['title','world','event_count','event_code','game_time']]
train_label=result[['accuracy_group']]


cat_features = np.where(train_.dtypes != np.float)[0]
train_pool = Pool(data=train_,label=train_label,cat_features=cat_features)
eval_dataset = Pool(data=test_df,cat_features=cat_features)


from catboost import CatBoost
model = CatBoost()

grid = {'learning_rate': [0.03],
        'depth': [10],
        'l2_leaf_reg': [0.001,3]}

grid_search_result = model.grid_search(grid,X=train_pool,plot=False,cv=3)

# Get predicted classes
preds_class = pd.DataFrame(model.predict(eval_dataset))
catboos_pred = pd.DataFrame(model.predict(train_pool))




params = {'n_estimators':300,
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'subsample': 0.75,
            'subsample_freq': 1,
            'learning_rate': 0.04,
            'feature_fraction': 0.9,
            'max_depth': 5,
            'lambda_l1': 1,  
            'lambda_l2': 1,
            'verbose': 100,
            
            }


data=catboos_pred
y=train_label
features = [f_ for f_ in data.columns ]


folds = StratifiedKFold(n_splits=5,random_state=42,shuffle=True)
oof_preds = np.zeros(data.shape[0])
sub_preds = np.zeros(preds_class.shape[0])
for n_fold, (trn_idx, val_idx) in enumerate(folds.split(data,y)):
    trn_x, trn_y = data[features].iloc[trn_idx], y.iloc[trn_idx]
    val_x, val_y = data[features].iloc[val_idx], y.iloc[val_idx]
    
    clf =LGBMClassifier(**params)
    
    clf.fit(trn_x, trn_y, 
            eval_set= [(trn_x, trn_y), (val_x, val_y)], 
             verbose=250, early_stopping_rounds=150
             
             
             
           )
    
    oof_preds[val_idx] = clf.predict(val_x, num_iteration=clf.best_iteration_) 
    sub_preds += clf.predict(preds_class, num_iteration=clf.best_iteration_)/ folds.n_splits
    
    print('Fold %2d ' % (n_fold + 1))

XT=test_dfa[['installation_id']]
XT['accuracy_group'] = np.around(sub_preds).astype(int)
submission = pd.merge(submission_sample.drop('accuracy_group',axis=1), XT,  how='inner', left_on=['installation_id'], right_on = ['installation_id'])
submission.to_csv('submission.csv', index = False)
submission['accuracy_group'].value_counts(dropna = False).sort_index()


