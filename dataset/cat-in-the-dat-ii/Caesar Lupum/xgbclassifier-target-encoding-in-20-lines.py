import pandas as pd
train = pd.read_csv('../input/cat-in-the-dat-ii/train.csv'); test = pd.read_csv('../input/cat-in-the-dat-ii/test.csv')
train.sort_index(inplace=True); train_y = train['target']; test_id = test['id']
train.drop(['target', 'id'], axis=1, inplace=True); test.drop('id', axis=1, inplace=True)
from sklearn.metrics import roc_auc_score
import category_encoders as ce
cat_feat_to_encode = train.columns.tolist();  smoothing=0.20; oof = pd.DataFrame([])
from sklearn.model_selection import StratifiedKFold
for tr_idx, oof_idx in StratifiedKFold(n_splits=5, random_state=2020, shuffle=True).split(train, train_y):
    ce_target_encoder = ce.TargetEncoder(cols = cat_feat_to_encode, smoothing=smoothing)
    ce_target_encoder.fit(train.iloc[tr_idx, :], train_y.iloc[tr_idx])
    oof = oof.append(ce_target_encoder.transform(train.iloc[oof_idx, :]), ignore_index=False)
ce_target_encoder = ce.TargetEncoder(cols = cat_feat_to_encode, smoothing=smoothing)
ce_target_encoder.fit(train, train_y);  train = oof.sort_index(); test = ce_target_encoder.transform(test)
from sklearn.model_selection import train_test_split
train_x_,val_x_,train_y_,val_y_=train_test_split(train,train_y,test_size=0.2)
import xgboost as xgb
model = xgb.XGBClassifier(max_depth=20,n_estimators=2020,colsample_bytree=0.20,learning_rate=0.020,objective='binary:logistic', n_jobs=-1)
model.fit(train_x_,train_y_,eval_set=[(val_x_,val_y_)],verbose=0,early_stopping_rounds=200)
pd.DataFrame({'id': test_id, 'target': model.predict_proba(test)[:,1]}).to_csv('submission.csv', index=False)