import pandas as pd
train = pd.read_csv('../input/cat-in-the-dat-ii/train.csv')
test = pd.read_csv('../input/cat-in-the-dat-ii/test.csv')
train.sort_index(inplace=True)
train_y = train['target']; test_id = test['id']
train.drop(['target', 'id'], axis=1, inplace=True); test.drop('id', axis=1, inplace=True)
from sklearn.metrics import roc_auc_score
cat_feat_to_encode = train.columns.tolist();  smoothing=0.20
import category_encoders as ce
oof = pd.DataFrame([])
from sklearn.model_selection import StratifiedKFold
for tr_idx, oof_idx in StratifiedKFold(n_splits=5, random_state=2020, shuffle=True).split(train, train_y):
    ce_target_encoder = ce.TargetEncoder(cols = cat_feat_to_encode, smoothing=smoothing)
    ce_target_encoder.fit(train.iloc[tr_idx, :], train_y.iloc[tr_idx])
    oof = oof.append(ce_target_encoder.transform(train.iloc[oof_idx, :]), ignore_index=False)
ce_target_encoder = ce.TargetEncoder(cols = cat_feat_to_encode, smoothing=smoothing)
ce_target_encoder.fit(train, train_y);  train = oof.sort_index(); test = ce_target_encoder.transform(test)
from sklearn import linear_model
glm = linear_model.LogisticRegression( random_state=1, solver='lbfgs', max_iter=2020, fit_intercept=True, penalty='none', verbose=0); glm.fit(train, train_y)
pd.DataFrame({'id': test_id, 'target': glm.predict_proba(test)[:,1]}).to_csv('submission.csv', index=False)
