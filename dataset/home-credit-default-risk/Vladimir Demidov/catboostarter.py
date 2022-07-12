import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

application_train = pd.read_csv("../input/application_train.csv")
application_test = pd.read_csv("../input/application_test.csv")
subm = pd.read_csv("../input/sample_submission.csv")


target_train = application_train['TARGET']
application_train.drop(['SK_ID_CURR', 'TARGET'], axis=1, inplace=True)
application_test.drop(['SK_ID_CURR'], axis=1, inplace=True)

cat_features = [f for f in application_train.columns if application_train[f].dtype == 'object']
def column_index(df, query_cols):
    cols = df.columns.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols, query_cols, sorter=sidx)]
cat_features_inds = column_index(application_train, cat_features)    
print("Cat features are: %s" % [f for f in cat_features])
print(cat_features_inds)

le = LabelEncoder()
for col in cat_features:
    application_train[col] = le.fit_transform(application_train[col].astype(str))
    application_test[col] = le.fit_transform(application_test[col].astype(str))
    
application_train.fillna(-1, inplace=True)
application_test.fillna(-1, inplace=True)
cols = application_train.columns

X_train, X_valid, y_train, y_valid = train_test_split(application_train, target_train,
                                                      test_size=0.1, random_state=17)
print(X_train.shape)
print(X_valid.shape)

                                     
print("\nCatBoost...")                                     
cb_model = CatBoostClassifier(iterations=101,
                              learning_rate=0.5,
                              depth=3,
                              l2_leaf_reg=40,
                              bootstrap_type='Bernoulli',
                              subsample=0.7,
                              scale_pos_weight=5,
                              eval_metric='AUC',
                              metric_period=50,
                              od_type='Iter',
                              od_wait=20,
                              random_seed=17,
                              allow_writing_files=False)

cb_model.fit(X_train, y_train,
             eval_set=(X_valid, y_valid),
             cat_features=cat_features_inds,
             use_best_model=True,
             verbose=True)
             
fea_imp = pd.DataFrame({'imp': cb_model.feature_importances_, 'col': cols})
fea_imp = fea_imp.sort_values(['imp', 'col'], ascending=[True, False]).iloc[-20:]
_ = fea_imp.plot(kind='barh', x='col', y='imp', figsize=(20, 10))
plt.savefig('catboost_feature_importance.png')             

print('AUC:', roc_auc_score(y_valid, cb_model.predict_proba(X_valid)[:,1]))
y_preds = cb_model.predict_proba(application_test)[:,1]
subm['TARGET'] = y_preds
subm.to_csv('submission.csv', index=False)