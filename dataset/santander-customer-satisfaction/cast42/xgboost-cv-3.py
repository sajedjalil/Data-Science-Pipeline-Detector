import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg") #Needed to save figures
from sklearn import cross_validation
import xgboost as xgb
from sklearn.metrics import roc_auc_score

training = pd.read_csv("../input/train.csv", index_col=0)
test = pd.read_csv("../input/test.csv", index_col=0)

print(training.shape)
print(test.shape)

# Replace -999999 in var3 column with most common value 2 
# See https://www.kaggle.com/cast42/santander-customer-satisfaction/debugging-var3-999999
# for details
training = training.replace(-999999,2)


X = training.iloc[:,:-1]
y = training.TARGET

# Add zeros per row as extra feature
X['n0'] = (X == 0).sum(axis=1)

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif,chi2
from sklearn.preprocessing import Binarizer, scale

p = 75 # 261 features validation_1-auc:0.848642

X_bin = Binarizer().fit_transform(scale(X))
selectChi2 = SelectPercentile(chi2, percentile=p).fit(X_bin, y)
selectF_classif = SelectPercentile(f_classif, percentile=p).fit(X, y)

chi2_selected = selectChi2.get_support()
chi2_selected_features = [ f for i,f in enumerate(X.columns) if chi2_selected[i]]
print('Chi2 selected {} features {}.'.format(chi2_selected.sum(),
   chi2_selected_features))
f_classif_selected = selectF_classif.get_support()
f_classif_selected_features = [ f for i,f in enumerate(X.columns) if f_classif_selected[i]]
print('F_classif selected {} features {}.'.format(f_classif_selected.sum(),
   f_classif_selected_features))
selected = chi2_selected & f_classif_selected
print('Chi2 & F_classif selected {} features'.format(selected.sum()))
features = [ f for f,s in zip(X.columns, selected) if s]
print (features)

X_sel = X[features]

test['n0'] = (test == 0).sum(axis=1)
# test['logvar38'] = test['var38'].map(np.log1p)
# # Encode var36 as category
# test['var36'] = test['var36'].astype('category')
# test = pd.get_dummies(test)
sel_test = test[features]   

xgtrain = xgb.DMatrix(X_sel, label=y)
clf = xgb.XGBClassifier(missing=9999999999,
                max_depth = 8,
                n_estimators=1000,
                learning_rate=0.02, 
                nthread=4,
                subsample=0.8,
                colsample_bytree=0.5,
                min_child_weight = 7,
                seed=4242)
xgb_param = clf.get_xgb_params()
#do cross validation
# print ('Start cross validation')
# cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=5000, nfold=5, metrics=['auc'],
#      early_stopping_rounds=50)
# print('Best number of trees = {}'.format(cvresult.shape[0]))
# clf.set_params(n_estimators=cvresult.shape[0])
clf.set_params(n_estimators=346)
print('Fit on the trainingsdata')
clf.fit(X_sel, y, eval_metric='auc')
print('Predict the probabilities based on features in the test set')
pred = clf.predict_proba(sel_test, ntree_limit=346)[:,1]

pred[np.where(sel_test['var15'] < 23)] = 0

submission = pd.DataFrame({"ID":test.index, "TARGET":pred})
submission.to_csv("submission.csv", index=False)

       
       