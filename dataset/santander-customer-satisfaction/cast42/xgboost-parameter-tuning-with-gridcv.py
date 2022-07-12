import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg") #Needed to save figures
from sklearn.cross_validation import *
from sklearn.grid_search import GridSearchCV
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


# Replace 9999999999 with NaN
# See https://www.kaggle.com/c/santander-customer-satisfaction/forums/t/19291/data-dictionary/111360#post111360
# training = training.replace(9999999999, np.nan)
# training.dropna(inplace=True)
# Leads to validation_0-auc:0.839577

X = training.iloc[:,:-1]
y = training.TARGET

# Add zeros per row as extra feature
X['n0'] = (X == 0).sum(axis=1)

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif,chi2
from sklearn.preprocessing import Binarizer, scale

p = 86 # 308 features validation_1-auc:0.848039
p = 80 # 284 features validation_1-auc:0.848414
p = 77 # 267 features validation_1-auc:0.848000
p = 75 # 261 features validation_1-auc:0.848642
# p = 73 # 257 features validation_1-auc:0.848338
# p = 70 # 259 features validation_1-auc:0.848588
# p = 69 # 238 features validation_1-auc:0.848547
# p = 67 # 247 features validation_1-auc:0.847925
# p = 65 # 240 features validation_1-auc:0.846769
# p = 60 # 222 features validation_1-auc:0.848581

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

# Based on https://www.kaggle.com/cast42/flavours-of-physics/gridsearchcv-with-feature-in-xgboost/code

xgb_model = xgb.XGBClassifier()

# Initial parameter
# parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
#               'objective':['binary:logistic'],
#               'learning_rate': [0.3], #so called `eta` value
#               'max_depth': [8],
#               'min_child_weight': [7],
#               'silent': [1],
#               'subsample': [0.8],
#               'colsample_bytree': [0.5],
#               'n_estimators': [300], #number of trees
#               'seed': [1337],
#               'missing': [9999999999]}
              
# Raw AUC score: 0.794133008324
# learning_rate = 0.02 -> 0.838619485896
# max_depth = 7 -> 0.840279516133
# max_depth = 6 -> 0.839682658184
# max_depth = 7, n_estimators=500 -> 

parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['binary:logistic'],
              'learning_rate': [0.02], #so called `eta` value
              'max_depth': [6],
              'min_child_weight': [7],
              'silent': [1],
              'subsample': [0.8],
              'colsample_bytree': [0.5],
              'n_estimators': [500], #number of trees
              'seed': [1337],
              'missing': [9999999999]}

clf = GridSearchCV(xgb_model, parameters, n_jobs=4, 
                   cv=StratifiedKFold(y, n_folds=4, shuffle=True), 
                   verbose=2, refit=True,scoring='roc_auc')

clf.fit(X_sel, y)

best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
print('Raw AUC score:', score)
for param_name in sorted(best_parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))
        
print('Overall AUC:', roc_auc_score(y, clf.predict_proba(X_sel)[:,1]))

test['n0'] = (test == 0).sum(axis=1)
sel_test = test[features]    
y_pred = clf.predict_proba(sel_test)

submission = pd.DataFrame({"ID":test.index, "TARGET":y_pred[:,1]})
submission.to_csv("submission.csv", index=False)
