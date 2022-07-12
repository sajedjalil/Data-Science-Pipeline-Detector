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

print('There are {} features.'.format(training.shape[1]))

X = training.iloc[:,:-1]
y = training.TARGET

# Add zeros per row as extra feature
X['n0'] = (X == 0).sum(axis=1)

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif,chi2
from sklearn.preprocessing import Binarizer, scale

# Let only chi2 and f_classif select features
p=25

# X_bin = Binarizer().fit_transform(scale(X))
# selectChi2 = SelectPercentile(chi2, percentile=p).fit(X_bin, y)
# selectF_classif = SelectPercentile(f_classif, percentile=p).fit(X, y)

# chi2_selected = selectChi2.get_support()
# chi2_selected_features = [ f for i,f in enumerate(X.columns) if chi2_selected[i]]
# print('Chi2 selected {} features {}.'.format(chi2_selected.sum(),
#   chi2_selected_features))
# f_classif_selected = selectF_classif.get_support()
# f_classif_selected_features = [ f for i,f in enumerate(X.columns) if f_classif_selected[i]]
# print('F_classif selected {} features {}.'.format(f_classif_selected.sum(),
#   f_classif_selected_features))
# selected = chi2_selected & f_classif_selected
# print('Chi2 & F_classif selected {} features'.format(selected.sum()))
# features = [ f for f,s in zip(X.columns, selected) if s]
# print (features)

features = ['var15', 'imp_op_var40_efect_ult1', 'imp_op_var40_efect_ult3', 
'imp_op_var41_efect_ult3', 'imp_op_var39_efect_ult1', 'imp_op_var39_efect_ult3',
'ind_var5', 'ind_var8_0', 'ind_var8', 'ind_var12_0', 'ind_var12', 'ind_var13_0',
'ind_var13_corto_0', 'ind_var13_corto', 'ind_var13_largo_0', 'ind_var13_largo',
'ind_var13', 'ind_var14_0', 'ind_var24_0', 'ind_var24', 'ind_var25_cte', 'ind_var26_0',
'ind_var26_cte', 'ind_var26', 'ind_var25_0', 'ind_var25', 'ind_var30', 'num_var4',
'num_var5', 'num_var8_0', 'num_var8', 'num_var12_0', 'num_var12', 'num_var13_0',
'num_var13_corto_0', 'num_var13_corto', 'num_var13_largo_0', 'num_var13_largo',
'num_var13', 'num_var24_0', 'num_var24', 'num_var26_0', 'num_var26', 'num_var25_0',
'num_var25', 'num_var30_0', 'num_var30', 'num_var35', 'num_var42_0', 'num_var42',
'saldo_var12', 'saldo_var13_corto', 'saldo_var13', 'saldo_var24', 'saldo_var30', 
'saldo_var42', 'var36', 'imp_aport_var13_hace3', 'ind_var43_recib_ult1', 
'num_aport_var13_hace3', 'num_meses_var5_ult3', 'num_meses_var8_ult3',
'num_meses_var12_ult3', 'num_meses_var13_corto_ult3', 'num_meses_var13_largo_ult3',
'num_op_var40_efect_ult1', 'num_op_var40_efect_ult3', 'num_var43_recib_ult1',
'saldo_medio_var5_hace2', 'saldo_medio_var5_ult3', 'saldo_medio_var12_hace2',
'saldo_medio_var12_ult1', 'saldo_medio_var12_ult3', 'saldo_medio_var13_corto_hace2',
'saldo_medio_var13_corto_ult1', 'saldo_medio_var13_corto_ult3', 'var38', 'n0']

print ('Features selected by Chi2 & f_classif')
print (features)

from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import StratifiedKFold

X_train, X_test, y_train, y_test = \
   cross_validation.train_test_split(X[features], y, random_state=1301, stratify=y, test_size=0.35)

# classifier = RandomForestClassifier(random_state=1301)
# classifier = ExtraTreesClassifier(random_state=1301)
classifier = xgb.XGBClassifier(
    objective='binary:logistic', 
    n_estimators=200, 
    learning_rate=0.08, 
    max_depth=5, 
    nthread=4,
    subsample=0.9,
#    colsample_bytree=0.8,
    reg_lambda=6,
    reg_alpha=5,
    seed=1301,
    silent=True
)
selector = RFECV(estimator=classifier, step=3, 
   cv=StratifiedKFold(y_train, shuffle=True, n_folds=3, random_state=1301), scoring='roc_auc')
selector.fit(X_train, y_train)

print('The optimal number of features is {}'.format(selector.n_features_))
features = [f for f,s in zip(X_train.columns, selector.support_) if s]
print('The selected features are:')
print ('{}'.format(features))

plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (roc auc)")
plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
plt.savefig('feature_auc_nselected.png', bbox_inches='tight', pad_inches=1)

X_train_s = X_train[features]

classifier.fit(X_train_s, y_train)

X_test_s = X_test[features]
print ('Out of sample auc: {}'.format(classifier.score(X_test_s, y_test)))

X_sel = X[features]

ratio = float(np.sum(y == 1)) / np.sum(y==0)
clf = xgb.XGBClassifier(missing=9999999999,
                max_depth = 7,
                n_estimators=1000,
                learning_rate=0.1, 
                nthread=4,
                subsample=1.0,
                colsample_bytree=0.5,
                min_child_weight = 3,
                scale_pos_weight = ratio,
                seed=1301)
                
clf.fit(X_train_s, y_train, early_stopping_rounds=50, eval_metric="auc",
        eval_set=[(X_train_s, y_train), (X_test_s, y_test)])
        
print('Overall AUC:', roc_auc_score(y, clf.predict_proba(X_sel, ntree_limit=clf.best_iteration)[:,1]))

test['n0'] = (test == 0).sum(axis=1)
sel_test = test[features]
y_pred = clf.predict_proba(sel_test, ntree_limit=clf.best_iteration)

submission = pd.DataFrame({"ID":test.index, "TARGET":y_pred[:,1]})
submission.to_csv("submission.csv", index=False)

mapFeat = dict(zip(["f"+str(i) for i in range(len(features))],features))
ts = pd.Series(clf.booster().get_fscore())
ts.sort_values()[-15:].plot(kind="barh", title=("features importance"))

featp = ts.sort_values()[-15:].plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
fig_featp = featp.get_figure()
fig_featp.savefig('feature_importance_xgb.png', bbox_inches='tight', pad_inches=1)



