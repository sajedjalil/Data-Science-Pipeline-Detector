# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib
from sklearn import cross_validation
import xgboost as xgb
from sklearn.metrics import roc_auc_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
#Read files
train = pd.read_csv("../input/train.csv", index_col=0)
test = pd.read_csv("../input/test.csv", index_col=0)
#Basic information about train file
#train.describe()
print(train.shape)
print(train.head(10))
#Take a look of the result
# Happy customers have TARGET==0, unhappy custormers have TARGET==1
# A little less then 4% are unhappy => unbalanced dataset
df = pd.DataFrame(train.TARGET.value_counts())
df['Percentage'] = df['TARGET']/train.shape[0]*100
df
# Feature Exploring
#var3 - Nationality of the customer
train.var3.describe()  
#Top-10 most common values
train.var3.value_counts()[:10]
# Judging from the range & frequency, var3 is supposed to be the nationality
# Replace -999999 in var3 column with most common value 2 
# See https://www.kaggle.com/cast42/santander-customer-satisfaction/debugging-var3-999999
# for details
train = train.replace(-999999,2)
train.loc[train.var3 == -999999].shape #make sure all outliers are replaced
# Model Building
X = train.iloc[:,:-1]
y = train.TARGET
# Add zeros per row as extra feature
X['n0'] = (X == 0).sum(axis=1)
# Add PCA components as features
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
X_normalized = normalize(X, axis=0)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_normalized) #normalization of the feature
X['PCA1'] = X_pca[:,0]
X['PCA2'] = X_pca[:,1]

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif,chi2
from sklearn.preprocessing import Binarizer, scale

#Select features with the best performance (two methods combined)
p = 75 #try different percentages
X_bin = Binarizer().fit_transform(scale(X))
selectChi2 = SelectPercentile(chi2, percentile=p).fit(X_bin, y)
selectF_classif = SelectPercentile(f_classif, percentile=p).fit(X, y)
chi2_selected = selectChi2.get_support()
chi2_selected_features = [ f for i,f in enumerate(X.columns) if chi2_selected[i]]
#print('Chi2 selected {} features {}.'.format(chi2_selected.sum(), chi2_selected_features))

f_classif_selected = selectF_classif.get_support()
f_classif_selected_features = [ f for i,f in enumerate(X.columns) if f_classif_selected[i]]
#print('F_classif selected {} features {}.'.format(f_classif_selected.sum(),
#   f_classif_selected_features))
selected = chi2_selected & f_classif_selected
print('Chi2 & F_classif selected {} features'.format(selected.sum()))
features = [ f for f,s in zip(X.columns, selected) if s] #zip - return a tuple
#print (features)
X_sel = X[features]
#Cross validation
X_train, X_test, y_train, y_test = \
  cross_validation.train_test_split(X_sel, y, random_state=1301, stratify=y, test_size=0.4)
ratio = float(np.sum(y == 1)) / np.sum(y==0)
clf = xgb.XGBClassifier(missing=9999999999,
                max_depth = 5,
                n_estimators=1000,
                learning_rate=0.1, 
                nthread=4,
                subsample=1.0,
                colsample_bytree=0.5,
                min_child_weight = 3,
                scale_pos_weight = ratio,
                reg_alpha=0.03,
                seed=1301)
clf.fit(X_train, y_train, early_stopping_rounds=50, eval_metric="auc",
        eval_set=[(X_train, y_train), (X_test, y_test)])
print('Overall AUC:', roc_auc_score(y, clf.predict_proba(X_sel, ntree_limit=clf.best_iteration)[:,1]))
#Prediction
test['n0'] = (test == 0).sum(axis=1)
test_normalized = normalize(test, axis=0)
pca = PCA(n_components=2)
test_pca = pca.fit_transform(test_normalized)
test['PCA1'] = test_pca[:,0]
test['PCA2'] = test_pca[:,1]
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
