import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.metrics import roc_auc_score
from collections import defaultdict

training = pd.read_csv("../input/train.csv", index_col=0)
test = pd.read_csv("../input/test.csv", index_col=0)

print(training.shape)
print(test.shape)

X = training.iloc[:,:-1]
y = training.TARGET
X['n0'] = (X == 0).sum(axis=1)

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif,chi2
from sklearn.preprocessing import Binarizer, scale


p = 30

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

# X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_sel,
#   y, random_state=1301, stratify=y, test_size=0.33)
   
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=400, random_state=1301, n_jobs=-1,
   criterion='gini', class_weight='balanced', max_depth=8)

scores = defaultdict(list)

# X = X.as_matrix()
# y = y.as_matrix()
# X_sel = X_sel.as_matrix()

X_train, X_valid, y_train, y_valid = \
  cross_validation.train_test_split(X_sel, y, random_state=1301, stratify=y, test_size=0.5)

# Based on http://blog.datadive.net/selecting-good-features-part-iii-random-forests/
#crossvalidate the scores on a number of different random splits of the data
nrfold = 10
for k in range(nrfold):
    print ('Train on fold {}'.format(k+1))
    X_train, X_valid, y_train, y_valid = \
      cross_validation.train_test_split(X_sel, y, stratify=y, test_size=0.5)
    r = rfc.fit(X_train, y_train)
    auc = roc_auc_score(y_valid, rfc.predict_proba(X_valid)[:,1])
    for i in range(X_sel.shape[1]):
        X_v = X_valid.copy().as_matrix()
        np.random.shuffle(X_v[:, i])
        shuff_auc = roc_auc_score(y_valid, rfc.predict_proba(X_v)[:,1])
        scores[features[i]].append((auc-shuff_auc)/auc)
print ("Features sorted by their score:")
print (sorted([(round(np.mean(score), 4), feat) for feat, score in scores.items()], reverse=True))

rfc_features = [feat for feat in scores.keys()]

print (rfc_features)

ts = pd.DataFrame({'feature': rfc_features,
                   'score': [np.mean(score) for score in scores.values()],
                   })

featp = ts.sort_values(by='score')[-20:].plot(kind='barh', x='feature', y='score', legend=False, figsize=(6, 10))
plt.title('Random Forest Classifier Feature Importance')
fig_featp = featp.get_figure()
fig_featp.savefig('feature_importance_rfc.png', bbox_inches='tight', pad_inches=1)


test['n0'] = (test == 0).sum(axis=1)
sel_test = test[features]
rfc.fit(X_sel, y)
y_pred = rfc.predict_proba(sel_test)

submission = pd.DataFrame({"ID":test.index, "TARGET":y_pred[:,1]})
submission.to_csv("submission_rfc.csv", index=False)