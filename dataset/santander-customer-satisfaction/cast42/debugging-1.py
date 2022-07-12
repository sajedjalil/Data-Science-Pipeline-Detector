import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn import cross_validation
import xgboost as xgb
from sklearn.metrics import roc_auc_score

training = pd.read_csv("../input/train.csv", index_col=0)
test = pd.read_csv("../input/test.csv", index_col=0)

print(training.shape)
print(test.shape)

X = training.iloc[:,:-1]
y = training.TARGET

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif

selectK = SelectKBest(f_classif, k=220)
selectK.fit(X, y)
X_sel = selectK.transform(X)

support = selectK.get_support()
features = X.columns[support]
print (features)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_sel, y, random_state=1301)
clf = xgb.XGBClassifier(max_depth   = 7,
                learning_rate       = 0.02,
                subsample           = 0.9,
                colsample_bytree    = 0.85,
                n_estimators        = 10)
clf.fit(X_train, y_train, early_stopping_rounds=500, eval_metric="auc",
        eval_set=[(X_test, y_test)])
        
print('Overall AUC:', roc_auc_score(y, clf.predict_proba(X_sel)[:,1]))
    
sel_test = selectK.transform(test)    
y_pred = clf.predict_proba(sel_test)

submission = pd.DataFrame({"ID":test.index, "TARGET":y_pred[:,1]})
submission.to_csv("submission.csv", index=False)

# Visualize
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(1, 1, 1)
colors = [(0.0, 0.63, 0.69), 'black']
markers = ["o", "D"]
classes = np.sort(np.unique(y_train))
labels = ["Satisfied customer", "Unsatisfied customer"]
y_pred = clf.predict(sel_test)

for class_ix, marker, color, label in zip(
            classes, markers, colors, labels):
    ax.scatter(X_train[np.where(y_train == class_ix), support[1]],
               X_train[np.where(y_train == class_ix), support[-1]],
               marker=marker, color=color, edgecolor='whitesmoke',
               linewidth='1', alpha=0.9, label=label)
    # ax.scatter(sel_test[np.where(y_pred == class_ix), support[1]],
    #           sel_test[np.where(y_pred == class_ix), support[-1]],
    #           marker=marker, color='r', edgecolor='whitesmoke',
    #           linewidth='1', alpha=0.9, label=label)
    ax.legend(loc='best')
plt.title(
        "Scatter plot of the training data examples projected on the "
        "2 most important features %s and %s" %(features[1], features[-1]))
plt.xlabel(features[1])
plt.ylabel(features[-1])
plt.show()

plt.savefig("var3-15.pdf", format='pdf')
plt.savefig("var3-15.png", format='png')

