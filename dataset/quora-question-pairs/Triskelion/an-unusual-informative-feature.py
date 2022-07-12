import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

df_train = pd.read_csv("../input/train.csv")
qid_abs_diff = list(abs(df_train["qid1"] - df_train["qid2"]))
df_leakage = pd.DataFrame({"qid_abs_diff": qid_abs_diff, "target": df_train["is_duplicate"]})
np.random.shuffle(qid_abs_diff)
df_leakage["qid_abs_diff_shuffled"] = qid_abs_diff

X = np.array(df_leakage["qid_abs_diff"]).reshape((df_leakage.shape[0], 1))
y = np.array(df_leakage["target"])

clf = RandomForestClassifier(random_state=1, n_estimators=50, n_jobs=-1)

skf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)

print("Absolute Difference between qid1 and qid2")
for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf.fit(X_train, y_train)
    preds = clf.predict_proba(X_test)[:,1]
    print("Fold: %d"%(i+1))
    print("AUC: %f"%(roc_auc_score(y_test, preds)))
    print("LOGLOSS: %f\n"%(log_loss(y_test, preds)))

print("Absolute Difference between qid1 and qid2 - shuffled")
X = np.array(df_leakage["qid_abs_diff_shuffled"]).reshape((df_leakage.shape[0], 1))    
for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf.fit(X_train, y_train)
    preds = clf.predict_proba(X_test)[:,1]
    print("Fold: %d"%(i+1))
    print("AUC: %f"%(roc_auc_score(y_test, preds)))
    print("LOGLOSS: %f\n"%(log_loss(y_test, preds)))