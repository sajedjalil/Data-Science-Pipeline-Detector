import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import pickle

X_train = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')
y_train_scored = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')
X_test = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')
submission = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')

dict_cp_type = {'trt_cp': 1, 'ctl_vehicle': 0}
dict_cp_dose = {'D1': 1, 'D2': 2}
X_train.cp_type = X_train.cp_type.apply(lambda x: dict_cp_type[x])
X_train.cp_dose = X_train.cp_dose.apply(lambda x: dict_cp_dose[x])
X_test.cp_type = X_test.cp_type.apply(lambda x: dict_cp_type[x])
X_test.cp_dose = X_test.cp_dose.apply(lambda x: dict_cp_dose[x])

X_train_no_id = X_train.iloc[:, 1:]
X_test_no_id = X_test.iloc[:, 1:]

X_train_no_id.cp_time /= 24
X_test_no_id.cp_time /= 24

for col in y_train_scored.columns[1:]:
    filename = '/kaggle/input/moa-v2/{}.sav'.format(col)
    
    clf = pickle.load(open(filename, 'rb'))
    submission[col] = clf.predict_proba(X_test_no_id)[:, 1]

submission.to_csv('submission.csv', index=False)
    
