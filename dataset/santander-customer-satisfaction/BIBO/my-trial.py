import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score, ShuffleSplit

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import safe_mask




plt.style.use('ggplot')
df = pd.read_csv("../input/train.csv", low_memory=False, index_col='ID')
df_test = pd.read_csv("../input/test.csv", low_memory=False, index_col='ID')
X_train = df.drop(['TARGET'],axis=1)
y_train = df['TARGET']
X_test = df_test
X_selection = SelectFromModel(GradientBoostingClassifier()).fit(X_train,y_train)
X_mask = X_selection.get_support()
X_mask = safe_mask(X_train,X_mask)
print(X_mask)
X_train_selected = np.array(X_train)[:,X_mask]
X_test_selected = np.array(X_test)[:,X_mask]
#SelectKBest(lambda X, Y: array(map(lambda x:mic(x, Y), X.T)).T, k=2).fit_transform(np.array(X_train), np.array(y_train))
print(np.array(X_train).shape)
print(X_train_selected.shape)

clf = RandomForestClassifier(n_estimators=60, random_state=0)
#score = cross_val_score(clf, X_train, y_train,cv=10)
clf.fit(X_train_selected, y_train)
y_pred = clf.predict(X_test_selected)
df_test['TARGET'] = pd.Series(y_pred,index=df_test.index)
result = df_test[['TARGET']]
result.to_csv("result.csv",encoding='utf-8')