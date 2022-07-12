from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import numpy as np

# 读取数据
data = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# 数据预处理
train = data.drop(columns=['Id', 'Cover_Type'])
targets = data['Cover_Type']

X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, 1:-1], data.iloc[:, -1], test_size=0.3, random_state=1)


def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv=5, scoring=scoring)
    return np.mean(xval)


def main(clf, output=False):
    # evaluate
    score = compute_score(clf, X=X_test, y=y_test, scoring='accuracy')
    print(str(clf.__class__) + str(score))

    if output:  # 是否生成结果
        clf.fit(train, targets)  # 使用所有的数据进行重新训练
        result = pd.DataFrame({"Id": test['Id'], "Cover_Type": clf.predict(test.iloc[:, 1:55])})
        result.to_csv('result.csv', index=False)


# classifier
# rf = RandomForestClassifier(n_estimators=500, max_features='sqrt', random_state=17, n_jobs=10)
# rf.fit(X_train, y_train)

# gb = GradientBoostingClassifier(n_estimators=100, random_state=17)
# gb.fit(X_train, y_train)

# adaboost = AdaBoostClassifier(n_estimators=500)


# XGBoost
import xgboost as xgb

xgb = xgb.XGBClassifier(max_depth=15)
xgb.fit(X_train, y_train)

models = [xgb]

for model in models:
    main(model, output=True)
