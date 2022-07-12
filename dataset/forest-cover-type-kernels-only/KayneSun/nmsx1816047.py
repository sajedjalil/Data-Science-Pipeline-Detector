import pandas as pd
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

# 数据导入
train_set = pd.read_csv('../input/train.csv')
test_set = pd.read_csv('../input/test.csv')
# print(train_set.head())

# 查看是否存在重复值与缺失值
# print(any(train_set.duplicated()))
# print(train_set.isnull().sum(axis=0))

# 删除Id列
train_set.drop('Id', axis=1, inplace=True)
test_set.drop('Id', axis=1, inplace=True)
# print(train_set.head())
# print('-' * 50)

# # 统计每种Cover_Type的情况
# cover_type = [1, 2, 3, 4, 5, 6, 7]
# for value in cover_type:
#     print(train_set[train_set['Cover_Type'] == value])
#     print('=' * 50)
print(train_set.describe())
print(train_set.dtypes)

# 绘制核密度图
col_train_1 = np.array(train_set.columns[0:10]).reshape(2, 5)
col_test_1 = np.array(test_set.columns[0:10]).reshape(2, 5)
# print(col_train_1)
# print(col_test_1)

plt.figure(figsize=(16, 16))
for i in range(2):
    for j in range(5):
        ax1 = plt.subplot2grid(shape=(2, 5), loc=(i, j))
        sns.distplot(train_set[col_train_1[i, j]], hist=False, kde=True,
                     kde_kws={'color': 'blue', 'linestyle': '-'}, label='train')
        sns.distplot(test_set[col_test_1[i, j]], hist=False, kde=True,
                     kde_kws={'color': 'red', 'linestyle': '--'}, label='test')

predectors = train_set.columns[:-1]
print(predectors)
X_train, X_test, y_train, y_test = model_selection.train_test_split(train_set[predectors],
                                                                    train_set.Cover_Type,
                                                                    test_size=0.25,
                                                                    random_state=1234)

cart_class = tree.DecisionTreeClassifier(criterion='gini', splitter='random',
                                         max_depth=100, min_samples_split=2, min_samples_leaf=1)
cart_class.fit(X_train, y_train)
pred = cart_class.predict(X_test)
print(metrics.accuracy_score(y_test, pred))
# print(cart_class.predict(test_forests))

predict_test = cart_class.predict(test_set)
sub = pd.read_csv('../input/sample_submission.csv')
sub.drop('Cover_Type', axis=1, inplace=True)
predict_result = pd.Series(predict_test, name='Cover_Type')
sub = pd.concat([sub, predict_result], axis=1)
sub.to_csv("sample_submission.csv", index=False)
