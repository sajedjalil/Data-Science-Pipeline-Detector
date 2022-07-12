# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from sklearn.preprocessing import minmax_scale
from sklearn import neighbors
from sklearn import metrics
from sklearn import model_selection
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
mpl.rcParams['font.sans-serif'] = ['SimHei']
# 数据导入
train_set = pd.read_csv('../input/train.csv')
pred_set = pd.read_csv('../input/test.csv')
# print(train_set.head())
# 查看是否存在重复值与缺失值
# print(any(train_set.duplicated()))
# print(train_set.isnull().sum(axis=0))
# 删除Id列
train_set.drop('Id', axis=1, inplace=True)
pred_set.drop('Id', axis=1, inplace=True)
# print(train_set.head())
# print(train_set.columns[:9])
# print('-' * 50)
# print(train_set.dtypes)
# print(pred_set.dtypes)
# # 绘制核密度图
# col_train_1 = np.array(train_set.columns[0:10]).reshape(2, 5)
# col_test_1 = np.array(pred_set.columns[0:10]).reshape(2, 5)
# plt.figure(figsize=(16, 16))
# for i in range(2):
#     for j in range(5):
#         ax1 = plt.subplot2grid(shape=(2, 5), loc=(i, j))
#         sns.distplot(train_set[col_train_1[i, j]], hist=False, kde=True,
#                      kde_kws={'color': 'blue', 'linestyle': '-'}, label='train')
#         sns.distplot(pred_set[col_test_1[i, j]], hist=False, kde=True,
#                      kde_kws={'color': 'red', 'linestyle': '--'}, label='test')
# 对训练集数据特征做归一化处理
train_pre = train_set.columns[:10]
train_scale = minmax_scale(train_set[train_pre].astype(float))
# print(train_scale)
train_post = train_set[train_set.columns[10:]]
# print(train_post)
train_scale = pd.DataFrame(train_scale, columns=train_pre)
# print(train_scale)
train_new = pd.concat([train_scale, train_post], axis=1)
# print(train_new)
# 对预测集数据特征做归一化处理
pred_pre = pred_set.columns[:10]
pred_scale = minmax_scale(pred_set[train_pre].astype(float))
# print(pred_scale)
pred_post = pred_set[pred_set.columns[10:]]
# print(pred_post)
pred_scale = pd.DataFrame(pred_scale, columns=train_pre)
# print(pred_scale)
pred_new = pd.concat([pred_scale, pred_post], axis=1)
# print(pred_new)
predictors = train_new.columns[:-1]
X_train, X_test, y_train, y_test = model_selection.train_test_split(train_new[predictors], train_new.Cover_Type,
                                                                    test_size=0.25, random_state=1234)
accuracy = []
k = np.arange(1, np.ceil(np.log2(train_new.shape[0]))).astype(int)
for i in k:
    cv_result = model_selection.cross_val_score(neighbors.KNeighborsClassifier(n_neighbors=i, weights='distance'),
                                                X_train, y_train, cv=10, scoring='accuracy')
    accuracy.append(cv_result.mean())
print(accuracy)
# arg_max = np.array(accuracy).argmax()
# plt.figure(figsize=(8, 8))
# plt.plot(k, accuracy)
# plt.scatter(k, accuracy)
# plt.text(k[arg_max], accuracy[arg_max], '最佳的K值为%s' % k[arg_max])
knn_class = neighbors.KNeighborsClassifier(n_neighbors=2, weights='distance')
knn_class.fit(X_train, y_train)
predict = knn_class.predict(X_test)
print(metrics.accuracy_score(y_test, predict))
print(metrics.classification_report(y_test, predict))
cm = pd.crosstab(predict, y_test)
cm = pd.DataFrame(cm, columns=[1, 2, 3, 4, 7, 6, 7],
                  index=[1, 2, 3, 4, 7, 6, 7])
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, cmap='GnBu')
plt.xlabel('Rel Lable')
plt.ylabel('Predict Lable')
plt.show()
predict_test = knn_class.predict(pred_new)
sub = pd.read_csv('../input/sample_submission.csv')
sub.drop('Cover_Type', axis=1, inplace=True)
predict_result = pd.Series(predict_test, name='Cover_Type')
sub = pd.concat([sub, predict_result], axis=1)
print(sub.head())
sub.to_csv('sample_submission.csv', index=False)