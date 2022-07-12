# -*- coding: UTF-8 -*-

# -----------------------------------------------------
# 一、加载数据部分
# 去掉不必要的warning警告，让展示更加整洁
import warnings
warnings.filterwarnings('ignore')


# 在文件中读取数据
import pandas as pd #用于数据结构处理
import numpy as np

print("loading train data....")
train_data = pd.read_csv("../input/train.csv")
print("loading test data....")
test_data = pd.read_csv("../input/test.csv")
print("Train data dimensions: ", train_data.shape)
print("Test data dimensions: ", test_data.shape)

# 将前几行数据打印出来
train_data.head()
print("Number of missing values",train_data.isnull().sum().sum())

# 获取数据的特征名称
contFeatureslist = []
for colName,x in train_data.iloc[1,:].iteritems():
    #print(x)
    if(not str(x).isalpha()):
        contFeatureslist.append(colName)

print(contFeatureslist)
contFeatureslist.remove("id")
contFeatureslist.remove("loss")

# -----------------------------------------------------
# 二、数据统计
print("describing the data....")

# 数据统计描述
train_data.describe()
# 显示数据分布的不对称度
# print(train_data.skew())

# -----------------------------------------------------
# 三、数据可视化处理

# 数据展示图标的库
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

plt.figure(figsize=(13,9))
sns.boxplot(train_data[contFeatureslist])
# Include  target variable also to find correlation between features and target feature as well
contFeatureslist.append("loss")

correlationMatrix = train_data[contFeatureslist].corr().abs()

plt.subplots(figsize=(13, 9))
sns.heatmap(correlationMatrix,annot=True)

# Mask unimportant features
sns.heatmap(correlationMatrix, mask=correlationMatrix < 1, cbar=False)
plt.show()