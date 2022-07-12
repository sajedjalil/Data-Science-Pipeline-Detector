# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from pylab import savefig
import seaborn as sns


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# print(train['price_doc'].describe())
# print("Skewness: %f" % train['price_doc'].skew())
# print("Kurtosis: %f" % train['price_doc'].kurt())

# sns.distplot(np.log1p(train['price_doc']),bins=50, kde=False)
# savefig('test.png')

# sns.distplot(train['price_doc'],bins=50, kde=False)
# savefig('test1.png')

# sns.set()
# cols = ['id', 'timestamp', 'full_sq', 'life_sq', 'floor', 'max_floor',
#       'material', 'build_year', 'num_room', 'kitch_sq',
#       'cafe_count_5000_price_2500', 'cafe_count_5000_price_4000',
#       'cafe_count_5000_price_high', 'big_church_count_5000',
#       'church_count_5000', 'mosque_count_5000', 'leisure_count_5000',
#       'sport_count_5000', 'market_count_5000', 'price_doc']
# sns.pairplot(train[cols], size = 2.5)
# plt.show();
# savefig('test4.png')

# total = train.isnull().sum().sort_values(ascending=False)
# percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
# missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# print(missing_data)
# print(missing_data.head(20))
print(train.columns)
cols = ['price_doc','full_sq', 'life_sq', 'floor', 'max_floor','build_year', 'num_room', 'kitch_sq', 'state','product_type','sub_area']
train.dropna(inplace=True)
train = pd.get_dummies(train)

print(train.iloc[0])

corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
savefig('test3.png')


from xgboost import plot_importance
import xgboost as xgb
from matplotlib import pyplot

def xBoost(x,y):
    xgb_regr = xgb.XGBRegressor(
                     colsample_bytree=0.2,
                     gamma=0.0,
                     learning_rate=0.01,
                     max_depth=4,
                     min_child_weight=1.5,
                     n_estimators=7200,                                                                  
                     reg_alpha=0.9,
                     reg_lambda=0.6,
                     subsample=0.2,
                     seed=42,
                     silent=1)
    
    xgb_regr.fit(x, y)
    plot_importance(xgb_regr)
    pyplot.show()
    savefig('test5.png')

# xBoost(train[['full_sq', 'life_sq', 'floor', 'max_floor','build_year', 'num_room', 'kitch_sq', 'state']], train[['price_doc']])