#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ----------------------------------------------- #
# ライブラリをインポート
# ----------------------------------------------- #
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import operator
from matplotlib import pylab as plt
plt.style.use('ggplot')

# XGBoostのversion確認
print('XGBoostのVersion')
print(xgb.__version__)
# 0.6

# ----------------------------------------------- #
# データ読み込み
# ----------------------------------------------- #
print('データ読み込み')
train = pd.read_csv('../input/train_users_2.csv')
test = pd.read_csv('../input/test_users.csv')


def freq(data, var):  # データ確認用関数を定義
    freq = data[var].value_counts().reset_index()
    freq.columns = [var, 'count']
    freq['percent'] = freq['count'] / freq['count'].sum() * 100
    freq['percent'] = freq['percent'].map('{:,.2f}%'.format)
    return(freq)

# 目的変数のcountry_destinationの分布を確認
print('目的変数のcountry_destinationの分布を確認')
print(freq(train, 'country_destination'))
#    country_destination   count percent
# 0                  NDF  124543  58.35%
# 1                   US   62376  29.22%
# 2                other   10094   4.73%
# 3                   FR    5023   2.35%
# 4                   IT    2835   1.33%
# 5                   GB    2324   1.09%
# 6                   ES    2249   1.05%
# 7                   CA    1428   0.67%
# 8                   DE    1061   0.50%
# 9                   NL     762   0.36%
# 10                  AU     539   0.25%
# 11                  PT     217   0.10%

labels = train['country_destination'].values
X_train = train.drop(
    ['id', 'date_first_booking', 'country_destination'], axis=1)
X_test = test.drop(['id', 'date_first_booking'], axis=1)
test_id = test['id']
n_labels = len(set(labels))
n_train = train.shape[0]
n_test = test.shape[0]

# ----------------------------------------------- #
# 特徴量抽出
# ----------------------------------------------- #
print('特徴量抽出')
X_all = pd.concat((X_train, X_test), axis=0, ignore_index=True)
X_all = X_all.fillna(-1)

# "age"に生年月日が混ざっているため、クリーニングを行う。
print('age変数の分布を確認')
print(X_all['age'].describe())
# count    275547.000000
# mean         26.725746
# std         110.820874
# min          -1.000000
# 25%          -1.000000
# 50%          25.000000
# 75%          35.000000
# max        2014.000000
# Name: age, dtype: float64

# 外れ値は-1として扱う
X_all.loc[X_all.age > 95, 'age'] = -1
X_all.loc[X_all.age < 13, 'age'] = -1

# "date_account_created"から、年(year)、月(month)、日(day)を抽出する
dac = np.vstack(
    X_all.date_account_created.astype(str).apply(
        lambda x: list(map(int, x.split('-')))
    ).values
)
X_all['dac_year'] = dac[:, 0]
X_all['dac_month'] = dac[:, 1]
X_all['dac_day'] = dac[:, 2]
X_all = X_all.drop(['date_account_created'], axis=1)

# "timestamp_first_active"から、年(year)、月(month)、日(day)を抽出する
tfa = np.vstack(
    X_all.timestamp_first_active.astype(str).apply(
        lambda x: list(map(int, [x[:4], x[4:6], x[6:8],
                                 x[8:10], x[10:12],
                                 x[12:14]]))
    ).values
)
X_all['tfa_year'] = tfa[:, 0]
X_all['tfa_month'] = tfa[:, 1]
X_all['tfa_day'] = tfa[:, 2]
X_all = X_all.drop(['timestamp_first_active'], axis=1)

# カテゴリカル変数のsignup_methodの分布を確認
print('カテゴリカル変数のsignup_methodの分布を確認')
print(freq(X_all, 'signup_method'))
#   signup_method   count percent
# 0         basic  198222  71.94%
# 1      facebook   74864  27.17%
# 2        google    2438   0.88%
# 3         weibo      23   0.01%

# カテゴリカル変数は、One-Hot Encodingを行う
ohe_features = ['gender', 'signup_method', 'signup_flow', 'language',
                'affiliate_channel', 'affiliate_provider',
                'first_affiliate_tracked', 'signup_app',
                'first_device_type', 'first_browser']
for feature in ohe_features:
    # print(freq(all, f)) # 全変数の分布を確認したい場合はコメントアウトを取り除く
    X_all_dummy = pd.get_dummies(X_all[feature], prefix=feature)
    X_all = X_all.drop([feature], axis=1)
    X_all = pd.concat((X_all, X_all_dummy), axis=1)

# モデル構築、モデル検証データを分割する
X_train = X_all.iloc[:n_train, :]
le = LabelEncoder()
y_train = le.fit_transform(labels)
X_test = X_all.iloc[n_train:, :]
print('モデル構築データ(件数, 変数の数)')
print(X_train.shape)
# (213451, 161)
print('モデル予測対象データ(件数, 変数の数)')
print(X_test.shape)
# (62096, 161)

# ----------------------------------------------- #
# 交差検証
# ----------------------------------------------- #
print('交差検証')
params = {
    'objective': 'multi:softprob',
    'eval_metric': 'merror',
    'num_class': n_labels,
    'eta': 0.3,
    'max_depth': 6,
    'subsample': 0.5,
    'colsample_bytree': 0.3,
    'silent': 1,
    'seed': 123
}

num_boost_round = 50

dtrain = xgb.DMatrix(X_train, y_train)
res = xgb.cv(params, dtrain, num_boost_round=num_boost_round, nfold=5,
             callbacks=[xgb.callback.print_evaluation(show_stdv=True),
                        xgb.callback.early_stop(1)])
# [0]	train-merror:0.412246+0.000480388	test-merror:0.412766+0.00256173
# Multiple eval metrics have been passed:
# 'test-merror' will be used for early stopping.
#
# Will train until test-merror hasn't improved in 1 rounds.
# [1]	train-merror:0.404883+0.00300017	test-merror:0.405322+0.00370093
# [2]	train-merror:0.399719+0.00471514	test-merror:0.400192+0.00550752
# [3]	train-merror:0.397167+0.00474924	test-merror:0.397714+0.0052585
# [4]	train-merror:0.391852+0.00498235	test-merror:0.392556+0.00498487
# [5]	train-merror:0.38987+0.0039126	test-merror:0.39101+0.00512646
# [6]	train-merror:0.387919+0.0040004	test-merror:0.388873+0.00514859
# [7]	train-merror:0.385036+0.00416564	test-merror:0.385884+0.00509722
# [8]	train-merror:0.381862+0.00507787	test-merror:0.383027+0.00583222
# [9]	train-merror:0.379536+0.00366889	test-merror:0.380871+0.00394551
# [10]	train-merror:0.377502+0.00452504	test-merror:0.379335+0.00365115
# [11]	train-merror:0.375121+0.00407585	test-merror:0.377039+0.00248762
# [12]	train-merror:0.372553+0.00473887	test-merror:0.375268+0.00249338
# [13]	train-merror:0.371378+0.00468253	test-merror:0.374073+0.00279706
# [14]	train-merror:0.370649+0.0046527	test-merror:0.373511+0.00270941
# [15]	train-merror:0.369+0.00325247	test-merror:0.372349+0.00119868
# [16]	train-merror:0.368002+0.00351359	test-merror:0.371366+0.00124609
# [17]	train-merror:0.366406+0.00317278	test-merror:0.37048+0.00145958
# [18]	train-merror:0.365491+0.00213682	test-merror:0.369464+0.00109974
# [19]	train-merror:0.364714+0.00176966	test-merror:0.368714+0.0017796
# [20]	train-merror:0.364264+0.0017978	test-merror:0.368311+0.00146618
# [21]	train-merror:0.363847+0.00180148	test-merror:0.368044+0.00173311
# [22]	train-merror:0.363585+0.00188765	test-merror:0.367651+0.0015153
# [23]	train-merror:0.363245+0.00175362	test-merror:0.367112+0.00128719
# [24]	train-merror:0.362954+0.0016427	test-merror:0.36699+0.00117697
# [25]	train-merror:0.36284+0.00155391	test-merror:0.36699+0.00125668
# [26]	train-merror:0.362441+0.00140342	test-merror:0.366854+0.00132489
# [27]	train-merror:0.362113+0.00116373	test-merror:0.36662+0.00150714
# [28]	train-merror:0.361672+0.00104634	test-merror:0.366128+0.00168629
# [29]	train-merror:0.361498+0.00101835	test-merror:0.366259+0.00162655
# Stopping. Best iteration:
# [28]	train-merror:0.361672+0.00104634	test-merror:0.366128+0.00168629


# ----------------------------------------------- #
# モデル構築
# ----------------------------------------------- #
print('モデル構築')
# Best iterationに従ってモデル構築する
num_boost_round = res['test-merror-mean'].idxmin()
print("交差検証の結果に従って{0}回でモデル構築を行う".format(num_boost_round))
clf = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_boost_round)

# ----------------------------------------------- #
# XGBoostの変数重要度の確認
# ----------------------------------------------- #
print('XGBoostの変数重要度の計算')
importance = clf.get_fscore()
importance_df = pd.DataFrame(
    sorted(importance.items(), key=operator.itemgetter(1)),
    columns=['feature', 'fscore']
)
importance_df = importance_df.iloc[-20:, :]

plt.figure()
importance_df.plot(kind='barh', x='feature', y='fscore',
                   legend=False, figsize=(30, 12))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
plt.gcf().savefig('feature_importance.png')

# ----------------------------------------------- #
# 予測
# ----------------------------------------------- #
print('XGBoostによる予測')
dtest = xgb.DMatrix(X_test)
y_pred = clf.predict(dtest).reshape(n_test, n_labels)

# ----------------------------------------------- #
# 提出ファイルの作成
# ----------------------------------------------- #
print('提出ファイルの作成')
ids = []
countries = []
for i in range(len(test_id)):
    idx = test_id[i]
    ids += [idx] * 5
    countries += le.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()

submission = pd.DataFrame(np.column_stack((ids, countries)),
                          columns=['id', 'country'])
submission.to_csv('submission.csv', index=False)
print('終了')
