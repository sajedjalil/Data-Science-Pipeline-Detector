# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import KFold
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression




# -----------------------------------
# データの読み込み
# -----------------------------------
# 学習データ、テストデータの読み込み
train = pd.read_csv("/kaggle/input/spaceship-titanic/train.csv")
test = pd.read_csv("/kaggle/input/spaceship-titanic/test.csv")

pd.set_option('display.max_rows', None)

#show head
# print(train.head())

#size
# print(train.shape)
#統計量，平均値，標準偏差，最大最小など
print(train.describe())
print(train.describe(include=['O']))#数値以外のデータ
#データ型
print(train.dtypes)
#欠損値
print(train.isnull().sum())

# 学習データを特徴量と目的変数に分ける
train_x = train.drop(['Transported'], axis=1)
train_y = train['Transported']

test_x = test.copy()



# -----------------------------------
# 特徴量作成
# -----------------------------------
# 変数PassengerIdを除外する
train_x = train_x.drop(['PassengerId', 'Name', 'Destination'], axis=1)
test_x = test_x.drop(['PassengerId', 'Name', 'Destination'], axis=1)


#Sのとき0，Pの時1
# where関数　https://note.nkmk.me/python-pandas-where-mask/
train_x['Cabin'] = train_x['Cabin'].where(train_x['Cabin'].str[-1:] == 'S',1)
train_x['Cabin'] = train_x['Cabin'].where(train_x['Cabin'] == 1,0)
test_x['Cabin'] = test_x['Cabin'].where(test_x['Cabin'].str[-1:] == 'S',1)
test_x['Cabin'] = test_x['Cabin'].where(test_x['Cabin'] == 1,0)
train_x['Cabin'] = train_x['Cabin'].astype(int)
test_x['Cabin'] = train_x['Cabin'].astype(int)

# 文字列->数値に変換する
for c in ['HomePlanet']:
    le = LabelEncoder()
    le.fit(train_x[c].fillna('NA'))

    # 学習データ、テストデータを変換する
    train_x[c] = le.transform(train_x[c].fillna('NA'))
    test_x[c] = le.transform(test_x[c].fillna('NA'))

# object to boolean
train_x['CryoSleep'] = train_x['CryoSleep'].astype(bool)
train_x['VIP'] = train_x['VIP'].astype(bool)
test_x['CryoSleep'] = test_x['CryoSleep'].astype(bool)
test_x['VIP'] = test_x['VIP'].astype(bool)



# -----------------------------------
# 標準化
# -----------------------------------

# 変換する数値変数をリストに格納
num_cols = ['Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']

# 学習データに基づいて複数列の標準化を定義
scaler = StandardScaler()
scaler.fit(train_x[num_cols])

# 変換後のデータで各列を置換
train_x[num_cols] = scaler.transform(train_x[num_cols])
test_x[num_cols] = scaler.transform(test_x[num_cols])



# -----------------------------------
# モデル作成
# -----------------------------------
print(train_x)
dtrain = xgb.DMatrix(train_x, label=train_y)
dtest = xgb.DMatrix(test_x)

#パラーメータの設定
params = {
        'objective': 'reg:squarederror', 'random_state':1234, 
        # 学習用の指標 (RMSE)
        'eval_metric': 'rmse',
    }

#学習
model = xgb.train(params,
                    dtrain,#訓練データ
                    100#設定した学習回数
                    )
#フィッティング
pred_xgb = model.predict(dtest, ntree_limit = model.best_ntree_limit)

#boolにする
# pred_xgb = np.where(pred > 0.5, 1, 0).astype(bool)
# #csvに出力
# submission = pd.DataFrame({"PassengerId": test['PassengerId'], "Transported": prediction_xgb})
# submission.to_csv('submission_first.csv', index=False)



# -----------------------------------
# ロジスティック回帰用の特徴量の作成
# -----------------------------------
# 元データをコピーする
train_x2 = train.drop(['Transported'], axis=1)
test_x2 = test.copy()

# 変数PassengerIdを除外する
train_x2 = train_x2.drop(['PassengerId', 'Name', 'Destination','Cabin'], axis=1)
test_x2 = test_x2.drop(['PassengerId', 'Name', 'Destination','Cabin'], axis=1)

# 文字列->数値に変換する
for c in ['HomePlanet']:
    le = LabelEncoder()
    le.fit(train_x2[c].fillna('NA'))

    # 学習データ、テストデータを変換する
    train_x2[c] = le.transform(train_x2[c].fillna('NA'))
    test_x2[c] = le.transform(test_x2[c].fillna('NA'))


# 数値変数の欠損値を学習データの平均で埋める
num_cols = ['Age']
for col in num_cols:
    train_x2[col].fillna(train_x2[col].mean(), inplace=True)
    test_x2[col].fillna(train_x2[col].mean(), inplace=True)


# 数値変数の欠損値を0で埋める
num_cols = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','CryoSleep','VIP']
for col in num_cols:
    train_x2[col].fillna(0, inplace=True)
    test_x2[col].fillna(0, inplace=True)

# object to boolean
train_x2['CryoSleep'] = train_x2['CryoSleep'].astype(int)
train_x2['VIP'] = train_x2['VIP'].astype(int)
test_x2['CryoSleep'] = test_x2['CryoSleep'].astype(int)
test_x2['VIP'] = test_x2['VIP'].astype(int)


# 変換する数値変数をリストに格納
num_cols = ['Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']

# 学習データに基づいて複数列の標準化を定義
scaler = StandardScaler()
scaler.fit(train_x2[num_cols])

# 変換後のデータで各列を置換
train_x2[num_cols] = scaler.transform(train_x2[num_cols])
test_x2[num_cols] = scaler.transform(test_x2[num_cols])

# -----------------------------------
# アンサンブル
# -----------------------------------
# ロジスティック回帰モデル
# xgboostモデルとは異なる特徴量を入れる必要があるので、別途train_x2, test_x2を作成した
model_lr = LogisticRegression(solver='lbfgs', max_iter=300)
model_lr.fit(train_x2, train_y)
pred_lr = model_lr.predict_proba(test_x2)[:, 1]

# 予測値の加重平均をとる
pred = pred_xgb * 0.8 + pred_lr * 0.2
pred_label = np.where(pred > 0.5, 1, 0).astype(bool)

#csvに出力
submission = pd.DataFrame({"PassengerId": test['PassengerId'], "Transported": pred_label})
submission.to_csv('submission.csv', index=False)




# -----------------------------------
# バリデーション
# -----------------------------------
# 各foldのスコアを保存するリスト
scores_accuracy = []
scores_logloss = []

# クロスバリデーションを行う
# 学習データを4つに分割し、うち1つをバリデーションデータとすることを、バリデーションデータを変えて繰り返す
kf = KFold(n_splits=4, shuffle=True, random_state=71)
for tr_idx, va_idx in kf.split(train_x):
    # 学習データを学習データとバリデーションデータに分ける
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
    tr_x2, va_x2 = train_x2.iloc[tr_idx], train_x2.iloc[va_idx]

    dtrain = xgb.DMatrix(tr_x, label=tr_y)
    dtest = xgb.DMatrix(va_x, label=va_y)
    

    params = {
        'objective': 'reg:squarederror', 'random_state':1234, 
        # 学習用の指標 (RMSE)
        'eval_metric': 'rmse',
    }

    #XGBで学習
    model = xgb.train(params,
                        dtrain,#訓練データ
                        100#設定した学習回数
                        )
    #フィッティング
    pred_xgb = model.predict(dtest, ntree_limit = model.best_ntree_limit)
    

    # ロジスティック回帰モデル
    model_lr = LogisticRegression(solver='lbfgs', max_iter=300)
    model_lr.fit(tr_x2, tr_y)
    pred_lr = model_lr.predict_proba(va_x2)[:, 1]

    #確率にする
    va_pred = pred_xgb.astype(float) * 0.8 + pred_lr * 0.2

    # バリデーションデータでのスコアを計算する
    logloss = log_loss(va_y, va_pred)
    accuracy = accuracy_score(va_y, va_pred > 0.5)

    # そのfoldのスコアを保存する
    scores_logloss.append(logloss)
    scores_accuracy.append(accuracy)

# 各foldのスコアの平均を出力する
logloss = np.mean(scores_logloss)
accuracy = np.mean(scores_accuracy)
print(f'logloss: {logloss:.4f}, accuracy: {accuracy:.4f}')

