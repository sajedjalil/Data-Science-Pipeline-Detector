import multiprocessing
from datetime import date, timedelta,datetime

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import os
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from scipy.stats import hmean,skew,kurtosis
import gc


df_train = pd.read_csv(
    '../input/train.csv', usecols=[1, 2, 3, 4, 5],
    dtype={'onpromotion': bool},
    # converters={'unit_sales': lambda u: np.log1p(
    #     float(u)) if float(u) > 0 else 0},
    parse_dates=["date"],
    skiprows=range(1, 66458909)  # 2016-01-01
)

df_test = pd.read_csv(
    "../input/test.csv", usecols=[0, 1, 2, 3, 4],
    dtype={'onpromotion': bool},
    parse_dates=["date"]  # , date_parser=parser
).set_index(
    ['store_nbr', 'item_nbr', 'date']
)

df_items = pd.read_csv(
    "../input/items.csv",dtype={'item_nbr':'int32','class':'int16','perishable':'int8'}
).set_index("item_nbr")
items=df_items.copy()
df_stores=pd.read_csv("../input/stores.csv")

# df_2017 = df_train[df_train.date.isin(
#     pd.date_range("2017-05-31", periods=7 * 11))].copy()
df_2017 = df_train.loc[df_train.date>=pd.datetime(2017,1,1)]
df_2017.loc[(df_2017.unit_sales<0),'unit_sales'] = 0 # eliminate negatives
df_2017['unit_sales'] =  df_2017['unit_sales'].apply(pd.np.log1p) #logarithm conversion
del df_train

promo_2017_train = df_2017.set_index(
    ["store_nbr", "item_nbr", "date"])[["onpromotion"]].unstack(
        level=-1).fillna(False)

promo_2017_train.columns = promo_2017_train.columns.get_level_values(1)
promo_2017_test = df_test[["onpromotion"]].unstack(level=-1).fillna(False)
promo_2017_test.columns = promo_2017_test.columns.get_level_values(1)
promo_2017_test = promo_2017_test.reindex(promo_2017_train.index).fillna(False)
promo_2017 = pd.concat([promo_2017_train, promo_2017_test], axis=1)
del promo_2017_test, promo_2017_train
df_2017_copy=df_2017.copy()

df_2017 = df_2017.set_index(
    ["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack(
        level=-1).fillna(0)
df_2017.columns = df_2017.columns.get_level_values(1)
df_2017_index=df_2017.index

items = items.reindex(df_2017.index.get_level_values(1))
stores_items = pd.DataFrame(index=df_2017.index)
df_2017=df_2017.reset_index()

df_test_forsub=df_test.copy()
# 确定每个商铺中每种商品的clsss的销量和促销情况
# 将商品信息merge
df_2017_copy=pd.merge(df_2017_copy,df_items[['class']],left_on=['item_nbr'],right_index=True)
df_2017_copy['onpromotion_mark']=df_2017_copy['onpromotion'].astype(np.uint8)
class_info=df_2017_copy.groupby(['date','store_nbr','class'])[['unit_sales','onpromotion_mark']].sum().reset_index()
class_info=class_info.rename(columns={'unit_sales': 'class_sales','onpromotion_mark':'class_onpromotion'})
df_test=df_test.reset_index()
df_test['onpromotion_mark']=df_test['onpromotion'].astype(np.uint8)
df_test=pd.merge(df_test,df_items[['class']],left_on=['item_nbr'],right_index=True)
class_info_test=df_test.groupby(['date','store_nbr','class'])[['onpromotion_mark']].sum().reset_index()
class_info_test=class_info_test.rename(columns={'onpromotion_mark':'class_onpromotion'})
promo_class_train = class_info.set_index(
    ["store_nbr", "class", "date"])[["class_onpromotion"]].unstack(
        level=-1).fillna(False)
promo_class_train.columns = promo_class_train.columns.get_level_values(1)
class_info_test=class_info_test.set_index(
    ['store_nbr', 'class', 'date']
)
promo_class_test = class_info_test[["class_onpromotion"]].unstack(level=-1).fillna(False)
promo_class_test.columns = promo_class_test.columns.get_level_values(1)
promo_class_test = promo_class_test.reindex(promo_class_train.index).fillna(0)
promo_class = pd.concat([promo_class_train, promo_class_test], axis=1)
del promo_class_test, promo_class_train

class_sales_info = class_info.set_index(
    ["store_nbr", "class", "date"])[["class_sales"]].unstack(
        level=-1).fillna(0)
class_sales_info.columns = class_sales_info.columns.get_level_values(1)

lbl=LabelEncoder()
lbl.fit(df_items['family'])
df_items['family']=lbl.transform(df_items['family']).astype(np.int8)
# 将每个class的商品数量merge上去
some=df_items.groupby(['class'])['perishable'].count().to_frame("class_items_number").reset_index()
df_items=df_items.reset_index()
df_items=pd.merge(df_items,some,on=['class'],how='left')
df_items=df_items.set_index("item_nbr")
df_2017=pd.merge(df_2017,df_items[['family','class','perishable','class_items_number']],how='left',left_on=['item_nbr'],right_index=True)

def get_timespan(df, dt, minus, periods, freq='D'):
    return df[pd.date_range(dt - timedelta(days=minus), periods=periods, freq=freq)]

def prepare_dataset(t2017, pool=True):
    df_class_sales = pd.DataFrame({
        "mean_3_class": get_timespan(class_sales_info, t2017, 3, 3).mean(axis=1).values,
        "mean_7_class": get_timespan(class_sales_info, t2017, 7, 7).mean(axis=1).values,
        "mean_14_class": get_timespan(class_sales_info, t2017, 14, 14).mean(axis=1).values,
        "mean_28_class": get_timespan(class_sales_info, t2017, 28, 28).mean(axis=1).values,
        "mean_63_class": get_timespan(class_sales_info, t2017, 63, 63).mean(axis=1).values,
        "promo_14_class": get_timespan(promo_class, t2017, 14, 14).sum(axis=1).values,
        "promo_28_class": get_timespan(promo_class, t2017, 28, 28).sum(axis=1).values,
        "promo_63_class": get_timespan(promo_class, t2017, 63, 63).sum(axis=1).values,
    })
    for i in range(16):
        df_class_sales["promo_class_day_{}".format(i)] = promo_class[t2017 + timedelta(days=i)].values.astype(np.uint8)
    df_class_sales.index = class_sales_info.index
    # 还应该加入的特征，促销的数量除以该class总数的比例， 这16天中 是否促销比促销的class商品数量

    df_sales_140 = df_2017_copy.loc[(df_2017_copy['date'] >= t2017 - timedelta(140)) & (df_2017_copy['date'] < t2017)]
    df_sales_63 = df_2017_copy.loc[(df_2017_copy['date'] >= t2017 - timedelta(63)) & (df_2017_copy['date'] < t2017)]
    df_sales_28 = df_2017_copy.loc[(df_2017_copy['date'] >= t2017 - timedelta(28)) & (df_2017_copy['date'] < t2017)]
    promo_sales_mean_140 = \
    df_sales_140.loc[df_sales_140['onpromotion'] == True, ['store_nbr', 'item_nbr', 'unit_sales']].groupby(
        ['store_nbr', 'item_nbr'])['unit_sales'].mean().to_frame("promo_mean_140").reset_index()
    norm_sales_mean_140 = \
    df_sales_140.loc[df_sales_140['onpromotion'] == False, ['store_nbr', 'item_nbr', 'unit_sales']].groupby(
        ['store_nbr', 'item_nbr'])['unit_sales'].mean().to_frame("norm_mean_140").reset_index()
    promo_sales_mean_63 = \
    df_sales_63.loc[df_sales_63['onpromotion'] == True, ['store_nbr', 'item_nbr', 'unit_sales']].groupby(
        ['store_nbr', 'item_nbr'])['unit_sales'].mean().to_frame("promo_mean_63").reset_index()
    norm_sales_mean_63 = \
    df_sales_63.loc[df_sales_63['onpromotion'] == False, ['store_nbr', 'item_nbr', 'unit_sales']].groupby(
        ['store_nbr', 'item_nbr'])['unit_sales'].mean().to_frame("norm_mean_63").reset_index()
    promo_sales_mean_28 = \
        df_sales_28.loc[df_sales_28['onpromotion'] == True, ['store_nbr', 'item_nbr', 'unit_sales']].groupby(
            ['store_nbr', 'item_nbr'])['unit_sales'].mean().to_frame("promo_mean_28").reset_index()
    norm_sales_mean_28 = \
        df_sales_28.loc[df_sales_28['onpromotion'] == False, ['store_nbr', 'item_nbr', 'unit_sales']].groupby(
            ['store_nbr', 'item_nbr'])['unit_sales'].mean().to_frame("norm_mean_28").reset_index()
    del df_sales_140, df_sales_63, df_sales_28;
    gc.collect()

    X = pd.DataFrame({
        'family': df_2017['family'].values,
        'class': df_2017['class'].values,
        'perishable': df_2017['perishable'].values,
        'class_items_number': df_2017['class_items_number'].values,
        'store_nbr': df_2017['store_nbr'].values,
        'item_nbr': df_2017['item_nbr'].values,
        "day_1_2017": get_timespan(df_2017, t2017, 1, 1).values.ravel(),
        "mean_3_2017": get_timespan(df_2017, t2017, 3, 3).mean(axis=1).values,
        "mean_7_2017": get_timespan(df_2017, t2017, 7, 7).mean(axis=1).values,
        "mean_14_2017": get_timespan(df_2017, t2017, 14, 14).mean(axis=1).values,
        "mean_28_2017": get_timespan(df_2017, t2017, 28, 28).mean(axis=1).values,
        "mean_63_2017": get_timespan(df_2017, t2017, 63, 63).mean(axis=1).values,
        "mean_140_2017": get_timespan(df_2017, t2017, 140, 140).mean(axis=1).values,
        "median_3_2017": get_timespan(df_2017, t2017, 3, 3).median(axis=1).values,
        "median_7_2017": get_timespan(df_2017, t2017, 7, 7).median(axis=1).values,
        "median_14_2017": get_timespan(df_2017, t2017, 14, 14).median(axis=1).values,
        "median_28_2017": get_timespan(df_2017, t2017, 28, 28).median(axis=1).values,
        "median_63_2017": get_timespan(df_2017, t2017, 63, 63).median(axis=1).values,
        "median_140_2017": get_timespan(df_2017, t2017, 140, 140).median(axis=1).values,
        "var_3_2017": get_timespan(df_2017, t2017, 3, 3).var(axis=1).values,
        "var_7_2017": get_timespan(df_2017, t2017, 7, 7).var(axis=1).values,
        "var_14_2017": get_timespan(df_2017, t2017, 14, 14).var(axis=1).values,
        "var_28_2017": get_timespan(df_2017, t2017, 28, 28).var(axis=1).values,
        "var_63_2017": get_timespan(df_2017, t2017, 63, 63).var(axis=1).values,
        "var_140_2017": get_timespan(df_2017, t2017, 140, 140).var(axis=1).values,
        "hmean_3_2017": (get_timespan(df_2017, t2017, 3, 3) + 1).apply(hmean, axis=1),
        "hmean_7_2017": (get_timespan(df_2017, t2017, 7, 7) + 1).apply(hmean, axis=1),
        "hmean_14_2017": (get_timespan(df_2017, t2017, 14, 14) + 1).apply(hmean, axis=1),
        "hmean_28_2017": (get_timespan(df_2017, t2017, 28, 28) + 1).apply(hmean, axis=1),
        "hmean_63_2017": (get_timespan(df_2017, t2017, 63, 63) + 1).apply(hmean, axis=1),
        "hmean_140_2017": (get_timespan(df_2017, t2017, 140, 140) + 1).apply(hmean, axis=1),

        "skew_7_2017": get_timespan(df_2017, t2017, 7, 7).apply(skew, axis=1),
        "skew_14_2017": get_timespan(df_2017, t2017, 14, 14).apply(skew, axis=1),
        "skew_28_2017": get_timespan(df_2017, t2017, 28, 28).apply(skew, axis=1),
        "skew_63_2017": get_timespan(df_2017, t2017, 63, 63).apply(skew, axis=1),
        "skew_140_2017": get_timespan(df_2017, t2017, 140, 140).apply(skew, axis=1),

        "kurtosis_7_2017": get_timespan(df_2017, t2017, 7, 7).apply(kurtosis, axis=1),
        "kurtosis_14_2017": get_timespan(df_2017, t2017, 14, 14).apply(kurtosis, axis=1),
        "kurtosis_28_2017": get_timespan(df_2017, t2017, 28, 28).apply(kurtosis, axis=1),
        "kurtosis_63_2017": get_timespan(df_2017, t2017, 63, 63).apply(kurtosis, axis=1),
        "kurtosis_140_2017": get_timespan(df_2017, t2017, 140, 140).apply(kurtosis, axis=1),

        "min_7_2017": get_timespan(df_2017, t2017, 7, 7).min(axis=1).values,
        "min_14_2017": get_timespan(df_2017, t2017, 14, 14).min(axis=1).values,
        "min_28_2017": get_timespan(df_2017, t2017, 28, 28).min(axis=1).values,
        "max_7_2017": get_timespan(df_2017, t2017, 7, 7).max(axis=1).values,
        "max_14_2017": get_timespan(df_2017, t2017, 14, 14).max(axis=1).values,
        "max_28_2017": get_timespan(df_2017, t2017, 28, 28).max(axis=1).values,

        'nosales_3days': get_timespan(df_2017, t2017, 3, 3).apply(lambda x: list(x).count(0), axis=1).values,
        'nosales_7days': get_timespan(df_2017, t2017, 7, 7).apply(lambda x: list(x).count(0), axis=1).values,
        'nosales_14days': get_timespan(df_2017, t2017, 14, 14).apply(lambda x: list(x).count(0), axis=1).values,
        'nosales_28days': get_timespan(df_2017, t2017, 28, 28).apply(lambda x: list(x).count(0), axis=1).values,
        'nosales_63days': get_timespan(df_2017, t2017, 63, 63).apply(lambda x: list(x).count(0), axis=1).values,
        'nosales_140days': get_timespan(df_2017, t2017, 140, 140).apply(lambda x: list(x).count(0), axis=1).values,

        "promo_14_2017": get_timespan(promo_2017, t2017, 14, 14).sum(axis=1).values,
        "promo_63_2017": get_timespan(promo_2017, t2017, 63, 63).sum(axis=1).values,
        "promo_140_2017": get_timespan(promo_2017, t2017, 140, 140).sum(axis=1).values,
        "unpromo_16aftsum_2017": (1 - get_timespan(promo_2017, t2017 + timedelta(16), 11, 11)).sum(axis=1).values
    })

    for i in range(7):
        X['mean_4_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 28 - i, 4, freq='7D').mean(axis=1).values
        X['min_4_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 28 - i, 4, freq='7D').min(axis=1).values
        X['max_4_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 28 - i, 4, freq='7D').max(axis=1).values
        X['mean_20_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 140 - i, 20, freq='7D').mean(axis=1).values
        X['min_20_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 140 - i, 20, freq='7D').min(axis=1).values
        X['max_20_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 140 - i, 20, freq='7D').max(axis=1).values

        X['nosales_4_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 28 - i, 4, freq='7D').apply(
            lambda x: list(x).count(0), axis=1).values
        # X['nosales_8_dow{}_2017'.format(i)]=get_timespan(df_2017, t2017, 56-i, 8, freq='7D').apply(lambda x: list(x).count(0), axis=1).values
        X['nosales_20_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 140 - i, 20, freq='7D').apply(
            lambda x: list(x).count(0), axis=1).values

        # 每个曜日所占的比例
        X['ratio_4_dow{}_2017'.format(i)] = X['mean_4_dow{}_2017'.format(i)] / X['mean_28_2017']

    for i in range(16):
        X["promo_{}".format(i)] = promo_2017[t2017 + timedelta(days=i)].values.astype(np.uint8)
    X['diff_7_14_sales'] = X['mean_7_2017'] - X['mean_14_2017']
    X['diff_14_28_sales'] = X['mean_14_2017'] - X['mean_28_2017']
    X['diff_28_63_sales'] = X['mean_28_2017'] - X['mean_63_2017']

    X = pd.merge(X, promo_sales_mean_140, on=['store_nbr', 'item_nbr'], how='left')
    X = pd.merge(X, norm_sales_mean_140, on=['store_nbr', 'item_nbr'], how='left')
    X['promo_scale_140'] = X['promo_mean_140'] / (X['norm_mean_140'] + 1e-6)
    X['promo_mean_diff_140'] = X['promo_mean_140'] - X['norm_mean_140']

    X = pd.merge(X, promo_sales_mean_63, on=['store_nbr', 'item_nbr'], how='left')
    X = pd.merge(X, norm_sales_mean_63, on=['store_nbr', 'item_nbr'], how='left')
    X['promo_scale_63'] = X['promo_mean_63'] / (X['norm_mean_63'] + 1e-6)
    X['promo_mean_diff_63'] = X['promo_mean_63'] - X['norm_mean_63']

    X = pd.merge(X, promo_sales_mean_28, on=['store_nbr', 'item_nbr'], how='left')
    X = pd.merge(X, norm_sales_mean_28, on=['store_nbr', 'item_nbr'], how='left')
    X['promo_scale_28'] = X['promo_mean_28'] / (X['norm_mean_28'] + 1e-6)
    X['promo_mean_diff_28'] = X['promo_mean_28'] - X['norm_mean_28']

    # df['promo_scale'].fillna(df['promo_scale'].median(), inplace=True)
    X.loc[X['promo_scale_140'] < 1, 'promo_scale_140'] = 1
    X.loc[X['promo_scale_63'] < 1, 'promo_scale_63'] = 1
    X.loc[X['promo_scale_28'] < 1, 'promo_scale_28'] = 1

    X = pd.merge(X, df_class_sales, left_on=['store_nbr', 'class'], right_index=True, how='left')
    X['promo_14_class_ratio'] = X['promo_14_class'] / X['class_items_number']
    X['promo_28_class_ratio'] = X['promo_28_class'] / X['class_items_number']
    X['promo_63_class_ratio'] = X['promo_63_class'] / X['class_items_number']

    for i in range(16):
        X["promo_class_day_{}_ratio".format(i)] = X["promo_{}".format(i)] / X["promo_class_day_{}".format(i)]

    if pool==False:
        # y = df_2017[
        #     pd.date_range(t2017, periods=16)
        # ].values
        return X
    y = df_2017[
        pd.date_range(t2017, periods=16)
    ].values
    X_l.append(X)
    y_l.append(y)
    print(t2017)

if __name__ == '__main__':
    print("Preparing dataset...")
    t2017 = date(2017, 6, 7)
    manager = multiprocessing.Manager()
    # X_l, y_l = [], []
    X_l = manager.list()
    y_l = manager.list()
    p = multiprocessing.Pool(8)
    for i in range(8):
        delta = timedelta(days=7 * i)
        # print(t2017 + delta)
        p.apply_async(prepare_dataset, args=(t2017 + delta,))
    p.close()
    p.join()
    X_l = list(X_l)
    y_l = list(y_l)
    X_train = pd.concat(X_l, axis=0)
    y_train = np.concatenate(y_l, axis=0)
    del X_l, y_l

    X_test = prepare_dataset(date(2017, 8, 16), pool=False)
    gc.collect()

    print("Training and predicting models...")
    params = {
        'num_leaves': 2**5 - 1,
        'objective': 'regression',
        'max_depth': 100,
        'min_data_in_leaf': 200,
        # 'learning_rate': 0.1,
        'learning_rate': 0.02,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 2,
        'metric': 'l2_root',
        'verbose':-1
    }

    MAX_ROUNDS = 3000
    val_pred = []
    test_pred = []
    cate_vars = []
    for i in range(16):
        print("=" * 50)
        print("Step %d" % (i+1))
        print("=" * 50)
        dtrain = lgb.Dataset(
            X_train, label=y_train[:, i],
            categorical_feature=cate_vars,
            weight= pd.concat([items["perishable"]] * 8) * 0.25 + 1
        )
        bst = lgb.train(
            params, dtrain, num_boost_round=MAX_ROUNDS,
            valid_sets=[dtrain], early_stopping_rounds=50, verbose_eval=100
        )
        # print("\n".join(("%s: %.2f" % x) for x in sorted(
        #     zip(X_train.columns, bst.feature_importance("gain")),
        #     key=lambda x: x[1], reverse=True
        # )))
        test_pred.append(bst.predict(
            X_test, num_iteration=bst.best_iteration or MAX_ROUNDS))

    print("Making submission...")
    y_test = np.array(test_pred).transpose()
    df_preds = pd.DataFrame(
        y_test, index=df_2017_index,
        columns=pd.date_range("2017-08-16", periods=16)
    ).stack().to_frame("unit_sales")
    df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)

    submission = df_test_forsub[["id"]].join(df_preds, how="left").fillna(0)
    
    submission["unit_sales"] = np.clip(np.expm1(submission["unit_sales"]), 0, 1000)
    submission.to_csv('lgb{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), float_format='%.4f', index=None)