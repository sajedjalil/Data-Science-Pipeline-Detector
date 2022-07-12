import pandas as pd
import numpy as np
import xgboost as xgb
from matplotlib import pyplot as plt
import operator

train = pd.read_csv("../input/train.csv", encoding= "utf_8")
test = pd.read_csv("../input/test.csv", encoding= "utf_8")

first_feat = ["id","timestamp","price_doc", "full_sq", "life_sq",
              "floor", "max_floor", "material", "build_year", "num_room",
              "kitch_sq", "state", "product_type", "sub_area"]

bad_index = train[train.life_sq > train.full_sq].index
train.ix[bad_index, "life_sq"] = np.NaN

equal_index = [601,1896,2791]
test.ix[equal_index, "life_sq"] = test.ix[equal_index, "full_sq"]

bad_index = test[test.life_sq > test.full_sq].index
test.ix[bad_index, "life_sq"] = np.NaN

bad_index = train[train.life_sq < 5].index
train.ix[bad_index, "life_sq"] = np.NaN

bad_index = test[test.life_sq < 5].index
test.ix[bad_index, "life_sq"] = np.NaN

bad_index = train[train.full_sq < 5].index
train.ix[bad_index, "full_sq"] = np.NaN

bad_index = test[test.full_sq < 5].index
test.ix[bad_index, "full_sq"] = np.NaN

kitch_is_build_year = [13117]
train.ix[kitch_is_build_year, "build_year"] = train.ix[kitch_is_build_year, "kitch_sq"]

bad_index = train[train.kitch_sq >= train.life_sq].index
train.ix[bad_index, "kitch_sq"] = np.NaN

bad_index = test[test.kitch_sq >= test.life_sq].index
test.ix[bad_index, "kitch_sq"] = np.NaN

bad_index = train[(train.kitch_sq == 0).values + (train.kitch_sq == 1).values].index
train.ix[bad_index, "kitch_sq"] = np.NaN

bad_index = test[(test.kitch_sq == 0).values + (test.kitch_sq == 1).values].index
test.ix[bad_index, "kitch_sq"] = np.NaN

##
bad_index = train[(train.full_sq > 210) * (train.life_sq / train.full_sq < 0.3)].index
train.ix[bad_index, "full_sq"] = np.NaN

bad_index = test[(test.full_sq > 150) * (test.life_sq / test.full_sq < 0.3)].index
test.ix[bad_index, "full_sq"] = np.NaN

bad_index = train[train.life_sq > 300].index
train.ix[bad_index, ["life_sq", "full_sq"]] = np.NaN

bad_index = test[test.life_sq > 200].index
test.ix[bad_index, ["life_sq", "full_sq"]] = np.NaN

train.product_type.value_counts(normalize= True)

test.product_type.value_counts(normalize= True)

bad_index = train[train.build_year < 1500].index
train.ix[bad_index, "build_year"] = np.NaN

bad_index = test[test.build_year < 1500].index
test.ix[bad_index, "build_year"] = np.NaN

bad_index = train[train.num_room == 0].index
train.ix[bad_index, "num_room"] = np.NaN

bad_index = test[test.num_room == 0].index
test.ix[bad_index, "num_room"] = np.NaN

bad_index = [10076, 11621, 17764, 19390, 24007, 26713, 29172]
train.ix[bad_index, "num_room"] = np.NaN

bad_index = [3174, 7313]
test.ix[bad_index, "num_room"] = np.NaN

bad_index = train[(train.floor == 0).values * (train.max_floor == 0).values].index
train.ix[bad_index, ["max_floor", "floor"]] = np.NaN

bad_index = train[train.floor == 0].index
train.ix[bad_index, "floor"] = np.NaN

bad_index = train[train.max_floor == 0].index
train.ix[bad_index, "max_floor"] = np.NaN

bad_index = test[test.max_floor == 0].index
test.ix[bad_index, "max_floor"] = np.NaN

bad_index = train[train.floor > train.max_floor].index
train.ix[bad_index, "max_floor"] = np.NaN

bad_index = test[test.floor > test.max_floor].index
test.ix[bad_index, "max_floor"] = np.NaN

train.floor.describe(percentiles= [0.9999])

bad_index = [23584]
train.ix[bad_index, "floor"] = np.NaN

train.material.value_counts()

test.material.value_counts()

train.state.value_counts()

bad_index = train[train.state == 33].index
train.ix[bad_index, "state"] = np.NaN

test.state.value_counts()

#test.to_csv("../input/test_clean.csv", index= False, encoding= "utf_8")
#train.to_csv("../input/train_clean.csv", index = False, encoding= "utf_8")









def rmsle(pred, price):
    assert len(pred) == len(price)
    return np.sqrt(np.mean(np.power(np.log1p(pred) - np.log1p(price), 2)))



train_all = train       #pd.read_csv("input/train_clean.csv", encoding="utf_8")
test_final = test       #pd.read_csv("input/test_clean.csv", encoding="utf_8")
macro = pd.read_csv("../input/macro.csv", parse_dates=['timestamp'])

N = int(train_all.shape[0]*0.7)
test_id = train_all.loc[N:, 'id']
y_train_all = train_all.price_doc
train_all = train_all.merge(macro, on='timestamp', how='left')
train_all.drop(["id", "price_doc", "timestamp"], axis=1, inplace=True)


factorize = lambda t: pd.factorize(t[1])[0]

df_obj = train_all.select_dtypes(include=['object'])

X_all = np.c_[
    train_all.select_dtypes(exclude=['object']).values,
    np.array(list(map(factorize, df_obj.iteritems()))).T
]

# Deal with categorical values
df_numeric = train_all.select_dtypes(exclude=['object'])
df_obj = train_all.select_dtypes(include=['object']).copy()

for c in df_obj:
    df_obj[c] = pd.factorize(df_obj[c])[0]

df_values = pd.concat([df_numeric, df_obj], axis=1)


# Convert to numpy values
X_all = df_values.values
df_columns = df_values.columns


X_train = X_all[:N, :]
X_test = X_all[N:, :]

y_train = y_train_all[:N]
y_test = y_train_all[N:]

xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'min_child_weight':1,
    'silent': 1
}
num_boost_round = 489

dtrain = xgb.DMatrix(X_train, y_train, feature_names=df_columns)
dtest = xgb.DMatrix(X_test, feature_names=df_columns)

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_round)
model.get_fscore()
#fig, ax = plt.subplots(1, 1, figsize=(8, 16))
#xgb.plot_importance(model, height=0.5, ax=ax)
#plt.show()


y_pred = model.predict(dtest)

RMSLE = rmsle(y_pred, y_test)
print("RMSLE: ", RMSLE)

############### PCA - doesn't change much
'''
importance = model.get_fscore()
importance = sorted(importance.items(), key=operator.itemgetter(1))

df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()

df_thresh = df.sort_values(by=['fscore'])
thresholds = np.sort(df.fscore)
N = len(thresholds)
errors = []
all_features = list(train_all.columns.values)
for i in range(1, int(N/4)):
    print('step: ', i, ' from: ', len(thresholds))
    thresh = thresholds[i]
    # select features using threshold

    selected_features = df_thresh[df_thresh['fscore']>thresh].feature
    feat_idx = []
    for feat in list(selected_features):
        feat_idx.append(all_features.index(feat))

    select_X_train = X_train[:,feat_idx]
    select_X_test = X_test[:, feat_idx]

    #selection = SelectFromModel(model, threshold=thresh, prefit=True)
    #select_X_train = selection.transform(X_train)
    #  train model
    dtrain = xgb.DMatrix(select_X_train, y_train, feature_names=list(selected_features))
    selection_model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_round)
    # eval model
    dtest = xgb.DMatrix(select_X_test, feature_names=list(selected_features))
    y_pred = selection_model.predict(dtest)
    err = rmsle(y_test, y_pred)
    print("Thresh=%.3f, n=%d, Err: %.5f" % (thresh, select_X_train.shape[1], err))
    print('\n'*5)
    errors.append(err)

plt.plot(range(int(N/2), N), errors)
plt.xlabel('indeks odciecia zmiennych')
plt.ylabel('RMSLE')
plt.savefig('rmsle.png')'''

df_sub = pd.DataFrame({'id': test_id, 'price_doc': y_pred})

df_sub.to_csv('sub.csv', index=False)

