import datetime
import math
import numpy as np
import pandas as pd
import sklearn as sk
import xgboost as xgb

from sklearn import ensemble
from sklearn import preprocessing
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import mean_squared_error

DATE_DELTA=5
K=10
NUM_TREES=1000
MAX_FEATS="sqrt"
NUM_TREES_CV=100
ONE_HOT_ENCODING=False
CROSS_VALIDATE=False

def split_date(df):
    df["year"]  = df.apply(lambda row: math.floor(row["quote_date"].year/DATE_DELTA)*DATE_DELTA, axis=1)
    df["month"] = df.apply(lambda row: row["quote_date"].month, axis=1)
    df["day"]   = df.apply(lambda row: row["quote_date"].day, axis=1)
    df["dow"]   = df.apply(lambda row: row["quote_date"].dayofweek, axis=1)
    return df.drop("quote_date", axis=1)

def filter_field(df, field, filtered):
    df[field] = df.apply(lambda row: row[field] if row[field] in filtered else float('NaN'), axis=1)

def remove_rare_events(train, test, field, K):
    data = train[field].values
    hist = {}
    for val in data: hist[val] = hist.get(val,0) + 1
    filtered = set( [ key for key,val in hist.items() if val > K ] )
    filter_field(train, field, filtered)
    filter_field(test, field, filtered)

def load_data():
    path = "../input"
    # load training and test datasets
    train = pd.read_csv(path + "/train_set.csv", parse_dates=[2], na_values="NONE")
    test = pd.read_csv(path + "/test_set.csv", parse_dates=[3], na_values="NONE")
    tube = pd.read_csv(path + "/tube.csv", na_values="NONE")
    tube.drop(["num_boss", "num_bracket", "other"], axis=1)
    
    # create some new features
    train = split_date(train)
    test  = split_date(test)

    train = pd.merge(train, tube, on="tube_assembly_id", how="left")
    test  = pd.merge(test, tube, on="tube_assembly_id", how="left")
    
    remove_rare_events(train, test, "supplier", K)
    remove_rare_events(train, test, "material_id", K)
    remove_rare_events(train, test, "end_a", K)
    remove_rare_events(train, test, "end_x", K)

    train_costs = np.log1p( train.cost.values )
    test_ids = test.id.values.astype(int)
    labels = train.cost.values

    test = test.drop(["id", "tube_assembly_id"], axis=1)
    train = train.drop(["tube_assembly_id", "cost"], axis=1)

    train = train.fillna("NA")
    test  = test.fillna("NA")

    for field in ["supplier", "year", "month", "day", "dow",
                  "material_id", "end_a", "end_x",
                  "end_a_1x", "end_a_2x", "end_x_1x",
                  "end_x_2x", "bracket_pricing"]:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[field].values))
        train[field] = lbl.transform(train[field].values)
        test[field]  = lbl.transform(test[field].values)

    if ONE_HOT_ENCODING:
        for field in ["supplier", "year", "month", "day", "dow",
                      "material_id", "end_a", "end_x"]:
            train_values = train[field].values
            test_values  = test[field].values
            lbl = preprocessing.OneHotEncoder()
            lbl.fit(np.resize(np.array(train_values).astype(float), (len(train_values),1)))
            train_values = lbl.transform(np.resize(np.array(train_values).astype(float), (len(train_values),1))).toarray()
            test_values  = lbl.transform(np.resize(np.array(test_values).astype(float), (len(test_values),1))).toarray()
            for i in range(train_values.shape[1]):
                train[field + "_" + str(i)] = train_values[:,i]
                test[field + "_" + str(i)]  = test_values[:,i]
            train = train.drop([ field ], axis=1)
            test  = test.drop([ field ], axis=1)

    for field in ["annual_usage", "min_order_quantity", "quantity",
                  "diameter", "wall", "length", "num_bends", "bend_radius"]:
        train[field] = np.log1p( train[field].values )
        test[field]  = np.log1p( test[field].values )

    train = np.array(train).astype(float)
    test  = np.array(test).astype(float)
    
    return (train, train_costs, test, test_ids, labels)

train,train_costs,test,test_ids,labels = load_data()

params = {}
params["objective"] = "reg:linear"
params["eta"] = 0.02
params["min_child_weight"] = 6
params["subsample"] = 0.7
params["colsample_bytree"] = 0.6
params["scale_pos_weight"] = 0.8
params["silent"] = 1
params["max_depth"] = 8
params["max_delta_step"]=2

label_log = np.log1p(labels)

plst = list(params.items())

xgtrain = xgb.DMatrix(train, label=label_log)
xgtest = xgb.DMatrix(test)

print('1500')


num_rounds = 1500
model = xgb.train(plst, xgtrain, num_rounds)
preds1 = model.predict(xgtest)

print('3000')

num_rounds = 3000
model = xgb.train(plst, xgtrain, num_rounds)
preds2 = model.predict(xgtest)

print('4000')

num_rounds = 4000
model = xgb.train(plst, xgtrain, num_rounds)
preds4 = model.predict(xgtest)

label_log = np.power(labels,1.0/16.0)

xgtrain = xgb.DMatrix(train, label=label_log)
xgtest = xgb.DMatrix(test)

print('power 1/16 4000')

num_rounds = 4000
model = xgb.train(plst, xgtrain, num_rounds)
preds3 = model.predict(xgtest)

preds = 0.4*np.expm1(preds4)+.1*np.expm1(preds1)+0.1*np.expm1(preds2)+0.4*np.power(preds3,16)


preds = pd.DataFrame({"id": test_ids, "cost": preds})
preds.to_csv('benchmark.csv', index=False)