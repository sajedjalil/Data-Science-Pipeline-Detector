import datetime
import gc
import time
import warnings
from itertools import chain

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from kaggle.competitions import twosigmanews
from matplotlib import pyplot as plt
from sklearn.ensemble import IsolationForest, VotingClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")

# Initialize the environment
if "env" not in globals():
    env = twosigmanews.make_env()
# Retrieve the data
mkt, news = env.get_training_data()

def mkt_fillna(mkt):
    
    mkt["returnsClosePrevMktres1"].fillna(mkt["returnsClosePrevRaw1"], inplace=True)
    mkt["returnsOpenPrevMktres1"].fillna(mkt["returnsOpenPrevRaw1"], inplace=True)
    mkt["returnsClosePrevMktres10"].fillna(mkt["returnsClosePrevRaw10"], inplace=True)
    mkt["returnsOpenPrevMktres10"].fillna(mkt["returnsOpenPrevRaw10"], inplace=True)

    return mkt

mkt = mkt_fillna(mkt)
print("Missing values filled.")

log_ret = np.log(mkt["close"].values / mkt["open"].values)
outlier_idx = ((log_ret > 0.5).astype(int) + (log_ret < -0.5).astype(int)).astype(bool)
mkt = mkt.loc[~outlier_idx, :]

mkt = mkt.loc[mkt["assetName"] != "Unknown", :]

short_ret_cols = ['returnsClosePrevRaw1', 'returnsOpenPrevRaw1', 'returnsClosePrevMktres1', 'returnsOpenPrevMktres1']
long_ret_cols = ['returnsClosePrevRaw10', 'returnsOpenPrevRaw10', 'returnsClosePrevMktres10', 'returnsOpenPrevMktres10']

for col in short_ret_cols:
    mkt = mkt.loc[mkt[col].abs() < 1, :]

for col in long_ret_cols:
    mkt = mkt.loc[mkt[col].abs() < 2, :]

mkt = mkt.loc[mkt["time"].dt.date > datetime.date(2009, 1, 1)]
print("Outliers removed.")

del log_ret
del outlier_idx
gc.collect()

def add_mkt_features(mkt):
    
    mkt["time"] = mkt["time"].dt.date
    mkt.rename(columns={"time": "date"}, inplace=True)
    mkt["returnsToday"] = np.log(mkt["close"].values / mkt["open"].values)
    mkt["relVol"] = mkt.groupby(["date"])["volume"].transform(lambda x: (x - x.mean()) / x.std())
    
    return mkt

mkt = add_mkt_features(mkt)
print("New market features added.")

def unwrap_assets(asset_codes):

    news_idx = list(chain.from_iterable([[i] * len(eval(asset_codes.values[i])) for i in range(asset_codes.shape[0])]))
    asset_codes = list(chain.from_iterable([list(eval(val)) for val in asset_codes.values]))
    codes_unwrap = pd.DataFrame({"newsIndex": news_idx, "assetCode": asset_codes})
    del asset_codes
    del news_idx
    gc.collect()

    return codes_unwrap

def add_news_features(news):
    
    news["sourceTimestamp"] = news["sourceTimestamp"].dt.date
    news.rename(columns={"sourceTimestamp": "date"}, inplace=True)
    news["rel1stMentionPos"] = news["firstMentionSentence"].values / news["sentenceCount"].values
    news["relSentimentWord"] = news["sentimentWordCount"].values / news["wordCount"].values
    news["relSentCnt"] = news.groupby(["date"])["sentenceCount"].transform(lambda x: (x - x.mean()) / x.std())
    news["relWordCnt"] = news.groupby(["date"])["wordCount"].transform(lambda x: (x - x.mean()) / x.std())
    news["relBodySize"] = news.groupby(["date"])["bodySize"].transform(lambda x: (x - x.mean()) / x.std())
    news["assetCode"] = news["assetCodes"].map(lambda x: list(eval(x))[0])
    news.drop(["assetCodes"], axis=1, inplace=True)
    
    return news

# news["newsIndex"] = news.index
# news = news.merge(unwrap_assets(news["assetCodes"]), how="right", on=["newsIndex"])
# news.drop(["newsIndex", "assetCodes"], axis=1, inplace=True)
news = add_news_features(news)
print("New news features added.")

mkt.drop(["assetName", "volume", "close", "open"], axis=1, inplace=True)

news.drop(["time", "sourceId", "headline", "provider", "subjects",
            "audiences", "bodySize", "sentenceCount", "wordCount",
            "assetName", "firstMentionSentence", "sentimentWordCount",
            "headlineTag"], axis=1, inplace=True)
print("Useless features dropped.")
gc.collect()

news = news.groupby(["date", "assetCode"], as_index=False).mean()

data = pd.merge(mkt, news, how="left", on=["assetCode", "date"])
print("Data merged.")

del mkt
del news
gc.collect()

fillna_dict = {}

for col in data.columns:
    
    fillna_dict[col] = 0 if col != "sentimentNeutral" else 1

data.fillna(value=fillna_dict, inplace=True)
print("Missing values filled.")

def add_lags(data, cols, windows):

    data_by_asset = data.groupby(["assetCode"])
    lag_df = data[["assetCode", "date"]]

    for col in cols:

        for win in windows:
            
            col_name = "_".join([col, str(win)])
            lag_df[col_name + "_avg"] = data_by_asset.rolling(win)[col].mean().reset_index(drop=True)
            lag_df[col_name + "_std"] = data_by_asset.rolling(win)[col].std().reset_index(drop=True)
            # lag_df[col_name + "_max"] = data_by_asset.rolling(win)[col].max().reset_index(drop=True)
            # lag_df[col_name + "_min"] = data_by_asset.rolling(win)[col].min().reset_index(drop=True)
            gc.collect()

    del data_by_asset
    del data
    gc.collect()

    return lag_df

# lag_cols = ["returnsClosePrevRaw1", "returnsClosePrevRaw10", "relVol", "returnsToday"]
# windows = [3, 7]
# data.sort_values(by=["assetCode", "date"], axis=0, inplace=True)
# lag_df = add_lags(data, lag_cols, windows)
# data = data.merge(lag_df, how="left", on=["assetCode", "date"])
# data.dropna(axis=0, how="any", inplace=True)
# print("Lags added.")

feature_cols = [col for col in data.columns.values if col not in
                ["date", "assetCode", "returnsOpenNextMktres10", "universe"]]
                
feature_scalers = [StandardScaler() for i in range(len(feature_cols))]

def normalize_data(data, cols, scalers):

    for i in range(len(cols)):

        data[cols[i]] = scalers[i].fit_transform(data[cols[i]].values.reshape((-1, 1)))
    
    gc.collect()

    return data

data = normalize_data(data, feature_cols, feature_scalers)
print("Data normalized.")

data["y"] = (data["returnsOpenNextMktres10"] > 0).astype(int)

seed = np.random.randint(1, 100)
data_train, data_test = train_test_split(data, random_state=seed, test_size=0.2)
data_val, data_test = train_test_split(data_test, random_state=seed, test_size=0.5)
del data
gc.collect()
print("Data split.")

class LGBModel(lgb.LGBMClassifier):
    
    def evaluate(self, y_true, y_pred):
        
        y_true = y_true.astype(int).reshape((-1, 1))
        y_pred = y_pred.astype(int).reshape((-1, 1))
        
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
        
        print("Accuracy: {:.4f}.".format(accuracy))
        print("Precision: {:.4f}.".format(precision))
        print("Recall: {:.4f}.".format(recall))
        print("F1 score: {:.4f}".format(f1_score))
        
        return accuracy, precision, recall, f1_score

    def get_confidence(self, x_test, ref_scaler):
        
        pred_prob = self.predict_proba(x_test)
        confidence = (pred_prob[:, 1] - pred_prob[:, 0]).reshape((-1, 1))
        conf_scaler = StandardScaler()
        confidence = conf_scaler.fit_transform(confidence)
        confidence = ref_scaler.inverse_transform(confidence)
        confidence = np.clip(confidence, -0.99999, 0.99999)
        
        return confidence.flatten()
    
    def score(self, x_test, data_test, ref_scaler):
        
        confidence = self.get_confidence(x_test, ref_scaler)
        
        y_ret_pred = np.zeros(confidence.shape[0])
        for i in range(y_ret_pred.shape[0]):
            y_ret_pred[i] = confidence[i] * data_test["returnsOpenNextMktres10"].values[i] * data_test["universe"].values[i]
        pred_data = pd.DataFrame({"date": data_test["date"], "y_ret_pred": y_ret_pred})
        pred_data = pred_data.groupby(["date"])["y_ret_pred"].sum().values.flatten()
        score = np.mean(pred_data) / np.std(pred_data)
        print("Validation score: {:.4f}.".format(score))
        
        return score, confidence

class XGBModel(xgb.XGBClassifier):
    
    def evaluate(self, y_true, y_pred):
        
        y_true = y_true.astype(int).reshape((-1, 1))
        y_pred = y_pred.astype(int).reshape((-1, 1))
        
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
        
        print("Accuracy: {:.4f}.".format(accuracy))
        print("Precision: {:.4f}.".format(precision))
        print("Recall: {:.4f}.".format(recall))
        print("F1 score: {:.4f}".format(f1_score))
        
        return accuracy, precision, recall, f1_score

    def get_confidence(self, x_test, ref_scaler):
        
        pred_prob = self.predict_proba(x_test)
        confidence = (pred_prob[:, 1] * 2 - 1).reshape((-1, 1))
        conf_scaler = StandardScaler()
        confidence = conf_scaler.fit_transform(confidence)
        confidence = ref_scaler.inverse_transform(confidence)
        confidence = np.clip(confidence, -0.99999, 0.99999)
        
        return confidence.flatten()
    
    def score(self, x_test, data_test, ref_scaler):
        
        confidence = self.get_confidence(x_test, ref_scaler)
        
        y_ret_pred = np.zeros(confidence.shape[0])
        for i in range(y_ret_pred.shape[0]):
            y_ret_pred[i] = confidence[i] * data_test["returnsOpenNextMktres10"].values[i] * data_test["universe"].values[i]
        pred_data = pd.DataFrame({"date": data_test["date"], "y_ret_pred": y_ret_pred})
        pred_data = pred_data.groupby(["date"])["y_ret_pred"].sum().values.flatten()
        score = np.mean(pred_data) / np.std(pred_data)
        print("Validation score: {:.4f}.".format(score))
        
        return score, confidence

class VotingClf(VotingClassifier):
    
    def evaluate(self, y_true, y_pred):
        
        y_true = y_true.astype(int).reshape((-1, 1))
        y_pred = y_pred.astype(int).reshape((-1, 1))
        
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
        
        print("Accuracy: {:.4f}.".format(accuracy))
        print("Precision: {:.4f}.".format(precision))
        print("Recall: {:.4f}.".format(recall))
        print("F1 score: {:.4f}".format(f1_score))
        
        return accuracy, precision, recall, f1_score
    
    def get_confidence(self, x_test, ref_scaler):
        
        pred_prob = self.predict_proba(x_test)
        confidence = (pred_prob[:, 1] - pred_prob[:, 0]).reshape((-1, 1))
        conf_scaler = StandardScaler()
        confidence = conf_scaler.fit_transform(confidence)
        confidence = ref_scaler.inverse_transform(confidence)
        confidence = np.clip(confidence, -0.99999, 0.99999)
        
        return confidence.flatten()
    
    def score(self, x_test, data_test, ref_scaler):
        
        confidence = self.get_confidence(x_test, ref_scaler)
        
        y_ret_pred = np.zeros(confidence.shape[0])
        for i in range(y_ret_pred.shape[0]):
            y_ret_pred[i] = confidence[i] * data_test["returnsOpenNextMktres10"].values[i] * data_test["universe"].values[i]
        pred_data = pd.DataFrame({"date": data_test["date"], "y_ret_pred": y_ret_pred})
        pred_data = pred_data.groupby(["date"])["y_ret_pred"].sum().values.flatten()
        score = np.mean(pred_data) / np.std(pred_data)
        print("Validation score: {:.4f}.".format(score))
        
        return score, confidence

lgbm_params = {
    "max_depth": 8,
    "num_leaves": 1000,
    "learning_rate": 0.1,
    "n_estimators": 200,
    "boosting_type": "dart",
    "n_jobs": -1,
    "reg_lambda": 0.01,
    "random_state": seed
}

model = LGBModel(**lgbm_params)

# seed = np.random.randint(1, 100)
# xgb_params = {
#     "max_depth": 6,
#     "learning_rate": 0.1,
#     "n_estimators": 200,
#     "booster": "dart",
#     # "n_jobs": -1,
#     "reg_lambda": 0.1,
#     "random_state": seed
# }
# model = XGBModel(**xgb_params)

# base_model_num = 20
# base_models = []

# for i in range(base_model_num):
    
#     seed = np.random.randint(1, 100)
#     lgbm_params = {
#         "max_depth": 6,
#         "num_leaves": 1000,
#         "learning_rate": 0.1,
#         "n_estimators": 100,
#         "boosting_type": "dart",
#         "reg_lambda": 0.1,
#         "random_state": seed
#     }
#     base_models.append((str(i), lgb.LGBMClassifier(**lgbm_params)))

# voting_params = {
#     "estimators": base_models,
#     "voting": "soft",
#     "n_jobs": -1
# }
# model = VotingClf(**voting_params)

x_train = data_train[feature_cols].values
y_train = data_train["y"].values
x_val = data_val[feature_cols].values 
y_val = data_val["y"].values
seed = np.random.randint(1, 100)

iforest = IsolationForest(n_estimators=50, contamination=0.01, random_state=seed)
inlier_idx = iforest.fit_predict(x_train) == 1
x_train = x_train[inlier_idx, :]
y_train = y_train[inlier_idx]

start = time.clock()
model.fit(x_train, y_train, eval_set=[(x_val, y_val)], early_stopping_rounds=20, verbose=1)
time_elapsed = int(time.clock() - start)
print("Total traninig time {} seconds.".format(time_elapsed))

x_test = data_test[feature_cols].values
y_test = data_test["y"].values
y_pred = model.predict(x_test)
model.evaluate(y_test, y_pred)

ref_scaler = StandardScaler()
ref_scaler.fit(data_train["returnsOpenNextMktres10"].values.reshape((-1, 1)))
_, confidence = model.score(x_test, data_test, ref_scaler)

plt.hist(confidence, bins="auto", label="Confidence")
plt.hist(data_test["returnsOpenNextMktres10"], bins="auto", alpha=0.8, label="True return")
plt.title("Confidence & True Return")
plt.legend(loc='best')
plt.xlim(-1,1)
plt.show()

del x_train
del x_test
del x_val
del y_train
del y_test
del y_pred
del y_val
del confidence

# Submission with the single LGBM model
if "days" not in globals():
    days = env.get_prediction_days()

# lag_start = datetime.date(year=2016, month=12, day=16)
# lag_train = data_train.loc[data_train["date"] >= lag_start]
# lag_val = data_val.loc[data_val["date"] >= lag_start]
# lag_test = data_test.loc[data_test["date"] >= lag_start]
# lag_data = pd.concat([lag_train, lag_val, lag_test], axis=0).sort_values(by=["assetCode", "date"], axis=0)
# lag_data = lag_data[["assetCode", "date"] + lag_cols]

del data_train
del data_val
del data_test
# del lag_train
# del lag_val
# del lag_test

for (mkt, news, pred) in days:
    
    start = time.clock()

    mkt = mkt_fillna(mkt)
    mkt = add_mkt_features(mkt)
    # news["newsIndex"] = news.index
    # news = news.merge(unwrap_assets(news["assetCodes"]), how="right", on=["newsIndex"])
    # news.drop(["newsIndex", "assetCodes"], axis=1, inplace=True)
    news = add_news_features(news)
    mkt.drop(["assetName", "volume", "close", "open"], axis=1, inplace=True)
    news.drop(["time", "sourceId", "headline", "provider", "subjects",
               "audiences", "bodySize", "sentenceCount", "wordCount",
               "assetName", "firstMentionSentence", "sentimentWordCount",
               "headlineTag"], axis=1, inplace=True)
    news = news.groupby(["assetCode", "date"], as_index=False).mean()
    data = pd.merge(mkt, news, how="left", left_on=["date", "assetCode"], right_on=["date", "assetCode"])
    del mkt
    del news
    gc.collect()
    data.fillna(value=fillna_dict, inplace=True)
    # lag_start = data["date"].unique()[0] - datetime.timedelta(days=14)
    # lag_data = lag_data.loc[lag_data["date"] >= lag_start, :]
    # lag_data = pd.concat([lag_data, data[["assetCode", "date"] + lag_cols]], axis=0)
    # lag_df = add_lags(lag_data, lag_cols, windows)
    # lag_df = lag_df.loc[lag_df["date"] == data["date"].unique()[0], :]
    # data = data.merge(lag_df, how="left", on=["assetCode", "date"])
    # del lag_df
    data = normalize_data(data, feature_cols, feature_scalers)

    x_test = data[feature_cols].values
    confidence = model.get_confidence(x_test, ref_scaler)
    confidence = pd.DataFrame({"assetCode": data["assetCode"].values, "confidenceValue": confidence})
    pred.drop(["confidenceValue"], axis=1, inplace=True)
    pred = pd.merge(pred, confidence, how="left", left_on=["assetCode"], right_on=["assetCode"])
    pred.fillna(0, inplace=True)
    env.predict(pred)
    # lag_data = pd.concat([lag_data, data[["assetCode", "date"] + lag_cols]], axis=0)
    print(data["date"].unique()[0])
    time_used = int(time.clock() - start)
    print("Time used: {} sec.".format(time_used))
    
    del data
    del x_test
    del confidence
    del pred
    gc.collect()

env.write_submission_file()
print("Finished.")
