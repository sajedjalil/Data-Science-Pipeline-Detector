import kagglegym
import numpy as np
np.random.seed(42)
import pandas as pd
from time import time

from xgboost import XGBRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline


def preprocess_data(obs, cols):
    print("Preprocessing data")
    t0 = time()
    y_min = obs.train.y.min()
    y_max = obs.train.y.max()
    idx = (obs.train.y<y_max) & (obs.train.y>y_min)

    #sX = StandardScaler()
    #sY = StandardScaler()
    
    X_train = obs.train.groupby('id').bfill().ffill().fillna(0).loc[idx, cols].values
    Y_train = obs.train.y.loc[idx].values#.reshape([-1,1])

    preprocess_kernels = {'y_min': y_min, 'y_max': y_max, 'idx': idx}
    
    print("Done! Preprocessing time:",time()- t0)
    print("Features shape:", X_train.shape)
    print("Targets shape:", Y_train.shape)
    return X_train, Y_train, preprocess_kernels
    
def train_model(X_train, Y_train):

#### MODEL HERE
    
    model = XGBRegressor(max_depth=4, learning_rate=0.1, n_estimators=33, silent=False, objective='reg:linear', nthread=-1, gamma=0.004, min_child_weight=1000,
    max_delta_step=0, subsample=1.0, colsample_bytree=0.5, colsample_bylevel=1, reg_alpha=0.0, reg_lambda=7.0, scale_pos_weight=1, base_score=0.0, seed=42,
    missing=None)
    
###############
    #pca = PCA(n_components=10)
    ss = StandardScaler()
    fs = SelectFromModel(model)

    pipe = Pipeline([('ss',ss),('fs',fs),('model',model)])
    print("Training pipeline")
    t0 = time()
    pipe.fit(X_train, Y_train)
    print("Done! R2 score for training: {0}, Training time: {1}".format(pipe.score(X_train, Y_train), time()-t0))
    print("Number of selected features: {0[1][0][1]}".format([i for i in zip(np.unique(pipe.named_steps['fs'].get_support(), return_counts=True))]))
    return pipe

# Predict-step-predict routine
def predict_targets(env, observation, model, preprocess_dict, features_to_use, print_info=True):

    reward = 0.0
    reward_log = []
    timestamp_log = []
    pred_log= []
    pos_count = 0
    neg_count = 0

    total_pos = []
    total_neg = []

    print("Predicting")
    t0= time()
    while True:
        # Predict with model
        features = observation.features.groupby('id').bfill().ffill().fillna(0).loc[:,features_to_use].values

        y_dnn = model.predict(features)#.clip(preprocess_dict['y_min'], preprocess_dict['y_max'])

        # Fill target df with predictions 
        observation.target.y = y_dnn
        observation.target.fillna(0, inplace=True)
        target = observation.target
        timestamp = observation.features["timestamp"][0]
        
        observation, reward, done, info = env.step(target)

        timestamp_log.append(timestamp)
        reward_log.append(reward)
        pred_log.append(y_dnn)

        if (reward < 0):
            neg_count += 1
        else:
            pos_count += 1

        total_pos.append(pos_count)
        total_neg.append(neg_count)
        
        if timestamp % 100 == 0:
            if print_info:
                print("Timestamp #{}".format(timestamp))
                print("Mean reward:", np.mean(reward_log[-timestamp:]))
                print("Positive rewards count: {0}, Negative rewards count: {1}".format(pos_count, neg_count))
                print("Positive reward %:", pos_count / (pos_count + neg_count) * 100)

            pos_count = 0
            neg_count = 0

        if done:
            break
    print("Done: %.1fs" % (time() - t0))
    print("Total reward sum:", np.sum(reward_log))
    print("Final reward mean:", np.mean(reward_log))
    print("Total positive rewards count: {0}, Total negative rewards count: {1}".format(np.sum(total_pos),
                                                                                        np.sum(total_neg)))
    print("Final positive reward %:", np.sum(total_pos) / (np.sum(total_pos) + np.sum(total_neg)) * 100)
    print(info)
    return np.array(pred_log), np.array(reward_log), np.array(timestamp_log)

def main():
    # Preprocess data, define and train model
    env = kagglegym.make()
    obs = env.reset()

    excl = ['id', 'sample', 'y', 'timestamp']
    cols = [c for c in obs.train.columns if c not in excl]
    
    data = preprocess_data(obs, cols)
    model = train_model(data[0],data[1])
    logs = predict_targets(env, obs, model, data[2], cols)
    
    return logs
    
logs = main()