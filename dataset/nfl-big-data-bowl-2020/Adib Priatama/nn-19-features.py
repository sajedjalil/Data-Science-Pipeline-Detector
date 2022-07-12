#feature selection : https://www.kaggle.com/coolcoder22/nfl-001-feature-selection

# IMPORTS 
import numpy as np
import pandas as pd
import sklearn.metrics as mtr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import Callback, EarlyStopping
from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Reshape, Dropout, merge, Add
from keras.layers.embeddings import Embedding
from sklearn.model_selection import KFold,GroupKFold
import scipy as sp
import warnings
import random as rn
import tensorflow as tf
from keras.models import load_model
import os
warnings.filterwarnings("ignore")
from kaggle.competitions import nflrush
env = nflrush.make_env()
iter_test = env.iter_test()


# evaluation metric
def crps(y_true, y_pred):
    y_true = np.clip(np.cumsum(y_true, axis=1), 0, 1)
    y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
    return ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * y_true.shape[0]) 


# author : nlgn
# Link : https://www.kaggle.com/kingychiu/keras-nn-starter-crps-early-stopping
class Metric(Callback):
    def __init__(self, model, callbacks, data):
        super().__init__()
        self.model = model
        self.callbacks = callbacks
        self.data = data

    def on_train_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_end(self, batch, logs=None):
        X_train, y_train = self.data[0][0], self.data[0][1]
        y_pred = self.model.predict(X_train)
        y_true = np.clip(np.cumsum(y_train, axis=1), 0, 1)
        y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
        tr_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * X_train[-1].shape[0])
        tr_s = np.round(tr_s, 6)
        logs['tr_CRPS'] = tr_s

        X_valid, y_valid = self.data[1][0], self.data[1][1]

        y_pred = self.model.predict(X_valid)
        y_true = np.clip(np.cumsum(y_valid, axis=1), 0, 1)
        y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
        val_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * X_valid[-1].shape[0])
        val_s = np.round(val_s, 6)
        logs['val_CRPS'] = val_s
        print('tr CRPS', tr_s, 'val CRPS', val_s)

        for callback in self.callbacks:
            callback.on_epoch_end(batch, logs)
            

# author : ryancaldwell
# Link : https://www.kaggle.com/ryancaldwell/location-eda
def create_features(df, deploy=False):
    def new_X(x_coordinate, play_direction):
        if play_direction == 'left':
            return 120.0 - x_coordinate
        else:
            return x_coordinate
        
    def new_Y(y_coordinate, play_direction):
        if play_direction == 'left':
            return 160/3 - y_coordinate
        else:
            return y_coordinate

    def new_line(rush_team, field_position, yardline):
        if rush_team == field_position:
            # offense starting at X = 0 plus the 10 yard endzone plus the line of scrimmage
            return 10.0 + yardline
        else:
            # half the field plus the yards between midfield and the line of scrimmage
            return 60.0 + (50 - yardline)

    def new_orientation(angle, play_direction):
        if play_direction == 'left':
            new_angle = 360.0 - angle
            if new_angle == 360.0:
                new_angle = 0.0
            return new_angle
        else:
            return angle

    def euclidean_distance(x1,y1,x2,y2):
        x_diff = (x1-x2)**2
        y_diff = (y1-y2)**2

        return np.sqrt(x_diff + y_diff)

    def back_direction(orientation):
        if orientation > 180.0:
            return 1
        else:
            return 0

    def update_yardline(df):
        new_yardline = df[df['NflId'] == df['NflIdRusher']]
        new_yardline['YardLine'] = new_yardline[['PossessionTeam','FieldPosition','YardLine']].apply(lambda x: new_line(x[0],x[1],x[2]), axis=1)
        new_yardline = new_yardline[['GameId','PlayId','YardLine']]

        return new_yardline

    def update_orientation(df, yardline):
        df['X'] = df[['X','PlayDirection']].apply(lambda x: new_X(x[0],x[1]), axis=1)
        df['Y'] = df[['Y','PlayDirection']].apply(lambda x: new_Y(x[0],x[1]), axis=1)
        df['Orientation'] = df[['Orientation','PlayDirection']].apply(lambda x: new_orientation(x[0],x[1]), axis=1)
        df['Dir'] = df[['Dir','PlayDirection']].apply(lambda x: new_orientation(x[0],x[1]), axis=1)

        df = df.drop('YardLine', axis=1)
        df = pd.merge(df, yardline, on=['GameId','PlayId'], how='inner')

        return df
    
    def get_sum_dist(df):
        df['SumLeastDist'] = pd.Series(np.nan)
        df['SumLeastDist2'] = pd.Series(np.nan)
        df['AvgLeastDist'] = pd.Series(np.nan)
        df['AvgLeastDist2'] = pd.Series(np.nan)
        for play_id, new_df in df.groupby(by='PlayId'):
            rush_t = new_df[new_df['NflIdRusher'] == new_df['NflId']]['Team'].values[0]
            mat = sp.spatial.distance.cdist(new_df[new_df['Team'] != rush_t][['X', 'Y']],
                                            new_df[new_df['Team'] == rush_t][['X', 'Y']])
            tes_spat = pd.DataFrame(mat, index=new_df[new_df['Team'] != rush_t]['DisplayName'],
                                    columns=new_df[new_df['Team'] == rush_t]['DisplayName'])
            df.loc[df['PlayId'] == play_id, 'SumLeastDist'] = tes_spat.min(axis=1).sum()
            df.loc[df['PlayId'] == play_id, 'SumLeastDist2'] = tes_spat.T.min(axis=1).sum()
            df.loc[df['PlayId'] == play_id, 'AvgLeastDist'] = tes_spat.min(axis=1).mean()
            df.loc[df['PlayId'] == play_id, 'AvgLeastDist2'] = tes_spat.T.min(axis=1).mean()
        
        return df

    def back_features(df):
        carriers = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','NflIdRusher','X','Y','Orientation','Dir','YardLine']]
        carriers['back_from_scrimmage'] = carriers['YardLine'] - carriers['X']
        carriers['back_oriented_down_field'] = carriers['Orientation'].apply(lambda x: back_direction(x))
        carriers['back_moving_down_field'] = carriers['Dir'].apply(lambda x: back_direction(x))
        carriers = carriers.rename(columns={'X':'back_X',
                                            'Y':'back_Y'})
        carriers = carriers[['GameId','PlayId','NflIdRusher','back_X','back_Y','back_from_scrimmage','back_oriented_down_field','back_moving_down_field']]

        return carriers

    def features_relative_to_back(df, carriers):
        player_distance = df[['GameId','PlayId','NflId','X','Y']]
        player_distance = pd.merge(player_distance, carriers, on=['GameId','PlayId'], how='inner')
        player_distance = player_distance[player_distance['NflId'] != player_distance['NflIdRusher']]
        player_distance['dist_to_back'] = player_distance[['X','Y','back_X','back_Y']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)

        player_distance = player_distance.groupby(['GameId','PlayId','back_from_scrimmage','back_oriented_down_field','back_moving_down_field'])\
                                         .agg({'dist_to_back':['min','max','mean','std']})\
                                         .reset_index()
        player_distance.columns = ['GameId','PlayId','back_from_scrimmage','back_oriented_down_field','back_moving_down_field',
                                   'min_dist','max_dist','mean_dist','std_dist']

        return player_distance

    def defense_features(df):
        rusher = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','Team','X','Y']]
        rusher.columns = ['GameId','PlayId','RusherTeam','RusherX','RusherY']

        defense = pd.merge(df,rusher,on=['GameId','PlayId'],how='inner')
        defense = defense[defense['Team'] != defense['RusherTeam']][['GameId','PlayId','X','Y','RusherX','RusherY']]
        defense['def_dist_to_back'] = defense[['X','Y','RusherX','RusherY']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)

        defense = defense.groupby(['GameId','PlayId'])\
                         .agg({'def_dist_to_back':['min','max','mean','std']})\
                         .reset_index()
        defense.columns = ['GameId','PlayId','def_min_dist','def_max_dist','def_mean_dist','def_std_dist']

        return defense

    def static_features(df):
        static_features = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','X','Y','S','A','Dis','Orientation','Dir',
                                                            'YardLine','Quarter','Down','Distance','DefendersInTheBox',
                                                                'SumLeastDist', 'SumLeastDist2', 'AvgLeastDist', 'AvgLeastDist2']].drop_duplicates()
        static_features['DefendersInTheBox'] = static_features['DefendersInTheBox'].fillna(np.mean(static_features['DefendersInTheBox']))

        return static_features
    
    def split_personnel(s):
        splits = s.split(',')
        for i in range(len(splits)):
            splits[i] = splits[i].strip()

        return splits

    def defense_formation(l):
        dl = 0
        lb = 0
        db = 0
        other = 0

        for position in l:
            sub_string = position.split(' ')
            if sub_string[1] == 'DL':
                dl += int(sub_string[0])
            elif sub_string[1] in ['LB','OL']:
                lb += int(sub_string[0])
            else:
                db += int(sub_string[0])

        counts = (dl,lb,db,other)

        return counts

    def offense_formation(l):
        qb = 0
        rb = 0
        wr = 0
        te = 0
        ol = 0

        sub_total = 0
        qb_listed = False
        for position in l:
            sub_string = position.split(' ')
            pos = sub_string[1]
            cnt = int(sub_string[0])

            if pos == 'QB':
                qb += cnt
                sub_total += cnt
                qb_listed = True
            # Assuming LB is a line backer lined up as full back
            elif pos in ['RB','LB']:
                rb += cnt
                sub_total += cnt
            # Assuming DB is a defensive back and lined up as WR
            elif pos in ['WR','DB']:
                wr += cnt
                sub_total += cnt
            elif pos == 'TE':
                te += cnt
                sub_total += cnt
            # Assuming DL is a defensive lineman lined up as an additional line man
            else:
                ol += cnt
                sub_total += cnt

        # If not all 11 players were noted at given positions we need to make some assumptions
        # I will assume if a QB is not listed then there was 1 QB on the play
        # If a QB is listed then I'm going to assume the rest of the positions are at OL
        # This might be flawed but it looks like RB, TE and WR are always listed in the personnel
        if sub_total < 11:
            diff = 11 - sub_total
            if not qb_listed:
                qb += 1
                diff -= 1
            ol += diff

        counts = (qb,rb,wr,te,ol)

        return counts
    
    def personnel_features(df):
        personnel = df[['GameId','PlayId','OffensePersonnel','DefensePersonnel']].drop_duplicates()
        personnel['DefensePersonnel'] = personnel['DefensePersonnel'].apply(lambda x: split_personnel(x))
        personnel['DefensePersonnel'] = personnel['DefensePersonnel'].apply(lambda x: defense_formation(x))
        personnel['num_DL'] = personnel['DefensePersonnel'].apply(lambda x: x[0])
        personnel['num_LB'] = personnel['DefensePersonnel'].apply(lambda x: x[1])
        personnel['num_DB'] = personnel['DefensePersonnel'].apply(lambda x: x[2])

        personnel['OffensePersonnel'] = personnel['OffensePersonnel'].apply(lambda x: split_personnel(x))
        personnel['OffensePersonnel'] = personnel['OffensePersonnel'].apply(lambda x: offense_formation(x))
        personnel['num_QB'] = personnel['OffensePersonnel'].apply(lambda x: x[0])
        personnel['num_RB'] = personnel['OffensePersonnel'].apply(lambda x: x[1])
        personnel['num_WR'] = personnel['OffensePersonnel'].apply(lambda x: x[2])
        personnel['num_TE'] = personnel['OffensePersonnel'].apply(lambda x: x[3])
        personnel['num_OL'] = personnel['OffensePersonnel'].apply(lambda x: x[4])

        # Let's create some features to specify if the OL is covered
        personnel['OL_diff'] = personnel['num_OL'] - personnel['num_DL']
        personnel['OL_TE_diff'] = (personnel['num_OL'] + personnel['num_TE']) - personnel['num_DL']
        # Let's create a feature to specify if the defense is preventing the run
        # Let's just assume 7 or more DL and LB is run prevention
        personnel['run_def'] = (personnel['num_DL'] + personnel['num_LB'] > 6).astype(int)

        personnel.drop(['OffensePersonnel','DefensePersonnel'], axis=1, inplace=True)
        
        return personnel

    def combine_features(relative_to_back, defense, static, personnel, deploy=deploy):
        df = pd.merge(relative_to_back,defense,on=['GameId','PlayId'],how='inner')
        df = pd.merge(df,static,on=['GameId','PlayId'],how='inner')
        df = pd.merge(df,personnel,on=['GameId','PlayId'],how='inner')

        if not deploy:
            df = pd.merge(df, outcomes, on=['GameId','PlayId'], how='inner')

        return df
    
    yardline = update_yardline(df)
    df = update_orientation(df, yardline)
    df = get_sum_dist(df)
    back_feats = back_features(df)
    rel_back = features_relative_to_back(df, back_feats)
    def_feats = defense_features(df)
    static_feats = static_features(df)
    personnel = personnel_features(df)
    basetable = combine_features(rel_back, def_feats, static_feats, personnel, deploy=deploy)
    return basetable


train = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', dtype={'WindSpeed': 'object'})
outcomes = train[['GameId','PlayId','Yards']].drop_duplicates()


train_basetable = create_features(train, False)


X = train_basetable.copy()
yards = X.Yards


y = np.zeros((yards.shape[0], 199))
for idx, target in enumerate(list(yards)):
    y[idx][99 + target] = 1
    
    
def process_two(t_):
    t_['fe1'] = pd.Series(np.sqrt(np.absolute(np.square(t_.X.values) - np.square(t_.Y.values))))
    t_['fe5'] = np.square(t_['S'].values) + 2 * t_['A'].values * t_['Dis'].values  # N
    t_['fe7'] = np.arccos(np.clip(t_['X'].values / t_['Y'].values, -1, 1))  # N
    t_['fe8'] = t_['S'].values / np.clip(t_['fe1'].values, 0.6, None)
    radian_angle = (90 - t_['Dir']) * np.pi / 180.0
    t_['fe10'] = np.abs(t_['S'] * np.cos(radian_angle))
    t_['fe11'] = np.abs(t_['S'] * np.sin(radian_angle))
    return t_


X = process_two(X)

important = ['back_from_scrimmage', 'min_dist', 'max_dist', 'mean_dist', 'std_dist',
       'def_min_dist', 'def_max_dist', 'def_mean_dist', 'def_std_dist', 'X',
       'Y', 'S', 'A', 'Dis', 'Orientation', 'Dir', 'YardLine', 'SumLeastDist',
             'SumLeastDist2', 'AvgLeastDist', 'AvgLeastDist2']

cat = ['back_oriented_down_field', 'back_moving_down_field']

num = ['back_from_scrimmage', 'min_dist', 'max_dist', 'mean_dist', 'std_dist', 'def_min_dist', 'def_max_dist', 'def_mean_dist', 'def_std_dist',
       'X', 'Y', 'S', 'A', 'Dis', 'Orientation', 'Dir', 'YardLine', 'Distance', 'SumLeastDist', 'SumLeastDist2', 'AvgLeastDist', 'AvgLeastDist2'] + ['fe1', 'fe5', 'fe7','fe8', 'fe10', 'fe11']
num = [i for i in num if i in important]
print(len(cat))
print(len(num))


scaler = StandardScaler()
X[num] = scaler.fit_transform(X[num])


def model_396_1():
    inputs = []
    embeddings = []
    for i in cat:
        input_ = Input(shape=(1,))
        embedding = Embedding(int(np.absolute(X[i]).max() + 1), 10, input_length=1)(input_)
        embedding = Reshape(target_shape=(10,))(embedding)
        inputs.append(input_)
        embeddings.append(embedding)
    input_numeric = Input(shape=(len(num),))
    embedding_numeric = Dense(512, activation='relu')(input_numeric) 
    inputs.append(input_numeric)
    embeddings.append(embedding_numeric)
    x = Concatenate()(embeddings)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(199, activation='softmax')(x)
    model = Model(inputs, output)
    return model


n_splits = 5
kf = GroupKFold(n_splits=n_splits)
score = []
for i_369, (tdx, vdx) in enumerate(kf.split(X, y, X['GameId'])):
    print(f'Fold : {i_369}')
    X_train, X_val, y_train, y_val = X.iloc[tdx], X.iloc[vdx], y[tdx], y[vdx]
    X_train = [np.absolute(X_train[i]) for i in cat] + [X_train[num]]
    X_val = [np.absolute(X_val[i]) for i in cat] + [X_val[num]]
    model = model_396_1()
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=[])
    es = EarlyStopping(monitor='val_CRPS', 
                   mode='min',
                   restore_best_weights=True, 
                   verbose=2, 
                   patience=5)
    es.set_model(model)
    metric = Metric(model, [es], [(X_train,y_train), (X_val,y_val)])
    for i in range(1):
        model.fit(X_train, y_train, verbose=False)
    for i in range(1):
        model.fit(X_train, y_train, batch_size=64, verbose=False)
    for i in range(1):
        model.fit(X_train, y_train, batch_size=128, verbose=False)
    for i in range(1):
        model.fit(X_train, y_train, batch_size=256, verbose=False)
    model.fit(X_train, y_train, callbacks=[metric], epochs=100, batch_size=1024, verbose=False)
    score_ = crps(y_val, model.predict(X_val))
    model.save(f'keras_369_{i_369}.h5')
    print(score_)
    score.append(score_)
    

print(np.mean(score))


models = []
for i in range(n_splits):
    models.append(load_model(f'keras_369_{i}.h5'))
    
for (test_df, sample_prediction_df) in iter_test:
    basetable = create_features(test_df, deploy=True)
    basetable = process_two(basetable)
    basetable[num] = scaler.transform(basetable[num])
    test_ = [np.absolute(basetable[i]) for i in cat] + [basetable[num]]
    
    y_pred = np.mean([model.predict(test_) for model in models], axis=0)
    y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1).tolist()[0]
    
    preds_df = pd.DataFrame(data=[y_pred], columns=sample_prediction_df.columns)
    env.predict(preds_df)
    
env.write_submission_file()