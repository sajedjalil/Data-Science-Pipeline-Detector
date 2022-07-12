import numpy as np
import pandas as pd
import math
import multiprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupShuffleSplit

print('loading data')

train = pd.read_csv('../input/X_train.csv')
test = pd.read_csv('../input/X_test.csv')
target = pd.read_csv('../input/y_train.csv')
submit = pd.read_csv('../input/sample_submission.csv')

print('start feature engineering')

# https://stackoverflow.com/questions/53033620/how-to-convert-euler-angles-to-quaternions-and-get-the-same-euler-angles-back-fr?rq=1
def quaternion_to_euler(x, y, z, w):

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = math.atan2(t3, t4)

    return X, Y, Z
    
def feature_engineering(df):
    
    df_new = pd.DataFrame()
    
    # calculate euclidean distance
    df['total_angular_velocity'] = np.sqrt(df['angular_velocity_X'] ** 2 + df['angular_velocity_Y'] ** 2 + df['angular_velocity_Z'] ** 2)
    df['total_linear_acceleration'] = np.sqrt(df['linear_acceleration_X'] ** 2 + df['linear_acceleration_Y'] ** 2 + df['linear_acceleration_Z'] ** 2)
    df['total_orientation'] = np.sqrt(df['orientation_X'] ** 2 + df['orientation_Y'] ** 2 + df['orientation_Z'] ** 2 + df['orientation_W'] ** 2)

    # calculate absolute value 
    df['linear_acceleration_X_abs'] = df['linear_acceleration_X'].where(df['linear_acceleration_X']>=0, - df['linear_acceleration_X'])
    df['linear_acceleration_Y_abs'] = df['linear_acceleration_Y'].where(df['linear_acceleration_Y']>=0, - df['linear_acceleration_Y'])
    df['linear_acceleration_Z_abs'] = df['linear_acceleration_Z'].where(df['linear_acceleration_Z']>=0, - df['linear_acceleration_Z'])
    
    # how much Robot have acceleration compared to velocity                                           
    df['acc_vs_vel'] = df['total_linear_acceleration'] / df['total_angular_velocity']
    
    x, y, z, w = df['orientation_X'].tolist(), df['orientation_Y'].tolist(), df['orientation_Z'].tolist(), df['orientation_W'].tolist()
    nx, ny, nz = [], [], []
    
    for i in range(len(x)):
        
        xx, yy, zz = quaternion_to_euler(x[i], y[i], z[i], w[i])
        nx.append(xx)
        ny.append(yy)
        nz.append(zz)
    
    df['euler_x'] = nx
    df['euler_y'] = ny
    df['euler_z'] = nz
    
    df['total_angle'] = np.sqrt(df['euler_x'] ** 2 + df['euler_y'] ** 2 + df['euler_z'] ** 2)
    df['angle_vs_acc'] = df['total_angle'] / df['total_linear_acceleration']
    df['angle_vs_vel'] = df['total_angle'] / df['total_angular_velocity']
    
    # add interaction feature
    df['acc_times_vel'] = df['total_linear_acceleration'] * df['total_angular_velocity']
    df['angle_times_acc'] = df['total_angle'] * df['total_linear_acceleration']
    df['angle_times_vel'] = df['total_angle'] * df['total_angular_velocity']
    df['angle_times_vel_times_acc'] = df['total_angle'] * df['total_angular_velocity'] * df['total_linear_acceleration']

    def f1(x):
        return np.mean(np.diff(np.abs(np.diff(x))))
    
    def f2(x):
        return np.mean(np.abs(np.diff(x)))
    
    for col in df.columns:
        
        if col in ['row_id', 'series_id', 'measurement_number']:
            continue
        
        df_new[col + '_mean'] = df.groupby(['series_id'])[col].mean()
        df_new[col + '_min'] = df.groupby(['series_id'])[col].min()
        df_new[col + '_max'] = df.groupby(['series_id'])[col].max()
        df_new[col + '_std'] = df.groupby(['series_id'])[col].std()
        df_new[col + '_max_to_min'] = df_new[col + '_max'] / df_new[col + '_min']

        df_new[col + '_mean_abs_change'] = df.groupby('series_id')[col].apply(f2)
        df_new[col + '_mean_change_of_abs_change'] = df.groupby('series_id')[col].apply(f1)
        
        df_new[col + '_abs_max'] = df.groupby('series_id')[col].apply(lambda x: np.max(np.abs(x)))
        df_new[col + '_abs_min'] = df.groupby('series_id')[col].apply(lambda x: np.min(np.abs(x)))

    return df_new

with multiprocessing.Pool() as pool: 
    train, test = pool.map(feature_engineering, [train, test])
    
le = LabelEncoder()
target['surface'] = le.fit_transform(target['surface'])

# replace NAN to 0
train.fillna(0, inplace=True)
test.fillna(0, inplace=True)

# replace infinite value to zero
train.replace(-np.inf, 0, inplace=True)
train.replace(np.inf, 0, inplace=True)
test.replace(-np.inf, 0, inplace=True)
test.replace(np.inf, 0, inplace=True)

#folds = StratifiedKFold(n_splits=100, shuffle=True, random_state=546789)
folds = GroupShuffleSplit(n_splits=10, test_size=0.2, random_state=13)

sub_preds_rf = np.zeros((test.shape[0], 9))
oof_preds_rf = np.zeros((train.shape[0]))
score = 0
counter = 0

print('start training')

#for train_index, test_index in folds.split(train, target['surface']):
for train_index, test_index in folds.split(train, target['surface'], train['group_id']):
    
    print('Fold {}'.format(counter+1))
    
    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
    clf.fit(train.iloc[train_index], target['surface'][train_index])
    oof_preds_rf[test_index] = clf.predict(train.iloc[test_index])
    sub_preds_rf += clf.predict_proba(test) / folds.n_splits
    score += clf.score(train.iloc[test_index], target['surface'][test_index])
    counter += 1
    
    print('score : {}'.format(clf.score(train.iloc[test_index], target['surface'][test_index])))

print('avg accuracy : {}'.format(score / folds.n_splits))

submit['surface'] = le.inverse_transform(sub_preds_rf.argmax(axis=1))
submit.to_csv('submit.csv', index=False)

print('done.')