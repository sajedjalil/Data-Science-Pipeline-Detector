# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

grid_size_x = 10000
grid_size_y = 10000

def generateOutput(pred_probs, le):
    classes = np.fliplr(np.argsort(pred_probs, axis=1)[:, -3:])
    Y_pred = le.inverse_transform(classes).astype(str)
    Y_pred = np.apply_along_axis(' '.join, 1, Y_pred)
    return Y_pred

def runOnGridCell(train, test, features):
    label_encoder = LabelEncoder()
    train['target'] = label_encoder.fit_transform(train['place_id'])
    xgb_model = XGBClassifier(learning_rate =0.1, n_estimators=100, max_depth=5, min_child_weight=5, gamma=0, subsample=0.8, colsample_bytree=0.8, objective= 'multi:softprob', scale_pos_weight=1, seed=32)
    xgb_model.fit(train[features], train['target'], eval_metric='merror')
    test_predprob = xgb_model.predict_proba(test[features])
    test['place_id'] = generateOutput(test_predprob, label_encoder)
    return test
    
def prepare_data(df):
    df['x'] *= 1000
    df['y'] *= 1000
    initial_date = np.datetime64('2000-01-01T01:01', dtype='datetime64[m]') 
    d_times = pd.DatetimeIndex(initial_date + np.timedelta64(int(mn), 'm') for mn in df.time.values)    
    df['minute'] = d_times.minute
    df['hour'] = d_times.hour
    df['weekday'] = d_times.weekday
    df['day'] = d_times.day
    df['month'] = d_times.month
    df['year'] = d_times.year
    df['week_no'] = d_times.weekofyear + 52 * (d_times.year - 2000)
    return df

def run(grid_x_n = 50, grid_y_n = 50):
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')
    train = prepare_data(train)
    test = prepare_data(test)
    print('Feature added')
    features = ['x', 'y', 'accuracy', 'minute', 'hour', 'day', 'weekday', 'week_no', 'month', 'year']
    test_pred = pd.DataFrame(index = test['row_id'].values, columns=['row_id', 'place_id'])
    split_x = grid_size_x / grid_x_n
    split_y = grid_size_y / grid_y_n
    for i in range(grid_x_n):
        x_min = i * split_x
        x_max = (i+1)*split_x
        if x_max == grid_size_x:
            x_max += 1
        df_train = train[(train['x'] >= x_min) & (train['x'] < x_max)]
        df_test = test[(test['x'] >= x_min) & (test['x'] < x_max)]
        for j in range(grid_y_n):
            y_min = j * split_y
            y_max = (j+1)*split_y
            if y_max == grid_size_y:
                y_max += 1
            cell_train = df_train[(df_train['y'] >= y_min) & (df_train['y'] < y_max)]
            cell_test = df_test[(df_test['y'] >= y_min) & (df_test['y'] < y_max)]
            cell_pred = runOnGridCell(cell_train, cell_test, features)
            cell_pred = cell_pred[['row_id', 'place_id']]
            cell_pred.index = cell_pred['row_id'].values
            test_pred.loc[cell_pred.index] = cell_pred
            print('Done for (' + str(i) + ',' + str(j) + ')')
    test_pred.to_csv('xgboost1_submission.csv.gz', index = False, compression='gzip')
    

if __name__ == '__main__':
    run()
    print('Done')