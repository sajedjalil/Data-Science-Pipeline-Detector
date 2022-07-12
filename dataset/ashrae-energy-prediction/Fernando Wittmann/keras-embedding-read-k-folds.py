"""
### Test 5
- 10 Epochs
- 100 iterations
- CLR
- Cat: ["site_id", "meter", "tm_hour_of_day", "tm_day_of_week",   "primary_use"]
- Num: ["square_feet", 'floor_count']
- LOG_FEATURES = ["square_feet", "floor_count"]
- Normalization: Yes
- Leak: 1.12
- Train: 1.1368
- Valid: 1.04109
- LB 1.28

### Test 4
- 10 Epochs
- 100 iterations
- CLR
- Cat: ["building_id", site_id", "meter", "tm_hour_of_day", "tm_day_of_week",   "primary_use"]
- Num: ["square_feet"]
- LOG_FEATURES = ["square_feet"]
- Normalization: No
- Leak: 1.129
- Train: 1.11 
- Valid: 1.02


### Test 3
- 10 Epochs
- 100 iterations
- CLR
- Cat: ["site_id", "meter", "tm_hour_of_day", "tm_day_of_week",   "primary_use"]
- Num: []
- Normalization: No
- Leak: 1.64
- Train: 1.71
- Valid: 1.64


### Test 2
- 10 Epochs
- 100 iterations
- CLR
- Cat: ["building_id", "site_id", "meter", "tm_hour_of_day", "tm_day_of_week",   "primary_use"]
- Num: []
- Normalization: No
- Leak: 1.1131
- Train: 1.209
- Valid: 1.16

### Test 1
- 10 Epochs
- 100 iterations
- CLR
- Cat: ["building_id", "site_id", "meter", "tm_hour_of_day", "tm_day_of_week",   "primary_use"]
- Num: ["square_feet", "had_air_temperature", "had_dew_temperature"]
- Normalization: No
- LB: 1.36
- Leak: 1.182
- Train: 1.2698
- Valid: 1.15494

"""
import pandas as pd
import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Dropout, Dense, Embedding, SpatialDropout1D, concatenate, BatchNormalization, Flatten
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from keras import backend as K
from keras.models import Model
from keras.losses import mean_squared_error as mse_loss
from keras import optimizers
from keras.optimizers import RMSprop, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from ashrae_utils import CyclicLR

def keras_model(dense_dim_1=64, dense_dim_2=32, dense_dim_3=32, dense_dim_4=16, dropout1=0.2, dropout2=0.1, dropout3=0.1, dropout4=0.1, lr=1e-3):
    print("Setting keras model")
    building_id = Input(shape=[1], name="building_id")
    meter = Input(shape=[1], name="meter")
    site_id = Input(shape=[1], name="site_id")
    primary_use = Input(shape=[1], name="primary_use")
    
    #num_cols = ["square_feet", "air_temperature", "dew_temperature"]
    square_feet = Input(shape=[1], name="square_feet")
    floor_count = Input(shape=[1], name="floor_count")
    
    had_air_temperature = Input(shape=[1], name="had_air_temperature")
    had_dew_temperature = Input(shape=[1], name="had_dew_temperature")
    
    
    had_cloud_temperature = Input(shape=[1], name="had_cloud_temperature")
    had_precip_depth_1_hr = Input(shape=[1], name="had_precip_depth_1_hr")
    had_sea_level_pressure = Input(shape=[1], name="had_sea_level_pressure")
    had_wind_direction = Input(shape=[1], name="had_wind_direction")
    had_wind_speed = Input(shape=[1], name="had_wind_speed")
    tm_day_of_week = Input(shape=[1], name="tm_day_of_week")
    tm_hour_of_day = Input(shape=[1], name="tm_hour_of_day")
    ts_month = Input(shape=[1], name="ts_month")

    
    emb_building_id = Embedding(1449, 50)(building_id)
    emb_site_id = Embedding(16, 8)(site_id)
    emb_meter = Embedding(4, 2)(meter)
    emb_hour = Embedding(24, 12)(tm_hour_of_day)
    emb_weekday = Embedding(7, 4)(tm_day_of_week)
    emb_primary_use = Embedding(16, 8)(primary_use)
    emb_ts_month = Embedding(12, 6)(ts_month)

    concat_emb = concatenate([
           Flatten() (emb_building_id),
           Flatten() (emb_site_id),
           Flatten() (emb_meter),
           Flatten() (emb_hour),
           Flatten() (emb_weekday),
           Flatten() (emb_primary_use),
           Flatten() (emb_ts_month)
    ])
    
    categ = Dropout(dropout1)(Dense(dense_dim_1,activation='relu') (concat_emb))
    categ = BatchNormalization()(categ)
    categ = Dropout(dropout2)(Dense(dense_dim_2,activation='relu') (categ))
    
    #main layer
        #main layer
    main_l = categ#concatenate([
          #categ,
          #is_holiday,
          #square_feet,
          #floor_count,
          #had_air_temperature,
          #had_dew_temperature
    #])
    
    main_l = Dropout(dropout3)(Dense(dense_dim_3,activation='relu') (main_l))
    main_l = BatchNormalization()(main_l)
    main_l = Dropout(dropout4)(Dense(dense_dim_4,activation='relu') (main_l))
    
    #output
    output = Dense(1) (main_l)
            
    model = Model([ 
                    building_id,
                    site_id,
                    meter, 
                    tm_hour_of_day,
                    tm_day_of_week,
                    primary_use,
                    ts_month
                    #is_holiday,
                    #square_feet,
                    #floor_count,
                    #had_air_temperature,
                    #had_dew_temperature,
                  ], 
                    output)

    model.compile(optimizer = Adam(lr=lr),
                  loss='mean_squared_error',
                  metrics=[root_mean_squared_error])
    return model

def read_train_valid(which_ds='export-k-folds'):
    print("## Reading Training and Validation Data")
    if which_ds == 'export-k-folds':
        train = pd.read_feather(f'/kaggle/input/{which_ds}/train.feather')
        train = train.set_index('index')
        y = train['meter_reading']
        X = train.drop('meter_reading', axis=1)
        
        if ADD_MONTH:
            month = get_ts_month('train')
            X['ts_month'] = month[X.index]
        X = check_log_features(X)
        X[numericals]=check_normalize_features(X[numericals])
        
        # For debuggin only
        y_train = y[X['tm_k_fold']!=0]
        X_train = X[X['tm_k_fold']!=0].drop('tm_k_fold', axis=1)
        
        y_valid = y[X['tm_k_fold']==0]
        X_valid = X[X['tm_k_fold']==0].drop('tm_k_fold', axis=1)
        
        X_train = X_train[columns]
        X_valid = X_valid[columns]
        
        X_train = {col: np.array(X_train[col]) for col in X_train.columns}
        X_valid = {col: np.array(X_valid[col]) for col in X_valid.columns}
        
        del train
        return X_train, y_train, X_valid, y_valid

def check_normalize_features(X):
    print("Check normalize features")
    if NORMALIZE_FEATURES:
        X = (X-np.mean(X, axis=0))/(np.std(X, axis=0)+1e-7)         
    return X

def check_log_features(X):
    print("Check log features")
    if LOG_FEATURES:
        for feature in logfy:
            X[feature] = np.log1p(X[feature] - X[feature].min())
            
    return X
    
def read_test(which_ds='export-k-folds'):
    print("## Reading Test Data")
    if which_ds == 'export-k-folds':
        test = pd.read_feather(f'/kaggle/input/{which_ds}/test.feather')
        if ADD_MONTH:
            month = get_ts_month('test')
            test['ts_month'] = month[test.index]
            
        row_ids = test.row_id
        test = test[columns]
        test = check_log_features(test)
        test[numericals] = check_normalize_features(test[numericals])
        test = {col: np.array(test[col]) for col in test.columns}
        
        return test, row_ids
    
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=0))

def get_callbacks(cbk_list):
    callbacks = []
    if 'clr' in cbk_list:
        clr = CyclicLR(base_lr=9e-4, max_lr=3e-2,
            step_size=4*N_ITER, mode='exp_range',
            gamma=0.99994)
        callbacks.append(clr)
        
    if 'model_checkpoint' in cbk_list:
        model_checkpoint = ModelCheckpoint("model.hdf5",
                                       save_best_only=True, verbose=1, monitor='val_root_mean_squared_error', mode='min')

        callbacks.append(model_checkpoint)
 
    return callbacks

def train_model(X_train, y_train, X_valid, y_valid):
    print("Train model")
    model = keras_model()
    callbacks = get_callbacks(['clr', 'model_checkpoint'])
    model.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=len(y_train)//N_ITER, epochs=EPOCHS,
             callbacks=callbacks)
    model.load_weights('model.hdf5')
    return model

def predict(model, test, row_ids):
    print("Predict")
    preds = model.predict(test, batch_size=len(test['meter'])//100, verbose=1)
    preds = np.expm1(np.clip(preds.T[0], 0, None))
    submission = pd.DataFrame({'row_id': row_ids, 'meter_reading': preds})
    
    return submission

def export(submission):
    if not DEBUG:
        print('## Saving to CSV')
        submission.to_csv("submission.csv", index=False, float_format='%g')
        
def rmsle(y_true, y_pred):
    return np.sqrt(np.mean((np.log1p(y_true)-np.log1p(y_pred))**2))

def read_leak():
    print("Reading leak data")
    y_test = pd.read_csv('/kaggle/input/leak-test-set/y_test.csv', names=['meter_reading'], index_col=0)
    y_test['meter_reading'] = np.clip(y_test['meter_reading'], 0, None)
    return y_test

def leak_benchmark(submission):
    print("## Benchmarking against leak data")
    y_test = read_leak()
    rmsle_error = rmsle(y_test.values.T[0], submission['meter_reading'][y_test.index].values)
    print(f'RMSLE in the leak data is {rmsle_error}')

def get_ts_month(ds = 'train'):
    print("Get Month")
    month = pd.read_csv(f'/kaggle/input/ashrae-energy-prediction/{ds}.csv')
    month = month['timestamp']
    month = pd.to_datetime(month)
    month = month.dt.month-1
    return month
 
    
def train_predict(X_train, y_train, X_valid, y_valid):
    print("Train predict")
    model = train_model(X_train, y_train, X_valid, y_valid)
    test, row_ids = read_test()
    submission = predict(model, test, row_ids)
    return submission
    
## Parameters
DEBUG=False
categorical_columns = [
    "building_id", "meter", "site_id", "primary_use", "had_air_temperature", "had_cloud_coverage",
    "had_dew_temperature", "had_precip_depth_1_hr", "had_sea_level_pressure", "had_wind_direction",
    "had_wind_speed", "tm_day_of_week", "tm_hour_of_day"]

categoricals = ["building_id", "site_id", "meter", "tm_hour_of_day", "tm_day_of_week",   "primary_use", 'ts_month']# 
numericals = []#"square_feet", "floor_count"]#, "had_air_temperature", "had_dew_temperature"]
columns = categoricals+numericals
logfy = ["square_feet", "floor_count"]
LOG_FEATURES=False
NORMALIZE_FEATURES=False
ADD_MONTH=True

# Optimizer Parameters
EPOCHS = 10
N_ITER = 100


## Main
if __name__=='__main__':
    
    ## 1. Read Datasets
    X_train, y_train, X_valid, y_valid = read_train_valid()
    
    # 2. Train and predict
    submission = train_predict(X_train, y_train, X_valid, y_valid)
    
    # 3. Leak Benchmark
    leak_benchmark(submission)

    # 4. Export Submission
    export(submission)