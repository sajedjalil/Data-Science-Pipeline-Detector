# %% [code]
# Used https://machinelearningmastery.com/seed-state-lstms-time-series-forecasting-python/ to forecasting time series with lstm
# Team name: BigData
# Team Members Sudha Guda, Soujanya Dawalagiri, Vaishnavi Garikipati

# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from datetime import timedelta
import os
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from matplotlib import pyplot as plt
plt.style.use('dark_background')
# Any results you write to the current directory are saved as output.

# %% [code]
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# %% [code]
df_wk1 = pd.read_csv("/kaggle/input/countrydatafile/coordinates.csv")
train_df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")
test_df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv")

# %% [code]
train_df["Date"] = train_df["Date"].astype("datetime64[ms]")
train_df["days"] = (train_df["Date"] - pd.to_datetime("2020-01-01")).dt.days
train_df["weekend"] = train_df["Date"].dt.dayofweek//5
test_df["Date"] = test_df["Date"].astype("datetime64[ms]")
test_df["days"] = (test_df["Date"] - pd.to_datetime("2020-01-01")).dt.days
test_df["weekend"] = test_df["Date"].dt.dayofweek//5

# %% [code]
train_df['Province_State'].fillna("", inplace=True)
test_df['Province_State'].fillna("", inplace=True)
df_wk1['Province/State'].fillna("", inplace=True)

# %% [code]
train_df['location'] = ['_'.join(x) for x in zip(train_df['Country_Region'], train_df['Province_State'])]
test_df['location'] = ['_'.join(x) for x in zip(test_df['Country_Region'], test_df['Province_State'])]
df_wk1['location'] = ['_'.join(x) for x in zip(df_wk1['Country/Region'], df_wk1['Province/State'])]

# %% [code]
# df_location_cord = df_wk1.groupby("location")[["location", "Lat", "Long"]]#.reset_index()
df_location_cord = df_wk1.drop_duplicates('location')[['location', 'Lat', 'Long']]#.drop('ConfirmedCases', 'Fatalities', 'Id', 'Province/State', 'Country/Region', 'Date')

# %% [code]
# # sub = train_df.merge(pred_cases, how='left', on=['geo', 'day'])
# df_location_cord.head(11)
test_df.shape

# %% [code]
train_df_cord = pd.merge(train_df, df_location_cord, on='location', how='inner')
test_df_cord = pd.merge(test_df, df_location_cord, on='location', how='inner')

# %% [code]
# loc_group = ["Country_Region", "Province_State"]

# %% [code]
# train_df.groupby('location')["ConfirmedCases"].shift(1)#.tail(10)

# %% [code]
TARGETS = ["ConfirmedCases", "Fatalities"]
features = ["Lat", "Long"]
for s in range(1, 6):
    for col in TARGETS:
        train_df_cord["prev_{}_{}".format(col, s)] = train_df_cord.groupby('location')[col].shift(s)
#         test_df_cord["prev_{}_{}".format(col, s)] = test_df_cord.groupby('location')[col].shift(s)
        features.append("prev_{}_{}".format(col, s))

# %% [code]
# df_location_cord.location.unique()
# test_df.head(11)
train_df_cord = train_df_cord[train_df_cord["Date"] >= train_df_cord["Date"].min() + timedelta(days=5)].copy()

# %% [code]
# dev_df.columns
tst_start_dt = test_df["Date"].min() # pd.to_datetime("2020-03-13") #
test_days = (train_df["Date"].max() - tst_start_dt).days + 1

dev_df, tst_df = train_df_cord[train_df_cord["Date"] < tst_start_dt].copy(), train_df_cord[train_df_cord["Date"] >= tst_start_dt].copy()

# %% [code]
# test_df = test_df.merge(tst_df[["Date","location"] + TARGETS], how="left", on=["Date", "location"])

# %% [code]
#     def nn_block(input_layer, size, dropout_rate, activation):
#         out_layer = KL.Dense(size, activation=None)(input_layer)
#         #out_layer = KL.BatchNormalization()(out_layer)
#         out_layer = KL.Activation(activation)(out_layer)
#         out_layer = KL.Dropout(dropout_rate)(out_layer)
#         return out_layer


#     def get_model():
#         inp = KL.Input(shape=(len(features),))

#         hidden_layer = nn_block(inp, 128, 0.0, "relu")
#         hidden_layer = nn_block(hidden_layer, 64, 0.0, "relu")
#         gate_layer = nn_block(hidden_layer, 32, 0.0, "sigmoid")
#         hidden_layer = nn_block(hidden_layer, 48, 0.0, "relu")
#         hidden_layer = nn_block(hidden_layer, 32, 0.0, "relu")
#         hidden_layer = KL.multiply([hidden_layer, gate_layer])

#         out = KL.Dense(len(TARGETS), activation="linear")(hidden_layer)

#         model = tf.keras.models.Model(inputs=[inp], outputs=out)
#         return model

# %% [code]
feat_confirm = ['Lat', 'Long', 'prev_ConfirmedCases_1', 'prev_ConfirmedCases_2', 'prev_ConfirmedCases_3', 'prev_ConfirmedCases_4', 'prev_ConfirmedCases_5']
feat_fatal = ['Lat', 'Long', 'prev_Fatalities_1', 'prev_Fatalities_2', 'prev_Fatalities_3', 'prev_Fatalities_4', 'prev_Fatalities_5']
model_confirm = Sequential()
model_confirm.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(len(feat_confirm), 1)))
model_confirm.add(LSTM(50, activation='relu'))
# model_confirm.add(LSTM(50, activation='relu'))
model_confirm.add(Dense(1))
model_confirm.compile(optimizer='adam', loss='mean_squared_logarithmic_error')

# %% [code]
def conv_raw(df_raw, features):
    df_tmp = df_raw[features]
    df_tmp['single_input_vector'] = df_tmp.apply(tuple, axis=1).apply(list)
    df_tmp_list = np.array(df_tmp['single_input_vector'].tolist())
    X_df_tmp_list = df_tmp_list.reshape((df_tmp_list.shape[0], df_tmp_list.shape[1], 1))
    return X_df_tmp_list

# %% [code]
# X_conf = dev_df[feat_confirm]
y_conf = dev_df['ConfirmedCases']

# %% [code]
train_X_conf = conv_raw(dev_df, feat_confirm)

# %% [code]
# X_conf['single_input_vector'] = X_conf.apply(tuple, axis=1).apply(list)
# list_X_conf = np.array(X_conf['single_input_vector'].tolist())

# %% [code]
# train_X_conf = list_X_conf.reshape((list_X_conf.shape[0], list_X_conf.shape[1], 1))

# %% [code]
# # Pad your sequences so they are the same length
# from keras.preprocessing.sequence import pad_sequences

# max_sequence_length = X.cumulative_input_vectors.apply(len).max()
# # Save it as a list   
# padded_sequences = pad_sequences(X.cumulative_input_vectors.tolist(), max_sequence_length).tolist()
# X['padded_input_vectors'] = pd.Series(padded_sequences).apply(np.asarray)

# %% [code]
# # Extract your training data
# X_train_init = np.asarray(X.padded_input_vectors)
# # Use hstack to and reshape to make the inputs a 3d vector
# X_train = np.hstack(X_train_init).reshape(len(X),max_sequence_length,len(input_cols))
# # y_train = np.hstack(np.asarray(df.output_vector)).reshape(len(df),len(output_cols))

# %% [code]
# # Extract your training data
# X_train_init = np.asarray(X.cumulative_input_vectors)
# # Use hstack to and reshape to make the inputs a 3d vector
# X_train = np.hstack(X_train_init).reshape(len(X_train_init),len(X_train_init[0]),1)
# # y_train = np.hstack(np.asarray(y.output_vector)).reshape(len(df),len(output_cols))

# %% [code]
# X_train_init.shape

# %% [code]
history_confirm = model_confirm.fit(train_X_conf, y_conf, epochs=30, batch_size=32, validation_split=0.1, verbose = 1)#,  shuffle=False)

# %% [code]
loss = history_confirm.history['loss']
val_loss = history_confirm.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %% [code]
tst_df_X_confirm = conv_raw(tst_df, feat_confirm)
ypred_test_df_confirm = model_confirm.predict(tst_df_X_confirm, verbose=1)

# %% [code]
X_fatal = dev_df[feat_fatal]
y_fatal = dev_df['Fatalities']
model_fatal = Sequential()
model_fatal.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(len(feat_fatal), 1)))
model_fatal.add(LSTM(50, activation='relu'))
# model_fatal.add(LSTM(50, activation='relu', return_sequences=True))
model_fatal.add(Dense(1))
model_fatal.compile(optimizer='adam', loss='mean_squared_logarithmic_error')

# %% [code]
# X_fatal['single_input_vector'] = X_fatal.apply(tuple, axis=1).apply(list)
# list_X_fatal = np.array(X_fatal['single_input_vector'].tolist())
# train_X_fatal = list_X_fatal.reshape((list_X_fatal.shape[0], list_X_fatal.shape[1], 1))
train_X_fatal = conv_raw(X_fatal, feat_fatal)

# %% [code]
history_fatal = model_fatal.fit(train_X_fatal, y_fatal, epochs=30, batch_size=32, validation_split=0.1, verbose = 1)#,  shuffle=False)

# %% [code]
loss = history_fatal.history['loss']
val_loss = history_fatal.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %% [code]
tst_df_X_fatal = conv_raw(tst_df, feat_fatal)
ypred_test_df_fatal = model_fatal.predict(tst_df_X_fatal, verbose=1)

# %% [code]
# temp_df = test_df_private.loc[test_df_private["Date"] == SUB_FIRST].copy()
# # print(temp_df.columns)
# temp_df_list_confirm = conv_raw(temp_df, feat_confirm)
# temp_df_list_fatal = conv_raw(temp_df, feat_fatal)
# y_pred_confirm = model_confirm.predict(temp_df_list_confirm, verbose=1)
# y_pred_fatal = model_confirm.predict(temp_df_list_fatal, verbose=1)

# %% [code]
test_df_public = test_df[test_df["Date"] <= train_df["Date"].max()].copy()
test_df_private = test_df[test_df["Date"] > train_df["Date"].max()].copy()

pred_cols = ["pred_{}".format(col) for col in TARGETS]
#sub_df_public = sub_df_public.merge(test_df[["Date"] + loc_group + pred_cols].rename(columns={col: col[5:] for col in pred_cols}), 
#                                    how="left", on=["Date"] + loc_group)
test_df_public = test_df_public.merge(tst_df[["Date","location"] + TARGETS], how="left", on=["Date","location"])

SUB_FIRST = test_df_private["Date"].min()
SUB_DAYS = (test_df_private["Date"].max() - test_df_private["Date"].min()).days + 1

test_df_private = train_df.append(test_df_private, sort=False)

for s in range(1, 6):
    for col in TARGETS:
        test_df_private["prev_{}_{}".format(col, s)] = test_df_private.groupby('location')[col].shift(s)

test_df_private = test_df_private[test_df_private["Date"] >= SUB_FIRST].copy()
test_df_private = pd.merge(test_df_private, df_location_cord, on='location', how='left')

# %% [code]
# # test_df_private.iloc[test_df_private.ForecastId==13419]
# test_df_private.loc[test_df_private['ForecastId'] == 14.0]

# %% [code]
# test_df_public.tail(11)

# %% [code]
def predict_cases(org_df, first_date, num_days, model_val, stat, features):
    temp_df = org_df.loc[org_df["Date"] == first_date].copy()
    # print(temp_df.columns)
    temp_df_list = conv_raw(temp_df, features)
    y_pred = model_val.predict(temp_df_list, verbose=1)
    print(y_pred)
    org_df.loc[org_df["Date"] == SUB_FIRST, "pred_{}".format(stat)] = y_pred
    y_prevs = [None]*5
    for i in range(1, 5):
        y_prevs[i] = temp_df[['prev_{}_{}'.format(stat, i)]].values
    #     y_prevs[i] = temp_df[['prev_ConfirmedCases_{}'.format(i), 'prev_Fatalities_{}'.format(i)]].values
    for d in range(1, num_days):
        date_val = first_date + timedelta(days=d)
    #     test_df_private[]
        temp_df = org_df.loc[org_df["Date"] == date_val].copy()
        temp_df['prev_'+stat+'_1'] = y_pred
#         print(temp_df[['prev_ConfirmedCases_1','prev_ConfirmedCases_2']].head(11))
        for i in range(2, 6):
            temp_df[['prev_{}_{}'.format(stat,i)]] = y_prevs[i-1]
#         print(temp_df[['prev_ConfirmedCases_1','prev_ConfirmedCases_2']].head(11))
        temp_df_list = conv_raw(temp_df, features)
        y_pred = model_val.predict(temp_df_list, verbose=1)
        y_prevs = [None, y_pred] + y_prevs[1:-1]
        print(y_pred)
        org_df.loc[org_df["Date"] == date_val, "pred_{}".format(stat)] = y_pred
    return org_df
    #     break
    # #     temp_df[prev_targets] = y_pred

# %% [code]
test_private_res_confirm = predict_cases(test_df_private, SUB_FIRST, SUB_DAYS, model_confirm, 'ConfirmedCases', feat_confirm)
test_private_res_fatal = predict_cases(test_df_private, SUB_FIRST, SUB_DAYS, model_fatal, 'Fatalities', feat_fatal)

# %% [code]
new_test_df = test_private_res_fatal[['ForecastId','pred_ConfirmedCases', 'pred_Fatalities']]
new_test_df = new_test_df.rename(columns={'pred_ConfirmedCases': 'ConfirmedCases', 'pred_Fatalities': 'Fatalities'})

# %% [code]
# test_private_res_confirm[['pred_ConfirmedCases']].head(100)
# test_private_res_fatal.head(11)

# %% [code]
# test_df_public[['ForecastId', 'ConfirmedCases', 'Fatalities']]

# %% [code]
# test_private_res_fatal[['ForecastId', 'ConfirmedCases', 'Fatalities']]

# %% [code]
# test_df.shape

# %% [code]
# final_dataset.shape

# %% [code]
# final_dataset = test_df_public[['ForecastId', 'ConfirmedCases', 'Fatalities']].append(test_private_res_fatal[['ForecastId', 'ConfirmedCases', 'Fatalities']])
final_dataset = pd.concat([test_df_public[['ForecastId', 'ConfirmedCases', 'Fatalities']],test_private_res_fatal[['ForecastId', 'ConfirmedCases', 'Fatalities']]], axis=0)

# %% [code]
final_dataset['ForecastId'] = final_dataset['ForecastId'].astype(int)
final_dataset.fillna(0, inplace=True)
final_dataset.to_csv("submission.csv", index=False)

# # %% [code]
# final_dataset.duplicated(subset=['ForecastId']).any()

# # %% [code]
# test_df_private.loc[test_df_private["Date"] == SUB_FIRST, "pred_{}".format('confirmed')] = y_pred_confirm
# y_prevs_confirm = [None]*5
# for i in range(1, 5):
#     y_prevs_confirm[i] = temp_df[['prev_ConfirmedCases_{}'.format(i)]].values
# #     y_prevs[i] = temp_df[['prev_ConfirmedCases_{}'.format(i), 'prev_Fatalities_{}'.format(i)]].values
# for d in range(1, SUB_DAYS):
#     date_val = SUB_FIRST + timedelta(days=d)
# #     test_df_private[]
#     temp_df = test_df_private.loc[test_df_private["Date"] == date_val].copy()
#     temp_df['prev_ConfirmedCases_1'] = y_pred_confirm
#     print(temp_df[['prev_ConfirmedCases_1','prev_ConfirmedCases_2']].head(11))
#     for i in range(2, 6):
#         temp_df[['prev_ConfirmedCases_{}'.format(i)]] = y_prevs_confirm[i-1]
#     print(temp_df[['prev_ConfirmedCases_1','prev_ConfirmedCases_2']].head(11))
#     temp_df_list = conv_raw(temp_df, feat_confirm)
#     y_pred_confirm = model_confirm.predict(temp_df_list, verbose=1)
#     y_prevs_confirm = [None, y_pred_confirm] + y_prevs_confirm[1:-1]
#     test_df_private.loc[test_df_private["Date"] == date_val, "pred_{}".format('confirmed')] = y_pred_confirm
# #     break
# # #     temp_df[prev_targets] = y_pred

# # %% [code]
# test_df_private.loc[test_df_private["Date"] == SUB_FIRST, "pred_{}".format('fatal')] = y_pred_fatal
# y_prevs_confirm = [None]*5
# for i in range(1, 5):
#     y_prevs_confirm[i] = temp_df[['prev_ConfirmedCases_{}'.format(i)]].values
#     y_prevs[i] = temp_df[['prev_ConfirmedCases_{}'.format(i), 'prev_Fatalities_{}'.format(i)]].values
# for d in range(1, SUB_DAYS):
#     date_val = SUB_FIRST + timedelta(days=d)
# #     test_df_private[]
#     temp_df = test_df_private.loc[test_df_private["Date"] == date_val].copy()
#     temp_df['prev_ConfirmedCases_1'] = y_pred_confirm
#     print(temp_df[['prev_ConfirmedCases_1','prev_ConfirmedCases_2']].head(11))
#     for i in range(2, 6):
#         temp_df[['prev_ConfirmedCases_{}'.format(i)]] = y_prevs_confirm[i-1]
#     print(temp_df[['prev_ConfirmedCases_1','prev_ConfirmedCases_2']].head(11))
#     temp_df_list = conv_raw(temp_df, feat_confirm)
#     y_pred_confirm = model_confirm.predict(temp_df_list, verbose=1)
#     y_prevs_confirm = [None, y_pred_confirm] + y_prevs_confirm[1:-1]
#     test_df_private.loc[test_df_private["Date"] == date_val, "pred_{}".format('confirmed')] = y_pred_confirm
#     break
# #     temp_df[prev_targets] = y_pred

# %% [code]
# temp_df = test_df_private.loc[test_df_private["Date"] == date_val].copy()

# %% [code]
# temp_df['prev_ConfirmedCases_1'] = y_pred_confirm
# print(temp_df[['prev_ConfirmedCases_1','prev_ConfirmedCases_2']].head(100))

# %% [code]
# for i in range(2, 6):
#     temp_df[['prev_ConfirmedCases_{}'.format(i)]] = y_prevs[i-1]
# print(temp_df[['prev_ConfirmedCases_1','prev_ConfirmedCases_2']].head(100))

# %% [code]
# y_prevs[1]

# %% [code]
# date_val = SUB_FIRST + timedelta(days=2)
# temp_df1 = test_df_private.loc[test_df_private["Date"] == date_val].copy()

# %% [code]
# temp_df1[['prev_ConfirmedCases_1','prev_Fatalities_1','prev_ConfirmedCases_2','prev_Fatalities_2']].head(50)

# %% [code]
# temp_df1.head(11)

# %% [code]
