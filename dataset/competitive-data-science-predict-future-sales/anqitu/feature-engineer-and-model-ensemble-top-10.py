Validation = False

reduce_size = False

num_first_level_models = 3 

SEED = 0



import time

start_time = time.time()



import pandas as pd

import numpy as np

import gc

from tqdm import tqdm



pd.set_option('display.max_rows', 99)

pd.set_option('display.max_columns', 50)

import warnings

warnings.filterwarnings('ignore')



# Data path

data_path = '../input'

submission_path = ''





def downcast_dtypes(df):

    float_cols = [c for c in df if df[c].dtype == "float64"]

    int_cols =   [c for c in df if df[c].dtype in ["int64", "int32"]]



    df[float_cols] = df[float_cols].astype(np.float32)

    df[int_cols]   = df[int_cols].astype(np.int16)



    return df





# 0. Load data ----------------------------------------------------------------

print('%0.2f min: Start loading result'%((time.time() - start_time)/60))

result = pd.read_csv('%s/result/ver6_lr_stacking.csv' % data_path)

result.to_csv('ver6_lr_stacking.csv', index = False)

print('%0.2f min: Finish loading result'%((time.time() - start_time)/60))



# It takes long to run the cope, so i comment them and upload my result



# print('%0.2f min: Start loading data'%((time.time() - start_time)/60))

# sale_train = pd.read_csv('%s/sales_train.csv' % data_path)

# test  = pd.read_csv('%s/test.csv' % data_path)



# sale_train[sale_train['item_id'] == 11373][['item_price']].sort_values(['item_price'])

# sale_train[sale_train['item_id'] == 11365].sort_values(['item_price'])

# # Correct sale_train values

# sale_train['item_price'][2909818] = np.nan

# sale_train['item_cnt_day'][2909818] = np.nan

# sale_train['item_price'][2909818] = sale_train[(sale_train['shop_id'] ==12) & (sale_train['item_id'] == 11373) & (sale_train['date_block_num'] == 33)]['item_price'].median()

# sale_train['item_cnt_day'][2909818] = round(sale_train[(sale_train['shop_id'] ==12) & (sale_train['item_id'] == 11373) & (sale_train['date_block_num'] == 33)]['item_cnt_day'].median())

# sale_train['item_price'][885138] = np.nan

# sale_train['item_price'][885138] = sale_train[(sale_train['item_id'] == 11365) & (sale_train['shop_id'] ==12) & (sale_train['date_block_num'] == 8)]['item_price'].median()





# test_nrow = test.shape[0]

# sale_train = sale_train.merge(test[['shop_id']].drop_duplicates(), how = 'inner')

# sale_train['date'] = pd.to_datetime(sale_train['date'], format = '%d.%m.%Y')

# print('%0.2f min: Finish loading data'%((time.time() - start_time)/60))



# # 1. Aggregate data ----------------------------------------------------------------

# from itertools import product



# # For every month we create a grid from all shops/items combinations from that month

# grid = []

# for block_num in sale_train['date_block_num'].unique():

#     cur_shops = sale_train[sale_train['date_block_num']==block_num]['shop_id'].unique()

#     cur_items = sale_train[sale_train['date_block_num']==block_num]['item_id'].unique()

#     grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))



# #turn the grid into pandas dataframe

# index_cols = ['shop_id', 'item_id', 'date_block_num']

# grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int32)



# print('%0.2f min: Finish creating the grid'%((time.time() - start_time)/60))





# index_cols = ['shop_id', 'item_id', 'date_block_num']

# sale_train['item_cnt_day'] = sale_train['item_cnt_day'].clip(0,20)

# gb_cnt = sale_train.groupby(index_cols)['item_cnt_day'].agg(['sum']).reset_index().rename(columns = {'sum': 'item_cnt_month'})

# gb_cnt['item_cnt_month'] = gb_cnt['item_cnt_month'].clip(0,20).astype(np.int)



# #join aggregated data to the grid

# train = pd.merge(grid,gb_cnt,how='left',on=index_cols).fillna(0)

# train['item_cnt_month'] = train['item_cnt_month'].astype(int)

# train = downcast_dtypes(train)



# #sort the data

# train.sort_values(['date_block_num','shop_id','item_id'],inplace=True)

# print('%0.2f min: Finish joining gb_cnt'%((time.time() - start_time)/60))



# # # Sanity check

# # sale_train['item_cnt_day'].sum()

# # train['item_cnt_month'].sum()

# # gb_cnt['item_cnt_month'].sum()





# item = pd.read_csv('%s/items.csv' % data_path)

# train = train.merge(item[['item_id', 'item_category_id']], on = ['item_id'], how = 'left')

# test = test.merge(item[['item_id', 'item_category_id']], on = ['item_id'], how = 'left')

# print('%0.2f min: Finish adding item_category_id'%((time.time() - start_time)/60))





# item_cat = pd.read_csv('%s/item_categories.csv' % data_path)

# l_cat = list(item_cat.item_category_name)

# for ind in range(0,1):

#     l_cat[ind] = 'PC Headsets / Headphones'

# for ind in range(1,8):

#     l_cat[ind] = 'Access'

# l_cat[8] = 'Tickets (figure)'

# l_cat[9] = 'Delivery of goods'

# for ind in range(10,18):

#     l_cat[ind] = 'Consoles'

# for ind in range(18,25):

#     l_cat[ind] = 'Consoles Games'

# l_cat[25] = 'Accessories for games'

# for ind in range(26,28):

#     l_cat[ind] = 'phone games'

# for ind in range(28,32):

#     l_cat[ind] = 'CD games'

# for ind in range(32,37):

#     l_cat[ind] = 'Card'

# for ind in range(37,43):

#     l_cat[ind] = 'Movie'

# for ind in range(43,55):

#     l_cat[ind] = 'Books'

# for ind in range(55,61):

#     l_cat[ind] = 'Music'

# for ind in range(61,73):

#     l_cat[ind] = 'Gifts'

# for ind in range(73,79):

#     l_cat[ind] = 'Soft'

# for ind in range(79,81):

#     l_cat[ind] = 'Office'

# for ind in range(81,83):

#     l_cat[ind] = 'Clean'

# l_cat[83] = 'Elements of a food'



# from sklearn import preprocessing

# lb = preprocessing.LabelEncoder()

# item_cat['item_cat_id_fix'] = lb.fit_transform(l_cat)



# train = train.merge(item_cat[['item_cat_id_fix', 'item_category_id']], on = ['item_category_id'], how = 'left')

# test = test.merge(item_cat[['item_cat_id_fix', 'item_category_id']], on = ['item_category_id'], how = 'left')



# del item, item_cat, grid, gb_cnt

# gc.collect()

# print('%0.2f min: Finish adding item_cat_id_fix'%((time.time() - start_time)/60))







# # 2. Add item/shop pair mean-encodings -----------------------------------------

# # For Trainset

# print('%0.2f min: Start adding mean-encoding for item_cnt_month'%((time.time() - start_time)/60))

# Target = 'item_cnt_month'

# global_mean =  train[Target].mean()

# y_tr = train[Target].values



# mean_encoded_col = ['shop_id', 'item_id', 'item_category_id', 'item_cat_id_fix']

# for col in tqdm(mean_encoded_col):



#     col_tr = train[[col] + [Target]]

#     corrcoefs = pd.DataFrame(columns = ['Cor'])



#     # 3.1.1 Mean encodings - KFold scheme

#     from sklearn.model_selection import KFold

#     kf = KFold(n_splits = 5, shuffle = False, random_state = SEED)



#     col_tr[col + '_cnt_month_mean_Kfold'] = global_mean

#     for tr_ind, val_ind in kf.split(col_tr):

#         X_tr, X_val = col_tr.iloc[tr_ind], col_tr.iloc[val_ind]

#         means = X_val[col].map(X_tr.groupby(col)[Target].mean())

#         X_val[col + '_cnt_month_mean_Kfold'] = means

#         col_tr.iloc[val_ind] = X_val

#         # X_val.head()

#     col_tr.fillna(global_mean, inplace = True)

#     corrcoefs.loc[col + '_cnt_month_mean_Kfold'] = np.corrcoef(y_tr, col_tr[col + '_cnt_month_mean_Kfold'])[0][1]



#     # 3.1.2 Mean encodings - Leave-one-out scheme

#     item_id_target_sum = col_tr.groupby(col)[Target].sum()

#     item_id_target_count = col_tr.groupby(col)[Target].count()

#     col_tr[col + '_cnt_month_sum'] = col_tr[col].map(item_id_target_sum)

#     col_tr[col + '_cnt_month_count'] = col_tr[col].map(item_id_target_count)

#     col_tr[col + '_target_mean_LOO'] = (col_tr[col + '_cnt_month_sum'] - col_tr[Target]) / (col_tr[col + '_cnt_month_count'] - 1)

#     col_tr.fillna(global_mean, inplace = True)

#     corrcoefs.loc[col + '_target_mean_LOO'] = np.corrcoef(y_tr, col_tr[col + '_target_mean_LOO'])[0][1]





#     # 3.1.3 Mean encodings - Smoothing

#     item_id_target_mean = col_tr.groupby(col)[Target].mean()

#     item_id_target_count = col_tr.groupby(col)[Target].count()

#     col_tr[col + '_cnt_month_mean'] = col_tr[col].map(item_id_target_mean)

#     col_tr[col + '_cnt_month_count'] = col_tr[col].map(item_id_target_count)

#     alpha = 100

#     col_tr[col + '_cnt_month_mean_Smooth'] = (col_tr[col + '_cnt_month_mean'] *  col_tr[col + '_cnt_month_count'] + global_mean * alpha) / (alpha + col_tr[col + '_cnt_month_count'])

#     col_tr[col + '_cnt_month_mean_Smooth'].fillna(global_mean, inplace=True)

#     corrcoefs.loc[col + '_cnt_month_mean_Smooth'] = np.corrcoef(y_tr, col_tr[col + '_cnt_month_mean_Smooth'])[0][1]





#     # 3.1.4 Mean encodings - Expanding mean scheme

#     cumsum = col_tr.groupby(col)[Target].cumsum() - col_tr[Target]

#     sumcnt = col_tr.groupby(col).cumcount()

#     col_tr[col + '_cnt_month_mean_Expanding'] = cumsum / sumcnt

#     col_tr[col + '_cnt_month_mean_Expanding'].fillna(global_mean, inplace=True)

#     corrcoefs.loc[col + '_cnt_month_mean_Expanding'] = np.corrcoef(y_tr, col_tr[col + '_cnt_month_mean_Expanding'])[0][1]



#     train = pd.concat([train, col_tr[corrcoefs['Cor'].idxmax()]], axis = 1)

#     print(corrcoefs.sort_values('Cor'))

#     print('%0.2f min: Finish encoding %s'%((time.time() - start_time)/60, col))



# print('%0.2f min: Finish adding mean-encoding'%((time.time() - start_time)/60))





# # 2. Feature Engineering -----------------------------------------

# # 2.1 Combine trainset and testset -----------------------------------------

# print('%0.2f min: Start combining data'%((time.time() - start_time)/60))

# if Validation == False:

#     test['date_block_num'] = 34

#     all_data = pd.concat([train, test], axis = 0)

#     all_data = all_data.drop(columns = ['ID'])

# else:

#     all_data = train



# del train, test, col_tr

# gc.collect()



# all_data = downcast_dtypes(all_data)







# # 2.2 Creating item/shop pair lags lag-based features ----------------------------

# print('%0.2f min: Start adding lag-based feature'%((time.time() - start_time)/60))

# index_cols = ['shop_id', 'item_id', 'item_category_id', 'item_cat_id_fix', 'date_block_num']

# cols_to_rename = list(all_data.columns.difference(index_cols))

# print(cols_to_rename)

# shift_range = [1, 2, 3, 4, 12]



# for month_shift in tqdm(shift_range):

#     train_shift = all_data[index_cols + cols_to_rename].copy()



#     train_shift['date_block_num'] = train_shift['date_block_num'] + month_shift



#     foo = lambda x: '{}_lag_{}'.format(x, month_shift) if x in cols_to_rename else x

#     train_shift = train_shift.rename(columns=foo)



#     all_data = pd.merge(all_data, train_shift, on=index_cols, how='left').fillna(0)



# del train_shift

# gc.collect()



# all_data = all_data[all_data['date_block_num'] >= 12] # Don't use old data from year 2013

# lag_cols = [col for col in all_data.columns if col[-1] in [str(item) for item in shift_range]]

# all_data = downcast_dtypes(all_data)

# print('%0.2f min: Finish generating lag features'%((time.time() - start_time)/60))





# # 2.3 Creating date features --------------------------------------------------------

# print('%0.2f min: Start getting date features'%((time.time() - start_time)/60))



# dates_train = sale_train[['date', 'date_block_num']].drop_duplicates()

# dates_test = dates_train[dates_train['date_block_num'] == 34-12]

# dates_test['date_block_num'] = 34

# dates_test['date'] = dates_test['date'] + pd.DateOffset(years=1)

# dates_all = pd.concat([dates_train, dates_test])



# dates_all['dow'] = dates_all['date'].dt.dayofweek

# dates_all['year'] = dates_all['date'].dt.year

# dates_all['month'] = dates_all['date'].dt.month

# dates_all = pd.get_dummies(dates_all, columns=['dow'])

# dow_col = ['dow_' + str(x) for x in range(7)]

# date_features = dates_all.groupby(['year', 'month', 'date_block_num'])[dow_col].agg('sum').reset_index()

# date_features['days_of_month'] = date_features[dow_col].sum(axis=1)

# date_features['year'] = date_features['year'] - 2013



# date_features = date_features[['month', 'year', 'days_of_month', 'date_block_num']]

# all_data = all_data.merge(date_features, on = 'date_block_num', how = 'left')

# date_columns = date_features.columns.difference(set(index_cols))

# print('%0.2f min: Finish getting date features'%((time.time() - start_time)/60))







# # 2.4 Scale feature columns --------------------------------------------------------

# from sklearn.preprocessing import StandardScaler

# train = all_data[all_data['date_block_num']!= all_data['date_block_num'].max()]

# test = all_data[all_data['date_block_num']== all_data['date_block_num'].max()]

# sc = StandardScaler()



# to_drop_cols = ['date_block_num']

# feature_columns = list(set(lag_cols + index_cols + list(date_columns)).difference(to_drop_cols))



# train[feature_columns] = sc.fit_transform(train[feature_columns])

# test[feature_columns] = sc.transform(test[feature_columns])

# all_data = pd.concat([train, test], axis = 0)

# all_data = downcast_dtypes(all_data)



# del train, test, date_features, sale_train

# gc.collect()

# print('%0.2f min: Finish scaling features'%((time.time() - start_time)/60))





# # 3. First-level model ------------------------------------------------------------------

# # Save `date_block_num`, as we can't use them as features, but will need them to split the dataset into parts

# dates = all_data['date_block_num']

# last_block = dates.max()

# print('Test `date_block_num` is %d' % last_block)

# print(feature_columns)



# print('%0.2f min: Start training First level models'%((time.time() - start_time)/60))

# start_first_level_total = time.perf_counter()



# scoringMethod = 'r2'; from sklearn.metrics import mean_squared_error; from math import sqrt





# # Train meta-features M = 15 (12 + 15 = 27)

# months_to_generate_meta_features = range(27,last_block +1)

# mask = dates.isin(months_to_generate_meta_features)

# Target = 'item_cnt_month'

# y_all_level2 = all_data[Target][mask].values

# X_all_level2 = np.zeros([y_all_level2.shape[0], num_first_level_models])





# # Now fill `X_train_level2` with metafeatures

# slice_start = 0



# for cur_block_num in tqdm(months_to_generate_meta_features):



#     print('-' * 50)

#     print('Start training for month%d'% cur_block_num)

#     start_cur_month = time.perf_counter()



#     cur_X_train = all_data.loc[dates <  cur_block_num][feature_columns]

#     cur_X_test =  all_data.loc[dates == cur_block_num][feature_columns]



#     cur_y_train = all_data.loc[dates <  cur_block_num, Target].values

#     cur_y_test =  all_data.loc[dates == cur_block_num, Target].values



#     # Create Numpy arrays of train, test and target dataframes to feed into models

#     train_x = cur_X_train.values

#     train_y = cur_y_train.ravel()

#     test_x = cur_X_test.values

#     test_y = cur_y_test.ravel()



#     preds = []



#     from sklearn.linear_model import (LinearRegression, SGDRegressor)

#     import lightgbm as lgb



#     sgdr= SGDRegressor(

#         penalty = 'l2' ,

#         random_state = SEED )

#     lgb_params = {

#                   'feature_fraction': 0.75,

#                   'metric': 'rmse',

#                   'nthread':1,

#                   'min_data_in_leaf': 2**7,

#                   'bagging_fraction': 0.75,

#                   'learning_rate': 0.03,

#                   'objective': 'mse',

#                   'bagging_seed': 2**7,

#                   'num_leaves': 2**7,

#                   'bagging_freq':1,

#                   'verbose':0

#                   }



#     estimators = [sgdr]



#     for estimator in estimators:

#         print('Training Model %d: %s'%(len(preds), estimator.__class__.__name__))

#         start = time.perf_counter()

#         estimator.fit(train_x, train_y)

#         pred_test = estimator.predict(test_x)

#         preds.append(pred_test)

#         # pred_train = estimator.predict(train_x)

#         # print('Train RMSE for %s is %f' % (estimator.__class__.__name__, sqrt(mean_squared_error(cur_y_train, pred_train))))

#         # print('Test RMSE for %s is %f' % (estimator.__class__.__name__, sqrt(mean_squared_error(cur_y_test, pred_test))))

#         run = time.perf_counter() - start

#         print('{} runs for {:.2f} seconds.'.format(estimator.__class__.__name__, run))

#         print()





#     print('Training Model %d: %s'%(len(preds), 'lightgbm'))

#     start = time.perf_counter()

#     estimator = lgb.train(lgb_params, lgb.Dataset(train_x, label=train_y), 300)

#     pred_test = estimator.predict(test_x)

#     preds.append(pred_test)

#     # pred_train = estimator.predict(train_x)

#     # print('Train RMSE for %s is %f' % ('lightgbm', sqrt(mean_squared_error(cur_y_train, pred_train))))

#     # print('Test RMSE for %s is %f' % ('lightgbm', sqrt(mean_squared_error(cur_y_test, pred_test))))

#     run = time.perf_counter() - start

#     print('{} runs for {:.2f} seconds.'.format('lightgbm', run))

#     print()





#     print('Training Model %d: %s'%(len(preds), 'keras'))

#     start = time.perf_counter()

#     from keras.models import Sequential

#     from keras.layers import Dense

#     from keras.wrappers.scikit_learn import KerasRegressor



#     def baseline_model():

#     	# create model

#         model = Sequential()

#         model.add(Dense(20, input_dim=train_x.shape[1], kernel_initializer='uniform', activation='softplus'))

#         model.add(Dense(1, kernel_initializer='uniform', activation = 'relu'))

#         # Compile model

#         model.compile(loss='mse', optimizer='Nadam', metrics=['mse'])

#         # model.compile(loss='mean_squared_error', optimizer='adam')

#         return model



#     estimator = KerasRegressor(build_fn=baseline_model, verbose=1, epochs=5, batch_size = 55000)



#     estimator.fit(train_x, train_y)

#     pred_test = estimator.predict(test_x)

#     preds.append(pred_test)



#     run = time.perf_counter() - start

#     print('{} runs for {:.2f} seconds.'.format('lightgbm', run))





#     cur_month_run_total = time.perf_counter() - start_cur_month

#     print('Total running time was {:.2f} minutes.'.format(cur_month_run_total/60))

#     print('-' * 50)



#     slice_end = slice_start + cur_X_test.shape[0]

#     X_all_level2[ slice_start : slice_end , :] = np.c_[preds].transpose()

#     slice_start = slice_end





# # Split train and test

# test_nrow = len(preds[0])

# X_train_level2 = X_all_level2[ : -test_nrow, :]

# X_test_level2 = X_all_level2[ -test_nrow: , :]

# y_train_level2 = y_all_level2[ : -test_nrow]

# y_test_level2 = y_all_level2[ -test_nrow : ]



# print('%0.2f min: Finish training First level models'%((time.perf_counter() - start_first_level_total)/60))





# # 4. Ensembling -------------------------------------------------------------------

# pred_list = {}



# # A. Second level learning model via linear regression

# print('Training Second level learning model via linear regression')



# from sklearn.linear_model import (LinearRegression, SGDRegressor)

# lr = LinearRegression()

# lr.fit(X_train_level2, y_train_level2)

# # Compute R-squared on the train and test sets.

# # print('Train R-squared for %s is %f' %('test_preds_lr_stacking', sqrt(mean_squared_error(y_train_level2, lr.predict(X_train_level2)))))

# test_preds_lr_stacking = lr.predict(X_test_level2)

# train_preds_lr_stacking = lr.predict(X_train_level2)

# print('Train R-squared for %s is %f' %('train_preds_lr_stacking', sqrt(mean_squared_error(y_train_level2, train_preds_lr_stacking))))



# pred_list['test_preds_lr_stacking'] = test_preds_lr_stacking

# if Validation:

#     print('Test R-squared for %s is %f' %('test_preds_lr_stacking', sqrt(mean_squared_error(y_test_level2, test_preds_lr_stacking))))





# # B. Second level learning model via SGDRegressor

# print('Training Second level learning model via SGDRegressor')

# sgdr= SGDRegressor(

#     penalty = 'l2' ,

#     random_state = SEED )



# sgdr.fit(X_train_level2, y_train_level2)

# # Compute R-squared on the train and test sets.

# # print('Train R-squared for %s is %f' %('test_preds_lr_stacking', sqrt(mean_squared_error(y_train_level2, lr.predict(X_train_level2)))))

# test_preds_sgdr_stacking = sgdr.predict(X_test_level2)

# train_preds_sgdr_stacking = sgdr.predict(X_train_level2)

# print('Train R-squared for %s is %f' %('train_preds_lr_stacking', sqrt(mean_squared_error(y_train_level2, train_preds_sgdr_stacking))))



# pred_list['test_preds_sgdr_stacking'] = test_preds_sgdr_stacking

# if Validation:

#     print('Test R-squared for %s is %f' %('test_preds_sgdr_stacking', sqrt(mean_squared_error(y_test_level2, test_preds_sgdr_stacking))))





# print('%0.2f min: Finish training second level model'%((time.time() - start_time)/60))







# # Submission -------------------------------------------------------------------

# if not Validation:

#     submission = pd.read_csv('%s/sample_submission.csv' % data_path)



#     ver = 6

#     for pred_ver in ['lr_stacking', 'sgdr_stacking']:

#         print(pred_list['test_preds_' + pred_ver].clip(0,20).mean())

#         submission['item_cnt_month'] = pred_list['test_preds_' + pred_ver].clip(0,20)

#         submission[['ID', 'item_cnt_month']].to_csv('%s/ver%d_%s.csv' % (submission_path, ver, pred_ver), index = False)



# print('%0.2f min: Finish running scripts'%((time.time() - start_time)/60))
