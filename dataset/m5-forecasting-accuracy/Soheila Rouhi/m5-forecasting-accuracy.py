import pandas as pd
import numpy as np
import lightgbm as lgb


###### prepare x_test and extract useful columns from calendar ################
cal = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')

useless_cols=['event_type_1', 'event_name_2', 'event_type_2']
cal=cal.drop(columns=useless_cols)
cal['event_name_1']=cal['event_name_1'].astype('category')
cal_evt=cal.iloc[0: 1941, 7]
## Encode event names
cal_evt_encoded=pd.get_dummies(cal_evt, 'event_name_1', dummy_na=False) # Convert all categorical features to numerical values; ignore NA values
cal_final=pd.merge(cal, cal_evt_encoded, left_index=True, right_index=True)
col_refined=[c.replace('event_name_1_', '') for c in cal_final.columns]
cal_final.columns=col_refined
#print(cal_final.columns)

cal_Cinco=cal_final.iloc[:, 13]
cal_MotherDay=cal_final.iloc[:, 26]
cal_OrthodoxEaster=cal_final.iloc[:, 31]
cal_Pesach=cal_final.iloc[:, 32]

cal_snap_CA=cal.iloc[0: 1941, 8]
cal_snap_TX=cal.iloc[0: 1941, 9]
cal_snap_WI=cal.iloc[0: 1941, 10]

cal_wday = cal.iloc[0: 1941, 3]
cal_month = cal.iloc[0: 1941, 4]
cal_year = cal.iloc[0: 1941, 5]

x_test = cal.iloc[1941:, [0, 3, 4, 5]]
x_test['date'] = pd.to_datetime(x_test['date'], format="%Y-%m-%d")
x_test = x_test.set_index('date')

# SUBMISSION Column names
Fs = []
for i in range(1, 29):
    Fs.append('F' + str(i))
	
##################################################################################
sales_eval = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv')
id_col=sales_eval[['id']]
real_data=sales_eval.iloc[:, 1919: ]
real_data.columns=Fs

d_columns = [dates for dates in sales_eval.columns if 'd_' in dates] # select date columns;
sales_eval=sales_eval.loc[:, d_columns]
df_submission=pd.DataFrame()

for i in range(0, sales_eval.shape[0]):
    print('this is i:'+ str(i))
    sample_sales= sales_eval.iloc[[i]]
    samp_sales=sample_sales.T.copy()
    samp_sales.columns = ['sold']
    samp_sales['date'] = cal.iloc[0: 1941, 0].values
    samp_sales['date'] = pd.to_datetime(samp_sales['date'], format="%Y-%m-%d")
    samp_sales = samp_sales.set_index('date')

    ##############################################################
    ######################## Light GBM ###########################
    samp_sales['weekday'] = cal_wday.values  ## without .values it adds NaN to weekday column
    samp_sales['month'] = cal_month.values
    samp_sales['year'] = cal_year.values
    samp_sales['snap_CA'] = cal_snap_CA.values
    samp_sales['snap_TX'] = cal_snap_TX.values
    samp_sales['snap_WI'] = cal_snap_WI.values
    samp_sales['Cinco'] = cal_Cinco.values
    samp_sales['MotherDay'] = cal_MotherDay.values
    samp_sales['OrthodoxEaster'] = cal_OrthodoxEaster.values
    samp_sales['Pesach'] = cal_Pesach.values

    X = samp_sales.iloc[:, 1:11].values ## in first submission it was 1:4
    y = samp_sales.sold.values

    params = {
        'num_leaves': 555,
        'min_child_weight': 0.034,
        'feature_fraction': 0.379,
        'bagging_fraction': 0.418,
        'min_data_in_leaf': 106,
        'objective': 'regression',
        'max_depth': -1,
        'learning_rate': 0.007,
        "boosting_type": "gbdt",
        "bagging_seed": 11,
        "metric": 'rmse',
        "verbosity": -1,
        'reg_alpha': 0.3899,
        'reg_lambda': 0.648,
        'random_state': 666,
    }

    lgb_train = lgb.Dataset(X, label=y)

    # Train LightGBM model
    m_lgb = lgb.train(params, lgb_train, 100)
    y_pred = m_lgb.predict(x_test)


    def col_value(a, Fcol_list, pred):
        colname = str(Fcol_list[a])
        value = np.round(pred[a], 6)
        dictionary = {colname: value}
        return dictionary
    list_dicts = []
    for i in range(0, 28):
        list_dicts.append(col_value(i, Fs, y_pred))
    full_dict = {}
    for dict in list_dicts:
        full_dict.update(dict)
    df_sub = pd.DataFrame(full_dict, index=[0])
    df_submission=df_submission.append(df_sub, ignore_index=True)

################ Create final submission file ##########################################################################
sales_valid = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')
eval_result=pd.concat([id_col, df_submission], axis=1)
valid_result=pd.concat([id_col, real_data], axis=1)
valid_result['id'] = valid_result.id.str.replace("evaluation", "validation")
submission=pd.concat([valid_result, eval_result],  ignore_index=True)
submission.to_csv('Submission-LGBM.csv', index=False)

########################################################################################################################