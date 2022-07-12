import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_log_error

def RMSLE(y_true, y_pred):
    del_idx = np.arange(len(y_true))[y_true == 0]
    y_true = np.delete(y_true, del_idx)
    y_pred = np.delete(y_pred, del_idx)
    y_pred = y_pred.clip(min=0.)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


def deal_with_outlier(df, columns, scale=2.0):
    df = df.copy()
    for si in df['air_store_id'].unique():
        store_idx = df['air_store_id'] == si
        for col in columns:
            q25 = df.loc[store_idx, col].quantile(.25)
            q75 = df.loc[store_idx, col].quantile(.75)
            IQR = q75 - q25
            threshold = q75 + scale * IQR
            df.loc[store_idx & (df[col] >= threshold), col] = q75 + scale * IQR
    return df


def calc_shifted_ewm(series, alpha=0.1, adjust=True):
    return series.shift().ewm(alpha=alpha, adjust=adjust).mean()


def calc_ewm_df(df, test_start, alpha=0.1, adjust=True):
    for store_id in range(821):
        store_idx = df['air_store_id'] == store_id
        store_df = df.loc[store_idx, ['visit_date', 'visitors', 'dow']]\
            .sort_values('visit_date').reset_index(drop=True)

        ewm = store_df.set_index('visit_date')\
            .groupby('dow')\
            .apply(lambda x: calc_shifted_ewm(x['visitors'], alpha, adjust))\
            .reset_index()\
            .rename(columns={'visitors': 'pred'})

        store_df = pd.merge(store_df, ewm)
        store_df['air_store_id'] = store_id
        store_train = store_df[store_df['visit_date'] < test_start]
        store_test = store_df[store_df['visit_date'] >= test_start].copy()

        # Test
        ewm_vals = []
        nan_dows = []
        for dow in store_test['dow'].unique():
            valid_first_day = store_test.loc[store_test['dow'] == dow, 'visit_date'].min(
            )
            ewm_val = ewm.loc[ewm['visit_date'] ==
                              valid_first_day, 'pred'].values[0]
            if np.isnan(ewm_val):
                nan_dows.append(dow)
            else:
                ewm_vals.append(ewm_val)
                store_test.loc[store_test['dow'] == dow, 'pred'] = ewm_val

        for dow in nan_dows:
            if not ewm_vals:
                ewm_val = store_df['visitors'].mean()
            else:
                ewm_val = np.mean(ewm_vals)
            store_test.loc[store_test['dow'] == dow, 'pred'] = ewm_val

        # Summary
        if store_id == 0:
            store_train_summary = store_train
            store_test_summary = store_test
        else:
            store_train_summary = pd.concat([store_train_summary, store_train])
            store_test_summary = pd.concat([store_test_summary, store_test])

    return store_train_summary, store_test_summary

lbl = LabelEncoder()

train_df = pd.read_csv('../input/air_visit_data.csv')
test_df = pd.read_csv('../input/sample_submission.csv')
holiday_df = pd.read_csv('../input/date_info.csv').rename(columns={'calendar_date':'visit_date'})

train_air_id = list(train_df['air_store_id'].unique())
test_air_id = list(set([l[:-11] for l in list(test_df['id'])]))
not_in_test = list(set(train_air_id) - set(test_air_id))
for nit in not_in_test:
    train_df = train_df[train_df['air_store_id'] != nit]

test_df['air_store_id'] = test_df.apply(lambda x: '_'.join(x['id'].split('_')[:2]), axis=1)
test_df['visit_date'] = test_df.apply(lambda x: x['id'].split('_')[-1], axis=1)
test_df.drop('id', axis=1, inplace=True)

train_df['visit_date'] = pd.to_datetime(train_df['visit_date']).dt.date
test_df['visit_date'] = pd.to_datetime(test_df['visit_date']).dt.date
test_df['visitors'] = np.NaN

all_df = pd.concat([train_df, test_df])
start_day = all_df['visit_date'].min()
end_day = all_df['visit_date'].max()

store_date_all = pd.concat([pd.DataFrame({
    'air_store_id': tai,
    'visit_date': pd.date_range(start_day, end_day).date,
    'id' : ['{}_{}'.format(tai, str(date)) for date in pd.date_range(
                start_day, end_day).date]
    }) for tai in test_air_id],
    axis=0,
    ignore_index=True).reset_index(drop=True)

all_df = pd.merge(store_date_all, all_df, how='left')

store_lbl = LabelEncoder()
all_df['air_store_id'] = store_lbl.fit_transform(all_df['air_store_id'])

date_lbl = LabelEncoder()
all_df['visit_date'] = date_lbl.fit_transform(all_df['visit_date'])

tra_idx = all_df['visit_date'] < 439
val_idx = (all_df['visit_date'] >= 439) & (all_df['visit_date'] < 478)
train_idx = all_df['visit_date'] < 478
test_idx = all_df['visit_date'] >= 478

all_df = deal_with_outlier(all_df, ['visitors'])
all_df.fillna(0, inplace=True)

valid_gt = all_df.loc[val_idx, ['air_store_id', 'visit_date', 'visitors']]

all_df['visitors'] = np.log1p(all_df['visitors'])
all_df_base = all_df.copy()

# Holidays
holiday_df['holiday_flg'] = holiday_df['holiday_flg'].astype('bool')
holiday_df['visit_date'] = pd.to_datetime(holiday_df['visit_date'])
holiday_df['dow'] = lbl.fit_transform(holiday_df['visit_date'].dt.dayofweek)
holiday_df['visit_date'] = date_lbl.transform(holiday_df['visit_date'].dt.date)
holiday_df.drop('day_of_week', axis=1, inplace=True)

holiday_df1 = holiday_df.copy()
holiday_df1.loc[holiday_df1['holiday_flg'], 'dow'] = 6
holiday_df1.loc[holiday_df1['dow'].isin([5, 6]), 'holiday_flg'] = True
holiday_flg = holiday_df1['holiday_flg'].values
before_holiday_flg = np.append(holiday_flg[1:], False)
holiday_df1['before_holiday_flg'] = before_holiday_flg
holiday_df1.loc[holiday_df1['holiday_flg'] & holiday_df1['before_holiday_flg'], 'dow'] = 5
holiday_df1.loc[holiday_df1['dow'].isin(np.arange(5)) & holiday_df1['before_holiday_flg'], 'dow'] = 4
holiday_df1 = holiday_df1[['visit_date', 'dow']]

holiday_df2 = holiday_df.copy()
holiday_df2.loc[holiday_df2['dow'].isin([5, 6]), 'holiday_flg'] = False
holiday_df2.loc[holiday_df2['holiday_flg'], 'dow'] = 7
holiday_df2 = holiday_df2[['visit_date', 'dow']]

holiday_df3 = holiday_df.copy()
holiday_df3.loc[holiday_df3['dow'].isin([5, 6]), 'holiday_flg'] = 1
holiday_flg = holiday_df3['holiday_flg'].values
before_holiday_flg = np.append(holiday_flg[1:], 0)
holiday_df3['before_holiday_flg'] = before_holiday_flg
num_days_off = []
tmp = []
for d in holiday_flg:
    if d == 0:
        num_days_off.extend(tmp+[0])
        tmp = []
    else:
        tmp = [len(tmp)+1]*(len(tmp)+1)
num_days_off = [ndo if ndo < 3 else 3 for ndo in num_days_off]
holiday_df3['dow'] = num_days_off
holiday_df3 = holiday_df3[['visit_date', 'dow']]

holiday_df4 = holiday_df.copy()
holiday_df4.loc[holiday_df4['dow'].isin([5, 6]), 'holiday_flg'] = 1
holiday_flg = holiday_df4['holiday_flg'].values
before_holiday_flg = np.append(holiday_flg[1:], 0)
holiday_df4['before_holiday_flg'] = before_holiday_flg
holiday_df4['new_dow'] = 0
holiday_df4.loc[holiday_df4['holiday_flg'], 'new_dow'] = 3
holiday_df4.loc[holiday_df4['holiday_flg'] & holiday_df4['before_holiday_flg'], 'new_dow'] = 2
holiday_df4.loc[holiday_df4['dow'].isin(np.arange(5)) & holiday_df4['before_holiday_flg'], 'new_dow'] = 1
holiday_df4 = holiday_df4[['visit_date', 'new_dow']].rename(columns={'new_dow':'dow'})

holiday_df5 = holiday_df.copy()
holiday_df5.loc[holiday_df5['dow'].isin([5, 6]), 'holiday_flg'] = 1
holiday_flg = holiday_df5['holiday_flg'].values
before_holiday_flg = np.append(holiday_flg[1:], 0)
holiday_df5['before_holiday_flg'] = before_holiday_flg
holiday_df5['dow'] = 0
holiday_df5.loc[holiday_df5['before_holiday_flg']==1, 'dow'] = 1
holiday_df5 = holiday_df5[['visit_date', 'dow']]

holiday_df6 = holiday_df.copy()
holiday_df6.loc[holiday_df6['dow'].isin([5, 6]), 'holiday_flg'] = 1
holiday_df6['dow'] = 0
holiday_df6.loc[holiday_df6['holiday_flg']==1, 'dow'] = 1
holiday_df6 = holiday_df6[['visit_date', 'dow']]

holiday_df7 = holiday_df.copy()
holiday_df7 = holiday_df7[['visit_date', 'dow']]

holiday_df8 = holiday_df.copy()
holiday_df8.loc[holiday_df8['dow'].isin([5, 6]), 'holiday_flg'] = False
holiday_df8['dow'] = lbl.fit_transform(holiday_df8.apply(lambda x: str(x['holiday_flg']) + str(x['dow']), axis=1))
holiday_df8 = holiday_df8[['visit_date', 'dow']]

# EWM
for i in range(1, 9):
    all_df = pd.merge(all_df_base, eval('holiday_df{}'.format(i)), how='left')

    # Validation    
    tra_df = all_df.drop(all_df.index[tra_idx & (all_df['visitors'] == 0)]).loc[tra_idx].copy()
    val_df = all_df.loc[val_idx].copy()

    df = pd.concat([tra_df, val_df])
    store_tra_summary, store_val_summary = calc_ewm_df(df, 439)

    tra_df = pd.merge(tra_df, store_tra_summary)
    val_df = pd.merge(val_df, store_val_summary)
    tra_df.dropna(inplace=True)

    # Test
    train_df = all_df.drop(all_df.index[train_idx & (all_df['visitors'] == 0)]).loc[train_idx].copy()
    test_df = all_df.loc[test_idx].copy()

    df = pd.concat([train_df, test_df])
    store_train_summary, store_test_summary = calc_ewm_df(df, 478)

    train_df = pd.merge(train_df, store_train_summary)
    test_df = pd.merge(test_df, store_test_summary)
    train_df.dropna(inplace=True)
    
    # Summarize
    valid_pred = store_val_summary[['air_store_id', 'visit_date', 'pred']].copy()
    valid_pred['pred'] = np.expm1(valid_pred['pred']).clip(lower=0.)
    valid_pred.rename(columns={'pred': 'type{}'.format(i)}, inplace=True)
    if i == 1:
        valid_summary_df = valid_pred
    else:
        valid_summary_df = pd.merge(valid_summary_df, valid_pred)

    test_pred = store_test_summary[['air_store_id', 'visit_date', 'pred']].copy()
    test_pred['pred'] = np.expm1(test_pred['pred']).clip(lower=0.)
    test_pred.rename(columns={'pred': 'type{}'.format(i)}, inplace=True)
    if i == 1:
        test_summary_df = test_pred
    else:
        test_summary_df = pd.merge(test_summary_df, test_pred)

# Best predictors for validation
predictor_list = np.array(valid_summary_df.columns[2:])

valid_summary_df = pd.merge(valid_summary_df, valid_gt)
valid_summary_df['best_pred'] = 0

num_top = 4
best_columns = []
for store_id in range(821):
    store_idx = valid_summary_df['air_store_id'] == store_id
    errs = []
    for pr in predictor_list:
        y_true = valid_summary_df.loc[store_idx, 'visitors'].values
        y_pred = valid_summary_df.loc[store_idx, pr].values
        errs.append(RMSLE(y_true, y_pred))
    best_columns.append(predictor_list[np.argsort(errs)[:num_top]].tolist())
    valid_summary_df.loc[store_idx, 'best_pred'] =\
        valid_summary_df.loc[store_idx, predictor_list[np.argsort(errs)[:num_top]].tolist()]\
            .mean(axis=1).values

# Prediction with best predictors
test_summary_df['best_pred'] = 0
for store_id, bi in enumerate(best_columns):
    store_idx = test_summary_df['air_store_id'] == store_id
    test_summary_df.loc[store_idx, 'best_pred'] =\
        test_summary_df.loc[store_idx, bi].mean(axis=1).values

submission_df = test_summary_df[['air_store_id', 'visit_date', 'best_pred']].copy()
submission_df['air_store_id'] = store_lbl.inverse_transform(submission_df['air_store_id'])
submission_df['visit_date'] = date_lbl.inverse_transform(submission_df['visit_date'])
submission_df.rename(columns={'best_pred': 'visitors'}, inplace=True)
submission_df['id'] = submission_df.apply(lambda x: '{}_{}'.format(x['air_store_id'], str(x['visit_date'])), axis=1)
submission_df = submission_df[['id', 'visitors']]

sub_sample = pd.read_csv('../input/sample_submission.csv')
submission_df = pd.merge(sub_sample[['id']], submission_df)
submission_df.to_csv('submission.csv', index=False)