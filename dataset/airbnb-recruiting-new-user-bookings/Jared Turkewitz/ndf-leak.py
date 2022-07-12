import pandas as pd
import numpy as np
import timeit
tic0=timeit.default_timer()
pd.options.mode.chained_assignment = None  # default='warn'

tic=timeit.default_timer()
#%%
age_gender_bkts = pd.read_csv('../input/age_gender_bkts.csv', header=0)
countries = pd.read_csv('../input/countries.csv', header=0)
sessions = pd.read_csv('../input/sessions.csv', header=0)
test_users = pd.read_csv('../input/test_users.csv', header=0)
train_users = pd.read_csv('../input/train_users.csv', header=0)
#%%
#is_sub_run = False
is_sub_run = True
#%%
if(is_sub_run):
    print('exploit leak of ndf and date_first_booking')
    test_users_ndf = test_users.loc[test_users['date_first_booking'].isnull()]
    test_users_ndf['pred0'] = 'NDF'
    test_users_ndf['pred1'] = 'US'
    test_users_ndf['pred2'] = 'other'
    test_users_ndf['pred3'] = 'FR'
    test_users_ndf['pred4'] = 'IT'
    test_users_ndf = test_users_ndf[['id','pred0','pred1','pred2','pred3','pred4']]
#%%
test_users['country_destination'] = 'dummy'
combined = pd.concat([train_users, test_users], axis=0,ignore_index=True)
user_id_dict = dict(zip(combined.id, combined.index))
user_id_rev_dict = {v: k for k, v in user_id_dict.items()}
combined['user_id'] = combined['id'].map(user_id_dict)
combined['date_account_created'] = pd.to_datetime(combined['date_account_created'])

combined['timestamp_first_active'] = pd.to_datetime(combined['timestamp_first_active'],
                                                        format='%Y%m%d%H%M%S')
combined = combined.loc[combined['date_first_booking'].notnull()]
combined['date_first_booking'] = pd.to_datetime(combined['date_first_booking'])
cond_age_null = combined['age'].isnull()
combined['age'][cond_age_null] = -1
cond_aff_tracked_null = combined['first_affiliate_tracked'].isnull()
combined['first_affiliate_tracked'][cond_aff_tracked_null] = 'unknown'

starting_date = pd.to_datetime('2014-02-01 00:00:00')
if (is_sub_run):
    train = combined.loc[combined.country_destination != 'dummy' ]
    test = combined.loc[combined.country_destination == 'dummy' ]
else:
    train = combined.loc[(combined['country_destination'] != 'dummy') &
                         (combined['timestamp_first_active'] < starting_date)]
    test = combined.loc[(combined['country_destination'] != 'dummy') &
                         (combined['timestamp_first_active'] >= starting_date)]
#%%
sessions['user_id'] = sessions['user_id'].map(user_id_dict)
#%%
users = set(combined['user_id'].unique())
sessions['has_user'] = sessions['user_id'].map(lambda x: x in users)
sessions = sessions.loc[sessions['has_user']]
for c in train_users.columns:
    if(train_users[c].unique().shape[0] > 100):
        continue
    print ('----',c,'---')
    print (train_users[c].value_counts())
#%%

if(is_sub_run):
    print('rank by most common destinations')
    test['pred0'] = 'US'
    test['pred1'] = 'other'
    test['pred2'] = 'FR'
    test['pred3'] = 'IT'
    test['pred4'] = 'GB'
    test = test[['id','pred0','pred1','pred2','pred3','pred4']]
    test_all = pd.concat([test, test_users_ndf], axis=0,ignore_index=True)
    test_long = test_all[['id','pred0']].rename(columns={'pred0':'country'})
    test_long = pd.concat([test_long,test_all[['id','pred1']].rename(columns={'pred1':'country'})],
                          axis=0,ignore_index=True)
    test_long = pd.concat([test_long,test_all[['id','pred2']].rename(columns={'pred2':'country'})],
                          axis=0,ignore_index=True)
    test_long = pd.concat([test_long,test_all[['id','pred3']].rename(columns={'pred3':'country'})],
                          axis=0,ignore_index=True)
    test_long = pd.concat([test_long,test_all[['id','pred4']].rename(columns={'pred4':'country'})],
                          axis=0,ignore_index=True)

    test_long.to_csv('basic_airbnb.csv', index=False)
    print('submission created')
#%%
toc=timeit.default_timer()
print('Total Time',toc - tic0)