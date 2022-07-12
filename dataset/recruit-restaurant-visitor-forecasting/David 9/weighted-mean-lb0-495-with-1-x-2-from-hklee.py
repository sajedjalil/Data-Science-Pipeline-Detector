import numpy as np, pandas as pd
import glob, re

dfs = { re.search('/([^/\.]*)\.csv', fn).group(1):pd.read_csv(fn) for fn in glob.glob('../input/*.csv')}
print('data frames read:{}'.format(list(dfs.keys())))

print('local variables with the same names are created.')
for k, v in dfs.items(): locals()[k] = v

print('holidays at weekends are not special, right?')
wkend_holidays = date_info.apply((lambda x:(x.day_of_week=='Sunday' or x.day_of_week=='Saturday') and x.holiday_flg==1), axis=1)
date_info.loc[wkend_holidays, 'holiday_flg'] = 0

print('add decreasing weights from now')
#date_info['weight'] = ((date_info.index + 1) / len(date_info))       # LB 0.509
#date_info['weight'] = ((date_info.index + 1) / len(date_info)) ** 2  # LB 0.503
#date_info['weight'] = ((date_info.index + 1) / len(date_info)) ** 3  # LB 0.500
#date_info['weight'] = ((date_info.index + 1) / len(date_info)) ** 4  # LB 0.498
# date_info['weight'] = ((date_info.index + 1) / len(date_info)) ** 5  # LB 0.497
date_info['weight'] = 1 / (len(date_info)-date_info.index) ** 2

# I've consumed all my submissions for day. I suggest to try weight 1/x, exp(-x).
# It seems to me that Recruit sales may be stochastic and depends only on recent means.

print('weighted mean visitors for each (air_store_id, day_of_week, holiday_flag) or (air_store_id, day_of_week)')
visit_data = air_visit_data.merge(date_info, left_on='visit_date', right_on='calendar_date', how='left')
visit_data.drop('calendar_date', axis=1, inplace=True)
visit_data['visitors'] = visit_data.visitors.map(pd.np.log1p)

wmean = lambda x:( (x.weight * x.visitors).sum() / x.weight.sum() )
visitors = visit_data.groupby(['air_store_id', 'day_of_week', 'holiday_flg']).apply(wmean).reset_index()
visitors.rename(columns={0:'visitors'}, inplace=True) # cumbersome, should be better ways.

print('prepare to merge with date_info and visitors')
sample_submission['air_store_id'] = sample_submission.id.map(lambda x: '_'.join(x.split('_')[:-1]))
sample_submission['calendar_date'] = sample_submission.id.map(lambda x: x.split('_')[2])
sample_submission.drop('visitors', axis=1, inplace=True)
sample_submission = sample_submission.merge(date_info, on='calendar_date', how='left')
sample_submission = sample_submission.merge(visitors, on=['air_store_id', 'day_of_week', 'holiday_flg'], how='left')

# fill missings with (air_store_id, day_of_week)
missings = sample_submission.visitors.isnull()
sample_submission.loc[missings, 'visitors'] = sample_submission[missings].merge(
    visitors[visitors.holiday_flg==0], on=('air_store_id', 'day_of_week'), how='left')['visitors_y'].values

# fill missings with (air_store_id)
missings = sample_submission.visitors.isnull()
sample_submission.loc[missings, 'visitors'] = sample_submission[missings].merge(
    visitors[['air_store_id', 'visitors']].groupby('air_store_id').mean().reset_index(), on='air_store_id', how='left')['visitors_y'].values
    
sample_submission['visitors'] = sample_submission.visitors.map(pd.np.expm1)

sample_submission[['id', 'visitors']].to_csv('dumb_result.csv', float_format='%.4f', index=None)
print("done")