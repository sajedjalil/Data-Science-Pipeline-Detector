import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict

df_train = pd.read_csv('../input/air_visit_data.csv').sort_values('visit_date')
date_info = pd.read_csv('../input/date_info.csv')

is_holiday = {}
for day, flag in date_info[['calendar_date', 'holiday_flg']].values:
    is_holiday[day] = flag

def get_day_of_week(visit_date):
    if is_holiday[visit_date]:
        return -1
    year, month, day = [int(i) for i in visit_date.split('-')]
    dt = datetime(day=day, month=month, year=year)
    return dt.weekday()
    
y_train = np.log1p(df_train['visitors'])
    
table = defaultdict(list)
backup_table = defaultdict(list)
backup_table2 = defaultdict(list)
for row, y in zip(list(zip(df_train['air_store_id'].values, df_train['visit_date'].apply(get_day_of_week).values)), y_train):
    table[hash(str(row))].append(y)
    backup_table[hash(str(row[0]))].append(y)
    backup_table2[hash(str(row[1]))].append(y)
   
df_test = pd.read_csv('../input/sample_submission.csv')
df_test['air_store_id'] = df_test['id'].apply(lambda x: '_'.join(x.split('_')[:2]))
df_test['visit_date'] = df_test['id'].apply(lambda x: x.split('_')[-1])

p_test = []
for row in zip(df_test['air_store_id'].values, df_test['visit_date'].apply(get_day_of_week).values):
    vals = table[hash(str(row))]
    if len(vals) == 0:
        vals = backup_table[hash(str(row[0]))]
    if len(vals) == 0:
        vals = backup_table2[hash(str(row[1]))]
    if len(vals) == 0:
        vals = [np.mean(y_train)]
    p_test.append(np.expm1(np.mean(vals)))
    
sub = pd.read_csv('../input/sample_submission.csv')
sub['visitors'] = p_test
sub.to_csv('hashtable_btb.csv', index=False)