# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

raw_data = pd.read_csv('../input/air_visit_data.csv', parse_dates=['visit_date'])
test_df = pd.read_csv('../input/sample_submission.csv')

test_df['store_id'], test_df['visit_date'] = test_df['id'].str[:20], test_df['id'].str[21:]
test_df.drop(['visitors'], axis=1, inplace=True)
test_df['visit_date'] = pd.to_datetime(test_df['visit_date'])

train = raw_data[raw_data['visit_date'] > '2017-01-28'].reset_index()
train['dow'] = train['visit_date'].dt.dayofweek
test_df['dow'] = test_df['visit_date'].dt.dayofweek
aggregation = {'visitors' :{'total_visitors' : 'median'}}

agg_data = train.groupby(['air_store_id', 'dow']).agg(aggregation).reset_index()
agg_data.columns = ['air_store_id','dow','visitors']
agg_data['visitors'] = agg_data['visitors']
merged = pd.merge(test_df, agg_data, how='left', left_on=['store_id','dow'], right_on=['air_store_id','dow'])
final = merged[['id','visitors']]
final.fillna(0, inplace=True)

final.to_csv('submission.csv', index=False)