import pandas as pd

data = pd.read_csv('../input/air_visit_data.csv').groupby('air_store_id')['visitors'].median().reset_index()
print(data.head(2))

sub = pd.read_csv('../input/sample_submission.csv')[['id']]
sub['air_store_id'] = sub['id'].apply(lambda x: '_'.join(x.split('_')[:2]))
print(sub.head(2))

sub = sub.merge(data, how='left', on='air_store_id')
sub[['id','visitors']].to_csv('median_baseline.csv', index=False)
print(sub.head(2))