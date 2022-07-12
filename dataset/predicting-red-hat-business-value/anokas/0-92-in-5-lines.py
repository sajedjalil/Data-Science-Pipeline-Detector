import pandas as pd
train = pd.merge(pd.read_csv('../input/act_train.csv'), pd.read_csv('../input/people.csv'), on='people_id')
test = pd.merge(pd.read_csv('../input/act_test.csv'), pd.read_csv('../input/people.csv'), on='people_id')
magic = train.groupby(train['group_1'] + train['date_x'])['outcome'].mean().to_dict()
pd.DataFrame({'activity_id':test['activity_id'], 'outcome':[magic[i] if i in magic else 0.5 for i in test['group_1'] + test['date_x']]}).to_csv('simplesub.csv', index=False)