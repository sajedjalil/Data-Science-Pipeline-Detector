import numpy as np
import pandas as pd

train = pd.read_csv('../input/act_train.csv')
test = pd.read_csv('../input/act_test.csv')
people = pd.read_csv('../input/people.csv')

train = pd.merge(train, people[['people_id', 'char_38']], on='people_id', how='left')
test = pd.merge(test, people[['people_id', 'char_38']], on='people_id', how='left')

means = {}
for i in range(101):
    means[i] = train[train.people_id.isin(people[people.char_38==i]['people_id'])]['outcome'].mean()

test['outcome'] = test['char_38'].apply(lambda x: means[x])
test[['activity_id', 'outcome']].to_csv('submission.csv', index=False)