import pandas as pd
print('Started')
train = pd.read_csv('../input/train_ver2.csv', usecols=['ncodpers','fecha_dato'])
print('Got train')
test = pd.read_csv('../input/test_ver2.csv', usecols=['ncodpers'])
print('Got test')
test_persons = test['ncodpers'].sort_values().values
train_persons = train.loc[(train['fecha_dato'] == '2016-05-28') & (train['ncodpers'].isin(test_persons)), 'ncodpers'].sort_values().values
print((test_persons == train_persons).all())
print("Done.")
