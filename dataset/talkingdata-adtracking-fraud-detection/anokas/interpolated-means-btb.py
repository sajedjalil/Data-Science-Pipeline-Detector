import pandas as pd
import mlcrate as mlc
from tqdm import tqdm

t = mlc.time.Timer()

dtypes = {
        'ip': 'uint32',
        'app': 'uint16',
        'device': 'uint16',
        'os': 'uint16',
        'channel': 'uint16',
        'is_attributed': 'uint8'
        }

df_train = pd.read_csv('../input/train.csv', usecols=['ip', 'app', 'device', 'os', 'channel', 'is_attributed'], nrows=100000000)
sub = pd.read_csv('../input/sample_submission.csv')

print('Read training data, {} elapsed'.format(t.fsince(0)))
t.add(1)

means = {}
weights = {}
cols = ['ip', 'app', 'device', 'os', 'channel']
for col in cols:
    means[col] = df_train.groupby(col)['is_attributed'].mean().to_dict()
    weights[col] = df_train[col].value_counts().to_dict()

print('Generated statistics, took {}'.format(t.fsince(1)))
t.add(2)

del df_train
df_test = pd.read_csv('../input/test.csv', usecols=['ip', 'app', 'device', 'os', 'channel'])
print('Read testing data, {} elapsed'.format(t.fsince(2)))
t.add(3)

preds = []
test_data = df_test[cols].values
for row in tqdm(test_data):
    total_weight = 0
    score = 0
    for val, col in zip(row, cols):
        weight = weights[col].get(val, 0)
        total_weight += weight
        score += means[col].get(val, 0) * weight
    preds.append(score / total_weight)

print('Generated predictions, took {}'.format(t.fsince(3)))

sub['is_attributed'] = preds
print(sub)
sub.to_csv('baseline_sub.csv', index=False)

print('Done, total time {}'.format(t.fsince(0)))
