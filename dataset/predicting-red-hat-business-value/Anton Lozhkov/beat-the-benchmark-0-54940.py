import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

df_train=pd.read_csv('../input/act_train.csv')
df_test=pd.read_csv('../input/act_test.csv')

probs = {}
print('Probabilities for each activity category:')
for a in df_train['activity_category'].unique():
    probs[a] = df_train.loc[df_train.activity_category == a]['outcome'].mean()
    print(a, probs[a])

preds_train = [probs[a] for a in df_train['activity_category']]
preds_test = [probs[a] for a in df_test['activity_category']]

print('Local AUC: ' + str(roc_auc_score(df_train['outcome'], preds_train)))

sub = pd.DataFrame()
sub['activity_id'] = df_test['activity_id']
sub['outcome'] = preds_test
sub.to_csv('beat_the_benchmark.csv', index=False)