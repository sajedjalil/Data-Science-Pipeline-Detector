__author__ = 'Tilii: https://kaggle.com/tilii7'

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
import gc

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        tmin, tsec = divmod((datetime.now() - start_time).total_seconds(), 60)
        print('\n Time taken: %i minutes and %s seconds.' % (tmin, round(tsec, 2)))

train = pd.read_csv('../input/train.csv')
y = train['target'].values
test = pd.read_csv('../input/test.csv')

train.drop(['target', 'ID_code'], axis=1, inplace=True)
test.drop(['ID_code'], axis=1, inplace=True)
all_features = [x for x in train.columns]

X = np.array(train)
Xt = np.array(test)
del train, test
gc.collect()

folds = 5
#step = 1
step = 4

# Define Random Forest Classifier and RFECV parameters
rfc = RandomForestClassifier(n_estimators=20, max_depth=5, n_jobs=4, max_features=None)
rfecv = RFECV(estimator=rfc,
              step=step,
              cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001).split(X,y),
              scoring='roc_auc',
              verbose=2)

# Estimate feature importance
starttime = timer(None)
start_time = timer(None)
rfecv.fit(X, y)
timer(start_time)

# Summarize the output
print(' Optimal number of features: %d' % rfecv.n_features_)
sel_features = [f for f, s in zip(all_features, rfecv.support_) if s]
print(' The selected features are: {}'.format(sel_features))

# Plot number of features vs CV scores
plt.figure(figsize=(12, 9))
plt.xlabel('Number of features tested (x%d)' % step)
plt.ylabel('Cross-validation score (roc_auc)')
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.tight_layout()
plt.savefig('RFECV-01.png', dpi=300)
plt.show()

# Save sorted feature rankings
ranking = pd.DataFrame({'Features': all_features})
ranking['Rank'] = np.asarray(rfecv.ranking_)
ranking.sort_values('Rank', inplace=True)
ranking.to_csv('RFECV-ranking-01.csv', index=False)
print(' Ranked features saved:  RFECV-ranking-01.csv')

# Make a prediction ; this is only a proof of principle
score = str(round(np.max(rfecv.grid_scores_), 6))
score = score.replace('.', '')
test = pd.read_csv('../input/sample_submission.csv')
test['target'] = rfecv.predict(Xt)
test = test[['ID_code', 'target']]
now = datetime.now()
sub_file = 'submission_RFECV-RandomForest-01_' + score + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
print("\n Writing submission file: %s" % sub_file)
test.to_csv(sub_file, index=False)
timer(starttime)
