__author__ = 'Tilii: https://kaggle.com/tilii7'
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import MinMaxScaler


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        tmin, tsec = divmod((datetime.now() - start_time).total_seconds(), 60)
        print('\n Time taken: %i minutes and %s seconds.' % (tmin, round(tsec, 2)))


def scale_data(X, scaler=None):
    if not scaler:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler


# Read data
train = pd.read_csv('../input/train.csv')
len_train = train.shape[0]
y = train['target'].values
test = pd.read_csv('../input/test.csv')

# Scale data to [-1, 1]
features = [x for x in train.columns if x not in ['id', 'target']]
scaled, scaler = scale_data(np.concatenate((train[features].values, test[features].values), axis=0))
train[features] = scaled[:len_train]
test[features] = scaled[len_train:]

train = train.drop(['target', 'id'], axis=1).values
test = test.drop(['id'], axis=1).values

folds = 10

# Define LinearSVC parameters
svc = LinearSVC(C=0.01, tol=0.0001, verbose=0, random_state=101, max_iter=2000, dual=False)
# Make this a calibrated classifier ; not necessary with AUC as metric
sigmoid = CalibratedClassifierCV(svc, cv=folds, method='sigmoid')

# Select between 10 and 20 features starting from 1
# One could do backward elimination but that would take longer
# For more details see:
# http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/
sfs = SFS(sigmoid,
          k_features=(10, 20),
          forward=True,
          floating=True,
          scoring='roc_auc',
          verbose=2,
          cv=list(StratifiedKFold(n_splits=folds, shuffle=True, random_state=101).split(train,y)),
          n_jobs=4)

# Estimate feature importance and time the whole process
starttime = timer(None)
start_time = timer(None)
sfs.fit(train, y, custom_feature_names=features)
timer(start_time)

# The whole run
print(sfs.subsets_)
print(sfs.k_feature_idx_)
print(sfs.k_feature_names_)

# Summarize the output
print(' Best score: .%6f' % sfs.k_score_)
print(' Optimal number of features: %d' % len(sfs.k_feature_idx_))
print(' The selected features are:')
print(sfs.k_feature_names_)

# Plot number of features vs CV scores
plt.figure(figsize=(12, 9))
fig1 = plot_sfs(sfs.get_metric_dict(), kind='std_dev', marker='.', ylabel='Cross-validation score (roc_auc)')
plt.title('Sequential Forward Floating Selection (StdDev)')
plt.xlabel('Number of features tested')
plt.ylabel('Cross-validation score (roc_auc)')
plt.tight_layout()
plt.savefig('SFFSCV-01.png', dpi=150)
#plt.show()
plt.show(block=False)

# Make a prediction ; this is only a proof of principle
score = str(round(sfs.k_score_, 6))
score = score.replace('.', '')
sample = pd.read_csv('../input/sample_submission.csv')
train = sfs.transform(train)
test = sfs.transform(test)
sigmoid.fit(train, y)
sample['target'] = sigmoid.predict_proba(test)[:, 1]
sample = sample[['id', 'target']]
now = datetime.now()
sub_file = 'submission_10x-SFFSCV-LinearSVC-01_' + score + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
print("\n Writing submission file: %s" % sub_file)
sample.to_csv(sub_file, index=False)

# Save sorted feature rankings
ranking = pd.DataFrame.from_dict(sfs.get_metric_dict()).T
ranking.to_csv('SFFSCV-ranking-01.csv', index=False)

timer(starttime)
