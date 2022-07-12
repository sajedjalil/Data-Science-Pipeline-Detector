__author__ = 'Tilii: https://kaggle.com/tilii7'

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        tmin, tsec = divmod((datetime.now() - start_time).total_seconds(), 60)
        print(' Time taken: %i minutes and %s seconds.' % (tmin, round(tsec, 2)))


train = pd.read_csv('../input/train.csv', dtype={'id': np.str, 'loss': np.float32})
y = np.array(train['loss'])
test = pd.read_csv('../input/test.csv', dtype={'id': np.str})

# Analyze all features ; modify categorical features
trainc = train.drop(['id', 'loss'], axis=1)
testc = test.drop(['id'], axis=1)
ntrain = trainc.shape[0]
ntest = testc.shape[0]
train_test = pd.concat((trainc, testc)).reset_index(drop=True)
all_features = [x for x in trainc.columns]
cat_features = [x for x in trainc.select_dtypes(include=['object']).columns]
num_features = [x for x in trainc.select_dtypes(exclude=['object']).columns]
print('\n Categorical features: %d' % len(cat_features))
print('\n Numerical features: %d\n' % len(num_features))
for c in range(len(cat_features)):
    train_test[cat_features[c]] = train_test[cat_features[c]].astype('category').cat.codes
trainc = train_test.iloc[:ntrain,:]
testc = train_test.iloc[ntrain:,:]
X = np.array(trainc)
Xt = np.array(testc)

# Define regressor and RFECV parameters
# To test the features properly, it is probably a good idea to change step=2, n_estimators to 200
# and max_depth=20 (or remove max_depth). It will take a long time, on the order of 5-10 hours
rfr = RandomForestRegressor(n_estimators=100, max_features='sqrt', max_depth=12, n_jobs=-1)
rfecv = RFECV(estimator=rfr,
              step=10,
              cv=KFold(y.shape[0],
                       n_folds=5,
                       shuffle=False,
                       random_state=101),
              scoring='neg_mean_absolute_error',
              verbose=2)

# Estimate feature importance and time the whole process
start_time = timer(None)
rfecv.fit(X, y)
timer(start_time)

# Summarize the output
print(' Optimal number of features: %d' % rfecv.n_features_)
sel_features = [f for f, s in zip(all_features, rfecv.support_) if s]
print(' The selected features are {}'.format(sel_features))

# Plot number of features vs CV scores
plt.figure()
plt.xlabel('Number of features tested x 10')
plt.ylabel('Cross-validation score (negative MAE)')
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.savefig('Allstate-RFECV.png')
plt.show()

# Save sorted feature rankings
ranking = pd.DataFrame({'Features': all_features})
ranking['Rank'] = np.asarray(rfecv.ranking_)
ranking.sort_values('Rank', inplace=True)
ranking.to_csv('./Allstate-RFECV-ranking.csv', index=False)
print(' Ranked features saved:  Allstate-RFECV-ranking.csv')

# Make a prediction ; this is only a proof of principle as
# the prediction will be poor until smaller step is are used
score = round(-np.max(rfecv.grid_scores_), 3)
test['loss'] = rfecv.predict(Xt)
test = test[['id', 'loss']]
now = datetime.now()
sub_file = 'submission_5xRFECV-RandomForest_' + str(score) + '_' + str(
    now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
print("\n Writing submission file: %s" % sub_file)
test.to_csv(sub_file, index=False)
