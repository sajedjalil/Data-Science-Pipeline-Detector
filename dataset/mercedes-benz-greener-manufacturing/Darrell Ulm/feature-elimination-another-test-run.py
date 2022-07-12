__author__ = 'Tilii: https://kaggle.com/tilii7'

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        tmin, tsec = divmod((datetime.now() - start_time).total_seconds(), 60)
        print(' Time taken: %i minutes and %s seconds.' % (tmin, round(tsec, 2)))


train = pd.read_csv('../input/train.csv', dtype={'ID': np.str, 'y': np.float32})
y = np.array(train['y'])
test = pd.read_csv('../input/test.csv', dtype={'ID': np.str})

# Analyze all features ; modify categorical features
trainc = train.drop(['ID', 'y'], axis=1)
testc = test.drop(['ID'], axis=1)
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
# To test the features properly, it is probably a good idea to change n_estimators to 200
# and max_depth=20 (or remove max_depth). It will take longer, on the order of 1-2 hours
rfr = RandomForestRegressor(n_estimators=100, max_features='sqrt', max_depth=20, n_jobs=-1)
rfecv = RFECV(estimator=rfr,
              step=2,
              cv=KFold(y.shape[0],
                       n_folds=5,
                       shuffle=False,
                       random_state=1001),
              scoring='r2',
              verbose=2)

# Estimate feature importance and time the whole process
starttime = timer(None)
start_time = timer(None)
rfecv.fit(X, y)
timer(start_time)

# Summarize the output
print(' Optimal number of features: %d' % rfecv.n_features_)
sel_features = [f for f, s in zip(all_features, rfecv.support_) if s]
print(' The selected features are {}'.format(sel_features))

# Plot number of features vs CV scores
plt.figure(figsize=(16, 12))
plt.xlabel('Number of features tested x 2')
plt.ylabel('Cross-validation score (R^2)')
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.savefig('Mercedes-RFECV-01.png')
#plt.show()
plt.show(block=False)

# Save sorted feature rankings
ranking = pd.DataFrame({'Features': all_features})
ranking['Rank'] = np.asarray(rfecv.ranking_)
ranking.sort_values('Rank', inplace=True)
ranking.to_csv('Mercedes-RFECV-ranking-01.csv', index=False)
print(' Ranked features saved:  Mercedes-RFECV-ranking-01.csv')

# Make a prediction ; this is only a proof of principle as
# the prediction will be poor until smaller step is are used
score = round(np.max(rfecv.grid_scores_), 6)
test['y'] = rfecv.predict(Xt)
test = test[['ID', 'y']]
now = datetime.now()
sub_file = 'submission_5fold-RFECV-RandomForest-01_' + str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
print("\n Writing submission file: %s" % sub_file)
test.to_csv(sub_file, index=False)
timer(starttime)
