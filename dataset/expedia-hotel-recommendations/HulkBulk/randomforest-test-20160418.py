import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score
train_cols = ['site_name', 'user_location_region', 'is_package', 'srch_adults_cnt', 'srch_children_cnt', 'srch_destination_id', 'hotel_market', 'hotel_country', 'hotel_cluster']
train = pd.DataFrame(columns=train_cols)
train_chunk = pd.read_csv('../input/train.csv', chunksize=100000)

for chunk in train_chunk:
    train = pd.concat( [ train, chunk[chunk['is_booking']==1][train_cols] ] )
    
train.head()
train_X = train[['site_name', 'user_location_region', 'is_package', 'srch_adults_cnt', 'srch_children_cnt', 'srch_destination_id', 'hotel_market', 'hotel_country']].values
train_y = train['hotel_cluster'].values
rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=2016, n_jobs=4)
clf = BaggingClassifier(rf, n_estimators=2, max_samples=0.1, random_state=2014, n_jobs=4)
clf.fit(train_X, train_y)

test_y = np.array([])
test_chunk = pd.read_csv('../input/test.csv', chunksize=50000)

for i, chunk in enumerate(test_chunk):
    test_X = chunk[['site_name', 'user_location_region', 'is_package', 'srch_adults_cnt', 'srch_children_cnt', 'srch_destination_id', 'hotel_market', 'hotel_country']].values
    if i > 0:
        test_y = np.concatenate( [test_y, clf.predict_proba(test_X)])
    else:
        test_y = clf.predict_proba(test_X)
    print(i)

def get5Best(x):
    result = []
    for z in x.argsort()[::-1][:5]:
        if z!=0:
            result.append(z)
    return " ".join([str(int(z)) for z in result])
submit = pd.read_csv('../input/sample_submission.csv')
submit['hotel_cluster'] = np.apply_along_axis(get5Best, 1, test_y)
submit.head()
submit.to_csv('submission_20160418_ent_1.csv', index=False)