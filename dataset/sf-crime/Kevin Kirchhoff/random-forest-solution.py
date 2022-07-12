import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

weekdays = {'Monday':0., 'Tuesday':1., 'Wednesday':2., 'Thursday': 3., 'Friday':4., 'Saturday':5., 'Sunday':6.}
categories = {c:i for i,c in enumerate(train['Category'].unique())}
cat_rev = {i:c for i,c in enumerate(train['Category'].unique())}
districts = {c:i for i,c in enumerate(train['PdDistrict'].unique())}
dis_rev = {i:c for i,c in enumerate(train['PdDistrict'].unique())}

# Extract features from given information
train['Hour'] = list(map(lambda x: float(int(x.split(' ')[1].split(':')[0])),
                                  train.Dates))
test['Hour'] = list(map(lambda x: float(int(x.split(' ')[1].split(':')[0])),
                                  test.Dates))

train['Minute'] = list(map(lambda x: float(int(x.split(' ')[1].split(':')[1])),
                                  train.Dates))
test['Minute'] = list(map(lambda x: float(int(x.split(' ')[1].split(':')[1])),
                                  test.Dates))

train['Month'] = list(map(lambda x: float(x.split(' ')[0].split('-')[1]), train.Dates))
test['Month'] = list(map(lambda x: float(x.split(' ')[0].split('-')[1]), test.Dates))

train['Year'] = list(map(lambda x: float(x.split(' ')[0].split('-')[0])-2003., train.Dates))
test['Year'] = list(map(lambda x: float(x.split(' ')[0].split('-')[0])-2003., test.Dates))

train['Day'] = list(map(lambda x: float(x.split(' ')[0].split('-')[2]), train.Dates))
test['Day'] = list(map(lambda x: float(x.split(' ')[0].split('-')[2]), test.Dates))

train['Day_Num'] = [float(weekdays[w]) for w in train.DayOfWeek]
test['Day_Num'] = [float(weekdays[w]) for w in test.DayOfWeek]

train['District_Num'] = [float(districts[t]) for t in train.PdDistrict]
test['District_Num'] = [float(districts[t]) for t in test.PdDistrict]

train['Category_Num'] = [float(categories[t]) for t in train.Category]

# Center X,Y coordinates
train['X'] = preprocessing.scale(list(map(lambda x: x+122.4194, train.X)))
train['Y'] = preprocessing.scale(list(map(lambda x: x-37.7749, train.Y)))

test['X'] = preprocessing.scale(list(map(lambda x: x+122.4194, test.X)))
test['Y'] = preprocessing.scale(list(map(lambda x: x-37.7749, test.Y)))

# Assign binary value to address by type
def define_address(addr):
    addr_type = 0.
    # Address types:
    #  Intersection: 1
    #  Residence: 0
    if '/' in addr and 'of' not in addr:
        addr_type = 1.
    else:
        add_type = 0.
    return addr_type
    
# Define address feature
train['Address_Num'] = list(map(define_address, train.Address))
test['Address_Num'] = list(map(define_address, test.Address))

# Feature selection
X_loc = ['X', 'Y', 'District_Num', 'Address_Num']
X_time = ['Minute', 'Hour']
X_date = ['Year','Month', 'Day', 'Day_Num']
X_all = X_loc + X_time + X_date

# Category column we want to predict
y = 'Category_Num'

print(train.head())

# Create random forest classifie
clf = RandomForestClassifier(max_features="log2", max_depth=11, n_estimators=24,
                             min_samples_split=1000, oob_score=True)
# Fit prediction
clf.fit(train[X_all], train[y])
pred = clf.predict_proba(test[X_all])

# Create submission
submission = pd.DataFrame({cat_rev[p] : [pred[i][p] for i in range(len(pred))] for p in range(len(pred[0]))})

submission['Id'] = [i for i in range(len(submission))]

submission = submission[['Id'] + sorted(train['Category'].unique())]
print(submission.head())

# Write submission
submission.to_csv('submission.csv.gz', index=False, compression='gzip')
