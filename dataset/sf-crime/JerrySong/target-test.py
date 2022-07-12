# Loading libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder as le
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import log_loss

# Global constants and variables

ROOT_FILENAME = "../input/"
TRAIN_FILENAME = 'train.csv'
TEST_FILENAME = 'test.csv'
RESULT_FILENAME = 'res2.cv'
train = pd.read_csv(ROOT_FILENAME+TRAIN_FILENAME, parse_dates=['Dates'], index_col=False)
test = pd.read_csv(ROOT_FILENAME+TEST_FILENAME, parse_dates=['Dates'], index_col=False)
# TRAIN_FILENAME = 'train.csv'
# TEST_FILENAME = 'test.csv'
# RESULT_FILENAME = 'res2.cv'
# train = pd.read_csv(TRAIN_FILENAME, parse_dates=['Dates'], index_col=False)
# test = pd.read_csv(TEST_FILENAME, parse_dates=['Dates'], index_col=False)
#train.info()
categories = {c:i for i,c in enumerate(train['Category'].unique())}
cat_rev = {i:c for i,c in enumerate(train['Category'].unique())}
districts = {c:i for i,c in enumerate(train['PdDistrict'].unique())}
weekdays = {'Monday':0., 'Tuesday':1., 'Wednesday':2., 'Thursday': 3., 'Friday':4., 'Saturday':5., 'Sunday':6.}
weekdays2 = {'Monday':0., 'Tuesday':0., 'Wednesday':0., 'Thursday': 0., 'Friday':0., 'Saturday':1., 'Sunday':1}
    
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

def getHourZn(hour):
    if(hour >= 2 and hour < 8): return 1;
    if(hour >= 8 and hour < 12): return 2;
    if(hour >= 12 and hour < 14): return 3;
    if(hour >= 14 and hour < 18): return 4;
    if(hour >= 18 and hour < 22): return 5;
    if(hour < 2 or hour >= 22): return 6;

def feature_engineering(data):
    data['Day'] = data['Dates'].dt.day
    data['Month'] = data['Dates'].dt.month
    data['Year'] = data['Dates'].dt.year-2003
    data['Hour'] = data['Dates'].dt.hour
    data['Minute'] = data['Dates'].dt.minute
    data['Day_Num'] = [float(weekdays[w]) for w in data.DayOfWeek]
    data['WeekOfYear'] = data['Dates'].dt.weekofyear
    data['District_Num'] = [float(districts[t]) for t in data.PdDistrict]
    data['HourZn'] = preprocessing.scale(list(map(getHourZn, data['Dates'].dt.hour)))
    data['isWeekday'] = [float(weekdays2[w]) for w in data.DayOfWeek]
    data['X'] = preprocessing.scale(list(map(lambda x: x+122.4194, data.X)))
    data['Y'] = preprocessing.scale(list(map(lambda x: x-37.7749, data.Y)))
    data['Address_Type'] = list(map(define_address, data.Address))
#   data['HourZn'] = getHourZn(data['Dates'].dt.hour);
    return data

X_loc = ['X', 'Y', 'District_Num', 'Address_Type']
X_time = ['Minute', 'Hour']
X_date = ['Year','Month', 'Day', 'Day_Num', 'HourZn']
X_all = X_loc + X_time + X_date

train = feature_engineering(train)
train['Category_Num'] = [float(categories[t]) for t in train.Category]
test = feature_engineering(test)

#print(train[X_all].info())
#print(train[X_all].describe())
#print(train['Category_Num'].describe())

# Print log loss
criminal_labels = train['Category_Num'].unique()
shuffle = np.random.permutation(np.arange(train.shape[0]))
shuffled_crime_data = train.iloc[shuffle]
shuffled_crime_data = shuffled_crime_data[X_all + ['Category_Num']]
shuffled_crime_data.info() 
mini_train_labels = shuffled_crime_data.iloc[:100000]['Category_Num']
mini_train_data = shuffled_crime_data.iloc[:100000].drop('Category_Num', axis=1)
mini_dev_labels = shuffled_crime_data.iloc[100000:110000]['Category_Num']
mini_dev_data = shuffled_crime_data.iloc[100000:110000].drop('Category_Num', axis=1)
print(mini_train_data.shape, mini_train_labels.shape, mini_dev_data.shape, mini_dev_labels.shape)
# Random Forest
random_forest = RandomForestClassifier(n_estimators=100, max_depth=15)
random_forest.fit(mini_train_data, mini_train_labels)
predictions = random_forest.predict_proba(mini_dev_data)
#print(mini_dev_labels.shape, predictions.shape, criminal_labels.shape)
total_loss = log_loss(mini_dev_labels, predictions)
print('The multiclass loss of Random Forest model is %.4f' % total_loss)

#print(type(train))
# X = train[:].values[:,:-1]
# Y = train[:].values[:,-1:]
# x = test[:].values[:,:]
# Y = MultiLabelBinarizer().fit_transform(Y)

'''
print(train.columns)
print(train['Year'])
clf = RandomForestClassifier(max_features="log2", max_depth=11, n_estimators=24,
                             min_samples_split=1000, oob_score=True).fit(train[X_all],train['Category_Num'])
y = clf.predict_proba(test[X_all])
print(y)
submission = pd.DataFrame({cat_rev[p] : [y[i][p] for i in range(len(y))] for p in range(len(y[0]))})

submission['Id'] = [i for i in range(len(submission))]

submission = submission[['Id'] + sorted(train['Category'].unique())]
print(submission.head())

# Write submission
submission.to_csv('res.csv.gz', index=False, compression='gzip')
# # 
# clf = svm.SVC(kernel='poly', degree=3).fit(X, Y)
# test['predict'] = clf.predict(test[:].values[:,:])
# print(test['predict'])
# test['Category'] = cEnc.inverse_transform(test['predict'])
# def field_to_columns(data, field, new_columns):
#     for i in range(len(new_columns)):
#         data[new_columns[i]] = (data[field] == new_columns[i]).astype(int)
#         print(1)
#     return data
# categories = list(cEnc.classes_)
# test = field_to_columns(test, 'Category', categories)
# print(test.columns)
print("end")
'''