# This script creates and evaluates a K-nearest-neighbor model in Python for the 'San Francisco Crime Classification' competition.
# This iteration is only to get the model up and running, so there is minimal feature engineering and parameter tuning.

# Import modules.
import pandas as pd
from sklearn import preprocessing, cross_validation
from sklearn.neighbors import KNeighborsClassifier

# Import training and test data.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# PRE-PROCESSING AND FEATURE ENGINEERING.

# Rename columns of training and testing data set.
train.columns = ['date', 'category_predict', 'description_ignore', 'day_of_week', 'pd_district', 'resolution', 'address', 'x', 'y']
test.columns = ['id', 'date', 'day_of_week', 'pd_district', 'address', 'x', 'y']

# Get hour of each crime.
train['hour'] = train['date'].str[11:13]
test['hour'] = test['date'].str[11:13]

# Convert categorical variables ('day_of_week' and 'pd_district') to numerical labels to add to 'train' and 'test'.
day_of_week_encoded = preprocessing.LabelEncoder()
day_of_week_encoded.fit(train['day_of_week'])
train['day_of_week_encoded'] = day_of_week_encoded.transform(train['day_of_week'])

pd_district_encoded = preprocessing.LabelEncoder()
pd_district_encoded.fit(train['pd_district'])
train['pd_district_encoded'] = pd_district_encoded.transform(train['pd_district'])

day_of_week_encoded = preprocessing.LabelEncoder()
day_of_week_encoded.fit(test['day_of_week'])
test['day_of_week_encoded'] = day_of_week_encoded.transform(test['day_of_week'])

pd_district_encoded = preprocessing.LabelEncoder()
pd_district_encoded.fit(test['pd_district'])
test['pd_district_encoded'] = pd_district_encoded.transform(test['pd_district'])

# Create scaler for each feature.
scaler = preprocessing.StandardScaler().fit(train[['day_of_week_encoded', 'pd_district_encoded', 'x', 'y', 'hour']])

# CREATE MODEL AND PREDICTIONS.

# Create classifier.
knn = KNeighborsClassifier(n_neighbors = 5)

# Fit classifier on scaled training data.
knn.fit(scaler.transform(train[['day_of_week_encoded', 'pd_district_encoded', 'x', 'y', 'hour']]),
       train['category_predict'])

# Generate predictions for (scaled) training and test data.
# For training data, I want accuracy.
# For test data, the submission format is specified by Kaggle.com.
train['pred'] = knn.predict(scaler.transform(train[['day_of_week_encoded', 'pd_district_encoded', 'x', 'y', 'hour']]))
test_pred = knn.predict_proba(scaler.transform(test[['day_of_week_encoded', 'pd_district_encoded', 'x', 'y', 'hour']]))

# CHECK TRAINING SET ACCURACY.

# Compute training set accuracy.
print('Training Set Accuracy :', sum(train['category_predict'] == train['pred']) / len(train['date']))

# CROSS VALIDATION.

# Get cross validation scores.
cv_scores = cross_validation.cross_val_score(knn,
                                             scaler.transform(train[['day_of_week_encoded', 'pd_district_encoded', 'x', 'y', 'hour']]),
                                             train['category_predict'],
                                             cv = 2)

# Take the mean accuracy across all cross validation segments.
print('Cross Validation Accuracy: ', cv_scores.mean())
                                            
# EXPORT TEST SET PREDICTIONS.
# This section exports test predictions to a csv in the format specified by Kaggle.com.

# Turn 'test_pred' into data frame.
test_pred = pd.DataFrame(test_pred)

# Add column names to 'test_pred'.
test_pred.columns = knn.classes_

# Name index column.
test_pred.index.name = 'Id'

# Write csv.
# This is commented out because I don't actually want to create a csv right now.
# test_pred.to_csv('test_pred_benchmark_knn.csv')
