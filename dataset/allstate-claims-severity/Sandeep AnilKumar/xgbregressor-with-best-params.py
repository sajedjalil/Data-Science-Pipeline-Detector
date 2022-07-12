import pandas as pd
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor as gBR
import time
from sklearn.model_selection import train_test_split
import csv
import xgboost

start_time = time.time()

train_data = pd.read_csv('../input/train.csv')
for i in range(1, 117):
    train_data['cat' + str(i)] = train_data['cat' + str(i)].astype('category')

categorical_columns = train_data.select_dtypes(['category']).columns
train_data[categorical_columns] = train_data[categorical_columns].apply(lambda x: x.cat.codes)

print("Time for encoding training data is := %.2f" % (time.time() - start_time))

train_data_vector = train_data.iloc[:, 1:131]
train_data_target = train_data.iloc[:, 131]

start_time = time.time()
test_data = pd.read_csv('../input/test.csv')
for i in range(1, 117):
    test_data['cat' + str(i)] = test_data['cat' + str(i)].astype('category')

categorical_columns = test_data.select_dtypes(['category']).columns
test_data[categorical_columns] = test_data[categorical_columns].apply(lambda x: x.cat.codes)

print("Time for encoding testing data is := %.2f" % (time.time() - start_time))

test_data_vector = test_data.iloc[:, 1:131]
test_data_ids = list(test_data['id'])

start_time = time.time()
#reg = gBR()
# reg = reg.fit(train_data_vector, train_data_target)

reg = xgboost.XGBRegressor(reg_alpha=0.2,  n_estimators=1000, learning_rate=0.2)
reg = reg.fit(train_data_vector, train_data_target)

lr_predict = reg.predict(test_data_vector)
print("Time for the regressor to train and predict is := %.2f" % (time.time() - start_time))

csv_file = open("submissions.csv", 'w', newline='')
wr = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_NONE)
wr.writerow(['id', 'loss'])

for index in range(0, len(test_data_ids)):
    wr.writerow([test_data_ids[index], lr_predict[index]])
    index += 1
print("done with predicting loss values for the test data")
csv_file.close()

