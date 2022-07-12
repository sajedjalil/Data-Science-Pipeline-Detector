import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import sys
import gc # We're gonna be clearing memory a lot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing, linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
le = preprocessing.LabelEncoder()
import statsmodels.api as sm

#matplotlib inline
print("started")
clicks_train = pd.read_csv("../input/clicks_train.csv", dtype={"display_id": int, "ad_id": int, "clicked": int})
print("End of reading")

feature_vector = []
class_value_vector = []

for index, row in clicks_train.iterrows():
    if index == 100000:
        break;
    feature_vector.append([row['display_id'], row['ad_id']])
    class_value_vector.append([row['clicked']])
    # print(row['display_id'], row['ad_id'], row['clicked'])
    
clicks_test = pd.read_csv('../input/clicks_test.csv')

test_values = []

for index, row in clicks_test.iterrows():
    if index == 1001:
        break;
    test_values.append([row['display_id'], row['ad_id']])
    
gc.collect()

# print(feature_vector)
# By this point I'll have a vector of training data set with 1 million record.
    
print("size of the vector")
print(sys.getsizeof(feature_vector))


print("Opening events file")
events = pd.read_csv("../input/events.csv")
events.set_index('display_id')

unique_country = events.geo_location.unique()
arr = []

for i in unique_country:
    if type(i) is str:
        if len(i) > 1:
            arr.append(''.join([i[0],i[1]]))

myset = set(arr)

uniqueVals = {}
counter = 0
for i in myset:
    if i != "--":
        counter += 1 
        uniqueVals[i] = counter
print("Done reading events file")

new_feature_vector = []
for row in feature_vector:
    
    event_platform_var = int(events.loc[row[0]]['platform'])
    event_timestamp_var = int(events.loc[row[0]]['timestamp'])
    if ''.join([events.loc[row[0]]['geo_location'][0], events.loc[row[0]]['geo_location'][1]]) == "--":
        event_location_var = 0
    else:
        event_location_var = int(uniqueVals[''.join([events.loc[row[0]]['geo_location'][0], events.loc[row[0]]['geo_location'][1]])])
    # event_uuid_var = events.loc[row[0]]['uuid']
    
    # new_feature_vector.append([int(row[0]), int(row[1]), int(row[2]), event_platform_var, event_timestamp_var, event_location_var, events.loc[row[0]]['uuid']])
    new_feature_vector.append([int(row[0]), int(row[1]), event_platform_var, event_timestamp_var, event_location_var])
    # print(events.loc[row[1]]['uuid'])

new_test_values = []

for row in test_values:
    
    event_platform_var = int(events.loc[row[0]]['platform'])
    event_timestamp_var = int(events.loc[row[0]]['timestamp'])
    event_location_var = int(uniqueVals[''.join([events.loc[row[0]]['geo_location'][0], events.loc[row[0]]['geo_location'][1]])])
    # event_uuid_var = events.loc[row[0]]['uuid']
    
    # new_feature_vector.append([int(row[0]), int(row[1]), int(row[2]), event_platform_var, event_timestamp_var, event_location_var, events.loc[row[0]]['uuid']])
    new_test_values.append([int(row[0]), int(row[1]), event_platform_var, event_timestamp_var, event_location_var])
    # print(events.loc[row[1]]['uuid'])

print(new_feature_vector[0])
print(new_test_values[0])
gc.collect()
print("size of the new vector")
print(sys.getsizeof(new_feature_vector))
    # print(row['display_id'], row['ad_id'], row['clicked'])

print(len(new_feature_vector))
print(len(test_values))

new_feature_vector = np.array(new_feature_vector).astype(np.int)
new_test_values = np.array(new_test_values).astype(np.int)
class_value_vector = np.array(class_value_vector).astype(np.int)

gnb = GaussianNB()

new_feature_vector=new_feature_vector.reshape(len(new_feature_vector), 5)
class_value_vector=class_value_vector.reshape(len(class_value_vector), 1)
new_test_values=new_test_values.reshape(len(new_test_values), 5)

print(new_test_values[0])
print(class_value_vector[0])
print(new_feature_vector[0])

gc.collect()


# Creating a linear regression object
regr = linear_model.LinearRegression()
# logReg = LogisticRegression()


# logReg.fit(new_feature_vector, class_value_vector)


regr.fit(new_feature_vector, class_value_vector)
gnb.fit(new_feature_vector,class_value_vector.ravel())

slope = regr.coef_[0][0]
intercept = regr.intercept_

print("y = %f + %f " %(intercept,slope))

print("Mean squared error: %.10f" % np.mean((regr.predict(new_feature_vector) -class_value_vector) ** 2))

i=0
for xt in new_test_values:
    if i > 999:
        break
    xp=np.array(xt).astype(np.int)
    xp=xp.reshape(1, 5)
    # xp=xp.reshape(-1, 1)
    print(xp)
    print(regr.predict(xp) - 80)
    # print(logReg.predict(xp))
    i=i+1


# page_view_sample = pd.read_csv("../input/page_views_sample.csv")
# page_view_sample.set_index('uuid')

# for row in new_feature_vector:
#     print(page_view_sample.loc[row[6]]['traffic_source'])


# print("reading page_views_sample")
# page_views = pd.read_csv("../input/page_views_sample.csv")
# page_views.set_index('uuid')

# for row in new_feature_vector:
#     print(page_views.loc[row[7]['traffic_source']]
