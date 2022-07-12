


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

seed = 1358
nanfill = -1000

# Train info preprocessing
trainInfo = pd.read_csv("../input/ItemInfo_train.csv")
# remove unacessary (for now) information
trainInfo = trainInfo.drop(['locationID', 'metroID', 'categoryID', 'attrsJSON'], axis = 1)
train = pd.read_csv("../input/ItemPairs_train.csv")
# merge ItemPairs and ItemInfo
train = pd.merge(pd.merge(train, trainInfo, how = 'inner', left_on = 'itemID_1', right_on = 'itemID'), trainInfo, how = 'inner', left_on = 'itemID_2', right_on = 'itemID')

# remove unacessary information
del trainInfo
train = train.drop(['itemID_1', 'itemID_2', 'generationMethod', 'itemID_x',  'itemID_y'], axis = 1)

print('Find distance of train pairs ...')
# create a column with the distance between pairs
train['dist'] = np.abs(train['lat_x']-train['lat_y'])+np.abs(train['lon_x']-train['lon_y'])
# remove latlon information
train = train.drop(['lat_x', 'lon_x', 'lat_y', 'lon_y'], axis = 1)

print('Find relative price difference of train pairs ...')
# create a column with the difference between prices
train['price_diff'] = np.abs(train['price_x']-train['price_y'])*1./np.min(train[['price_x','price_y']], axis =1)
# set null possitions on price_diff to zero
train.loc[(train.price_x.isnull()) & (train.price_y.isnull()),'price_diff'] = 0
# remove prices information
train = train.drop(['price_x', 'price_y'], axis = 1)

print('Find relative title difference of train pairs ...')
# create a column with the difference between titles
train['title_diff'] = train[['title_x', 'title_y']].apply(lambda x:(x[0]==x[1])+0.0, axis = 1)
# remove titles information
train = train.drop(['title_x', 'title_y'], axis = 1)

print('Find relative description difference of train pairs ...')
# create a column with the difference between descriptions
train['description_diff'] = train[['description_x', 'description_y']].apply(lambda x:(x[0]==x[1])+0.0, axis = 1)
# remove descriptions information
train = train.drop(['description_x', 'description_y'], axis = 1)

print('Find difference of number of images in train pairs ...')
train['images_array_x'] =  train['images_array_x'].apply(lambda x:len(x.split()) if isinstance(x, str) else 0)
train['images_array_y'] =  train['images_array_y'].apply(lambda x:len(x.split()) if isinstance(x, str) else 0)
# create a column with the difference between number of images
train['images_num_diff'] = train[['images_array_x', 'images_array_y']].apply(lambda x: abs(x[0]-x[1]), axis = 1)
# remove images information
train = train.drop(['images_array_x', 'images_array_y'], axis = 1)

# get labels
y = train.isDuplicate.values
train = train.drop('isDuplicate', axis = 1)
train = train.fillna(nanfill)
print(train.columns)

# Test info preprocessing
testInfo =  pd.read_csv("../input/ItemInfo_test.csv")
# remove unacessary (for now) information
testInfo = testInfo.drop(['locationID', 'metroID', 'categoryID', 'attrsJSON'], axis = 1)
test  = pd.read_csv("../input/ItemPairs_test.csv")
# merge ItemPairs and ItemInfo
test = pd.merge(pd.merge(test, testInfo, how = 'inner', left_on = 'itemID_1', right_on = 'itemID'), testInfo, how = 'inner', left_on = 'itemID_2', right_on = 'itemID')

# remove unacessary information
del testInfo
ids = test['id'].values
test = test.drop(['id', 'itemID_1', 'itemID_2', 'itemID_x',  'itemID_y'], axis = 1)

print('Find distance of test pairs ...')
# create a column with the distance between pairs
test['dist'] = np.abs(test['lat_x']-test['lat_y'])+np.abs(test['lon_x']-test['lon_y'])
# remove latlon information
test = test.drop(['lat_x', 'lon_x', 'lat_y', 'lon_y'], axis = 1)

print('Find relative price difference of test pairs ...')
# create a column with the difference between prices
test['price_diff'] = np.abs(test['price_x']-test['price_y'])*1./np.min(test[['price_x','price_y']], axis =1)
test.loc[(test.price_x.isnull()) & (test.price_y.isnull()),'price_diff'] = 0
# remove prices information
test = test.drop(['price_x', 'price_y'], axis = 1)

print('Find relative title difference of test pairs ...')
# create a column with the difference between titles
test['title_diff'] = test[['title_x', 'title_y']].apply(lambda x:(x[0]==x[1])+0.0, axis = 1)
# remove titles information
test = test.drop(['title_x', 'title_y'], axis = 1)

print('Find relative description difference of test pairs ...')
# create a column with the difference between descriptions
test['description_diff'] = test[['description_x', 'description_y']].apply(lambda x:(x[0]==x[1])+0.0, axis = 1)
# remove descriptions information
test = test.drop(['description_x', 'description_y'], axis = 1)

print('Find difference of number of images in test pairs ...')
test['images_array_x'] =  test['images_array_x'].apply(lambda x:len(x.split()) if isinstance(x, str) else 0)
test['images_array_y'] =  test['images_array_y'].apply(lambda x:len(x.split()) if isinstance(x, str) else 0)
# create a column with the difference between number of images
test['images_num_diff'] = test[['images_array_x', 'images_array_y']].apply(lambda x: abs(x[0]-x[1]), axis = 1)
# remove images information
test = test.drop(['images_array_x', 'images_array_y'], axis = 1)

# Save train and test information in a CSV file
train.to_csv('train.csv')
test.to_csv('test.csv')

# fill empty features
test = test.fillna(nanfill)
print(test.columns)

# scale the data
scaler = StandardScaler()
train = scaler.fit_transform(train.values)
test = scaler.transform(test.values)

np.random.seed(seed)

# shuffle the dataset
shflidx = np.random.permutation(train.shape[0])
train = train[shflidx, :]
y = y[shflidx]


#clf = LogisticRegression()
clf = RandomForestClassifier()

# train the model
clf.fit(train, y)


preds = clf.predict_proba(test)[:,1]

# save the results
sub = pd.DataFrame()
sub['id'] = ids
sub['probability'] = preds
sub.to_csv('submission.csv', index = False)



