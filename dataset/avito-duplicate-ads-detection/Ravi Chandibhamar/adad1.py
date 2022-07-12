
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

seed = 1358
nanfill = -1000

trainInfo = pd.read_csv("../input/ItemInfo_train.csv")
trainInfo = trainInfo.drop(['locationID', 'metroID', 'categoryID', 'attrsJSON'], axis = 1)
train = pd.read_csv("../input/ItemPairs_train.csv")
train = pd.merge(pd.merge(train, trainInfo, how = 'inner', left_on = 'itemID_1', right_on = 'itemID'), trainInfo, how = 'inner', left_on = 'itemID_2', right_on = 'itemID')

del trainInfo
train = train.drop(['itemID_1', 'itemID_2', 'generationMethod', 'itemID_x',  'itemID_y'], axis = 1)

print('Find distance of train pairs ...')
train['dist'] = np.abs(train['lat_x']-train['lat_y'])+np.abs(train['lon_x']-train['lon_y'])
train = train.drop(['lat_x', 'lon_x', 'lat_y', 'lon_y'], axis = 1)

print('Find relative price difference of train pairs ...')
train['price_diff'] = np.abs(train['price_x']-train['price_y'])*1./np.min(train[['price_x','price_y']], axis =1)
train.loc[(train.price_x.isnull()) & (train.price_y.isnull()),'price_diff'] = 0
train = train.drop(['price_x', 'price_y'], axis = 1)

print('Find relative title difference of train pairs ...')
train['title_diff'] = train[['title_x', 'title_y']].apply(lambda x:(x[0]==x[1])+0.0, axis = 1)
train = train.drop(['title_x', 'title_y'], axis = 1)

print('Find relative description difference of train pairs ...')
train['description_diff'] = train[['description_x', 'description_y']].apply(lambda x:(x[0]==x[1])+0.0, axis = 1)
train = train.drop(['description_x', 'description_y'], axis = 1)

print('Find difference of number of images in train pairs ...')
train['images_array_x'] =  train['images_array_x'].apply(lambda x:len(x.split()) if isinstance(x, str) else 0)
train['images_array_y'] =  train['images_array_y'].apply(lambda x:len(x.split()) if isinstance(x, str) else 0)
train['images_num_diff'] = train[['images_array_x', 'images_array_y']].apply(lambda x: abs(x[0]-x[1]), axis = 1)
train = train.drop(['images_array_x', 'images_array_y'], axis = 1)

y = train.isDuplicate.values
with open('y.csv', 'wb') as pickle_file:
   pickle.dump(y, pickle_file)

train = train.drop('isDuplicate', axis = 1)
train = train.fillna(nanfill)
print(train.columns)

testInfo =  pd.read_csv("../input/ItemInfo_test.csv")
testInfo = testInfo.drop(['locationID', 'metroID', 'categoryID', 'attrsJSON'], axis = 1)
test  = pd.read_csv("../input/ItemPairs_test.csv")
test = pd.merge(pd.merge(test, testInfo, how = 'inner', left_on = 'itemID_1', right_on = 'itemID'), testInfo, how = 'inner', left_on = 'itemID_2', right_on = 'itemID')

del testInfo
ids = test['id'].values
test = test.drop(['id', 'itemID_1', 'itemID_2', 'itemID_x',  'itemID_y'], axis = 1)

print('Find distance of test pairs ...')
test['dist'] = np.abs(test['lat_x']-test['lat_y'])+np.abs(test['lon_x']-test['lon_y'])
test = test.drop(['lat_x', 'lon_x', 'lat_y', 'lon_y'], axis = 1)

print('Find relative price difference of test pairs ...')
test['price_diff'] = np.abs(test['price_x']-test['price_y'])*1./np.min(test[['price_x','price_y']], axis =1)
test.loc[(test.price_x.isnull()) & (test.price_y.isnull()),'price_diff'] = 0
test = test.drop(['price_x', 'price_y'], axis = 1)

print('Find relative title difference of test pairs ...')
test['title_diff'] = test[['title_x', 'title_y']].apply(lambda x:(x[0]==x[1])+0.0, axis = 1)
test = test.drop(['title_x', 'title_y'], axis = 1)

print('Find relative description difference of test pairs ...')
test['description_diff'] = test[['description_x', 'description_y']].apply(lambda x:(x[0]==x[1])+0.0, axis = 1)
test = test.drop(['description_x', 'description_y'], axis = 1)

print('Find difference of number of images in test pairs ...')
test['images_array_x'] =  test['images_array_x'].apply(lambda x:len(x.split()) if isinstance(x, str) else 0)
test['images_array_y'] =  test['images_array_y'].apply(lambda x:len(x.split()) if isinstance(x, str) else 0)
test['images_num_diff'] = test[['images_array_x', 'images_array_y']].apply(lambda x: abs(x[0]-x[1]), axis = 1)
test = test.drop(['images_array_x', 'images_array_y'], axis = 1)
train.to_csv('train.csv')
test.to_csv('test.csv')

test = test.fillna(nanfill)
print(test.columns)

scaler = StandardScaler()
train = scaler.fit_transform(train.values)
test = scaler.transform(test.values)

np.random.seed(seed)

shflidx = np.random.permutation(train.shape[0])
train = train[shflidx, :]
y = y[shflidx]


clf = LogisticRegression(C=1e6,penalty='l1',max_iter=5000)
clf.fit(train, y)
preds = clf.predict_proba(test)[:,1]
sub = pd.DataFrame()
sub['id'] = ids
sub['probability'] = preds
sub.to_csv('submission.csv', index = False)