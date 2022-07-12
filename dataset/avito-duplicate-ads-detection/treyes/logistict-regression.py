import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance as dist
import random
import cv2

seed = 1358
nanfill = -1000

def compararHistograma(arr1, arr2):
    index = {}
    results = []
    print("img")
    print(arr1)
    if arr1 != nanfill and arr2 != nanfill:
        arr1 = str(arr1).split(',')
        arr2 = str(arr2).split(',')
        
        hist1 = obtenerHistograma(random.choice(arr1), index)
        
        for item in arr2:
            index[item] = obtenerHistograma(item)
    
        for img2 in arr2:
            if img2 in index:
                d = dist.cityblock(hist1, index[img2])
                results.append(d)
                
        print(index)
        
        if len(results) > 0:
            diferencia = np.min(results)
        else:
            diferencia = -1
    else:
        diferencia = -1

    return diferencia

def obtenerHistograma(item, index):
    item = item.strip()
    cadena = '../input/Images_'+item[-2:-1]+'/'+str(int(item[-2:]))+'/'+item+'.jpg'

    img = cv2.imread(cadena)
    if img is not None:
        hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = hist.flatten()
        cv2.normalize(hist, hist, 1.0, 0)

        return hist
    else:
        return 0
        
trainInfo = pd.read_csv("../input/ItemInfo_train.csv")
trainInfo = trainInfo.drop(['locationID', 'metroID', 'categoryID', 'attrsJSON'], axis = 1)
train = pd.read_csv("../input/ItemPairs_train.csv", nrows= 50)
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
train = train.fillna(nanfill)
print('Find difference in images in train pairs ...')
train['images_diff'] = train[['images_array_x', 'images_array_y']].apply(lambda x: compararHistograma(x[0], x[1]), axis = 1)
train = train.drop(['images_array_x', 'images_array_y'], axis = 1)

y = train.isDuplicate.values
train = train.drop('isDuplicate', axis = 1)

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
test = test.fillna(nanfill)
print('Find difference of number of images in test pairs ...')
test['images_diff'] = test[['images_array_x', 'images_array_y']].apply(lambda x: compararHistograma(x[0], x[1]), axis = 1)
test = test.drop(['images_array_x', 'images_array_y'], axis = 1)
train.to_csv('train.csv')
test.to_csv('test.csv')


print(test.columns)

scaler = StandardScaler()
train = scaler.fit_transform(train.values)
test = scaler.transform(test.values)

np.random.seed(seed)

shflidx = np.random.permutation(train.shape[0])
train = train[shflidx, :]
y = y[shflidx]


clf = LogisticRegression()
clf.fit(train, y)
preds = clf.predict_proba(test)[:,1]
sub = pd.DataFrame()
sub['id'] = ids
sub['probability'] = preds
sub.to_csv('submission.csv', index = False)