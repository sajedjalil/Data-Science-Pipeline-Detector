#the1owl's notebook in script form
from multiprocessing import Pool, cpu_count
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import fbeta_score
import xgboost as xgb
import pandas as pd
import numpy as np
import glob, cv2
import random

random.seed(1)
np.random.seed(1)

def get_features(path):
    img = cv2.imread(path)
    hist = cv2.calcHist([cv2.imread(path,0)],[0],None,[256],[0,256])
    m, s = cv2.meanStdDev(img)
    img = cv2.resize(img, (20, 20), cv2.INTER_LINEAR)
    img = np.append(img.flatten(), m.flatten())
    img = np.append(img, s.flatten())
    img = np.append(img, hist.flatten()) #/ 255
    return [path, img]

def normalize_img(paths):
    imf_d = {}
    p = Pool(cpu_count())
    ret = p.map(get_features, paths)
    for i in range(len(ret)):
        imf_d[ret[i][0]] = ret[i][1]
    ret = []
    fdata = [imf_d[f] for f in paths]
    fdata = np.array(fdata, dtype=np.uint8)
    return fdata

in_path = '../input/'
train = pd.read_csv(in_path + 'train.csv')
train['path'] = train['image_name'].map(lambda x: in_path + 'train-jpg/' + x + '.jpg')
y = train['tags'].str.get_dummies(sep=' ')
xtrain = normalize_img(train['path']); print('train...')

test_jpg = glob.glob(in_path + 'test-jpg/*')
test = pd.DataFrame([[p.split('/')[3].replace('.jpg',''),p] for p in test_jpg])
test.columns = ['image_name','path']
xtest = normalize_img(test['path']); print('test...')

etr = ExtraTreesRegressor(n_estimators=18, max_depth=12, n_jobs=-1, random_state=1)
etr.fit(xtrain, y); print('etr fit...')

train_pred = etr.predict(xtrain)
train_pred[train_pred > 0.20] = 1
train_pred[train_pred < 1] = 0
print(fbeta_score(y,train_pred,beta=2, average='samples'))

pred1 = etr.predict(xtest); print('etr predict...')
etr_test = pd.DataFrame(pred1, columns=y.columns)
etr_test['image_name'] =  test[['image_name']]

xgb_train = pd.DataFrame(train[['path']], columns=['path'])
xgb_test = pd.DataFrame(test[['image_name']], columns=['image_name'])
print('xgb fit...')
for c in y.columns:
    model = xgb.XGBClassifier(n_estimators=10, max_depth=7, seed=1)
    model.fit(xtrain, y[c])
    xgb_train[c] = model.predict_proba(xtrain)[:, 1]
    xgb_test[c] = model.predict_proba(xtest)[:, 1]
    print(c)

train_pred = xgb_train[y.columns].values
train_pred[train_pred >0.20] = 1
train_pred[train_pred < 1] = 0
print(fbeta_score(y,train_pred,beta=2, average='samples')) 
print('xgb predict...')

xgb_test.columns = [x+'_' if x not in ['image_name'] else x for x in xgb_test.columns]
blend = pd.merge(etr_test, xgb_test, how='left', on='image_name')

for c in y.columns:
    blend[c] = (blend[c] * 0.45)  + (blend[c+'_'] * 0.55)

blend = blend[etr_test.columns]

tags = []
for r in blend[y.columns].values:
    r = list(r)
    tags.append(' '.join([j[1] for j in sorted([[r[i],y.columns[i]] for i in range(len(y.columns)) if r[i]>.20], reverse=True)]))

test['tags'] = tags
test[['image_name','tags']].to_csv('submission_blend.csv', index=False)
test.head()