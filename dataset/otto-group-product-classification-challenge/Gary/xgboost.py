import pandas as pd
import os
import xgboost as xgb
import operator
from matplotlib import pylab as plt
from sklearn import preprocessing

# import data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
sample = pd.read_csv('../input/sampleSubmission.csv')

# map Class_n to n-1
#class_range = range(1, 10)
#class_dict = {}
#for n in class_range:
#    class_dict['Class_{}'.format(n)] = n-1
#train['target'] = train['target'].map(class_dict)

# drop ids and get labels
labels = train.target.values
labels = preprocessing.LabelEncoder().fit_transform(labels)
train = train.drop(["id", "target"], axis=1)
features=list(train.columns[0:])
test = test.drop("id", axis = 1)

# train a xgboost classifier
params = {"objective": "multi:softprob", "eval_metric":"mlogloss", "num_class": 9}
train_xgb = xgb.DMatrix(train, labels)
test_xgb  = xgb.DMatrix(test)
trainRound = 100
gbm = xgb.train(params, train_xgb, trainRound)
pred = gbm.predict(test_xgb)

# create a feature map
outfile = open('xgb.fmap', 'w')
i = 0
for feat in features:
    outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    i = i + 1
outfile.close()

# plot feature importance
importance = gbm.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))
df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()
plt.figure()
df.plot()
df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 20))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
plt.gcf().savefig('feature_importance_xgb.png')

# create submission file
pred = pd.DataFrame(pred, index=sample.id.values, columns=sample.columns[1:])
pred.to_csv('prediction.csv', index_label='id')
