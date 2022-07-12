# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing
import xgboost as xgb
import datetime, operator
#now = datetime.datetime.now()

input_folder = '../input/'

train = pd.read_csv(input_folder + 'train.csv')
test = pd.read_csv(input_folder + 'test.csv')
macro = pd.read_csv(input_folder + 'macro.csv')
id_test = test.id
train.sample(3)

sample = train.sample(frac=0.5)

price_per_sq = sample.price_doc / sample['full_sq']
price_per_sq = price_per_sq[ np.isinf(price_per_sq) == False ].mean()
# create the price_by_sq parameters by using the train data, cross validation will do the same with inner train and inner validation
train['price_by_sq'] = train['full_sq'] * price_per_sq
test['price_by_sq'] = test['full_sq'] * price_per_sq


# label encode categorical variables
print('converting categorical to numerical variables...')
categoricals = []
for f in test.columns:
    if test[f].dtype=='object':
        categoricals.append(f)
        print(f)
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[f].values.astype('str')) + list(test[f].values.astype('str')))
            
        train[f] = lbl.transform(list(train[f].values.astype('str')))
        test[f] = lbl.transform(list(test[f].values.astype('str')))


y_train = train["price_doc"]
x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)
x_test = test.drop(["id", "timestamp"], axis=1)

#best_features = pd.read_csv(input_folder + 'feature_importance_reynaldo.csv')
#x_train = x_train[ best_features.feature.iloc[ len(best_features) - 100: ]  ]
       
xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1

}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)


cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
    verbose_eval=50, show_stdv=False)
cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()

num_boost_rounds = len(cv_output)
model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round= num_boost_rounds)

#fig, ax = plt.subplots(1, 1, figsize=(8, 13))
#xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)

importance = model.get_fscore()
importance = sorted(importance.items(), key=operator.itemgetter(1))

#df = pd.DataFrame(importance, columns=['feature', 'fscore'])
#df['fscore'] = df['fscore'] / df['fscore'].sum()
#output.to_csv('feature_importance.csv', index=False)

y_predict = model.predict(dtest)
output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})
output.head()

output.to_csv('xgbSub.csv', index=False)

# Any results you write to the current directory are saved as output.