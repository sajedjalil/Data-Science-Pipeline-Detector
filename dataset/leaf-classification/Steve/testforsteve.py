import numpy
import pandas
import xgboost as xgb
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
#load train dataset
#dataframe = pandas.read_csv("C:/Users/sound/Desktop/Kaggle/Leaf Classfication/Leaf Classfication in XGBOOST/data/train.csv")
dataframe = pandas.read_csv("../input/train.csv")
dataset = dataframe.values
X = dataset[:,2:].astype(float)
scaler = StandardScaler().fit(X)
X = scaler.transform(X)
Y = dataset[:,1]
encoder = LabelEncoder()
le=encoder.fit(Y)
#encoded_Y = encoder.transform(Y)
#convert integers to dummy variables 
#dummy_y = np_utils.to_categorical(encoded_Y)

#load test dataset
#test = pandas.read_csv('C:/Users/sound/Desktop/Kaggle/Leaf Classfication/Leaf Classfication in XGBOOST/data/test.csv')
test = pandas.read_csv('../input/test.csv')
test_ids = test.pop('id')
x_test = test.values
scaler = StandardScaler().fit(x_test)
x_test = scaler.transform(x_test)

model = xgb.XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, silent=True, objective='binary:logistic', nthread=-1, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, seed=0, missing=None)
model.fit(X,Y)

yprob = model.predict_proba(x_test)
submission = pandas.DataFrame(yprob, index=test_ids, columns=le.classes_)
submission.to_csv('submission.csv')
