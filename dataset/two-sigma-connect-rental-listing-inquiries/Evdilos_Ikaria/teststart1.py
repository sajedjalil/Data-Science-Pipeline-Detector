
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy import sparse

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
from sklearn import model_selection, preprocessing, ensemble
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
pd.set_option('display.max_columns', None)

df_train = pd.read_json(open("../input/train.json", "r"))
target_num_map = {'high':0, 'medium':1, 'low':2}
yy = np.array(df_train['interest_level'].apply(lambda x: target_num_map[x]))

df_test = pd.read_json(open("../input/test.json", "r"))
lista=np.array(df_test["listing_id"])
print(df_train.shape)
print(df_test.shape)

#COLUMNS....
#'bathrooms', 'bedrooms', 'building_id', 'created', 'description',
#       'display_address', 'features', 'interest_level', 'latitude',
#       'listing_id', 'longitude', 'manager_id', 'photos', 'price',
#       'street_address']

df_train["num_photos"] = df_train["photos"].apply(len)
df_train["num_features"] = df_train["features"].apply(len)
df_train["num_description_words"] = df_train["description"].apply(lambda x: len(x.split(" ")))
df_train["created"] = pd.to_datetime(df_train["created"])
df_train["created_year"] = df_train["created"].dt.year
df_train["created_month"] = df_train["created"].dt.month
df_train["created_day"] = df_train["created"].dt.day

df_test["num_photos"] = df_test["photos"].apply(len)
df_test["num_features"] = df_test["features"].apply(len)
df_test["num_description_words"] = df_test["description"].apply(lambda x: len(x.split(" ")))
df_test["created"] = pd.to_datetime(df_test["created"])
df_test["created_year"] = df_test["created"].dt.year
df_test["created_month"] = df_test["created"].dt.month
df_test["created_day"] = df_test["created"].dt.day

num_feats = ["bathrooms", "bedrooms", "latitude", "longitude", "price",
             "num_photos", "num_features", "num_description_words",
             "created_year", "created_month", "created_day"]
             
categorical = ["display_address", "manager_id", "building_id", "street_address"]
for f in categorical:
        if df_train[f].dtype=='object':
            #print(f)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(df_train[f].values) + list(df_test[f].values))
            df_train[f] = lbl.transform(list(df_train[f].values))
            df_test[f] = lbl.transform(list(df_test[f].values))
            num_feats.append(f)


df_train['features'] = df_train["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
df_test['features'] = df_test["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
#print(df_train["features"].head())

tfidf = CountVectorizer(stop_words='english', max_features=200)
tr_sparse = tfidf.fit_transform(df_train["features"])
te_sparse = tfidf.transform(df_test["features"])
#===================================================================================
df_train = df_train[num_feats]
df_test =  df_test[num_feats]
print(df_train.shape, df_test.shape)
print ('FINAL COLUMNS...',num_feats)
X = df_train.copy()
test=df_test.copy()
#==========================================================================
#XGB Section...
param = {}
param['objective'] = 'multi:softprob'
param['eta'] = 0.1
param['max_depth'] = 5#6
param['silent'] = 1
param['num_class'] = 3
param['eval_metric'] = "mlogloss"
param['min_child_weight'] = 1
param['subsample'] = 0.7
param['colsample_bytree'] = 0.7
param['seed'] = 0
num_rounds = 10

plst = list(param.items())

#==========================================================================
#CV...

X_train, X_val, y_train, y_val = train_test_split(X, yy, test_size=0.33)

clf = RandomForestClassifier(n_estimators=10)
clf.fit(X_train, y_train)
pred_test_rf = clf.predict_proba(X_val)

xgtrain = xgb.DMatrix(X_train, y_train)
xgtest = xgb.DMatrix(X_val, label=y_val)
watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)

pred_test_xgb = model.predict(xgtest)
#================================================
y_val_pred1=(0*pred_test_rf + 1*pred_test_xgb)
score=log_loss(y_val, y_val_pred1)
print (' *** score...RF 0  XGB 1',score)

y_val_pred2=(0.2*pred_test_rf + 0.8*pred_test_xgb)
score=log_loss(y_val, y_val_pred2)
print (' *** score...0.2  0.8',score)

y_val_pred3=(0.1*pred_test_rf + 0.9*pred_test_xgb)
score=log_loss(y_val, y_val_pred3)
print (' *** score...0.1  0.9',score)

#================================================
#Game

pred_test_rf = clf.predict_proba(test)

xgtrain = xgb.DMatrix(X, label=yy)
xgtest = xgb.DMatrix(test)
pred_test_xgb = model.predict(xgtest)

y=(0.1*pred_test_rf+0.9*pred_test_xgb)
#================================================
#Output
sub = pd.DataFrame()
sub["listing_id"] = lista
ff0=np.zeros(len(y))
ff1=np.zeros(len(y))
ff2=np.zeros(len(y))

for rr in range(len(y)):
    ff0[rr] = y[rr][0]
    ff1[rr] = y[rr][1]
    ff2[rr] = y[rr][2]


sub['high']=ff0
sub['medium']=ff1  
sub['low']=ff2  

sub.to_csv("submission_rf.csv", index=False)





