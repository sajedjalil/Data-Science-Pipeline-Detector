import numpy as np 
import pandas as pd 
from subprocess import check_output
import xgboost as xgb
import sklearn
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
import csv
from keras.utils import np_utils
from sklearn.cross_validation import StratifiedKFold 
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation

def save_result(result,ID,preds):
    print ("save result")
    with open(result, 'w') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerow(["ID", "TARGET" ])
        data = zip(ID,preds)
        a.writerows(data)



# Read data
print ("Loading the data")
ID =pd.read_csv("../input/test.csv")['ID']
df_train = pd.read_csv("../input/train.csv", index_col='ID')
feature_cols = list(df_train.columns)
feature_cols.remove("TARGET")
df_test = pd.read_csv("../input/test.csv", index_col='ID')

# Split up the data
X_all = df_train[feature_cols]
y_all = df_train["TARGET"]
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3, random_state=5, stratify=y_all)

# Get top features from xgb model
model = xgb.XGBRegressor(
    learning_rate=0.1,
    n_estimators=1000,
    max_depth=5,
    min_child_weight=9,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=5
)

print ("Start training using XG boost")
# Train cv
xgb_param = model.get_xgb_params()
dtrain = xgb.DMatrix(X_train.values, label=y_train.values, missing=np.nan)
cv_result = xgb.cv(
    xgb_param,
    dtrain,
    num_boost_round=model.get_params()['n_estimators'],
    nfold=5,
    metrics=['auc'],
    early_stopping_rounds=50)
best_n_estimators = cv_result.shape[0]
model.set_params(n_estimators=best_n_estimators)

# Train model
model.fit(X_train, y_train, eval_metric='auc')

# Predict training data
y_hat_train = model.predict(X_train)

# Predict test data
y_hat_test = model.predict(X_test)

# Print model report:
print("\nModel Report of XGboost")
print("best n_estimators: {}".format(best_n_estimators))
print("AUC Score (Train): %f" % roc_auc_score(y_train, y_hat_train))
print("AUC Score (Test) : %f" % roc_auc_score(y_test,  y_hat_test))


print ("Get the important feature")
# Get important features
feat_imp = list(pd.Series(model.booster().get_fscore()).sort_values(ascending=False).index)

# Even out the targets
df_train_1 = df_train[df_train["TARGET"] == 1]
df_train_0 = df_train[df_train["TARGET"] == 0].head(df_train_1.shape[0])
df_train = df_train_1.append(df_train_0)

print ("Feature for the train set")
# Scale data
X_all = df_train[feat_imp].copy(deep=True)
y_all = df_train["TARGET"]
X_all[feat_imp] = sklearn.preprocessing.scale(X_all, axis=0, with_mean=True, with_std=True, copy=True)
X_all['TARGET'] = y_all


df_test = df_test[feat_imp].copy(deep=True)
df_test[feat_imp] = sklearn.preprocessing.scale(df_test, axis=0, with_mean=True, with_std=True, copy=True)


def build_model():
    layer_1 = 400
    layer_2 = 200
    model = Sequential()
    model.add(Dense(input_dim=train.shape[1], 
                    output_dim=layer_1, 
                    init='uniform',
                    activation='tanh'))

    model.add(Dense(input_dim=layer_1,
                    output_dim=layer_2,
                    init='uniform',
                    activation='tanh'))

    model.add(Dense(input_dim=layer_2,
                    output_dim=1,
                    init='uniform',
                    activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam' , metrics=['accuracy'])
    return model



print ('Start of Learning using Keras')
train = X_all
test = df_test
target = train['TARGET']
train = train.drop('TARGET',axis=1)


print ("Start cross validation")

visibletrain = blindtrain = train
n = 5 # number of folds in strattified cv
kfolder=StratifiedKFold(target, n_folds= n,shuffle=False, random_state=42)
index=0
num_rounds = 560
avg_auc = 0
for train_index, test_index in kfolder:
    print('Fold:', index)
    visibletrain = train.iloc[train_index].values
    blindtrain = train.iloc[test_index].values
    visibletest = target.iloc[train_index].values
    blindtest = target.iloc[test_index].values

    model = build_model()    
    model.fit(visibletrain, visibletest, nb_epoch=30, batch_size=50)
    valid_preds = model.predict_proba(blindtrain, verbose=0)[:,0]
    roc = roc_auc_score(blindtest, valid_preds)
    print("AUC Score:", roc)
    avg_auc += roc
    index+=1

print ("Average AUC score : ", avg_auc/n)

train = train.values
test = test.values
target = target.values

model = build_model()
model.fit(train, target, nb_epoch=30, batch_size=50)
train_pred = model.predict_proba(train)[:,0]
pred = model.predict_proba(test)[:,0]

print ("Training AUC",roc_auc_score(target, train_pred) )


save_result("keras.csv", ID , pred)