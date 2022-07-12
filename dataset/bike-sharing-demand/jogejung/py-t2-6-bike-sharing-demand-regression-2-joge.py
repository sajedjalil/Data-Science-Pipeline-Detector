# 시험환경 세팅 (코드 변경 X)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def exam_data_load(df, target, id_name="", null_name=""):
    if id_name == "":
        df = df.reset_index().rename(columns={"index": "id"})
        id_name = 'id'
    else:
        id_name = id_name
    
    if null_name != "":
        df[df == null_name] = np.nan
    
    X_train, X_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=2021)
    y_train = X_train[[id_name, target]]
    X_train = X_train.drop(columns=[id_name, target])
    y_test = X_test[[id_name, target]]
    X_test = X_test.drop(columns=[id_name, target])
    return X_train, X_test, y_train, y_test 
    
df = pd.read_csv("../input/bike-sharing-demand/train.csv")
    
X_train, X_test, y_train, y_test = exam_data_load(df, target='count')#, id_name='Id')

# print(X_train.head())
# print(y_train.head())

X_train_len = len(X_train)
X_all = pd.concat([X_train,X_test],axis = 0)

# print(X_all.info())
# print(X_all["datetime"])
# print(X_all.head())
# print(X_all.tail())
# print(X_all.columns)

# print(pd.to_datetime(X_all["datetime"]))
Day_time = X_all[["datetime"]]
X_all = X_all.drop(["datetime"],axis = 1)
X_all_1 = pd.get_dummies(X_all)

# X_all_1 = pd.concat([X_all_1,Day_time],axis = 1)
print(X_all_1.head())

from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
X_all_2 = SS.fit_transform(X_all_1)



X_train_2 = X_all_2[:X_train_len]
X_test_2 = X_all_2[X_train_len:]

from xgboost import XGBRegressor
model = XGBRegressor()

model.fit(X_train_2,y_train["count"])

print(model.score(X_train_2,y_train["count"]))

pred = model.predict(X_test_2)

print(model.score(X_test_2,y_test["count"]))
