
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

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

X_train.head()
y_train.head() # id, count

X_train.describe()
X_train.info()
X_train.columns # ['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered']

X_train['datetime'] = pd.to_datetime(X_train['datetime']) # 변환 했지만 이 부분을 제외
# datetime이므로 train_test_split 할때 앞에서 부터 자르기

# season -> category // nominal
# holiday -> category // binary
# workingday -> cateogry // holiday 와 반대이므로 제외
# weather -> cateogry // nominal
# temp -> numeric //
# atemp -> numeric
# humidity -> numeric
# windspeed -> numeric
# casual -> numeric
# registered -> numeric
#X_train[X_train.columns[10]].value_counts()

column_cat = ['season', 'holiday', 'weather'] # workingday 는 holiday와 반대이므로 drop
column_num = ['temp', 'humidity', 'windspeed', 'casual', 'registered'] # temp와 atemp의 연관성이 너무 높으므로 atemp drop

X_train[column_cat + column_num].corr()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train[column_num])
X_test_scaled = scaler.transform(X_test[column_num])

X_train_scaled = pd.DataFrame(X_train_scaled, columns=column_num, index=X_train[column_num].index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=column_num, index=X_test[column_num].index)

X_train_cat = pd.get_dummies(X_train[column_cat], drop_first=True)
X_test_cat = pd.get_dummies(X_test[column_cat], drop_first=True)

X_train_merge = pd.concat([X_train_scaled, X_train_cat], axis=1)
X_test_merge = pd.concat([X_test_scaled, X_test_cat], axis=1)

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor

voting_reg = VotingRegressor([('xgb', XGBRegressor()), ('rf', RandomForestRegressor())])
voting_reg.fit(X_train_merge, y_train['count'].values)

y_predict = voting_reg.predict(X_test_merge)
result = pd.DataFrame({'id':y_test['id'], 'count':y_predict})
result.to_csv('result.csv', index=False)

#채점
from sklearn.metrics import mean_squared_error
print(np.sqrt(mean_squared_error(y_predict, y_test['count'])))