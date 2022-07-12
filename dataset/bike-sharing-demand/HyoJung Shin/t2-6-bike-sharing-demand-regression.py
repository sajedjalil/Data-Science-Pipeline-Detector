# ## 자전거 수요 예측 
# 매 시간마다 렌탈된 자전거 수량 예측 

# 컬럼 구성
    # datetime - hourly date + timestamp  
    # season -  1 = spring, 2 = summer, 3 = fall, 4 = winter 
    # holiday - whether the day is considered a holiday
    # workingday - whether the day is neither a weekend nor holiday
    # weather - 1: Clear, Few clouds, Partly cloudy, Partly cloudy
    # 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
    # 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
    # 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog 
    # temp - temperature in Celsius
    # atemp - "feels like" temperature in Celsius
    # humidity - relative humidity
    # windspeed - wind speed
    # casual - number of non-registered user rentals initiated
    # registered - number of registered user rentals initiated
    # count - number of total rentals


# ## 라이브러리, 데이터 불러오기

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

X_train = pd.read_csv('../input/bike-sharing-demand/train.csv')
X_test = pd.read_csv('../input/bike-sharing-demand/test.csv')
submission = pd.read_csv('../input/bike-sharing-demand/sampleSubmission.csv', engine='python') # 제출용


# ## EDA

# datetime은 object형임, 예측 단위가 시간인데 날짜로 입력 -> 시간 분리 필요
# 그 외 int, float64형
# 범주형 변수 있음, season, weather
# X_train, X_test 컬럼 개수 다름, X_train에만 등록정보 추가되어 있음 -> 삭제 필요
X_train.info() # 결측치 없음
X_test.info() # 결측치 없음 
X_train.tail()
X_train.describe() # 자동으로 집계된 내용일 것이므로 이상치 처리하지 않음

# ## 데이터 전처리 : 결측치, 이상치 처리, 변수 종류 및 원핫인코딩 필요 여부, y 변수 및 정규화 등의 처리, 데이터 나누기

# 필요한 컬럼 구분 : datetime에서 hour 분리 (분리 후 원래 컬럼은 삭제)
X_train['datetime'] = pd.to_datetime(X_train['datetime'])
X_train['hour'] = X_train['datetime'].dt.hour
X_train = X_train.drop(columns=['datetime'])
X_test['datetime'] = pd.to_datetime(X_test['datetime'])
X_test['hour'] = X_test['datetime'].dt.hour
X_test = X_test.drop(columns=['datetime'])
type(X_test['hour'][0]) # int형이지만 범주형 변수 -> 원핫인코딩 수행 필요

# 참고 : str 사용 시 datetime 형 변환 없이 자릿수로 추출할 수 있음
#X_train['hour'] = X_train['datetime'].str[-8:-6]
#X_test['hour'] = X_test['datetime'].str[-8:-6]

# 범주형 변수 onehotEncoding 수행 : season, weather, hour
X_train = pd.get_dummies(X_train, columns=['season', 'weather', 'hour'])
X_test = pd.get_dummies(X_test, columns=['season', 'weather', 'hour'])
X_train.columns

# 불필요한 컬럼 삭제
X_train = X_train.drop(columns=['casual', 'registered'])
X_train.columns

# y 변수 지정 : count
y_train = X_train[['count']]
X_train = X_train.drop(columns=['count'])
X_train.columns

# 데이터 전처리 하기 전에 정규분포 형태 확인 후 처리
y_train['count'].value_counts() # 왼쪽으로 치우치고 오른쪽 꼬리 긴 형태 -> 정규화 수행 (np.log1p)
y_train['count'] = np.log1p(y_train['count'])

# 데이터 나누기
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, shuffle=True, test_size=0.15, random_state=2021)
X_tr.shape, X_val.shape, y_tr.shape, y_val.shape


# ## 모델 학습 및 평가하기 : rmse_xgb 선정

model_rf = RandomForestRegressor()
model_rf.fit(X_tr, y_tr)
pred_rf = model_rf.predict(X_val)

model_xgb = XGBRegressor()
model_xgb.fit(X_tr, y_tr)
pred_xgb = model_xgb.predict(X_val)

# 모델 평가 : RMSE 사용
rmse_rf = np.sqrt(mean_squared_error(y_val, pred_rf))
print('RandomForestRegressor MSE', rmse_rf) # 0.419

rmse_xgb = np.sqrt(mean_squared_error(y_val, pred_xgb))
print('XGBRegressor MSE', rmse_xgb) # 0.403



# ## 하이퍼 파라미터 적용 : default XGB가 가장 성능 우수

# n_estimators = [10, 50, 100]
# max_depths = [10, 50, 100]

# print("============= RandomForestRegressor")  

# for n_estimator in n_estimators: # 50, 50, 0.422
#     for max_depth in max_depths:
#         print(n_estimator, max_depth)
#         model = RandomForestRegressor(n_estimators=n_estimator, max_depth=max_depth)
#         model.fit(X_tr, y_tr)
#         pred = model.predict(X_val)
#         print(np.sqrt(mean_squared_error(y_val, pred)))
        
# print("============= XGB")        

# for n_estimator in n_estimators: # 50, 10, 0.417
#     for max_depth in max_depths:
#         print(n_estimator, max_depth)
#         model = XGBRegressor(n_estimators=n_estimator, max_depth=max_depth)
#         model.fit(X_tr, y_tr)
#         pred = model.predict(X_val)
#         print(np.sqrt(mean_squared_error(y_val, pred)))


# ## 선정한 모델로 X_test 예측 후 결과 파일 생성하기

pred_y = model_xgb.predict(X_test)

# y 변수에 np.log1p 적용했으므로 예측 결과에 np.expm1 적용해서 죄종 결과 도출
pred_result = np.expm1(pred_y)

X_test.columns

# 결과 파일
output = pd.DataFrame({'count': pred_result})
output.head()
output.to_csv("수험번호.csv", index=False)




###################### 예제 코드


# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_log_error
# from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor

# train = pd.read_csv('../input/bike-sharing-demand/train.csv')
# test = pd.read_csv('../input/bike-sharing-demand/test.csv')

# train['hour'] = train['datetime'].str[11:13]
# test['hour'] = test['datetime'].str[11:13]

# train_x_raw = train.drop(['datetime', 'casual', 'registered', 'count'], axis=1)
# train_y = train['count']

# test_x_raw = test.drop(['datetime'], axis=1)
# test_y = test['datetime']

# total_x = pd.concat([train_x_raw, test_x_raw])
# total_x[['season', 'weather']] = total_x[['season', 'weather']].astype('str')
# total_x = pd.get_dummies(total_x, ['hour', 'season', 'weather'])

# train_x = total_x[:10886]
# test_x = total_x[10886:]

# def rmsle(y1, y2):
#     return np.sqrt(np.mean(np.square(np.log1p(y1) - np.log1p(y2))))

# tr_x, val_x, tr_y, val_y = train_test_split(train_x, train_y, test_size=0.7, random_state=777)

# model = XGBRegressor(n_estimators=50, max_depth=50)
# model.fit(tr_x, tr_y)
# print(model.score(tr_x, tr_y))
# print(model.score(val_x, val_y))
# print(rmsle(model.predict(val_x), val_y)) # 0.5943
# # print(np.sqrt(mean_squared_log_error(model.predict(val_x), val_y))) #음수 오류 발생

# result = test_y.copy()
# result['count'] = model.predict(test_x)
# result.to_csv('0000.csv', index=False)

# # 결과값(RMSLE) : 0.59

# # n_estimators = [10, 50, 100]
# # max_depths = [10, 50, 100]

# # for n_estimator in n_estimators: # 100, 50, 0.78, 0.60
# #     for max_depth in max_depths:
# #         print(n_estimator, max_depth)
# #         model = RandomForestRegressor(n_estimators=n_estimator, max_depth=max_depth)
# #         model.fit(tr_x, tr_y)
# #         print(model.score(tr_x, tr_y))
# #         print(model.score(val_x, val_y))
# #         print(np.sqrt(mean_squared_log_error(model.predict(val_x), val_y)))
        
# # for n_estimator in n_estimators:    #50, 50, 0.77, 0.59
# #     for max_depth in max_depths:
# #         print(n_estimator, max_depth)
# #         model = XGBRegressor(n_estimators=n_estimator, max_depth=max_depth)
# #         model.fit(tr_x, tr_y)
# #         print(model.score(tr_x, tr_y))
# #         print(model.score(val_x, val_y))
# #         print(np.sqrt(mean_squared_log_error(np.abs(model.predict(val_x)), val_y)))