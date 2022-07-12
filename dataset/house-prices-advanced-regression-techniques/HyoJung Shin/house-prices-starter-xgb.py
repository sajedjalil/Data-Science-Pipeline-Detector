# ## 목표 : 주택 가격 예측 -> 수치형 변수 예측, 회귀 중 XGBRegressor 사용

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


# ## 0. 목표
    # 주택 가격 예측 : 회귀 - RandomForestRegressor, XGBRegressor -> 평가 RMSE
    
# ## 1. 라이브러리(7개), 데이터 로드
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# 시험용
# X_train = pd.reda_csv('data/X_train.csv')
# X_test = pd.reda_csv('data/X_test.csv')
# y_train = pd.reda_csv('data/y_train.csv')

df = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
X_train, X_test, y_train, y_test = exam_data_load(df, target='SalePrice', id_name='Id')

X_train.shape, X_test.shape, y_train.shape, y_test.shape

# ## 2. EDA : X_train, y_train, X_test 수행
    # info() 
        # 컬럼 개수 : X는 79개, y는 2개
X_train.info() 
X_test.info()
y_train.info()
        # 컬럼 유형
            # 범주형, 수치형(범주 표현된 수치형 컬럼 있는지 확인) : object, int, float 있음, y='Sales'는 int로 되어 있음
        # 결측치 : 있는데, 컬럼이 너무 많아서 뭘 사용해야 할지?
    # describe()
        # 이상치 (범위, 형태)
            # mean, std 차이 : YearBuilt, YearRemodAdd, 3SsnPorch, PoolArea, MiscVal, MoSold, YrSold 차이 많이 남
            # 수치형 변수 값 단위 (로그 적용 필요한지 확인) : 차이 큼, StandardScaler 적용
X_train.describe() 
X_test.describe()
y_train.describe()
    # tail()
        # 실제 값 확인 : 잘못된 수치형 타입(나이인데 소수점 등)
X_train.tail() 
X_test.tail()
y_train.tail()
    # corr()
        # 상관관계 높은 변수
X_train.corr() 
X_test.corr()

        
# ## 3. 데이터 전처리(7개) : X_train, y_train 수행 
    # 이상치 처리 (범위, 형태) : 이번에는 컬럼이 너무 많아서 진행하지 않음
    # 결측치 처리 : 평균으로 대치
num_cols = X_train.select_dtypes(exclude='object').columns
num_cols
X_train[num_cols] = X_train[num_cols].fillna(X_train[num_cols].mean)
X_test[num_cols] = X_test[num_cols].fillna(X_test[num_cols].mean)
X_test.isna().sum()
    # 수치형 컬럼 처리 : 이번에는 수치형만 활용해 봄(회귀여도 범주형 원핫인코딩으로 변환해서 사용함)
        # 수치형 컬럼 외 삭제
    # 불필요한 컬럼 삭제 : 범주형 모두 삭제
del_cols =  X_train.select_dtypes(include='object').columns
X_train = X_train.drop(columns=del_cols)
X_test = X_test.drop(columns=del_cols)
X_train.shape, X_test.shape

    # y 변수 처리
        # 변수 값 형태 : 바로 사용할 수 없으면 추출하거나 형변환 하는 등 처리 : 바로 사용 가능
        # 변수 왜도 확인 : 정렬 후 value_counts()로 왜도 확인해 치우친 경우 로그 변환(np.log1p) -> 결과 예측 후 복원(np.expm1)
y_train['SalePrice'].sort_values(ascending=True).value_counts() # 오른쪽 꼬리 긴 형태로 로그 변환 필요
y_train['SalePrice'].hist()
y_train['SalePrice'] = np.log1p(y_train['SalePrice'])
y_train['SalePrice'].sort_values(ascending=True).value_counts()
y_train['SalePrice'].hist()
        # y 변수만 할당
y_train = y_train['SalePrice']
X_test
    # 범주형 컬럼 처리 : OneHotEncoding : 이번엔 범주형 사용하지 않음
    
    # 수치형 컬럼 처리 : SandardScaler, log 적용이랑 둘 다 하나?
        # 컬럼별 단위 차이 나면 StandardScaler 적용 (X_train만 fit_transform, X_test는 transform만)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled =  scaler.transform(X_test)
    
    # 데이터 나누기 (tr, val) : X_train_scaled, X_test_scaled
X_tr, X_val, y_tr, y_val = train_test_split(X_train_scaled, y_train, test_size=0.15, shuffle=True, random_state=2021)
X_tr.shape, X_val.shape, y_tr.shape, y_val.shape

# ## 4. 모델 학습 및 평가 : RMSE 평가 (np.sqrt(MSE)), 적은 게 좋음
    # 기본 모델 학습
# model_rf = RandomForestRegressor()
# model_rf.fit(X_tr, y_tr)
# pred_rf = model_rf.predict(X_val)
# print('model_rf : ', np.sqrt(mean_squared_error(pred_rf, y_val))) # 0.117

# model_xgb = XGBRegressor()
# model_xgb.fit(X_tr, y_tr)
# pred_xgb = model_xgb.predict(X_val)
# print('model_xgb :', np.sqrt(mean_squared_error(pred_xgb, y_val))) # 0.123

#     # 하이퍼 파라미터 적용
# estimators = [10, 50, 100]
# max_depths = [10, 50, 100]

# for estimator in estimators:
#     for max_depth in max_depths:
#         model1 = RandomForestRegressor(n_estimators=estimator, max_depth=max_depth)
#         model1.fit(X_tr, y_tr)
#         pred1 = model1.predict(X_val)
#         print('RF : ', estimator, max_depth, ' / ', np.sqrt(mean_squared_error(pred1, y_val)))
        
# for estimator in estimators:
#     for max_depth in max_depths:
#         model2 = XGBRegressor(n_estimators=estimator, max_depth=max_depth)
#         model2.fit(X_tr, y_tr)
#         pred2 = model2.predict(X_val)
#         print('XGB : ', estimator, max_depth, ' / ', np.sqrt(mean_squared_error(pred2, y_val)))
    
    # 최적 모델 선정 : RF 100 100 0.116
model_rf = RandomForestRegressor(n_estimators=100, max_depth=100)
model_rf.fit(X_tr, y_tr)
pred_rf = model_rf.predict(X_val)
print('model_rf : ', np.sqrt(mean_squared_error(pred_rf, y_val))) # 0.117

    
# ## 5. X_test 예측 및 파일 제출 : X_test_scaled
    # X_test_scaled 예측 후 복원(np.expm1)
pred_result = model_rf.predict(X_test_scaled)
#pred_result = np.expm1(pred_result)

    # 파일 생성 (y_test[변수], pred_result)
output = pd.DataFrame({'Id': y_test['Id'], 'SalePrice': pred_result})
output.to_csv('수험번호.csv', index=False)

    # 파일 확인
import os

path = os.getcwd()
result = pd.read_csv(path +'/수험번호.csv')
result.tail()

print(len(X_test_scaled), len( y_test['SalePrice']))

print(X_test_scaled.shape, y_test['SalePrice'].shape)

# ## 채점
print('output = ', np.sqrt(mean_squared_error(pred_result, y_test['SalePrice'])))

############################## v1 코드    
    
# # ## 시험 문제 세팅 후 데이터 업로드 하기   

# import pandas as pd
    
# df = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
# X_train, X_test, y_train, y_test = exam_data_load(df, target='SalePrice', id_name='Id')

# X_train.shape, X_test.shape, y_train.shape, y_test.shape


# # ## EDA : y는 SalePrice 컬럼

# X_train.info()
# X_test.info()

# X_train.isnull().sum() # 결측치 있음 -> 처리 필요
# y_test.isnull().sum() # 결측치 없음

# # 참고 : 컬럼이 많은데 생략되지 않고 모두 출력하도록 표시
# pd.set_option("display.max_columns", 100)
# display(X_train.head(3))
# display(X_test.head(3))

# # 둘 다 오른쪽 꼬리 긴 형태
# y_train['SalePrice'].hist()
# y_test['SalePrice'].hist()


# # ## 데이터 전처리 : 이상치, 결측치 처리 및 target 변수 y 지정, 수치형 데이터(회귀)만 구분

# # 회귀 적용 시 수치형 데이터만 사용하도록 준비, y 변수 준비
# X_train = X_train.select_dtypes(exclude=['object'])
# X_test = X_test.select_dtypes(exclude=['object'])
# target = y_train['SalePrice']

# # 이상치 처리 가능 - IQR 사용

# # 결측치 처리 : SimpleImputer 사용 (특정값으로 대치 : impute)
# # default 조건 : missing_values=nan, strategy='mean'
# # X_train에는 fit, transform 모두 사용
# # X_test에는 trainsform만 사용
# from sklearn.impute import SimpleImputer

# imp = SimpleImputer()
# X_train = imp.fit_transform(X_train)
# X_test = imp.transform(X_test)

# # 데이터 나누기
# from sklearn.model_selection import train_test_split

# X_tr, X_val, y_tr, y_val = train_test_split(X_train, target, test_size=0.15, random_state=2021)
# X_tr.shape, X_val.shape, y_tr.shape, y_val.shape


# # ## 모델 학습하기

# from xgboost import XGBRegressor
# from sklearn.metrics import mean_squared_error

# from sklearn.ensemble import RandomForestRegressor 

# model = RandomForestRegressor()
# model.fit(X_tr, y_tr)
# pred = model.predict(X_val)


# # # default 설정
# # model = XGBRegressor()
# # model.fit(X_tr, y_tr, verbose=False) # verbose는 함수 과정 출력 옵션, 0(False) 은 출력 안 함 1(True)은 자세히, 2는 함축적인 정보만 출력
# # pred = model.predict(X_val)

# # 하이퍼 파라미터 설정
# model2 = XGBRegressor(n_estimators=100, max_depth=4, colsample_bytree=0.9)
# model2.fit(X_tr, y_tr)
# pred2 = model2.predict(X_val)


# # RMSE : MSE에 root(sqrt) 취한 값, 회귀 예측 시 오차 확인 방법
# from sklearn.metrics import mean_squared_error
# import numpy as np

# def rmse(y, y_pred):
#     return np.sqrt(mean_squared_error(y, y_pred))

# print("Train RMSE = ", str(rmse(y_val, pred)))
# print("Train RMSE V2 = ", str(rmse(y_val, pred2)))


# # ## 결과 CSV 출력 : X_test로 예측해서 결과 추정
# pred = model.predict(X_test)
# output = pd.DataFrame({'Id': y_test['Id'], 'SalePrice': pred})
# output.to_csv('수험번호.csv', index=False)


# # ## 결과 채점 : V2가 성능 더 우수, 하이퍼 파라미터 사용해서 최적화 수행 (n_estimators=100, max_depth=4, colsample_bytree=0.9)
# print("Test RMSE : " + str(rmse(y_test['SalePrice'], pred)))

# pred2 = model2.predict(X_test)
# print("Test RMSE V2 : " + str(rmse(y_test['SalePrice'], pred2)))


###########################  예제 코드

# df = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
# X_train, X_test, y_train, y_test = exam_data_load(df, target='SalePrice', id_name='Id')

# X_train.shape, X_test.shape, y_train.shape, y_test.shape

# # # Data Load & Simple EDA

# import pandas as pd

# X_train.shape, X_test.shape

# pd.set_option("display.max_columns", 100)
# display(X_train.head(3))
# display(X_test.head(3))

# y_train['SalePrice'].hist()

# y_test['SalePrice'].hist()

# X_train.isnull().sum().sort_values(ascending=False)[:20]

# X_test.isnull().sum().sort_values(ascending=False)[:20]

# X_train.info()

# # # Preprocessing

# X_train = X_train.select_dtypes(exclude=['object'])
# X_test = X_test.select_dtypes(exclude=['object'])
# target = y_train['SalePrice']

# from sklearn.impute import SimpleImputer

# imp = SimpleImputer()
# X_train = imp.fit_transform(X_train)
# X_test = imp.transform(X_test)

# from sklearn.model_selection import train_test_split
# X_tr, X_val, y_tr, y_val = train_test_split(X_train, target, test_size=0.15, random_state=2022)
# X_tr.shape, X_val.shape, y_tr.shape, y_val.shape

# # # Model

# from xgboost import XGBRegressor

# model = XGBRegressor()
# model.fit(X_tr, y_tr, verbose=False)
# pred = model.predict(X_val)

# from sklearn.metrics import mean_squared_error

# def rmsle(y, y_pred):
#     return np.sqrt(mean_squared_error(y, y_pred))

# print("RMSLE : " + str(rmsle(y_val, pred)))

# # # Simple Preprocessing

# X_train, X_test, y_train, y_test = exam_data_load(df, target='SalePrice', id_name='Id')

# idx1 = y_train['SalePrice'].quantile(0.005)>y_train['SalePrice']
# idx2 = y_train['SalePrice'].quantile(0.995)<y_train['SalePrice']

# y_train = y_train[~(idx1 + idx2)]
# X_train = X_train[~(idx1 + idx2)]

# X_train = X_train.select_dtypes(exclude=['object'])
# X_test = X_test.select_dtypes(exclude=['object'])
# target = y_train['SalePrice']

# imp = SimpleImputer()
# X_train = imp.fit_transform(X_train)
# X_test = imp.transform(X_test)

# X_tr, X_val, y_tr, y_val = train_test_split(X_train, target, test_size=0.15, random_state=20222)

# model = XGBRegressor()
# model.fit(X_tr, y_tr)
# pred = model.predict(X_val)

# print("RMSLE : " + str(rmsle(y_val, pred)))

# # ## Simple Tuning

# X_train, X_test, y_train, y_test = exam_data_load(df, target='SalePrice', id_name='Id')

# idx1 = y_train['SalePrice'].quantile(0.005)>y_train['SalePrice']
# idx2 = y_train['SalePrice'].quantile(0.995)<y_train['SalePrice']

# y_train = y_train[~(idx1 + idx2)]
# X_train = X_train[~(idx1 + idx2)]

# X_train = X_train.select_dtypes(exclude=['object'])
# X_test = X_test.select_dtypes(exclude=['object'])
# target = y_train['SalePrice']

# imp = SimpleImputer()
# X_train = imp.fit_transform(X_train)
# X_test = imp.transform(X_test)

# X_tr, X_val, y_tr, y_val = train_test_split(X_train, target, test_size=0.15, random_state=20222)

# model = XGBRegressor(n_estimators=100, max_depth=4, colsample_bytree=0.9)
# model.fit(X_tr, y_tr)
# pred = model.predict(X_val)

# print("RMSLE : " + str(rmsle(y_val, pred)))



# # # # Simple Preprocessing

# X_train, X_test, y_train, y_test = exam_data_load(df, target='SalePrice', id_name='Id')

# idx1 = y_train['SalePrice'].quantile(0.005)>y_train['SalePrice']
# idx2 = y_train['SalePrice'].quantile(0.995)<y_train['SalePrice']

# y_train = y_train[~(idx1 + idx2)]
# X_train = X_train[~(idx1 + idx2)]

# X_train = X_train.select_dtypes(exclude=['object'])
# X_test = X_test.select_dtypes(exclude=['object'])
# target = y_train['SalePrice']

# imp = SimpleImputer()
# X_train = imp.fit_transform(X_train)
# X_test = imp.transform(X_test)

# X_tr, X_val, y_tr, y_val = train_test_split(X_train, target, test_size=0.15, random_state=20222)


# model = XGBRegressor()
# model.fit(X_tr, y_tr)
# pred = model.predict(X_val)

# print("RMSLE : " + str(rmse(y_val, pred)))
# len(y_val), len(pred)

# # ## Simple Tuning

# X_train, X_test, y_train, y_test = exam_data_load(df, target='SalePrice', id_name='Id')

# idx1 = y_train['SalePrice'].quantile(0.005)>y_train['SalePrice']
# idx2 = y_train['SalePrice'].quantile(0.995)<y_train['SalePrice']

# y_train = y_train[~(idx1 + idx2)]
# X_train = X_train[~(idx1 + idx2)]

# X_train = X_train.select_dtypes(exclude=['object'])
# X_test = X_test.select_dtypes(exclude=['object'])
# target = y_train['SalePrice']

# imp = SimpleImputer()
# X_train = imp.fit_transform(X_train)
# X_test = imp.transform(X_test)

# X_tr, X_val, y_tr, y_val = train_test_split(X_train, target, test_size=0.15, random_state=20222)

# model = XGBRegressor(n_estimators=100, max_depth=4, colsample_bytree=0.9)
# model.fit(X_tr, y_tr)
# pred = model.predict(X_val)

# print("RMSLE : " + str(rmse(y_val, pred)))

# # # Predict & to CSV

# pred = model.predict(X_test)
# output = pd.DataFrame({'Id': y_test['Id'], 'SalePrice': pred})
# output.head()
# output.to_csv("000000.csv", index=False)

# # # 결과 체점

# pred = model.predict(X_test)
# print("RMSLE : " + str(rmsle(y_test['SalePrice'], pred)))