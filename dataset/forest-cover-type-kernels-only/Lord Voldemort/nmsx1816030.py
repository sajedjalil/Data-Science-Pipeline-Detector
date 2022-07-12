import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# 读取训练数据集和测试数据集
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")

# 处理Distance_To_Hydrology
df_train['Distance_To_Hydrology'] = \
df_train.apply(lambda x: abs((x['Horizontal_Distance_To_Hydrology']**2\
                        + x['Vertical_Distance_To_Hydrology']**2)**0.5), axis=1)

df_test['Distance_To_Hydrology'] = \
df_test.apply(lambda x: abs((x['Horizontal_Distance_To_Hydrology']**2\
                        + x['Vertical_Distance_To_Hydrology']**2)**0.5), axis=1)

# 处理Hillshade
df_train['Hillshade_avg'] = (df_train['Hillshade_9am']+df_train['Hillshade_Noon']+df_train['Hillshade_3pm']).mean()

df_test['Hillshade_avg'] = (df_train['Hillshade_9am']+df_train['Hillshade_Noon']+df_train['Hillshade_3pm']).mean()

# 处理Wilderness_Area
df_train['Wilderness_Area'] = df_train['Wilderness_Area1'] * 1 + df_train['Wilderness_Area2'] * 2\
                              + df_train['Wilderness_Area3'] * 3 + df_train['Wilderness_Area4'] * 4

df_test['Wilderness_Area'] = df_test['Wilderness_Area1'] * 1 + df_test['Wilderness_Area2'] * 2\
                              + df_test['Wilderness_Area3'] * 3 + df_test['Wilderness_Area4'] * 4
# 处理Soil_Type
def handle_Soil_Type(x):
    Soil_Type = 0
    for i in range(1, 41):
        if x['Soil_Type'+str(i)] != 0:
            Soil_Type += x['Soil_Type'+str(i)]*i
    return Soil_Type

df_train['Soil_Type'] = df_train.apply(lambda x: handle_Soil_Type(x), axis=1)

df_test['Soil_Type'] = df_test['Soil_Type1']*1 + df_test['Soil_Type2']*2 + df_test['Soil_Type3'] * 3\
                        + df_test['Soil_Type4']*4 + df_test['Soil_Type5']*5 + df_test['Soil_Type6'] * 6\
                        + df_test['Soil_Type7']*7 + df_test['Soil_Type8']*8 + df_test['Soil_Type9'] * 9\
                        + df_test['Soil_Type10']*10 + df_test['Soil_Type11']*11 + df_test['Soil_Type12'] * 12\
                        + df_test['Soil_Type13']*13 + df_test['Soil_Type14']*14 + df_test['Soil_Type15'] * 15\
                        + df_test['Soil_Type16']*16 + df_test['Soil_Type17']*17 + df_test['Soil_Type18'] * 18\
                        + df_test['Soil_Type19']*19 + df_test['Soil_Type20']*20 + df_test['Soil_Type21'] * 21\
                        + df_test['Soil_Type22']*22 + df_test['Soil_Type23']*23 + df_test['Soil_Type24'] * 24\
                        + df_test['Soil_Type25']*25 + df_test['Soil_Type26']*26 + df_test['Soil_Type27'] * 27\
                        + df_test['Soil_Type28']*28 + df_test['Soil_Type29']*29 + df_test['Soil_Type30'] * 30\
                        + df_test['Soil_Type31']*31 + df_test['Soil_Type32']*32 + df_test['Soil_Type33'] * 33\
                        + df_test['Soil_Type34']*34 + df_test['Soil_Type35']*35 + df_test['Soil_Type36'] * 36\
                        + df_test['Soil_Type37']*37 + df_test['Soil_Type38']*38 + df_test['Soil_Type39'] * 39\
                        + df_test['Soil_Type40']*40

# 选定特征
selected_features = [
    'Elevation'
    , 'Aspect'
    , 'Slope'
    , 'Distance_To_Hydrology'
    , 'Horizontal_Distance_To_Roadways'
    , 'Hillshade_avg'
    , 'Horizontal_Distance_To_Fire_Points'
    , 'Wilderness_Area'
    , 'Soil_Type']

# 特征
X_train = df_train[selected_features]
X_test = df_test[selected_features]
y_train = df_train['Cover_Type']

# 标准化数据
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# etc

etc = ensemble.ExtraTreesClassifier(n_estimators=350)
etc.fit(X_train,y_train)
print("accuracy for etc:", cross_val_score(etc, X_train, y_train, cv=5).mean())
etc_y_predict = etc.predict(X_test)
etc_submission = pd.DataFrame({'Id': df_test['Id'], 'Cover_Type': etc_y_predict})
etc_submission.to_csv('submission.csv', index=False)