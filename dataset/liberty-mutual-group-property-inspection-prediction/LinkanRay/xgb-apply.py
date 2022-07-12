import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing 

train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
hazard = train_df["Hazard"]
data_train = train_df.drop(['Hazard','Id'], axis=1)
data_test = test_df.drop(['Id'], axis=1)

xgb_params = {"objective": "reg:linear", "eta": 0.01, "max_depth": 8, "seed": 42, "silent": 1}
num_rounds = 1000

for i in range(data_train.shape[1]):
    lbl_enc = preprocessing.LabelEncoder()
    if data_train[data_train.columns[i]].dtype != 'int64':
        data_train[data_train.columns[i]] = lbl_enc.fit_transform(data_train[data_train.columns[i]])




for i in range(data_test.shape[1]):
    lbl_enc = preprocessing.LabelEncoder()
    if data_test[data_test.columns[i]].dtype != 'int64':
        data_test[data_test.columns[i]] = lbl_enc.fit_transform(data_test[data_test.columns[i]])



model = RandomForestRegressor(max_depth=10, random_state=0).fit(data_train,hazard)
predicted = model.predict(data_test)


submission = pd.DataFrame({"Id": test_df['Id'], "Hazard": predicted})
submission = submission.set_index('Id')
submission.to_csv('submit_xtreereg.csv')
