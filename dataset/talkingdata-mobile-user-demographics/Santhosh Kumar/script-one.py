# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#from datetime import datetime
from time import strptime
#import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
#import seaborn as sn
#import matplotlib.pyplot as mplt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

def parse_timestamp(time_stamp):
    return strptime(str(time_stamp), "%Y-%m-%d %H:%M:%S")

gd_train = pd.read_csv('../input/gender_age_train.csv')
gd_test = pd.read_csv('../input/gender_age_test.csv')
phone = pd.read_csv('../input/phone_brand_device_model.csv')

df = gd_train.merge(phone, on = 'device_id', how = 'left')
del gd_train
tdf = gd_test.merge(phone, on = 'device_id', how = 'left')
del phone
del gd_test

#print(df.info())
#print(df.head())

df = df.merge(
                pd.read_csv('../input/events.csv',
                            usecols = ['timestamp', 'device_id'],
                            #parse_dates = [1],
                            #date_parser = parse_timestamp
                            ), 
                on = 'device_id', 
                how = 'left')

tdf = tdf.merge(
                pd.read_csv('../input/events.csv',
                            usecols = ['timestamp', 'device_id']),
                on = 'device_id',
                how = 'left')

df['gd'] = -1
df.gd = df.gender.apply(lambda x: 1 if x == "M" else 0)
#df.timestamp = pd.to_datetime(df.timestamp)
#tdf.timestamp = pd.to_datetime(tdf.timestamp)
phone_le = LabelEncoder()
phone_le.fit(df.phone_brand)
device_le = LabelEncoder()
device_le.fit(df.device_model)
group_le = LabelEncoder()
group_le.fit(df.group)
df['phone'] = phone_le.transform(df.phone_brand)
tdf['phone'] = phone_le.fit_transform(tdf.phone_brand)
df['device'] = device_le.transform(df.device_model)
tdf['device'] = device_le.fit_transform(tdf.device_model)
df['grp'] = group_le.transform(df.group)

df = df.drop(['phone_brand', 'device_model', 'gender', 'group'], axis = 1)
tdf = tdf.drop(['phone_brand', 'device_model'], axis = 1)
print(df.info())
print(df.head())
#print(len(df.timestamp.unique())) # 497664

#train_index, test_index = train_test_split(df.index.values, train_size = 0.8, test_size = 0.2)
#train_df = df.iloc[train_index]
#test_df = df.iloc[test_index]
features = ['phone', 'device']
target = ['gd']
lg_reg_gd = LogisticRegression()
lg_reg_gd.fit(df[features], df[target])
gender = lg_reg_gd.predict(tdf[features])
#print(predicted)

target = ['age']
lg_reg_age = LogisticRegression()
lg_reg_age.fit(df[features], df[target])
age =  lg_reg_age.predict(tdf[features])


grp_train = pd.DataFrame({"age": df.age,
                        "gender": df.gd,
                        "grp": df.grp,
                        "device_id": df.device_id})

grp_test = pd.DataFrame({"age": age,
                        "gender": gender,
                        "device_id": tdf.device_id})

del df
del tdf

f_features = ['age', 'gender']
f_target = ['grp']

lg_reg_grp = LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs')
lg_reg_grp.fit(grp_train[f_features], grp_train[f_target])
#final_predictions = lg_reg_grp.predict(grp_test[f_features])
final_predictions = lg_reg_grp.predict_proba(grp_test[f_features])

#predictions_grp = group_le.inverse_transform(final_predictions)
header_cols = [group_le.inverse_transform(cls)
                        for cls in lg_reg_grp.classes_]
submission = pd.DataFrame(
                            final_predictions, 
                            columns = header_cols)

submission = submission.round(3)
submission['device_id'] = grp_test.device_id
print(submission.info())
print(submission.head())
submission.to_csv('submission_early_morn.csv', columns = ['device_id'].extend(header_cols), index = False)

#print(len(grp_test.device_id), len(final_predictions)) 2035414 2035414
#total = len(test_df)
#wrong = len(test_df.grp != predicted)
#correct = total - wrong

#print("Model score: ", lg_reg.score(test_df[features], predicted))
#print("Total: ", total)
#print("Correct prediction: ", correct)
#print("Wrong prediction: ", wrong)
#print("Percentage of correctness: ", correct / total * 100)
