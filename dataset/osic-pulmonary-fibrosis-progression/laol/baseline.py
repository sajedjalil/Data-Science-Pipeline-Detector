import pandas as pd
from sklearn.preprocessing import LabelEncoder

weeks_range = range(-12, 134)

test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
sample_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')
# test.SmokingStatus = LabelEncoder().fit_transform(test.SmokingStatus)
# df.Sex = LabelEncoder().fit_transform(df.Sex)
smoking_status = pd.get_dummies(test.SmokingStatus)
del test['SmokingStatus']
# df.SmokingStatus = LabelEncoder().fit_transform(df.SmokingStatus)
test = pd.concat([test, smoking_status], axis=1)
# df['FullFVC'] = df.FVC / (df.Percent / 100)
# train_columns = ["FVC", "Percent", "Age", "Sex", "SmokingStatus",, 
#                  "Weeks", "Currently smokes", "Ex-smoker", "Never smoked"]

for i in range(len(test)):
    try:
        sample_df.loc[sample_df['Patient_Week'].str.contains(test.Patient[i]), 'FVC'] = \
        test.FVC[i] *  0.9427272727272727 +  test["Ex-smoker"].values[i] * (-24)
    except:
        pass

sample_df['Confidence'] = 250

sample_df.to_csv('submission.csv', index=False)

print(1)

