import pandas as pd 
import numpy as np
import sklearn as skl
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
sample_submission = pd.read_csv("../input/sample_submission.csv")

drops = ["Id","Response"]

df_all = train.append(test)

#factorize categorical variable
df_all["Product_Info_2"] = pd.factorize(df_all["Product_Info_2"])[0]

# Use -1 for any others
df_all.fillna(-1, inplace=True)

# fix the dtype on the label column
df_all['Response'] = df_all['Response'].astype(int)

train = df_all[df_all['Response']>0].copy()
test = df_all[df_all['Response']<0].copy()

target_vars = [col for col in train.columns if col not in drops]

rf = RandomForestClassifier(n_estimators=500, random_state=0)

rf_fit = rf.fit(train[target_vars], train["Response"])

preds = rf_fit.predict(test[target_vars])

preds_sub = pd.DataFrame({"Id": test['Id'].values, "Response": preds})
preds_sub = preds_sub.set_index('Id')
preds_sub.to_csv('rf_submission.csv')

