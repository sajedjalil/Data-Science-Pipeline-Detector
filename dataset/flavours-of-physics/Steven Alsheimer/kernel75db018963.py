# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import roc_auc_score

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from xgboost import XGBClassifier
exec(open("../input/eval-lepton-test/evaluation2.py").read())
print(os.listdir("../input/flavours-of-physics"))

XGBmodel = XGBClassifier(subsample=1.0,
                        nthread=4,
                        n_estimators = 300,
                        min_child_weight = 5,
                        max_depth = 5,
                        learning_rate = 0.1,
                        gamma = 0.5,
                        colsample_bytree = 0.6)

df_main = pd.read_csv('../input/flavours-of-physics/training.csv.zip')
pd.set_option('display.max_columns', None)
df_main = df_main.drop(['production','min_ANNmuon', 'id','mass','SPDhits','p1_p','p2_IPSig','p0_p','p2_pt','p1_IP',
                        'p2_IP','p0_track_Chi2Dof','CDF1', 'CDF2', 'CDF3','isolationb','isolationc'], axis=1)
df_main.head(3)

TrainingDF, TestDF = train_test_split(df_main, test_size = 0.2)

TrainingDF_Input = TrainingDF.copy()
TrainingDF_Input = TrainingDF_Input.drop(['signal'], axis=1)
TrainingDF_Output = TrainingDF['signal']

scalar = StandardScaler()

TrainingDF_Input = scalar.fit_transform(TrainingDF_Input)

TestDF_Input = TestDF.copy()
TestDF_Input = TestDF_Input.drop(['signal'], axis=1)
TestDF_Output = TestDF['signal']
TestDF_Input = scalar.transform(TestDF_Input)
XGBmodel.fit(TrainingDF_Input, TrainingDF_Output)
check_agreement = pd.read_csv('../input/flavours-of-physics/check_agreement.csv.zip')
features = []
check_agreement_1 = check_agreement.drop(['id', 'weight'], axis=1)
Check_0 = check_agreement_1[check_agreement_1['signal'].values == 0]
Check_1 = check_agreement_1[check_agreement_1['signal'].values == 1]
for col in df_main.columns:
    if col != 'signal' and col != 'mass':
        features.append(col)

check_A = scalar.transform(Check_0[features])
agreement_probs_0= (XGBmodel.predict_proba(check_A)[:,1])

check_B = scalar.transform(Check_1[features])
agreement_probs_1= (XGBmodel.predict_proba(check_B)[:,1])
ks = compute_ks(
            agreement_probs_0,
            agreement_probs_1,
            check_agreement[check_agreement['signal'] == 0]['weight'].values,
            check_agreement[check_agreement['signal'] == 1]['weight'].values)
print ('KS metric', ks, ks < 0.09)



check_correlation = pd.read_csv('../input/flavours-of-physics/check_correlation.csv.zip')
check_corr_1 = check_correlation.drop(['id'], axis=1)
check_Corr = scalar.transform(check_corr_1[features])
corr_probs = (XGBmodel.predict_proba(check_Corr)[:,1])


cvm = compute_cvm(corr_probs, check_correlation['mass'])
print ('CvM metric', cvm, cvm < 0.002)

df_main_2 = pd.read_csv('../input/flavours-of-physics/training.csv.zip')
train_evaluation_M = df_main_2[df_main_2['min_ANNmuon'] > 0.4]
train_evaluation = train_evaluation_M.drop(['id'], axis=1)
train_evaluation = scalar.transform(train_evaluation[features])
train_probs = XGBmodel.predict_proba(train_evaluation)[:,1]
train_eval = train_evaluation_M['signal']
AUC = roc_auc_truncated(train_eval, train_probs)
print ('AUC', AUC)


df_F_test = pd.read_csv('../input/flavours-of-physics/test.csv.zip')
df_F_test_M = df_F_test.drop(['id'],axis=1)
df_F_test_M = scalar.transform(df_F_test_M[features])
test_probs  = (XGBmodel.predict_proba(df_F_test_M)[:,1])
Final_submission = pd.DataFrame({"id": df_F_test["id"], "prediction": test_probs})
Final_submission.to_csv("xgboost_ML_SA_submission.csv", index=False)
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session