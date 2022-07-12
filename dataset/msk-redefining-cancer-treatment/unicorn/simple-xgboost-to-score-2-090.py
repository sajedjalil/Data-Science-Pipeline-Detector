import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures,LabelEncoder
import xgboost as xgb
import matplotlib.pyplot  as plt

from sklearn.metrics import accuracy_score,roc_auc_score


xgb_params = {
    'eta': 0.05,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'multi:softprob',
    'silent': 1,
    'seed' : 0
}


input_df = pd.read_csv('../input/training_variants')

test_df = pd.read_csv('../input/test_variants')

test_index = test_df ['ID']



le = LabelEncoder()
input_df['Gene'] =  le.fit_transform(input_df['Gene'])
input_df['Variation'] =  le.fit_transform(input_df['Variation'])
target = le.fit_transform(input_df['Class'])

input_df = input_df.drop(['ID','Class'],axis=1)
test_df = test_df.drop(['ID'],axis=1)


# poly = PolynomialFeatures(3)
#
# input_df_narray = poly.fit_transform(input_df)
#
# target_feature_names = ['x'.join(['{}^{}'.format(pair[0],pair[1]) for pair in tuple if pair[1]!=0]) for tuple in [zip(input_df.columns,p) for p in poly.powers_]]
# output_df = pd.DataFrame(input_df_narray, columns = target_feature_names)
#
#
# output_df = np.log(output_df)

print(set(target))

num_class = len(set(target))

xgb_params['num_class'] = num_class
xgtrain =  xgb.DMatrix(input_df, target)

test_df['Gene'] =  le.fit_transform(test_df['Gene'])
test_df['Variation'] =  le.fit_transform(test_df['Variation'])

xgtest =  xgb.DMatrix(test_df)

cvresult = xgb.train( xgb_params,  xgtrain )


prdt = cvresult.predict(xgtest)


""" Submission """
submission = pd.DataFrame(prdt)

print(submission)

submission['id'] = test_index
submission.columns = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9', 'id']
#submission.to_csv("../submit/my_submission.csv",index=False)