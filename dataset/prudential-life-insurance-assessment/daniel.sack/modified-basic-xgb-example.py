import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split

print('Importing data...')
df_train = pd.read_csv('../input/train.csv', index_col='Id')
df_test = pd.read_csv('../input/test.csv', index_col='Id')

# encode categorical columns (i.e. encode Product_Info_2)
for col in df_test.columns:
    if df_test[col].dtype == np.dtype('O'):
        le = LabelEncoder()
        l1 = df_train[col].unique().tolist()
        l2 = df_test[col].unique().tolist()
        le.fit(l1+l2)
        df_train[col] = le.transform(df_train[col])
        df_test[col] = le.transform(df_test[col])

# set up data
dtrain = xgb.DMatrix(
	data=df_train.drop('Response', axis=1),
	label=df_train.Response,
	missing=np.nan
	)
evallist = [(dtrain, 'train')]

# set up and fit estimator
print('Fitting classifier...')
params = {
	'objective': 'reg:linear',
	'eval_metric': 'rmse',
	'silent': 1,
}
bst = xgb.train(params, dtrain, 200, evallist)

# generate predictions
dtest = xgb.DMatrix(data=df_test, missing=np.nan)
y_pred = bst.predict(dtest).round().astype(int)
out = pd.DataFrame({'Response': y_pred}, index=df_test.index)

# restrict to range of options
out.Response[out.Response<1] = 1
out.Response[out.Response>8] = 8

# write file
outfile = 'prediction.csv'
print('\nWriting %s predictions to %s' % (str(out.shape), outfile))
out.to_csv(outfile, index_label='Id')