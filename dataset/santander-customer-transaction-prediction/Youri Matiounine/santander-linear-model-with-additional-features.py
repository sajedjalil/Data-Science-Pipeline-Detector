import pandas as pd
import numpy as np
import gc
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import norm, rankdata


# Load data
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# Merge test/train datasets into a single one and separate unneeded columns
target = train_df.pop('target')
len_train = len(train_df)
merged_df = pd.concat([train_df, test_df])
ID = merged_df.pop('ID_code')[len_train:]
del test_df, train_df
gc.collect()



# Add more features
for col in merged_df.columns:
    # Normalize the data, so that it can be used in norm.cdf(), as though it is a standard normal variable
    merged_df[col] = ((merged_df[col] - merged_df[col].mean()) / merged_df[col].std()).astype('float32')

    # Square
    merged_df[col+'_s'] = merged_df[col] * merged_df[col]

    # Cube
    merged_df[col+'_c'] = merged_df[col] * merged_df[col] * merged_df[col]

    # 4th power
    merged_df[col+'_q'] = merged_df[col] * merged_df[col] * merged_df[col] * merged_df[col]

    # Cumulative percentile (not normalized)
    merged_df[col+'_r'] = rankdata(merged_df[col]).astype('float32')

    # Cumulative normal percentile
    merged_df[col+'_n'] = norm.cdf(merged_df[col]).astype('float32')
    


# Normalize the data. Again.
for col in merged_df.columns:
    merged_df[col] = ((merged_df[col] - merged_df[col].mean()) / merged_df[col].std()).astype('float32')



# Do linear regression
clf = LinearRegression().fit(merged_df.iloc[:len_train], target)
preds = clf.predict(merged_df.iloc[:len_train])
print('AUC: ', roc_auc_score(target, preds) )
print('R2: ', clf.score(merged_df.iloc[:len_train], target) )



# Write submission file
sub_preds = clf.predict(merged_df.iloc[len_train:])
out_df = pd.DataFrame({'ID_code': ID, 'target': sub_preds.astype('float32')})
out_df.to_csv('submission.csv', index=False)